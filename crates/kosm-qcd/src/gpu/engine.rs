//! The GPU Monte Carlo engine: generates configurations, measures them, and
//! streams the results into [`Stats`] and the viewer.
//!
//! One cycle is `sep` compound updates and the plaquette, then per tab:
//! - the bare Polyakov loop (every tab);
//! - Wilson loops on spatially smeared links, time links untouched, so the
//!   static potential keeps the Wilson action's transfer matrix;
//! - the three-quark loops and their flux correlation after light 4D stout
//!   smoothing (flux tab), or topology after heavy 4D smoothing (vacuum tab).
//!
//! Cycles are queued as operations and encoded a few per frame under a work
//! budget that adapts to the frame time, so the GPU stays busy without
//! stalling the page. Each cycle's results come back in one asynchronous
//! readback; the next cycle's updates run while it is in flight.

use std::collections::VecDeque;

use eframe::egui_wgpu::wgpu;
use phyz_qft::su3::staircase;
use web_time::Instant;

use super::{Batch, GpuLattice, Kernels, Links, Readback, Reduce, groups};
use crate::field::FieldFile;
use crate::physics::{FluxSums, Stats, instanton_radii, lattice_spacing_fm};

/// What the current tab needs measured.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Measure {
    Vacuum,
    Flux,
    Temperature,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Settings {
    /// Spatial extent.
    pub size: usize,
    /// Temporal extent; small N_t is high temperature.
    pub nt: usize,
    pub beta: f64,
    pub n_or: usize,
    /// Compound updates between measured configurations.
    pub sep: usize,
    pub therm: usize,
    /// Spatial stout steps for Wilson loops; 4D stout steps before the flux
    /// measurement and before topology. All even.
    pub smear_spatial: usize,
    pub smear_flux: usize,
    pub smear_topo: usize,
    pub rho: f32,
    /// Time extent of the static three-quark loop.
    pub t_len: usize,
    pub seed: u64,
}

impl Settings {
    pub fn vacuum() -> Self {
        Self {
            size: 16,
            nt: 16,
            beta: 6.0,
            n_or: 3,
            sep: 1,
            therm: 60,
            smear_spatial: 16,
            smear_flux: 6,
            smear_topo: 30,
            rho: 0.1,
            t_len: 4,
            seed: 1,
        }
    }

    pub fn a_fm(&self) -> f64 {
        lattice_spacing_fm(self.beta)
    }

    /// Temperature T = 1 / (N_t a) in MeV.
    pub fn temperature_mev(&self) -> f64 {
        crate::physics::HBARC / (self.nt as f64 * self.a_fm())
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    /// A compound update; `measured` cycles are counted for display.
    Update,
    Plaquette,
    BeginSmoothing,
    StoutPair,
    SpatialPair,
    Polyakov,
    Wilson,
    Flux,
    Topology,
    Finish,
}

impl Op {
    fn cost(self, s: &Settings, ns: usize) -> f32 {
        match self {
            Op::Update => (1 + s.n_or) as f32,
            Op::StoutPair => 2.0,
            Op::SpatialPair => 1.6,
            Op::Wilson => 8.0,
            Op::Flux => 3.0 + ns as f32 / 1024.0,
            Op::Plaquette | Op::Topology => 1.6,
            Op::Polyakov | Op::BeginSmoothing | Op::Finish => 0.1,
        }
    }

    /// Whether the op copies into the readback buffer, which must not happen
    /// while the previous cycle's readback is mapped or pending.
    fn writes_readback(self) -> bool {
        matches!(
            self,
            Op::Plaquette | Op::Polyakov | Op::Wilson | Op::Flux | Op::Topology | Op::Finish
        )
    }
}

/// Where each measurement lands in the readback buffer, in f32s.
#[derive(Clone, Copy, Default)]
struct Layout {
    plaq: usize,
    poly: usize,
    wilson: usize,
    corr: usize,
    w: usize,
    fsum: usize,
    dens: usize,
    total: usize,
}

impl Layout {
    fn new(lat: &GpuLattice) -> Self {
        let g = groups(lat.n);
        let mut at = 0;
        let mut take = |len: usize| {
            let o = at;
            at += len;
            o
        };
        let plaq = take(g);
        let poly = take(2 * lat.ns);
        let wilson = take(lat.rmax * lat.tmax * g);
        let corr = take(2 * lat.ns);
        let w = take(g);
        let fsum = take(6 * g);
        let dens = take(2 * lat.n);
        Self {
            plaq,
            poly,
            wilson,
            corr,
            w,
            fsum,
            dens,
            total: at,
        }
    }
}

/// What the running cycle has queued a copy for.
#[derive(Clone, Copy, Default)]
struct Pending {
    poly: bool,
    wilson: bool,
    flux: bool,
    topo: bool,
    /// Quark placement the flux measurement used.
    quark_gen: u64,
}

pub struct Engine {
    k: Kernels,
    pub lat: GpuLattice,
    pub settings: Settings,
    pub measure: Measure,
    ops: VecDeque<Op>,
    pass_id: u32,
    seed: u32,
    rb: Readback,
    layout: Layout,
    /// What the cycle being encoded has copied, and what the readback in
    /// flight holds.
    building: Pending,
    inflight: Pending,
    awaiting: bool,
    /// The readback in flight belongs to a lattice that was restarted.
    discard: bool,
    pub therm_left: usize,
    pub stats: Stats,
    pub flux: FluxSums,
    /// Quark positions in lattice coordinates (x, y, z).
    pub quarks: [[i32; 3]; 3],
    junction: [i32; 3],
    quark_gen: u64,
    /// Latest topology snapshot (4D): fields "topo", "action".
    pub vacuum: Option<FieldFile>,
    pub vacuum_gen: u64,
    /// Latest local Polyakov loops (3D, phase-aligned): "Re P", "|P|".
    pub polyakov: Option<FieldFile>,
    /// Work units encoded per frame, adapted to the frame time.
    pub budget: f32,
    pub rate: f32,
    last_cycle: Option<Instant>,
}

impl Engine {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: Settings,
        measure: Measure,
    ) -> Self {
        let k = Kernels::new(device);
        let lat = GpuLattice::new(
            device,
            &k,
            [settings.nt, settings.size, settings.size, settings.size],
        );
        let layout = Layout::new(&lat);
        let rb = Readback::new(device, (layout.total * 4) as u64);
        let ns = lat.ns;
        let mut e = Self {
            k,
            lat,
            settings,
            measure,
            ops: VecDeque::new(),
            pass_id: 0,
            seed: settings.seed as u32,
            rb,
            layout,
            building: Pending::default(),
            inflight: Pending::default(),
            awaiting: false,
            discard: false,
            therm_left: 0,
            stats: Stats::default(),
            flux: FluxSums::new(ns),
            quarks: [[0; 3]; 3],
            junction: [0; 3],
            quark_gen: 0,
            vacuum: None,
            vacuum_gen: 0,
            polyakov: None,
            budget: 8.0,
            rate: 0.0,
            last_cycle: None,
        };
        e.start(queue);
        e.set_quarks(queue, e.default_quarks());
        e
    }

    fn start(&mut self, queue: &wgpu::Queue) {
        self.discard = self.awaiting;
        self.lat
            .upload_links(queue, &self.lat.hot_links(self.settings.seed));
        self.stats = Stats {
            rmax: self.lat.rmax,
            tmax: self.lat.tmax,
            ..Default::default()
        };
        self.flux = FluxSums::new(self.lat.ns);
        self.ops.clear();
        self.therm_left = self.settings.therm;
        self.ops
            .extend(std::iter::repeat_n(Op::Update, self.settings.therm));
        self.vacuum = None;
        self.polyakov = None;
        self.last_cycle = None;
    }

    /// New settings: rebuild buffers if the lattice shape changed, then
    /// restart from a hot configuration.
    pub fn restart(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, settings: Settings) {
        let dims = [settings.nt, settings.size, settings.size, settings.size];
        if dims != self.lat.dims {
            self.lat = GpuLattice::new(device, &self.k, dims);
            self.layout = Layout::new(&self.lat);
            // a mapped readback can't be resized; wait for it to drain first
            if !self.awaiting {
                self.rb = Readback::new(device, (self.layout.total * 4) as u64);
            }
        }
        self.settings = settings;
        self.seed = settings.seed as u32;
        self.start(queue);
        let q = self.clamp_quarks(self.quarks);
        self.set_quarks(queue, q);
    }

    pub fn default_quarks(&self) -> [[i32; 3]; 3] {
        let c = (self.settings.size / 2) as i32;
        let r = (self.settings.size as i32 / 2 - 2).clamp(2, 5);
        let at = |ang: f64| {
            let a = ang.to_radians();
            [
                c + (r as f64 * a.cos()).round() as i32,
                c + (r as f64 * a.sin()).round() as i32,
                c,
            ]
        };
        [at(0.0), at(126.87), at(-126.87)]
    }

    fn clamp_quarks(&self, q: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
        let l = self.settings.size as i32;
        q.map(|p| p.map(|c| c.clamp(0, l - 1)))
    }

    /// Move the quarks: the junction goes to their centroid, the staircases
    /// are rebuilt, and the flux-tube averages start over.
    pub fn set_quarks(&mut self, queue: &wgpu::Queue, quarks: [[i32; 3]; 3]) {
        let quarks = self.clamp_quarks(quarks);
        let j: [i32; 3] = std::array::from_fn(|i| {
            ((quarks[0][i] + quarks[1][i] + quarks[2][i]) as f64 / 3.0).round() as i32
        });
        let paths = quarks.map(|q| staircase([q[0] - j[0], q[1] - j[1], q[2] - j[2]]));
        self.lat.set_paths(queue, &paths);
        self.quarks = quarks;
        self.junction = j;
        self.quark_gen += 1;
        self.flux = FluxSums::new(self.lat.ns);
    }

    fn queue_cycle(&mut self) {
        let s = self.settings;
        let ops = &mut self.ops;
        ops.extend(std::iter::repeat_n(Op::Update, s.sep));
        ops.push_back(Op::Plaquette);
        // bare Polyakov loop: the smoothing buffer still holds the rough links
        ops.push_back(Op::BeginSmoothing);
        ops.push_back(Op::Polyakov);
        if self.measure != Measure::Temperature {
            ops.extend(std::iter::repeat_n(Op::SpatialPair, s.smear_spatial / 2));
            ops.push_back(Op::Wilson);
            ops.push_back(Op::BeginSmoothing);
            let steps = if self.measure == Measure::Flux {
                s.smear_flux
            } else {
                s.smear_topo
            };
            ops.extend(std::iter::repeat_n(Op::StoutPair, steps / 2));
            ops.push_back(if self.measure == Measure::Flux {
                Op::Flux
            } else {
                Op::Topology
            });
        }
        ops.push_back(Op::Finish);
        self.building = Pending {
            quark_gen: self.quark_gen,
            ..Default::default()
        };
    }

    /// Advance: collect a finished readback, queue the next cycle, and encode
    /// this frame's share of work. `frame_dt` steers the budget.
    pub fn tick(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame_dt: f32) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = device.poll(wgpu::PollType::Poll);
        if self.awaiting {
            match self.rb.take() {
                Some(data) => {
                    self.awaiting = false;
                    if !std::mem::take(&mut self.discard) && data.len() >= self.layout.total {
                        self.parse(&data);
                    }
                }
                None if self.rb.idle() => self.awaiting = false,
                None => {}
            }
            // a lattice resize while mapped left the old readback in place
            if !self.awaiting && (self.rb.buf.size() as usize) < self.layout.total * 4 {
                self.rb = Readback::new(device, (self.layout.total * 4) as u64);
            }
        }
        if frame_dt > 1.0 / 40.0 {
            self.budget = (self.budget * 0.85).max(1.0);
        } else if frame_dt < 1.0 / 55.0 {
            self.budget = (self.budget * 1.05).min(400.0);
        }
        if self.ops.is_empty() {
            self.queue_cycle();
        }
        let mut batch = Batch::new(device, &self.k, &self.lat);
        let mut spent = 0.0;
        let (lat, layout, rb) = (&self.lat, self.layout, &self.rb);
        let g = groups(lat.n) as u64;
        let mut finished = false;
        while let Some(&op) = self.ops.front() {
            let cost = op.cost(&self.settings, lat.ns);
            if spent > 0.0 && spent + cost > self.budget || batch.room() < 64 {
                break;
            }
            if self.awaiting && op.writes_readback() {
                break;
            }
            spent += cost;
            self.ops.pop_front();
            match op {
                Op::Update => {
                    batch.update(
                        self.settings.beta as f32,
                        self.seed,
                        &mut self.pass_id,
                        self.settings.n_or,
                    );
                    self.therm_left = self.therm_left.saturating_sub(1);
                }
                Op::Plaquette => {
                    batch.fields(Links::Rough);
                    batch.reduce(Reduce::Fields, 1);
                    batch.copy(&lat.partial, 0, &rb.buf, (layout.plaq * 4) as u64, g * 4);
                }
                Op::BeginSmoothing => batch.begin_smoothing(),
                Op::StoutPair => batch.stout_pair(self.settings.rho, false),
                Op::SpatialPair => batch.stout_pair(self.settings.rho, true),
                Op::Polyakov => {
                    batch.polyakov();
                    batch.copy(
                        &lat.poly,
                        0,
                        &rb.buf,
                        (layout.poly * 4) as u64,
                        (lat.ns * 8) as u64,
                    );
                    self.building.poly = true;
                }
                Op::Wilson => {
                    let k = lat.rmax * lat.tmax;
                    batch.wilson();
                    batch.reduce(Reduce::Wilson, k);
                    batch.copy(
                        &lat.partial,
                        0,
                        &rb.buf,
                        (layout.wilson * 4) as u64,
                        k as u64 * g * 4,
                    );
                    self.building.wilson = true;
                }
                Op::Flux => {
                    batch.fields(Links::Smooth);
                    batch.baryon(self.settings.t_len as u32);
                    batch.copy(
                        &lat.corr,
                        0,
                        &rb.buf,
                        (layout.corr * 4) as u64,
                        (2 * lat.ns * 4) as u64,
                    );
                    batch.reduce(Reduce::BaryonW, 1);
                    batch.copy(&lat.partial, 0, &rb.buf, (layout.w * 4) as u64, g * 4);
                    batch.reduce(Reduce::Fields, 6);
                    batch.copy(
                        &lat.partial,
                        0,
                        &rb.buf,
                        (layout.fsum * 4) as u64,
                        6 * g * 4,
                    );
                    self.building.flux = true;
                }
                Op::Topology => {
                    batch.fields(Links::Smooth);
                    // action (component 1) and charge (component 2), contiguous
                    batch.copy(
                        &lat.fields,
                        (lat.n * 4) as u64,
                        &rb.buf,
                        (layout.dens * 4) as u64,
                        (2 * lat.n * 4) as u64,
                    );
                    self.building.topo = true;
                }
                Op::Finish => {
                    finished = true;
                    break;
                }
            }
        }
        batch.submit(queue);
        if finished {
            self.rb.request();
            self.awaiting = true;
            self.inflight = self.building;
        }
    }

    fn parse(&mut self, d: &[f32]) {
        let lat = &self.lat;
        let (n, ns, g) = (lat.n, lat.ns, groups(lat.n));
        let l = self.layout;
        let sum = |a: &[f32]| a.iter().map(|&x| x as f64).sum::<f64>();
        let now = Instant::now();
        if let Some(t) = self.last_cycle {
            let dt = now.duration_since(t).as_secs_f32().max(1e-3);
            self.rate = if self.rate == 0.0 {
                1.0 / dt
            } else {
                0.9 * self.rate + 0.1 / dt
            };
        }
        self.last_cycle = Some(now);
        self.stats.configs += 1;
        Stats::push_bounded(&mut self.stats.plaq, sum(&d[l.plaq..l.plaq + g]) / n as f64);
        let p = self.inflight;
        if p.poly {
            let loops = &d[l.poly..l.poly + 2 * ns];
            let (re, im) = (
                sum(&loops.iter().step_by(2).copied().collect::<Vec<_>>()),
                sum(&loops.iter().skip(1).step_by(2).copied().collect::<Vec<_>>()),
            );
            let avg = (re / ns as f64, im / ns as f64);
            Stats::push_bounded(&mut self.stats.poly, avg);
            self.polyakov = Some(self.polyakov_field(loops, avg));
        }
        if p.wilson {
            let k = lat.rmax * lat.tmax;
            let w: Vec<f64> = (0..k)
                .map(|c| sum(&d[l.wilson + c * g..l.wilson + (c + 1) * g]) / n as f64)
                .collect();
            Stats::push_bounded(&mut self.stats.wilson, w);
        }
        if p.flux && p.quark_gen == self.quark_gen {
            for i in 0..ns {
                self.flux.ws[0][i] += d[l.corr + i] as f64;
                self.flux.ws[1][i] += d[l.corr + ns + i] as f64;
            }
            self.flux.w += sum(&d[l.w..l.w + g]);
            let comp = |c: usize| sum(&d[l.fsum + c * g..l.fsum + (c + 1) * g]);
            self.flux.s[0] += comp(1);
            self.flux.s[1] += comp(3) + comp(4) + comp(5);
            self.flux.samples += n as f64;
            self.flux.configs += 1;
        }
        if p.topo {
            let action = &d[l.dens..l.dens + n];
            let topo = &d[l.dens + n..l.dens + 2 * n];
            Stats::push_bounded(&mut self.stats.q, sum(topo));
            for r in instanton_radii(topo, lat.dims, 4e-4) {
                Stats::push_bounded(&mut self.stats.radii, r);
            }
            self.vacuum = Some(FieldFile {
                dims: lat.dims,
                a_fm: self.settings.a_fm(),
                names: vec!["topo".into(), "action".into()],
                data: vec![topo.to_vec(), action.to_vec()],
                quarks: vec![],
            });
            self.vacuum_gen += 1;
        }
    }

    /// Local Polyakov loops rotated so the volume average is real and
    /// positive: aligned (deconfined) regions stand out as a plateau.
    fn polyakov_field(&self, loops: &[f32], avg: (f64, f64)) -> FieldFile {
        let ns = self.lat.ns;
        let norm = (avg.0 * avg.0 + avg.1 * avg.1).sqrt().max(1e-12);
        let (c, s) = ((avg.0 / norm) as f32, (-avg.1 / norm) as f32);
        let mut re = Vec::with_capacity(ns);
        let mut abs = Vec::with_capacity(ns);
        for i in 0..ns {
            let (x, y) = (loops[2 * i], loops[2 * i + 1]);
            re.push(x * c - y * s);
            abs.push((x * x + y * y).sqrt());
        }
        let l = self.settings.size;
        FieldFile {
            dims: [1, l, l, l],
            a_fm: self.settings.a_fm(),
            names: vec!["Re P".into(), "|P|".into()],
            data: vec![re, abs],
            quarks: vec![],
        }
    }

    /// C(r) as a 3D field in display coordinates (r = 0 at the junction),
    /// with the quark positions: "action" and "electric".
    pub fn flux_field(&self) -> Option<FieldFile> {
        let l = self.settings.size;
        let j = self.junction;
        let shift = |c: Vec<f64>| -> Vec<f32> {
            let mut out = vec![0.0; l * l * l];
            for z in 0..l {
                for y in 0..l {
                    for x in 0..l {
                        let r = |v: usize, jc: i32| (v as i32 - jc).rem_euclid(l as i32) as usize;
                        out[x + l * (y + l * z)] =
                            c[r(x, j[0]) + l * (r(y, j[1]) + l * r(z, j[2]))] as f32;
                    }
                }
            }
            out
        };
        let action = self.flux.correlation(0)?;
        let electric = self.flux.correlation(1)?;
        Some(FieldFile {
            dims: [1, l, l, l],
            a_fm: self.settings.a_fm(),
            names: vec!["action".into(), "electric".into()],
            data: vec![shift(action), shift(electric)],
            quarks: self.quarks.iter().map(|q| q.map(|c| c as f32)).collect(),
        })
    }

    pub fn status(&self) -> String {
        if self.therm_left > 0 {
            format!("thermalizing: {} updates left", self.therm_left)
        } else {
            format!("{} configurations, {:.1}/s", self.stats.configs, self.rate)
        }
    }
}
