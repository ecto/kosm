//! Every GPU kernel against phyz-qft on the same gauge configuration.
//! Tests skip (pass trivially, with a note) when no compute-capable adapter
//! is present.

#![allow(clippy::needless_range_loop)]

use super::*;
use phyz_qft::su3::{Su3, Su3Lattice, staircase};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    if !supported(&adapter) {
        return None;
    }
    let desc = wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    };
    pollster::block_on(adapter.request_device(&desc)).ok()
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let rb = Readback::new(device, bytes);
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, 0, &rb.buf, 0, bytes);
    queue.submit([enc.finish()]);
    rb.request();
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rb.take().expect("readback")
}

fn to_gpu(lat: &Su3Lattice) -> Vec<[f32; 2]> {
    lat.links
        .iter()
        .flatten()
        .flat_map(|u| {
            u.m.iter()
                .flatten()
                .map(|c| [c.re as f32, c.im as f32])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn from_gpu(data: &[f32], n: usize) -> [Vec<Su3>; 4] {
    std::array::from_fn(|mu| {
        (0..n)
            .map(|site| {
                let b = (mu * n + site) * 18;
                let mut m = Su3::ZERO;
                for k in 0..9 {
                    m.m[k / 3][k % 3] =
                        phyz_qft::su3::C64::new(data[b + 2 * k] as f64, data[b + 2 * k + 1] as f64);
                }
                m
            })
            .collect()
    })
}

/// A thermalized-ish, far-from-trivial configuration.
fn config() -> Su3Lattice {
    let mut lat = Su3Lattice::hot([4, 6, 4, 4], 5.8, 11);
    for _ in 0..3 {
        lat.update(1);
    }
    lat
}

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    k: Kernels,
    g: GpuLattice,
    cpu: Su3Lattice,
}

fn rig() -> Option<Rig> {
    let Some((device, queue)) = device() else {
        eprintln!("no compute-capable GPU adapter; skipping");
        return None;
    };
    let cpu = config();
    let k = Kernels::new(&device);
    let g = GpuLattice::new(&device, &k, cpu.dims);
    g.upload_links(&queue, &to_gpu(&cpu));
    Some(Rig {
        device,
        queue,
        k,
        g,
        cpu,
    })
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{what}: gpu {a} vs cpu {b}"
    );
}

#[test]
fn fields_match_cpu() {
    let Some(r) = rig() else { return };
    let mut b = Batch::new(&r.device, &r.k, &r.g);
    b.fields(Links::Rough);
    b.submit(&r.queue);
    let n = r.g.n;
    let out = read(&r.device, &r.queue, &r.g.fields, (6 * n * 4) as u64);
    let fs = r.cpu.field_strength();
    let (action, topo) = (fs.action_density(), fs.topological_charge_density());
    let [ex, ey, ez] = fs.electric_sq();
    for site in 0..n {
        let plaq: f64 = (0..4)
            .flat_map(|mu| (mu + 1..4).map(move |nu| (mu, nu)))
            .map(|(mu, nu)| r.cpu.plaquette(site, mu, nu).re_tr())
            .sum::<f64>()
            / 18.0;
        close(out[site] as f64, plaq, 1e-5, "plaquette");
        close(out[n + site] as f64, action[site], 1e-4, "action");
        close(
            out[2 * n + site] as f64,
            topo[site],
            1e-3,
            "topological charge",
        );
        close(out[3 * n + site] as f64, ex[site], 1e-4, "E_x²");
        close(out[4 * n + site] as f64, ey[site], 1e-4, "E_y²");
        close(out[5 * n + site] as f64, ez[site], 1e-4, "E_z²");
    }
}

#[test]
fn reduce_sums_components() {
    let Some(r) = rig() else { return };
    let mut b = Batch::new(&r.device, &r.k, &r.g);
    b.fields(Links::Rough);
    b.reduce(Reduce::Fields, 6);
    b.submit(&r.queue);
    let n = r.g.n;
    let g = groups(n);
    let fields = read(&r.device, &r.queue, &r.g.fields, (6 * n * 4) as u64);
    let partial = read(&r.device, &r.queue, &r.g.partial, (6 * g * 4) as u64);
    for c in 0..6 {
        let want: f64 = fields[c * n..(c + 1) * n].iter().map(|&v| v as f64).sum();
        let got: f64 = partial[c * g..(c + 1) * g].iter().map(|&v| v as f64).sum();
        close(got, want, 1e-5, "reduced sum");
    }
}

#[test]
fn stout_matches_cpu() {
    let Some(r) = rig() else { return };
    let mut b = Batch::new(&r.device, &r.k, &r.g);
    b.begin_smoothing();
    b.stout_pair(0.1, false);
    b.submit(&r.queue);
    let n = r.g.n;
    let got = from_gpu(
        &read(&r.device, &r.queue, &r.g.smooth_a, (4 * n * 72) as u64),
        n,
    );
    let mut cpu = r.cpu.clone();
    cpu.stout_smear(0.1, 2);
    for mu in 0..4 {
        for site in 0..n {
            let d = got[mu][site].dist_sqr(&cpu.links[mu][site]).sqrt();
            assert!(d < 2e-5, "stout link ({mu}, {site}) off by {d}");
        }
    }
}

#[test]
fn polyakov_matches_cpu() {
    let Some(r) = rig() else { return };
    let mut b = Batch::new(&r.device, &r.k, &r.g);
    b.begin_smoothing();
    b.polyakov();
    b.submit(&r.queue);
    let out = read(&r.device, &r.queue, &r.g.poly, (r.g.ns * 8) as u64);
    let [nt, nx, ny, nz] = r.cpu.dims;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (line, _) = r.cpu.time_line(r.cpu.index([0, x, y, z]), nt);
                let tr = line.trace();
                let s = x + nx * (y + ny * z);
                close(out[2 * s] as f64, tr.re / 3.0, 1e-5, "Re P");
                close(out[2 * s + 1] as f64, tr.im / 3.0, 1e-5, "Im P");
            }
        }
    }
}

#[test]
fn wilson_loops_match_cpu() {
    let Some(r) = rig() else { return };
    let mut b = Batch::new(&r.device, &r.k, &r.g);
    b.begin_smoothing();
    b.wilson();
    b.submit(&r.queue);
    let (n, rmax, tmax) = (r.g.n, r.g.rmax, r.g.tmax);
    let out = read(
        &r.device,
        &r.queue,
        &r.g.wilson,
        (rmax * tmax * n * 4) as u64,
    );
    for site in [0, 7, n / 2 + 3, n - 1] {
        for rr in 1..=rmax {
            for tt in 1..=tmax {
                let mut w = 0.0;
                for i in 1..4 {
                    let spatial = vec![(i, true); rr];
                    let (s_low, x_r) = r.cpu.path(site, &spatial);
                    let (t_far, _) = r.cpu.time_line(x_r, tt);
                    let (t_near, top) = r.cpu.time_line(site, tt);
                    let (s_high, _) = r.cpu.path(top, &spatial);
                    w += (s_low * t_far).mul_dag(&s_high).mul_dag(&t_near).re_tr() / 3.0;
                }
                close(
                    out[((rr - 1) * tmax + tt - 1) * n + site] as f64,
                    w / 3.0,
                    1e-5,
                    &format!("W({rr},{tt})"),
                );
            }
        }
    }
}

#[test]
fn baryon_and_correlation_match_cpu() {
    let Some(r) = rig() else { return };
    let quarks = [[2, 0, 0], [-1, 2, 0], [-1, -2, 1]];
    let paths = quarks.map(staircase);
    let t_len = 2;
    r.g.set_paths(&r.queue, &paths);
    let mut b = Batch::new(&r.device, &r.k, &r.g);
    b.begin_smoothing();
    b.fields(Links::Smooth);
    b.baryon(t_len as u32);
    b.submit(&r.queue);
    let n = r.g.n;
    let w = read(&r.device, &r.queue, &r.g.bw, (n * 4) as u64);
    let corr = read(&r.device, &r.queue, &r.g.corr, (2 * r.g.ns * 4) as u64);
    let cpu_w: Vec<f64> = (0..n)
        .map(|s| r.cpu.baryon_loop(s, &paths, t_len))
        .collect();
    for s in 0..n {
        close(w[s] as f64, cpu_w[s], 1e-4, "W3Q");
    }
    // numerator Σ W(x0) S(x0 + r) on the middle time slice
    let action = r.cpu.field_strength().action_density();
    let [nt, nx, ny, nz] = r.cpu.dims;
    for (rx, ry, rz) in [(0, 0, 0), (1, 2, 3), (5, 1, 0)] {
        let mut want = 0.0;
        for s in 0..n {
            let [t, x, y, z] = r.cpu.coords(s);
            let far = r.cpu.index([
                (t + t_len / 2) % nt,
                (x + rx) % nx,
                (y + ry) % ny,
                (z + rz) % nz,
            ]);
            want += cpu_w[s] * action[far];
        }
        close(
            corr[rx + nx * (ry + ny * rz)] as f64,
            want,
            1e-4,
            "correlation",
        );
    }
}

#[test]
fn heatbath_reaches_known_plaquette() {
    let Some((device, queue)) = device() else {
        eprintln!("no compute-capable GPU adapter; skipping");
        return;
    };
    let k = Kernels::new(&device);
    let g = GpuLattice::new(&device, &k, [8, 8, 8, 8]);
    g.upload_links(&queue, &g.hot_links(3));
    let mut pass_id = 0;
    let measure = |pass_id: &mut u32, updates: usize| {
        for _ in 0..updates / 5 {
            let mut b = Batch::new(&device, &k, &g);
            for _ in 0..5 {
                b.update(6.0, 7, pass_id, 4);
            }
            b.submit(&queue);
        }
        let mut b = Batch::new(&device, &k, &g);
        b.fields(Links::Rough);
        b.submit(&queue);
        let p = read(&device, &queue, &g.fields, (g.n * 4) as u64);
        p.iter().map(|&v| v as f64).sum::<f64>() / g.n as f64
    };
    measure(&mut pass_id, 100);
    let n = 40;
    let mean = (0..n).map(|_| measure(&mut pass_id, 5)).sum::<f64>() / n as f64;
    eprintln!("GPU β=6.0 8⁴ ⟨P⟩ = {mean:.5}");
    assert!((mean - 0.5937).abs() < 0.002, "⟨P⟩ = {mean}");
}

/// Per-kernel GPU timings on a 16⁴ lattice.
/// `cargo test -p kosm-qcd --release -- --ignored --nocapture kernel_timings`
#[test]
#[ignore = "benchmark"]
fn kernel_timings() {
    let Some((device, queue)) = device() else {
        return;
    };
    let k = Kernels::new(&device);
    let g = GpuLattice::new(&device, &k, [16, 16, 16, 16]);
    g.upload_links(&queue, &g.hot_links(1));
    g.set_paths(&queue, &[[5, 0, 0], [-3, 4, 0], [-3, -4, 0]].map(staircase));
    let time = |label: &str, reps: usize, f: &dyn Fn(&mut Batch)| {
        let wait = || {
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
        };
        let mut pass = 0;
        let mut b = Batch::new(&device, &k, &g);
        b.update(6.0, 1, &mut pass, 0);
        b.submit(&queue);
        wait();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let mut b = Batch::new(&device, &k, &g);
            f(&mut b);
            b.submit(&queue);
        }
        wait();
        eprintln!(
            "{label:>22}: {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3 / reps as f64
        );
    };
    time("heatbath sweep", 10, &|b| {
        let mut p = 0;
        b.update(6.0, 1, &mut p, 0)
    });
    time("stout pair", 10, &|b| b.stout_pair(0.1, false));
    time("spatial stout pair", 10, &|b| b.stout_pair(0.1, true));
    time("fields", 10, &|b| b.fields(Links::Smooth));
    time("wilson", 10, &|b| b.wilson());
    time("polyakov", 10, &|b| b.polyakov());
    time("baryon + correlate", 5, &|b| b.baryon(4));
    time("reduce 6", 10, &|b| b.reduce(Reduce::Fields, 6));
}

#[test]
fn spatial_stout_keeps_time_links() {
    let Some(r) = rig() else { return };
    let mut b = Batch::new(&r.device, &r.k, &r.g);
    b.begin_smoothing();
    b.stout_pair(0.1, true);
    b.submit(&r.queue);
    let n = r.g.n;
    let got = from_gpu(
        &read(&r.device, &r.queue, &r.g.smooth_a, (4 * n * 72) as u64),
        n,
    );
    // CPU reference: two steps of spatial stout with spatial staples only
    let mut cpu = r.cpu.clone();
    for _ in 0..2 {
        let old = cpu.clone();
        for mu in 1..4 {
            for site in 0..n {
                let n_mu = old.fwd(site, mu);
                let mut a = Su3::ZERO;
                for nu in 1..4 {
                    if nu == mu {
                        continue;
                    }
                    let n_nu = old.fwd(site, nu);
                    a += old.links[nu][n_mu]
                        .mul_dag(&old.links[mu][n_nu])
                        .mul_dag(&old.links[nu][site]);
                    let (n_mnu, n_mu_mnu) = (old.bwd(site, nu), old.bwd(n_mu, nu));
                    a += old.links[nu][n_mu_mnu].dag_mul(&old.links[mu][n_mnu].dagger())
                        * old.links[nu][n_mnu];
                }
                let u = old.links[mu][site];
                let x = (u * a).traceless_antihermitian();
                cpu.links[mu][site] = Su3::exp_antihermitian(&x.scale(-0.1)) * u;
            }
        }
    }
    for mu in 0..4 {
        for site in 0..n {
            let d = got[mu][site].dist_sqr(&cpu.links[mu][site]).sqrt();
            assert!(d < 2e-5, "spatial stout link ({mu}, {site}) off by {d}");
        }
    }
}

/// Run the engine headlessly until `configs` measured cycles have been parsed.
fn run_engine(device: &wgpu::Device, queue: &wgpu::Queue, e: &mut engine::Engine, configs: usize) {
    let target = e.stats.configs + configs;
    let mut spins = 0;
    while e.stats.configs < target {
        e.tick(device, queue, 1.0 / 60.0);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        spins += 1;
        assert!(
            spins < 200_000,
            "engine stalled at {} configs",
            e.stats.configs
        );
    }
}

/// Pure SU(3) at N_t = 4 deconfines at β_c = 5.6925: the Polyakov loop is
/// ~0 below and jumps to a Z(3) direction above.
/// `cargo test -p kosm-qcd --release -- --ignored --nocapture deconfinement`
#[test]
#[ignore = "tens of seconds"]
fn deconfinement_transition() {
    let Some((device, queue)) = device() else {
        return;
    };
    let mean_abs = |beta: f64| {
        let s = engine::Settings {
            size: 12,
            nt: 4,
            beta,
            therm: 200,
            sep: 2,
            ..engine::Settings::vacuum()
        };
        let mut e = engine::Engine::new(&device, &queue, s, engine::Measure::Temperature);
        run_engine(&device, &queue, &mut e, 200);
        let p = &e.stats.poly;
        let tail = &p[p.len() / 2..];
        let abs = tail.iter().map(|(a, b)| a.hypot(*b)).sum::<f64>() / tail.len() as f64;
        let phase = tail.iter().map(|(a, b)| b.atan2(*a)).sum::<f64>() / tail.len() as f64;
        eprintln!(
            "β = {beta}: ⟨|P|⟩ = {abs:.4}, mean arg = {phase:+.2}, rate {:.0}/s",
            e.rate
        );
        abs
    };
    let cold = mean_abs(5.5);
    let hot = mean_abs(5.9);
    assert!(hot > 4.0 * cold, "confined {cold} vs deconfined {hot}");
}

/// The flux tube follows the quarks: after moving them, the vacuum is most
/// suppressed near the new junction, and C(r) ≈ 1 far from all quarks.
#[test]
#[ignore = "tens of seconds"]
fn flux_tube_follows_quarks() {
    let Some((device, queue)) = device() else {
        return;
    };
    let s = engine::Settings {
        size: 12,
        nt: 12,
        therm: 80,
        ..engine::Settings::vacuum()
    };
    let mut e = engine::Engine::new(&device, &queue, s, engine::Measure::Flux);
    for quarks in [
        [[9, 6, 6], [4, 9, 6], [4, 3, 6]],
        [[3, 3, 5], [8, 3, 5], [5, 8, 5]],
    ] {
        e.set_quarks(&queue, quarks);
        run_engine(&device, &queue, &mut e, 40);
        let f = e.flux_field().expect("flux field");
        let c = &f.data[0];
        let l = 12;
        let at = |x: i32, y: i32, z: i32| c[(x as usize) + l * ((y as usize) + l * (z as usize))];
        let j: [i32; 3] = std::array::from_fn(|k| {
            ((quarks[0][k] + quarks[1][k] + quarks[2][k]) as f64 / 3.0).round() as i32
        });
        let centre = at(j[0], j[1], j[2]);
        // the point farthest from the junction on the periodic box
        let far = at((j[0] + 6) % 12, (j[1] + 6) % 12, (j[2] + 6) % 12);
        eprintln!(
            "quarks {quarks:?}: C(junction) = {centre:.3}, C(far) = {far:.3}, {} configs",
            e.flux.configs
        );
        assert!(
            centre < far - 0.03,
            "no suppression at the junction: {centre} vs {far}"
        );
        assert!(
            (far - 1.0).abs() < 0.05,
            "far field should be ~1, got {far}"
        );
    }
}
