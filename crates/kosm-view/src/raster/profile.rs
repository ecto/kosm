//! What each pass cost, off the GPU's own clock.
//!
//! A wall-clock timer around [`Raster::draw`](super::pipeline::Raster::draw)
//! measures the *encode*, not the draw: `queue.submit` returns before the
//! device has run a triangle, so the number it gives is CPU work plus whatever
//! backpressure the driver applied, and it says nothing about which pass is
//! expensive. Timestamps the device writes at the beginning and end of every
//! pass do say, and they are the only honest way to decide what to trim.
//!
//! Two queries a pass, resolved into a buffer at the end of the frame and
//! mapped **one frame later** — reading them back in the same frame would
//! stall the pipeline and change the thing being measured. So the table a
//! caller prints is a frame or two old, which for a pace line that reports
//! once a second is the same table.
//!
//! `Features::TIMESTAMP_QUERY` is not universal. When the device was not given
//! it, [`Profiler::new`] returns [`None`], every pass is encoded with no
//! timestamp writes, and the caller prints the wall clock alone.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// The passes, in the order the encoder runs them. The index into the query
/// set is twice the ordinal, and the end timestamp is one past it.
pub const PASSES: [&str; 11] = [
    "shadow", "prepass", "ao", "ao blur1", "ao blur2", "sky", "scene", "bright", "bloom1",
    "bloom2", "resolve",
];

/// The first slot of each group; the ao and bloom chains take three each.
pub const SHADOW: usize = 0;
pub const PREPASS: usize = 1;
pub const AO: usize = 2;
pub const SKY: usize = 5;
pub const SCENE: usize = 6;
pub const BLOOM: usize = 7;
pub const RESOLVE: usize = 10;

const QUERIES: u32 = PASSES.len() as u32 * 2;
const BYTES: u64 = QUERIES as u64 * 8;

/// How many triangles, instances and draw calls a pass submitted.
///
/// Counted on the CPU as the passes are encoded, because that is where the
/// answer is: a `draw` of `v` vertices over `n` instances is one call, `n`
/// instances and `n · v / 3` triangles, and no device statistic is needed to
/// say so.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub draws: u32,
    pub instances: u32,
    pub triangles: u32,
}

impl Counts {
    pub fn add(&mut self, vertices: u32, instances: u32) {
        if instances == 0 {
            return;
        }
        self.draws += 1;
        self.instances += instances;
        self.triangles += vertices / 3 * instances;
    }
}

/// One frame's per-pass milliseconds, and what it drew.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Milliseconds a pass, indexed as [`PASSES`]. A pass that did not run is
    /// zero.
    pub ms: [f64; PASSES.len()],
    /// The first pass's beginning to the last pass's end, milliseconds.
    ///
    /// **This is the frame's GPU time, and the sum of `ms` is not.** A tiled
    /// GPU overlaps passes that do not depend on each other — the shadow map
    /// and the prepass write different targets, and one's vertex work runs
    /// under the other's fragments — so the per-pass intervals overlap and
    /// their sum over-counts. Each one is still the right *ranking*.
    pub span: f64,
    /// The two geometry passes' counts: the sun's map, and the camera's.
    pub shadow: Counts,
    pub camera: Counts,
}

impl Report {
    pub fn total_ms(&self) -> f64 {
        self.ms.iter().sum()
    }

    /// The table as one line. The order is the encoder's and not the
    /// expensive-first order, because the question a reader has is where in
    /// the chain the time went.
    pub fn line(&self) -> String {
        let mut s = String::new();
        for (name, ms) in PASSES.iter().zip(self.ms.iter()) {
            if *ms <= 0.0 {
                continue;
            }
            if !s.is_empty() {
                s.push_str("  ");
            }
            s.push_str(&format!("{name} {ms:.2}"));
        }
        format!(
            "{s}  |  gpu span {:.2} ms (pass sum {:.2}); camera {} draws / {} instances / {} tris, shadow {} draws / {} tris",
            self.span,
            self.total_ms(),
            self.camera.draws,
            self.camera.instances,
            self.camera.triangles,
            self.shadow.draws,
            self.shadow.triangles,
        )
    }
}

/// The query set and the two buffers a readback needs.
pub struct Profiler {
    set: wgpu::QuerySet,
    /// Where `resolve_query_set` puts the ticks.
    resolved: wgpu::Buffer,
    /// The mappable copy of it.
    staging: wgpu::Buffer,
    /// Nanoseconds a tick.
    period: f32,
    /// Set by the map callback, cleared when the bytes have been taken.
    ready: Arc<AtomicBool>,
    mapping: bool,
    last: Report,
}

impl Profiler {
    /// A profiler, or [`None`] when the device was not given the feature.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        Some(Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("raster passes"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            }),
            resolved: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raster timestamps"),
                size: BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raster timestamps read"),
                size: BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period: queue.get_timestamp_period(),
            ready: Arc::new(AtomicBool::new(false)),
            mapping: false,
            last: Report::default(),
        })
    }

    /// The timestamp writes for pass `i`, to hand a `RenderPassDescriptor`.
    pub fn writes(&self, i: usize) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(i as u32 * 2),
            end_of_pass_write_index: Some(i as u32 * 2 + 1),
        })
    }

    /// Take the ticks a finished map left behind, before this frame encodes.
    ///
    /// **This has to run before the encoder touches the staging buffer.** A
    /// copy into a buffer that is still mapped, or has a map pending, fails
    /// validation at submit and takes the whole frame's command buffer with
    /// it — which is what a first version of this did, and it read as a frame
    /// that took twice as long as it should.
    pub fn collect(&mut self, device: &wgpu::Device, ran: [bool; PASSES.len()]) {
        if !self.mapping {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        if !self.ready.swap(false, Ordering::Acquire) {
            return;
        }
        {
            let view = self.staging.slice(..).get_mapped_range();
            if let Ok(bytes) = view {
                let ticks: &[u64] = bytemuck::cast_slice(&bytes);
                let (mut first, mut last) = (u64::MAX, 0u64);
                for (i, slot) in self.last.ms.iter_mut().enumerate() {
                    let (a, b) = (ticks[i * 2], ticks[i * 2 + 1]);
                    // A skipped pass's pair still holds whatever an earlier
                    // frame wrote there, so it is zeroed rather than read.
                    if !ran[i] || b <= a {
                        *slot = 0.0;
                        continue;
                    }
                    first = first.min(a);
                    last = last.max(b);
                    *slot = (b - a) as f64 * self.period as f64 * 1e-6;
                }
                self.last.span = if last > first {
                    (last - first) as f64 * self.period as f64 * 1e-6
                } else {
                    0.0
                };
            }
        }
        self.staging.unmap();
        self.mapping = false;
    }

    /// Copy the ticks out at the end of the frame's encoder — if the staging
    /// buffer is free. When the last readback has not landed yet this frame's
    /// ticks are simply not read, which is the price of never waiting.
    /// Returns whether it encoded the copy.
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) -> bool {
        if self.mapping {
            return false;
        }
        enc.resolve_query_set(&self.set, 0..QUERIES, &self.resolved, 0);
        enc.copy_buffer_to_buffer(&self.resolved, 0, &self.staging, 0, BYTES);
        true
    }

    /// After `submit`: ask for the map of a copy that was just encoded, and
    /// note what the frame drew.
    pub fn after_submit(&mut self, copied: bool, shadow: Counts, camera: Counts) {
        self.last.shadow = shadow;
        self.last.camera = camera;
        if copied && !self.mapping {
            self.mapping = true;
            let ready = self.ready.clone();
            self.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                ready.store(r.is_ok(), Ordering::Release);
            });
        }
    }

    /// The last table that came back.
    pub fn report(&self) -> &Report {
        &self.last
    }
}
