//! The court on the GPU: the same picture, traced by `vcad-kernel-raytrace`'s
//! compute shader instead of its CPU integrator.
//!
//! Nothing about the *scene* is decided here either. `kosm_spike::court::render`
//! still owns what the court is made of — which roots, which materials, where
//! the balls are, which panels are lights — and this module only packs that
//! into the buffers the shader reads. Every solid is packed once, with
//! `GpuScene::from_brep`; every frame says where the instances of it are, with
//! `GpuScene::placed`. So a ball moving costs a transform of its packed
//! surfaces, not a re-pack and not a BVH rebuild.
//!
//! The device is the *viewport's* device (see `viewport::Scene::init`). vcad
//! is on wgpu 30 now, so the surface and the tracer are one set of wgpu types
//! and there is one adapter in the process rather than two.
//!
//! ## nothing comes back
//!
//! [`Stage::accumulate`] is the whole of a pass, and it reads nothing back.
//! vcad's `accumulate_and_denoise_resident` traces one raw sample, folds it
//! into a per-pixel mean and count that live in device buffers, runs the
//! à-trous filter against the resident guide planes, and tonemaps the result
//! into a storage texture — [`Stage::target`] — which is on the viewport's own
//! device and which the blit samples directly. There is no `Film`, no
//! readback, and no CPU-side history on this tier at all.
//!
//! What this side still owns is the **keep mask**: one byte a pixel, 1 to go
//! on accumulating and 0 to start over, computed by [`crate::history::Mask`]
//! from the same geometry the CPU tier masks with. Reprojection is not on the
//! device, so a moved camera uploads an all-restart mask — the CPU tier still
//! carries its samples through a moved camera and this one does not.
//!
//! It also owns the **scissor**, and now uses it. `GpuRenderState::set_scissor`
//! used to size the *trace* alone, while vcad's accumulate pass ran over every
//! pixel of the frame and would have folded the stale raw sample outside the
//! rectangle into the history as if it were fresh. It honours the same
//! rectangle now — outside it the mean, the count and the variance are left
//! exactly as they were, and the resolve pass still covers the frame so the
//! target texture stays whole. So a pass whose keep mask fits in a box worth
//! less than half the frame traces and folds that box and nothing else.
//!
//! ## the panels are in the picture
//!
//! `set_camera_visible_lights` is off in vcad's default state, and with it off
//! the shader would not shade a light the camera can see. That was the whole
//! of the closed room's brightness gap: the walls already agreed with the CPU
//! integrator to 0.04%, and the ten ceiling panels the gym is lit by came back
//! black. This tier turns it on every pass, along with the level's own
//! `max_depth` and `ground_enabled = 0` — `GpuRenderState::new` re-derives a
//! frame-dependent depth and an implicit ground plane, so all three have to be
//! re-stated on every pass or the two tiers are not tracing the same picture.
//!
//! ## residency
//!
//! The court is uploaded once and stays there. `ResidentScene` holds the
//! surface, face, BVH, material and light buffers; a frame rewrites only the
//! bytes that moved (`update_scene`, once per *frame*, not once per pass) and
//! a pass rewrites only the camera and the render state.
//!
//! ## what the GPU picture is not
//!
//! - A root with no BRep — the painted markings, which are drawn and not
//!   modelled — cannot be packed and is not in the GPU picture at all.
//! - The shader's implicit ground plane is switched off because the level
//!   authors its own floor, while the CPU path also gets an infinite one at
//!   the slab's underside.
//!
//! The environment is no longer among them. It used to be: the shader's
//! analytic gradient is a different colour in every direction, and scaling it
//! by the level's `env_radiance` gave the GPU about 45% of the light the CPU's
//! flat constant gives, because `GpuRenderState` had no way to be told the
//! gradient's own colours. It has one now — `set_gradient_env` — so this tier
//! sends the very `Environment::constant(env_radiance)` the CPU tier builds.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use kosm_spike::court::render::{self, Snapshot};
use vcad_kernel::Solid;
use vcad_kernel_gpu::GpuContext;
use vcad_kernel_raytrace::gpu::{
    GpuAreaLight, GpuCamera, GpuDenoiseParams, GpuMaterial, GpuRenderState, GpuScene,
    HistoryPipeline, RayTracePipeline, ResidentScene, DEFAULT_FIREFLY_CLAMP, DEFAULT_RR_START,
};
use vcad_kernel_raytrace::pathtrace::{Environment, Pbr};

use crate::court::Camera;

/// The court, packed. Built once; every frame after that is placements.
pub struct Stage {
    ctx: GpuContext,
    pipeline: RayTracePipeline,
    /// The level's roots and the gym, already placed and merged into one.
    statics: GpuScene,
    /// The ball's parts at the origin — its solid and its seams — one packed
    /// scene each, placed once per ball per frame.
    ball: Vec<GpuScene>,
    /// Packed extras (the net), kept across frames by the solid's identity,
    /// since a net that only moves is not a new solid. `None` is one that
    /// would not pack — remembered so it is complained about once rather than
    /// once a frame.
    extras: HashMap<usize, Option<GpuScene>>,
    lights: Vec<GpuAreaLight>,
    max_depth: u32,
    /// The level's environment, built exactly as `render::Scene` builds the
    /// CPU tier's: `Environment::constant(env_radiance)`. It reaches the
    /// shader through `GpuRenderState::set_gradient_env`, so the two tiers are
    /// lit by the same sky.
    env: Environment,
    /// The merged scene for the frame on screen, and which frame that was.
    /// Assembling it is a clone of the statics and a placement per instance,
    /// which costs the same whatever the resolution — so it is done once per
    /// frame and not once per pass, and a paused window pays for it once.
    scene: Option<(u64, GpuScene)>,
    /// The court on the device. Built on the first pass, kept across every
    /// one after it: a frame rewrites the placements, a pass rewrites the
    /// camera. `uploaded` is the frame whose placements are currently in it.
    resident: Option<ResidentScene>,
    uploaded: Option<u64>,
    /// The history and denoise passes, compiled once.
    history: HistoryPipeline,
    /// How the device filters the running mean. The default fades the filter
    /// out as a pixel reaches thirty-two samples, which is
    /// `History::resolve`'s `DENOISE_UNTIL` and right for a window that keeps
    /// converging. A still that stops at thirty-two wants the filter at full
    /// strength instead — see [`Stage::always_denoise`].
    denoise: GpuDenoiseParams,
    /// What the pass tonemaps into and the blit samples: an `Rgba8Unorm`
    /// storage texture on the viewport's device, remade only on a resize. It
    /// carries an `Rgba8UnormSrgb` view format because the blit decodes on the
    /// way in when the surface will re-encode on the way out.
    target: Option<(Arc<wgpu::Texture>, wgpu::TextureView)>,
    /// Passes since the stage was built. Nothing accumulates across them —
    /// this only drives the shader's jitter and its RNG, so that two passes
    /// of the same frame are two different samples.
    passes: u32,
    size: (u32, u32),
}

/// One solid, packed with its material. A packed scene carries a single
/// material, which `merge` then re-indexes, so a root's `Pbr` goes in here.
fn pack(solid: &Solid, pbr: Pbr) -> Option<GpuScene> {
    let brep = solid.as_brep()?;
    let mut scene = GpuScene::from_brep(brep).ok()?;
    if scene.faces.is_empty() {
        return None;
    }
    scene.materials = vec![GpuMaterial::from_pbr(pbr)];
    for f in &mut scene.faces {
        f.material_idx = 0;
    }
    // The studio rig `from_brep` sizes to each solid is not this gym's
    // lighting; the level's ceiling panels are, and they go on after merging.
    scene.lights.clear();
    Some(scene)
}

fn merge_all(mut scenes: impl Iterator<Item = GpuScene>) -> Option<GpuScene> {
    let first = scenes.next()?;
    Some(scenes.fold(first, GpuScene::merge))
}

impl Stage {
    /// Pack the court for the GPU, on a device that already exists.
    ///
    /// A solid with no BRep — one that only tessellates — is left out rather
    /// than silently traced as nothing: the shader reads analytic surfaces,
    /// and a mesh is not one. The count is reported so a picture missing a
    /// part says so on stderr instead of just looking wrong.
    pub fn new(
        stage: &render::Scene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_depth: u32,
        env_radiance: f32,
    ) -> anyhow::Result<Self> {
        let ctx = GpuContext { device: device.clone(), queue: queue.clone() };
        let pipeline =
            RayTracePipeline::new(&ctx).map_err(|e| anyhow::anyhow!("the tracer's pipeline: {e}"))?;
        let history = HistoryPipeline::new(&ctx)
            .map_err(|e| anyhow::anyhow!("the history's pipelines: {e}"))?;

        // The statics are instances: sixty of the court's bars are one cube,
        // and packing that cube once and placing it sixty times is the whole
        // point of walking the level to placements. Keyed by the solid's
        // identity, which is what sharing it made equal.
        let mut dropped = 0usize;
        let mut by_solid: HashMap<usize, Option<GpuScene>> = HashMap::new();
        let packed: Vec<GpuScene> = stage
            .static_parts()
            .filter_map(|(solid, pbr, to_world)| {
                let key = solid as *const Solid as usize;
                let base = by_solid.entry(key).or_insert_with(|| pack(solid, pbr));
                match base {
                    Some(s) => Some(s.placed(to_world)),
                    None => {
                        dropped += 1;
                        None
                    }
                }
            })
            .collect();
        let kept = packed.len();
        let statics = merge_all(packed.into_iter())
            .ok_or_else(|| anyhow::anyhow!("nothing in the court packs for the GPU"))?;

        // The ball's seams are on the GPU again. They are four thin tori, and
        // the shader used to trace a torus wider than the solid said it was —
        // a ring that covered 330 pixels on the CPU covered 603 here, which on
        // a seam meant each near-black ring swelling until it engulfed the
        // ball it was drawn on. vcad's torus intersection is fixed (its
        // silhouettes now agree with the CPU integrator's to an IoU of 0.993),
        // so the seams are packed like any other part.
        let ball: Vec<GpuScene> = stage
            .ball_parts()
            .filter_map(|(_, solid, pbr, local)| pack(solid, pbr).map(|s| s.placed(local)))
            .collect();
        anyhow::ensure!(!ball.is_empty(), "the ball does not pack for the GPU");

        let lights: Vec<GpuAreaLight> =
            stage.lights().iter().map(GpuAreaLight::from_area_light).collect();

        eprintln!(
            "court  gpu: {kept} solids packed ({dropped} skipped, no BRep), \
             {} surfaces, {} faces, {} bvh nodes, {} panels",
            statics.surfaces.len(),
            statics.faces.len(),
            statics.bvh_nodes.len(),
            lights.len(),
        );

        Ok(Self {
            ctx,
            pipeline,
            statics,
            ball,
            extras: HashMap::new(),
            lights,
            scene: None,
            max_depth,
            env: Environment::constant([env_radiance; 3]),
            resident: None,
            uploaded: None,
            history,
            denoise: GpuDenoiseParams::default(),
            target: None,
            passes: 0,
            size: (0, 0),
        })
    }


    /// The whole scene at one instant: the statics, plus every ball part at
    /// every ball's pose, plus the net.
    ///
    /// The statics are cloned rather than shared because `merge` re-indexes
    /// what it merges into. That clone is the per-frame cost of doing without
    /// a shader-side instance table, and it is a memcpy of a few arrays.
    fn at(&mut self, snap: &Snapshot, stage: &render::Scene) -> GpuScene {
        let mut scene = self.statics.clone();
        for at in stage.ball_placements(snap) {
            for part in &self.ball {
                scene = scene.merge(part.placed(&at));
            }
        }
        for (solid, pbr, to_world) in stage.extra_parts(snap) {
            let key = solid as *const Solid as usize;
            if !self.extras.contains_key(&key) {
                let packed = pack(solid, pbr);
                if packed.is_none() {
                    eprintln!("court  gpu: an extra has no BRep to pack; it is not in the picture");
                }
                self.extras.insert(key, packed);
            }
            if let Some(p) = &self.extras[&key] {
                let placed = p.placed(to_world);
                scene = scene.merge(placed);
            }
        }
        scene.lights = self.lights.clone();
        scene
    }

    /// One pass, folded in, denoised and tonemapped — on the device.
    ///
    /// The court is already there. A *frame* rewrites the placements
    /// (`update_scene`, with the clone-and-merge in [`Stage::at`] behind it)
    /// and a *pass* rewrites the camera, the render state and the keep mask,
    /// so a pass of a still picture is a dispatch and four small compute
    /// passes. `frame_index` still climbs: it is what moves the shader's
    /// Halton jitter and its RNG, so two passes of one frame are two samples.
    ///
    /// `keep` is one byte a pixel: 1 to go on accumulating that pixel's mean,
    /// 0 to start it over at this pass's sample. Empty means keep everything.
    /// [`crate::history::Mask`] builds it from the poses, before a ray is cast.
    ///
    /// What comes back is the texture the picture is now in, on the viewport's
    /// own device. Nothing was read back to make it.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate(
        &mut self,
        stage: &render::Scene,
        snap: &Snapshot,
        frame_id: u64,
        camera: &Camera,
        size: (u32, u32),
        keep: &[u8],
        scissor: Option<[u32; 4]>,
        samples: u32,
    ) -> anyhow::Result<Arc<wgpu::Texture>> {
        let n = (size.0 as u64) * (size.1 as u64);
        anyhow::ensure!(n > 0, "an empty picture");
        let assembled = Instant::now();
        if self.scene.as_ref().is_none_or(|(id, _)| *id != frame_id) {
            let scene = self.at(snap, stage);
            self.scene = Some((frame_id, scene));
        }
        let assembly = assembled.elapsed();

        let uploaded = Instant::now();
        let (_, scene) = self.scene.as_ref().expect("just assembled");
        match &mut self.resident {
            Some(res) => {
                res.resize(&self.ctx, size.0, size.1);
                if self.uploaded != Some(frame_id) || self.size != size {
                    res.update_scene(&self.ctx, scene);
                }
            }
            None => {
                self.resident = Some(self.pipeline.resident_scene(&self.ctx, scene, size.0, size.1));
            }
        }
        self.uploaded = Some(frame_id);
        if self.size != size {
            self.target = None;
        }
        self.size = size;
        let upload = uploaded.elapsed();

        let (texture, view) = self.ensure_target(size);
        let cam = GpuCamera::new(
            [camera.eye.x as f32, camera.eye.y as f32, camera.eye.z as f32],
            [camera.target.x as f32, camera.target.y as f32, camera.target.z as f32],
            [0.0, 0.0, 1.0],
            (camera.fov_deg as f32).to_radians(),
            size.0,
            size.1,
        );
        let denoise = GpuDenoiseParams { exposure: camera.exposure, ..self.denoise };
        let traced = Instant::now();
        let res = self.resident.as_mut().expect("just built");
        // `samples` samples, each its own call: vcad's accumulate folds one
        // raw sample per call, so a pass of several is several calls with a
        // climbing `frame_index` to move the jitter and the RNG. Only the
        // first carries the keep mask — the pixels this pass restarts are
        // restarted once, and the rest of the pass accumulates onto them.
        for k in 0..samples.max(1) {
            self.passes += 1;
            let mut state = GpuRenderState::new(self.passes);
            // A photoreal viewport: no edge overlay, no stylisation, and no
            // implicit ground plane — the level authors its own floor.
            state.enable_edges = 0;
            state.stylize = 0;
            state.ground_enabled = 0;
            state.max_depth = self.max_depth;
            // Draw the panels to camera rays. Off by default, and with it off
            // the shader refused to shade a light it was standing under at
            // all, which is the whole of the closed room's brightness gap:
            // the walls already agreed with the CPU integrator to 0.04%, and
            // the ceiling the picture is lit by was black.
            state.set_camera_visible_lights(true);
            state.rr_start = DEFAULT_RR_START;
            state.firefly_clamp = DEFAULT_FIREFLY_CLAMP;
            // The same sky the CPU tier integrates against, colours and all.
            if let Environment::Gradient(g) = &self.env {
                state.set_gradient_env(g);
            }
            // The scissor sizes the trace *and* the fold: vcad's accumulate
            // pass honours the same rectangle, so every pixel outside keeps
            // the mean, the count and the variance it had. See the module
            // docs.
            if let Some(rect) = scissor {
                state.set_scissor(rect);
            }
            self.pipeline
                .accumulate_and_denoise_resident(
                    &self.ctx,
                    &self.history,
                    res,
                    &cam,
                    state,
                    if k == 0 { keep } else { &[] },
                    &denoise,
                    &view,
                )
                .map_err(|e| anyhow::anyhow!("the tracer: {e}"))?;
        }

        // Wait for the passes to land. Not a readback — no pixel comes back —
        // but with nothing else synchronising the two sides the worker would
        // queue passes faster than the device retires them, and the window's
        // tuner would be timing `queue.submit` rather than the render. A pass
        // has to be a pass before it can be measured.
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely())?;

        // Where a pass goes, when anyone asks.
        if std::env::var("KOSM_GPU_TIMING").is_ok() {
            eprintln!(
                "court  gpu: {}\u{d7}{} pass \u{2014} {:.1} ms assembling, {:.1} ms uploading, {:.1} ms tracing",
                size.0,
                size.1,
                assembly.as_secs_f64() * 1e3,
                upload.as_secs_f64() * 1e3,
                traced.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(texture)
    }

    /// Filter every pass at full strength, however many samples a pixel has.
    ///
    /// For `--shot`, which takes a fixed number of passes and then stops: the
    /// fade exists so a window that goes on converging is not softened once it
    /// no longer needs the filter, and a still that ends at exactly the cutoff
    /// would get the fade with none of the convergence.
    pub fn always_denoise(&mut self) {
        self.denoise.count_cutoff = u32::MAX;
    }

    /// The storage texture the passes write and the blit reads, made once per
    /// size.
    fn ensure_target(&mut self, size: (u32, u32)) -> (Arc<wgpu::Texture>, wgpu::TextureView) {
        if let Some((t, v)) = &self.target {
            return (t.clone(), v.clone());
        }
        let texture = Arc::new(self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("court gpu target"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            // The shader stores through the unorm view; the blit samples
            // through the sRGB one when the surface re-encodes.
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Rgba8UnormSrgb],
        }));
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            ..Default::default()
        });
        self.target = Some((texture.clone(), view.clone()));
        (texture, view)
    }

    /// The target texture as sRGB bytes.
    ///
    /// The one readback on this tier, and it is not in the window: `--shot`
    /// takes its passes and then asks once, for the PNG.
    pub fn read_target(&self) -> anyhow::Result<Vec<u8>> {
        let (texture, _) = self.target.as_ref().ok_or_else(|| anyhow::anyhow!("no pass yet"))?;
        let (w, h) = self.size;
        // A texture-to-buffer copy wants its rows aligned; the padding comes
        // straight back out below.
        let row = (4 * w).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("court gpu readback"),
            size: (row as u64) * (h as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.ctx.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let mapped = slice.get_mapped_range()?;
        let mut out = Vec::with_capacity((4 * w * h) as usize);
        for y in 0..h as usize {
            let start = y * row as usize;
            out.extend_from_slice(&mapped[start..start + (4 * w) as usize]);
        }
        drop(mapped);
        staging.unmap();
        Ok(out)
    }
}
