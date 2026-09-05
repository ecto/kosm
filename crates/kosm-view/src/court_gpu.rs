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
//! from the same geometry the CPU tier masks with.
//!
//! A moved camera used to upload an all-restart mask, because reprojection
//! was the CPU tier's alone and nothing on the device knew where last frame's
//! pixel had gone. It is on the device now: a pass whose camera moved calls
//! `accumulate_and_denoise_resident_reprojected` with the previous pass's
//! camera, and each pixel is unprojected through this pass's depth, projected
//! back into the previous view, and keeps that pixel's mean and count where
//! the surfaces agree. The keep mask on such a pass is the *still-camera*
//! one — only the rectangles the world moved under — and the reprojection
//! settles the rest. Only disocclusions restart, so an orbit no longer looks
//! like a frame of noise per mouse move.
//!
//! The previous camera is only offered when it is worth offering: a pass whose
//! camera did not move passes `None` (reprojecting a view onto itself is two
//! dispatches for nothing), and so does the first pass at a new size, since
//! vcad reallocates the history on a resize and there is no previous plane to
//! test against.
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
//! ## one denoise a frame, not one a box
//!
//! vcad's scissor is a single rectangle, so `k` dirty boxes are `k` calls.
//! Fused, that was `k` of *everything*: `k` traces, which is what was wanted,
//! and also `k` demodulate/à-trous/resolve chains over the whole frame, which
//! was not — the denoise cannot be scissored (the filter reaches 32 pixels off
//! a box's edge and the resolve has to leave the texture whole), so a viewer
//! with four small boxes paid four full-frame filters to show one frame.
//!
//! [`Stage::accumulate`] uses vcad's split now: `accumulate_resident` once per
//! box — trace and fold, both scissored, and the fold's *dispatch* is the
//! box's workgroups rather than the frame's — then
//! `denoise_and_resolve_resident` once for the pass. Measured in vcad's own
//! suite at 512x288 with four boxes of a tenth of the frame each: 5.4 ms fused
//! against 3.9 ms split. `KOSM_GPU_TIMING` prints the two halves separately.
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
use std::time::{Duration, Instant};

use kosm_spike::court::render::{self, Snapshot};
use vcad_kernel::Solid;
use vcad_kernel_gpu::GpuContext;
use vcad_kernel_raytrace::gpu::{
    DEFAULT_FIREFLY_CLAMP, DEFAULT_RR_START, GpuAreaLight, GpuCamera, GpuDenoiseParams,
    GpuMaterial, GpuRenderState, GpuScene, HistoryPipeline, RayTracePipeline, ResidentScene,
};
use vcad_kernel_raytrace::pathtrace::{Environment, Pbr, PixelFilter};

use crate::court::Camera;

/// `--filter box|gaussian|blackman`, read once when the tracer is built.
///
/// Box is the default and is the uniform jitter the tracer has always used,
/// so a shot taken with no flag is bit-for-bit the shot that was there
/// before. The other two importance-sample a real reconstruction filter,
/// which costs nothing per sample and shows up on the rim and the net.
fn pixel_filter_from_args() -> PixelFilter {
    let want = std::env::args()
        .skip_while(|a| a != "--filter")
        .nth(1)
        .unwrap_or_default()
        .to_ascii_lowercase();
    match want.as_str() {
        "gaussian" => PixelFilter::Gaussian,
        "blackman" | "blackman-harris" | "bh" => PixelFilter::BlackmanHarris,
        _ => PixelFilter::Box,
    }
}

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
    /// Where in the pixel a primary ray is aimed. `--filter gaussian` or
    /// `--filter blackman` picks a real reconstruction filter; the default is
    /// the uniform jitter every earlier frame was drawn with, so a shot taken
    /// without the flag is the shot that was there before.
    filter: PixelFilter,
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
    /// The camera the last pass rendered from, and the size it rendered at —
    /// what a reprojected pass unprojects into. `None` until the first pass.
    last_view: Option<((u32, u32), GpuCamera)>,
    /// Whether the last pass carried its history across a camera move, and
    /// whether that has ever been said out loud.
    reprojected: bool,
    said_reprojected: bool,
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
        let ctx = GpuContext {
            device: device.clone(),
            queue: queue.clone(),
        };
        let pipeline = vcad_kernel_raytrace::gpu::brep_pipeline(&ctx)
            .map_err(|e| anyhow::anyhow!("the tracer's pipeline: {e}"))?;
        let history = HistoryPipeline::new(&ctx)
            .map_err(|e| anyhow::anyhow!("the history's pipelines: {e}"))?;
        let filter = pixel_filter_from_args();

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

        let lights: Vec<GpuAreaLight> = stage
            .lights()
            .iter()
            .map(GpuAreaLight::from_area_light)
            .collect();

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
            filter,
            env: Environment::constant([env_radiance; 3]),
            resident: None,
            uploaded: None,
            history,
            denoise: GpuDenoiseParams::default(),
            target: None,
            passes: 0,
            size: (0, 0),
            last_view: None,
            reprojected: false,
            said_reprojected: false,
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
    /// `reproject` says the camera moved and the history should follow it
    /// rather than start over — the previous pass's camera goes to vcad as
    /// `prev_view` and the device carries every pixel whose surface it can
    /// find again. It is honoured only when there *is* a previous pass at
    /// this same size, and only for the pass's first sample: the camera does
    /// not move between the samples of one pass.
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
        boxes: &[[u32; 4]],
        samples: u32,
        reproject: bool,
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
                self.resident = Some(
                    self.pipeline
                        .resident_scene(&self.ctx, scene, size.0, size.1),
                );
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
            [
                camera.eye.x as f32,
                camera.eye.y as f32,
                camera.eye.z as f32,
            ],
            [
                camera.target.x as f32,
                camera.target.y as f32,
                camera.target.z as f32,
            ],
            [0.0, 0.0, 1.0],
            (camera.fov_deg as f32).to_radians(),
            size.0,
            size.1,
        );
        let denoise = GpuDenoiseParams {
            exposure: camera.exposure,
            ..self.denoise
        };
        // The view the history is currently in. Only a previous pass at the
        // same size can be reprojected from — vcad reallocates the history on
        // a resize, so a stepped size has no previous plane to test against
        // and is a restart whatever the caller asked for.
        let prev_view = self
            .last_view
            .filter(|(s, _)| reproject && *s == size)
            .map(|(_, c)| c);
        self.reprojected = prev_view.is_some();
        if self.reprojected && !self.said_reprojected {
            self.said_reprojected = true;
            eprintln!(
                "court  gpu: the history follows the camera — passes on a moved camera reproject"
            );
        }
        let traced = Instant::now();
        // Timing splits the pass in two, which costs a sync point the render
        // path does not otherwise want.
        let timing = std::env::var("KOSM_GPU_TIMING").is_ok();
        let res = self.resident.as_mut().expect("just built");
        // `samples` samples, each its own call: vcad's accumulate folds one
        // raw sample per call, so a pass of several is several calls with a
        // climbing `frame_index` to move the jitter and the RNG. Only the
        // first carries the keep mask — the pixels this pass restarts are
        // restarted once, and the rest of the pass accumulates onto them.
        // One dispatch per box. vcad's scissor is a single rectangle, so k
        // boxes are k calls: the *trace* shrinks to each box, though the fold
        // and the denoise chain behind it do not. An empty list is the whole
        // frame, in one call, exactly as before.
        let dispatches: Vec<Option<[u32; 4]>> = if boxes.is_empty() {
            vec![None]
        } else {
            boxes.iter().map(|&b| Some(b)).collect()
        };
        let mut accumulated = Duration::ZERO;
        for k in 0..samples.max(1) {
            for (b, rect) in dispatches.iter().enumerate() {
            let box_started = Instant::now();
            self.passes += 1;
            let mut state = GpuRenderState::new(self.passes);
            // A photoreal viewport: no edge overlay, no stylisation, and no
            // implicit ground plane — the level authors its own floor.
            state.enable_edges = 0;
            state.stylize = 0;
            state.ground_enabled = 0;
            state.max_depth = self.max_depth;
            state.set_pixel_filter(self.filter);
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
            if let Some(rect) = *rect {
                state.set_scissor(rect);
            }
            self.pipeline
                .accumulate_resident(
                    &self.ctx,
                    &self.history,
                    res,
                    &cam,
                    state,
                    // The boxes are disjoint and the fold is scissored, so
                    // each box restarts its own pixels once and no box can
                    // touch another's.
                    if k == 0 { keep } else { &[] },
                    // Only the first sample of the pass, and only its first
                    // box: after that the history is already in this pass's
                    // view. The reprojection gathers over the whole frame and
                    // every later box reads that gather out of the scratch
                    // pair, so it is a once-per-pass thing whatever the boxes
                    // are.
                    if k == 0 && b == 0 {
                        prev_view.as_ref()
                    } else {
                        None
                    },
                )
                .map_err(|e| anyhow::anyhow!("the tracer: {e}"))?;
            accumulated += box_started.elapsed();
            }
        }

        // With no sync here `accumulated` is the cost of *submitting* the box
        // passes, which is not what anyone reading a timing line wants to
        // know. Close them out on the device first, and take the whole
        // elapsed time rather than the sum of the per-box submits.
        if timing {
            self.ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
            accumulated = traced.elapsed();
        }

        // One denoise for the whole pass, however many boxes went into it.
        // This is the half that cannot be scissored — the filter reaches 32
        // pixels off a box's edge and the resolve has to leave the target
        // texture whole — and it used to run once per box per sample. At four
        // boxes that was four demodulate/à-trous/resolve chains over the full
        // frame to show one frame.
        let denoised = Instant::now();
        self.pipeline
            .denoise_and_resolve_resident(&self.ctx, &self.history, res, &denoise, &view)
            .map_err(|e| anyhow::anyhow!("the denoiser: {e}"))?;

        // Wait for the passes to land. Not a readback — no pixel comes back —
        // but with nothing else synchronising the two sides the worker would
        // queue passes faster than the device retires them, and the window's
        // tuner would be timing `queue.submit` rather than the render. A pass
        // has to be a pass before it can be measured.
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
        let denoise_time = denoised.elapsed();

        // Where a pass goes, when anyone asks.
        if timing {
            eprintln!(
                "court  gpu: {}\u{d7}{} pass \u{2014} {:.1} ms assembling, {:.1} ms uploading, \
                 {:.1} ms tracing ({} box{} \u{d7} {} sample{}, {:.1} ms tracing and \
                 accumulating, {:.1} ms denoising once)",
                size.0,
                size.1,
                assembly.as_secs_f64() * 1e3,
                upload.as_secs_f64() * 1e3,
                traced.elapsed().as_secs_f64() * 1e3,
                dispatches.len(),
                if dispatches.len() == 1 { "" } else { "es" },
                samples.max(1),
                if samples.max(1) == 1 { "" } else { "s" },
                accumulated.as_secs_f64() * 1e3,
                denoise_time.as_secs_f64() * 1e3,
            );
        }
        self.last_view = Some((size, cam));
        Ok(texture)
    }

    /// Whether the last pass carried its history across a camera move.
    pub fn reprojected(&self) -> bool {
        self.reprojected
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
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            // The shader stores through the unorm view; the blit samples
            // through the sRGB one when the surface re-encodes.
            view_formats: &[
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            ],
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
    /// The device's own sample count for every pixel, read back.
    ///
    /// Nothing in the render path wants this — a pass reads nothing back —
    /// but a test that asks what a camera move cost has to ask the device,
    /// since the host's mirror of the counts cannot know which pixels the
    /// reprojection failed to match.
    pub fn history_counts(&mut self) -> anyhow::Result<Vec<u32>> {
        let res = self
            .resident
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no pass yet"))?;
        let hist = pollster::block_on(self.pipeline.read_history(&self.ctx, res))
            .map_err(|e| anyhow::anyhow!("the tracer: {e}"))?
            .ok_or_else(|| anyhow::anyhow!("no history yet"))?;
        Ok(hist.count)
    }

    pub fn read_target(&self) -> anyhow::Result<Vec<u8>> {
        let (texture, _) = self
            .target
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no pass yet"))?;
        let (w, h) = self.size;
        // A texture-to-buffer copy wants its rows aligned; the padding comes
        // straight back out below.
        let row = (4 * w).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
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
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
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
