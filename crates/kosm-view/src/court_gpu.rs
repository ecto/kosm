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
//! What comes out of [`Stage::sample`] is one *raw* linear sample of the
//! frame, with the denoiser's guide planes, packed into the same
//! `pathtrace::Film` the CPU integrator produces. Nothing accumulates on the
//! device: the accumulator in this program is [`crate::history`] — one
//! per-pixel running mean, one geometric change mask, one reprojection — for
//! both tiers. The shader's own progressive average, its spatial denoise and
//! its tonemap all happen on the way to an output texture that is thrown
//! away.
//!
//! ## residency
//!
//! The court is uploaded once and stays there. `ResidentScene` holds the
//! surface, face, BVH, material and light buffers; a frame rewrites only the
//! bytes that moved (`update_scene`, once per *frame*, not once per pass) and
//! a pass rewrites only the camera and the render state. So a pass is a
//! dispatch and a readback, not a re-upload of the whole court — which is
//! what `RayTracePipeline::render_with_render_state` used to make it, and what
//! the fixed term in the window's cost model was mostly paying for.
//!
//! `render_resident_linear` is the exit that makes this usable: it forces the
//! shader into raw-sample mode and hands back linear radiance plus depth,
//! normal and albedo, in exactly `pathtrace::render`'s conventions. That is
//! why the GPU tier now reprojects through a moved camera and runs the à-trous
//! denoiser, like the CPU one, instead of throwing its whole history away
//! whenever the camera turns.
//!
//! ## what the GPU picture is not
//!
//! - A root with no BRep — the painted markings, which are drawn and not
//!   modelled — cannot be packed and is not in the GPU picture at all.
//! - The shader's implicit ground plane is switched off because the level
//!   authors its own floor, while the CPU path also gets an infinite one at
//!   the slab's underside.
//! - The environment is the shader's analytic studio gradient scaled by the
//!   level's `env_radiance`, where the CPU's is that constant flat. Making
//!   them agree exactly is possible — a 1x1 lat-long map is a constant
//!   environment the shader will take — and it was tried: at
//!   `env_radiance = 0.05` in a closed gym lit by ten panels at 18 it changed
//!   the 960x540 still by **less than one code value anywhere in the frame**,
//!   and cost 60% more per pass, because an environment *image* is a light
//!   the shader draws a next-event sample towards on every bounce. So the
//!   gradient stays. It is not why the two tiers differ in brightness.

use std::collections::HashMap;
use std::time::Instant;

use kosm_spike::court::render::{self, Snapshot};
use vcad_kernel::Solid;
use vcad_kernel_gpu::GpuContext;
use vcad_kernel_raytrace::gpu::{
    GpuAreaLight, GpuCamera, GpuMaterial, GpuRenderState, GpuScene, RayTracePipeline, ResidentScene,
    DEFAULT_FIREFLY_CLAMP, DEFAULT_RR_START,
};
use vcad_kernel_raytrace::pathtrace::{Film, Pbr};

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
    /// The level's `env_radiance`: the brightness of the shader's analytic
    /// environment. See [`Stage::new`] for why it is not the CPU's constant.
    env_radiance: f32,
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

        let mut dropped = 0usize;
        let packed: Vec<GpuScene> = stage
            .static_parts()
            .filter_map(|(solid, pbr, to_world)| match pack(solid, pbr) {
                Some(s) => Some(s.placed(to_world)),
                None => {
                    dropped += 1;
                    None
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
            .filter_map(|(_, solid, pbr)| pack(solid, pbr))
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
            env_radiance,
            resident: None,
            uploaded: None,
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

    /// One fresh sample of the frame, in linear radiance, with guides.
    ///
    /// The court is already on the device. A *frame* rewrites the placements
    /// — `update_scene`, with the clone-and-merge in [`Stage::at`] behind it
    /// — and a *pass* rewrites the camera and the render state and nothing
    /// else, so the second and later passes of a still picture cost a
    /// dispatch and a readback. `frame_index` still climbs: it is what moves
    /// the shader's Halton jitter and its RNG, so two passes of one frame are
    /// two samples.
    ///
    /// `render_resident_linear` forces raw-sample mode, so what comes back is
    /// one unweighted sample and not a step of the shader's running average,
    /// and it fills `depth`, `normal` and `albedo` in `pathtrace::render`'s
    /// conventions. That is what lets [`crate::history`] reproject this tier
    /// through a moved camera and denoise it, as it always could the CPU's.
    ///
    /// `scissor` is `[x, y, w, h]`, and it is what makes a pass cost what
    /// moved: the dispatch is sized to the rectangle and every invocation is
    /// offset into it. **Pixels outside come back stale** — whatever the
    /// previous pass left in the device's buffers, not zero — which is
    /// exactly the CPU tier's `render_into` contract, and the history is told
    /// which rectangle was fresh.
    pub fn sample(
        &mut self,
        stage: &render::Scene,
        snap: &Snapshot,
        frame_id: u64,
        camera: &Camera,
        size: (u32, u32),
        scissor: Option<[u32; 4]>,
    ) -> anyhow::Result<Film> {
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
                self.resident =
                    Some(self.pipeline.resident_scene(&self.ctx, scene, size.0, size.1));
            }
        }
        self.uploaded = Some(frame_id);
        self.size = size;
        let upload = uploaded.elapsed();
        self.passes += 1;

        let mut state = GpuRenderState::new(self.passes);
        // A photoreal viewport: no edge overlay, no stylisation, and no
        // implicit ground plane — the level authors its own floor.
        state.enable_edges = 0;
        state.stylize = 0;
        state.ground_enabled = 0;
        state.max_depth = self.max_depth;
        state.rr_start = DEFAULT_RR_START;
        state.firefly_clamp = DEFAULT_FIREFLY_CLAMP;
        state.env_intensity = self.env_radiance;
        if let Some(r) = scissor.filter(|r| r[2] > 0 && r[3] > 0) {
            state.set_scissor(r);
        }

        let cam = GpuCamera::new(
            [camera.eye.x as f32, camera.eye.y as f32, camera.eye.z as f32],
            [camera.target.x as f32, camera.target.y as f32, camera.target.z as f32],
            [0.0, 0.0, 1.0],
            (camera.fov_deg as f32).to_radians(),
            size.0,
            size.1,
        );
        let traced = Instant::now();
        let res = self.resident.as_mut().expect("just built");
        let film =
            pollster::block_on(self.pipeline.render_resident_linear(&self.ctx, res, &cam, state))
                .map_err(|e| anyhow::anyhow!("the tracer: {e}"))?;

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
        Ok(film)
    }
}
