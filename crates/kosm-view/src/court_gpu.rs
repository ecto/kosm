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
//! What the GPU picture is not: the shader's environment is an analytic studio
//! gradient where the CPU's is the level's constant grey, and the shader's
//! implicit ground plane is switched off because the level authors its own
//! floor while the CPU path also gets an infinite one at the slab's underside.
//! Neither shows through a closed gym at `env_radiance = 0.05`, but they are
//! why the two images are alike and not identical.

use std::collections::HashMap;
use std::time::Instant;

use kosm_spike::court::render::{self, Snapshot};
use vcad_kernel::Solid;
use vcad_kernel_gpu::GpuContext;
use vcad_kernel_math::Transform;
use vcad_kernel_raytrace::gpu::{
    GpuAreaLight, GpuCamera, GpuMaterial, GpuRenderState, GpuScene, RayTracePipeline,
    DEFAULT_FIREFLY_CLAMP, DEFAULT_RR_START,
};
use vcad_kernel_raytrace::pathtrace::Pbr;

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
    /// The analytic environment's brightness, from the level's `env_radiance`.
    env_intensity: f32,
    /// The merged scene for the frame being accumulated. Assembling it is a
    /// clone of the statics and a placement per instance, which costs the
    /// same whatever the resolution — so it is done once per subject and not
    /// once per pass, and a paused window pays for it once.
    scene: Option<GpuScene>,
    /// Progressive accumulation, and how many passes are in it. Thrown away
    /// whenever the subject or the size changes.
    accum: Option<wgpu::Buffer>,
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
        env_intensity: f32,
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

        // The ball's seams do not go on the GPU. They are four thin tori, and
        // the shader traces a torus wider than the solid says it is — an
        // untrimmed one, near enough: a torus that covers 330 pixels under the
        // CPU integrator covers 603 under the compute shader, at the identity
        // transform, with nothing placed. On a seam that means each ring swells
        // until it engulfs the ball it is drawn on, and a ball whose seams are
        // near-black comes out a black blob. Better a ball with no seams on it
        // than a ball that is not a ball. The rim is a torus too and is drawn:
        // at its size the difference does not read, and losing the hoop would
        // cost more than it saves.
        let ball: Vec<GpuScene> = stage
            .ball_parts()
            .filter(|(name, _, _)| !matches!(*name, "ball-seams" | "seam"))
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
            env_intensity,
            accum: None,
            passes: 0,
            size: (0, 0),
        })
    }

    /// Throw the accumulator away: the picture is now of something else.
    pub fn reset(&mut self) {
        self.accum = None;
        self.scene = None;
        self.passes = 0;
    }

    /// How many passes are in the picture now.
    pub fn passes(&self) -> u32 {
        self.passes
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

    /// One more pass on the picture, accumulated onto whatever is already
    /// there. Returns sRGB bytes at `size`, ready for the blit.
    pub fn pass(
        &mut self,
        stage: &render::Scene,
        snap: &Snapshot,
        camera: &Camera,
        size: (u32, u32),
    ) -> anyhow::Result<Vec<u8>> {
        if self.size != size {
            self.size = size;
            self.reset();
        }
        let assembled = Instant::now();
        if self.scene.is_none() {
            self.scene = Some(self.at(snap, stage));
        }
        let assembly = assembled.elapsed();
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
        state.env_intensity = self.env_intensity;

        let cam = GpuCamera::new(
            [camera.eye.x as f32, camera.eye.y as f32, camera.eye.z as f32],
            [camera.target.x as f32, camera.target.y as f32, camera.target.z as f32],
            [0.0, 0.0, 1.0],
            (camera.fov_deg as f32).to_radians(),
            size.0,
            size.1,
        );
        let traced = Instant::now();
        let scene = self.scene.as_ref().expect("just assembled");
        let (pixels, accum) = pollster::block_on(self.pipeline.render_with_render_state(
            &self.ctx,
            scene,
            &cam,
            size.0,
            size.1,
            self.accum.take(),
            state,
        ))
        .map_err(|e| anyhow::anyhow!("the tracer: {e}"))?;
        self.accum = Some(accum);
        // Where a pass goes, when anyone asks. The trace includes the upload
        // of every buffer and the readback of the image: the pipeline builds
        // its buffers per call, so a pass pays for the whole scene crossing
        // the bus whether or not it changed.
        if std::env::var("KOSM_GPU_TIMING").is_ok() {
            eprintln!(
                "court  gpu: {}×{} pass — {:.1} ms assembling, {:.1} ms tracing",
                size.0,
                size.1,
                assembly.as_secs_f64() * 1e3,
                traced.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(pixels)
    }
}

/// The identity, for a part that is already where it belongs.
#[allow(dead_code)]
pub fn identity() -> Transform {
    Transform::identity()
}
