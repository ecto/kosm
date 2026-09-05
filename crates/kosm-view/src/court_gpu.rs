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
//! What this side owns is **what moved**. There is no keep mask left here and
//! no scissor: every pass is the whole frame, and every pixel decides for
//! itself whether last pass's mean is still about the same thing.
//!
//! The device can carry a pixel across a *camera* move on its own — it has
//! this pass's depth and both views. It cannot carry it across an *object*
//! move, because nothing on the device knows the ball is somewhere else than
//! it was. This module does: it placed it both times. So every frame it notes,
//! for each placed instance — every ball part, every net segment — the range
//! of *faces* it owns in the merged scene, because a face index is exactly the
//! primitive id the integrator writes into the guide planes. Differencing this
//! frame's poses against **the pass the history is in** gives one
//! `prev_T · cur_T⁻¹` per instance that actually moved, and `InstanceMotion`'s
//! id table points that instance's faces at it. The court itself never moves
//! and stays `InstanceMotion::STATIC`.
//!
//! It is the pass and not the frame that is differenced because two passes of
//! one frame moved nothing: the second declares no motion, and the history is
//! already in its poses.
//!
//! The reprojection is offered only when something moved — the eye or an
//! object. A still camera over a still frame, which is what `--shot` is pass
//! after pass, would otherwise put every pixel through a depth and normal gate
//! it can only lose by, for two dispatches and no picture.
//!
//! `--history-cap`, `--clamp-k`, `--clamp-reset` and `--no-spatial-variance`
//! are the temporal knobs, over kosm-render's own defaults.
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
//! sends the very environment the CPU tier builds, and the sun with it: both
//! come off `render::Scene` rather than being rebuilt from knobs here, so
//! `sky 1` cannot mean one thing on the CPU and another on the GPU.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use kosm_render::gpu::{InstanceMotion, SampleBudget};
use kosm_spike::court::render::{self, Snapshot};
use vcad_kernel::Solid;
use vcad_kernel_gpu::GpuContext;
use vcad_kernel_math::{Point3, Transform};
use vcad_kernel_raytrace::gpu::{
    DEFAULT_FIREFLY_CLAMP, DEFAULT_RR_START, GpuAreaLight, GpuCamera, GpuDenoiseParams,
    GpuMaterial, GpuRenderState, GpuScene, HistoryPipeline, RayTracePipeline, ResidentScene,
};
use vcad_kernel_raytrace::pathtrace::{Environment, Pbr, PixelFilter, Sun};
// The learned denoiser is kosm-render's own; vcad re-exports the a-trous half
// of `gpu` and has no reason to know about this one.
use kosm_render::gpu::{NeuralDenoiser, NeuralPipeline};
use kosm_render::neural::Weights;

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

/// `--key value` or `--key=value`, for the temporal knobs.
fn flag(key: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().enumerate().find_map(|(i, a)| {
        a.strip_prefix(&format!("--{key}="))
            .map(str::to_owned)
            .or_else(|| {
                (a == &format!("--{key}"))
                    .then(|| args.get(i + 1).cloned())
                    .flatten()
            })
    })
}

/// The temporal knobs, off the command line, over kosm-render's own defaults.
///
/// `--history-cap N` bounds the exponential moving average, `--clamp-k K` is
/// how many standard errors of disagreement between the history and this
/// pass's neighbourhood it takes to shorten a pixel's history, and
/// `--no-spatial-variance` turns off SVGF's spatial estimate for the pixels
/// too young to have an error bar of their own. Defaults are
/// [`GpuDenoiseParams::default`]'s — 64, 4.0 and on.
fn denoise_from_args() -> GpuDenoiseParams {
    let mut d = GpuDenoiseParams::default();
    if let Some(v) = flag("history-cap").and_then(|v| v.parse::<u32>().ok()) {
        d.history_cap = v.max(1);
    }
    if let Some(v) = flag("clamp-k").and_then(|v| v.parse::<f32>().ok()) {
        d.clamp_k = v.max(0.0);
    }
    if let Some(v) = flag("clamp-reset").and_then(|v| v.parse::<u32>().ok()) {
        d.clamp_reset = v.max(1);
    }
    if std::env::args().any(|a| a == "--no-spatial-variance") {
        d.spatial_variance = false;
    }
    d
}

/// The gradient-directed sample budget off the command line.
///
/// `--budget` turns it on fully directed; `--budget=0.4` names the bias, where
/// 0 is the uniform spend the viewer always had and 1 is entirely where the
/// budget says. `--rays-per-frame N` is the frame's total, in samples folded,
/// and defaults to one per pixel — the same number a uniform pass folds, so
/// the flag moves the samples without spending more of them.
/// `--budget-radius R` is how far a moved instance's drive is dilated, over
/// the à-trous filter's own 32-pixel footprint, `--budget-floor K` guarantees
/// every pixel a sample once every K frames, and `--budget-rounds R` is how
/// many trace rounds a pass is split into — the range a pixel's share can
/// span, four by default.
///
/// Absent, nothing changes: every pixel folds every sample of the pass, which
/// is what it did before any of this.
fn budget_from_args() -> Option<SampleBudget> {
    let asked = std::env::args().any(|a| a == "--budget" || a.starts_with("--budget="));
    if !asked {
        return None;
    }
    let mut b = SampleBudget {
        bias: flag("budget")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0),
        ..SampleBudget::default()
    };
    if let Some(v) = flag("budget-radius").and_then(|v| v.parse::<u32>().ok()) {
        b.radius = v;
    }
    if let Some(v) = flag("budget-floor").and_then(|v| v.parse::<u32>().ok()) {
        b.floor_k = v.max(1);
    }
    if let Some(v) = flag("budget-rounds").and_then(|v| v.parse::<u32>().ok()) {
        b.rounds = v.max(1);
    }
    Some(b)
}

/// ReSTIR DI off the command line.
///
/// `--restir` turns it on at sixteen candidates; `--restir=32` names M.
/// `--restir-spatial N` and `--restir-radius R` are the reuse knobs, over the
/// renderer's own one pass at four pixels. Absent, nothing changes: the
/// shader takes the next-event path it always took and the picture is the one
/// it always was.
fn restir_from_args() -> Option<(u32, u32, f32)> {
    let asked = std::env::args().any(|a| a == "--restir" || a.starts_with("--restir="));
    if !asked {
        return None;
    }
    let m = flag("restir")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(16)
        .max(1);
    let spatial = flag("restir-spatial")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1);
    let radius = flag("restir-radius")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(4.0);
    Some((m, spatial, radius))
}

/// The court's own trained denoiser, shipped inside the binary.
///
/// Under a megabyte, so it is embedded rather than looked up beside the
/// executable: a viewer that has to find a file next to itself is a viewer
/// that breaks when someone moves it.
const BUNDLED_WEIGHTS: &[u8] = include_bytes!("../assets/denoise-court.bin");

/// Which filter `--denoise` asks for.
///
/// `--denoise atrous` (the default) is the a-trous wavelet chain in
/// `history.wgsl`. `--denoise neural` is the network in
/// `kosm_render::gpu::neural`, trained on this court's own reference renders
/// by kosm-spike's `denoise_dataset` example; `--denoise neural=PATH` runs a
/// different set of weights, which is how a new fit is looked at without a
/// rebuild.
fn neural_weights_from_args() -> anyhow::Result<Option<Weights>> {
    let Some(v) = flag("denoise") else {
        return Ok(None);
    };
    let (kind, path) = match v.split_once('=') {
        Some((k, p)) => (k, Some(p.to_owned())),
        None => (v.as_str(), None),
    };
    match kind {
        "atrous" | "none" => Ok(None),
        "neural" => {
            let w = match path {
                Some(p) => {
                    Weights::load(&p).map_err(|e| anyhow::anyhow!("the weights at {p}: {e}"))?
                }
                None => Weights::from_bytes(BUNDLED_WEIGHTS)
                    .map_err(|e| anyhow::anyhow!("the bundled weights: {e}"))?,
            };
            eprintln!(
                "denoise: neural, {} hidden channels, {} parameters",
                w.hidden,
                w.parameters()
            );
            Ok(Some(w))
        }
        other => anyhow::bail!("--denoise {other}: expected `atrous` or `neural[=weights.bin]`"),
    }
}

/// One placed instance in the merged scene: which faces it owns, and the
/// pose that put them where they are.
///
/// The faces are a contiguous range because `GpuScene::merge` appends, and
/// the *face index* is the primitive id the integrator writes into the guide
/// planes — so this range is exactly what
/// [`InstanceMotion`]'s id table wants filling in.
#[derive(Clone)]
struct Placement {
    faces: (u32, u32),
    to_world: Transform,
}

/// A rigid or affine transform as the row-major 3x4 the reprojection reads,
/// worked out by where it sends the origin and the three axes rather than by
/// reaching into the matrix.
fn row_major(t: &Transform) -> [f32; 12] {
    let o = t.apply_point(&Point3::new(0.0, 0.0, 0.0));
    let c = |x: f64, y: f64, z: f64| {
        let p = t.apply_point(&Point3::new(x, y, z));
        [p.x - o.x, p.y - o.y, p.z - o.z]
    };
    let (cx, cy, cz) = (c(1.0, 0.0, 0.0), c(0.0, 1.0, 0.0), c(0.0, 0.0, 1.0));
    [
        cx[0] as f32,
        cy[0] as f32,
        cz[0] as f32,
        o.x as f32, //
        cx[1] as f32,
        cy[1] as f32,
        cz[1] as f32,
        o.y as f32, //
        cx[2] as f32,
        cy[2] as f32,
        cz[2] as f32,
        o.z as f32,
    ]
}

/// Whether a 3x4 is close enough to the identity that declaring it would be
/// three more vec4s for nothing. Translations are in millimetres, so a
/// hundredth of one is well under a pixel at any size this tier renders.
fn is_identity(m: &[f32; 12]) -> bool {
    let id = InstanceMotion::IDENTITY;
    m.iter()
        .zip(id.iter())
        .enumerate()
        .all(|(i, (a, b))| (a - b).abs() <= if i % 4 == 3 { 1e-2 } else { 1e-5 })
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
    /// ReSTIR DI's `(candidates, spatial passes, radius)`, or `None` for the
    /// next-event path. `--restir` turns it on; see `restir_from_args`.
    restir: Option<(u32, u32, f32)>,
    /// The level's environment and sun, taken off the CPU tier's own
    /// `render::Scene` rather than rebuilt from the level's knobs. They reach
    /// the shader through `set_gradient_env` and `set_sun`, so the two tiers
    /// are lit by the same sky and the same daylight.
    env: Environment,
    sun: Option<Sun>,
    /// The merged scene for the frame on screen, and which frame that was.
    /// Assembling it is a clone of the statics and a placement per instance,
    /// which costs the same whatever the resolution — so it is done once per
    /// frame and not once per pass, and a paused window pays for it once.
    scene: Option<(u64, GpuScene)>,
    /// Where every moving instance's faces are in that merged scene, and the
    /// pose that put them there. Rebuilt with the scene, once a frame.
    placements: Vec<Placement>,
    /// How many faces the merged scene has — the length of
    /// [`InstanceMotion`]'s id table.
    face_count: u32,
    /// The placements the *previous pass* folded into the history. What the
    /// reprojection has to be differenced against is the pass, not the frame:
    /// two passes of one frame moved nothing, and the second declares no
    /// motion at all.
    prev_placements: Option<Vec<Placement>>,
    /// How many of `placements` are ball parts. They come first, they are the
    /// same count every frame, and the net's are what changes — so when the
    /// two passes' instance lists do not line up, this is how much of them
    /// still does.
    ball_places: usize,
    prev_ball_places: usize,
    /// The court on the device. Built on the first pass, kept across every
    /// one after it: a frame rewrites the placements, a pass rewrites the
    /// camera. `uploaded` is the frame whose placements are currently in it.
    resident: Option<ResidentScene>,
    uploaded: Option<u64>,
    /// The history and denoise passes, compiled once.
    history: HistoryPipeline,
    /// The learned filter, when `--denoise neural` asked for one: the three
    /// compute pipelines and the weights and activations they run over. It
    /// stands exactly where the a-trous chain stands - see
    /// `RayTracePipeline::denoise_and_resolve_resident_neural` - so
    /// everything else about a pass is the same either way.
    neural: Option<(NeuralPipeline, NeuralDenoiser)>,
    /// How the device filters the running mean. The default fades the filter
    /// out as a pixel reaches thirty-two samples, which is
    /// `History::resolve`'s `DENOISE_UNTIL` and right for a window that keeps
    /// converging. A still that stops at thirty-two wants the filter at full
    /// strength instead — see [`Stage::always_denoise`].
    denoise: GpuDenoiseParams,
    /// The gradient-directed sample budget, or `None` for the uniform spend.
    budget: Option<SampleBudget>,
    /// Whether the budget line has been printed.
    said_budget: bool,
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
    /// The camera that pass was taken from, in the level's own units, so
    /// "did the camera move?" is answered exactly rather than by comparing
    /// two packed bases.
    last_camera: Option<Camera>,
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
    ) -> anyhow::Result<Self> {
        let ctx = GpuContext {
            device: device.clone(),
            queue: queue.clone(),
        };
        let pipeline = vcad_kernel_raytrace::gpu::brep_pipeline(&ctx)
            .map_err(|e| anyhow::anyhow!("the tracer's pipeline: {e}"))?;
        let history = HistoryPipeline::new(&ctx)
            .map_err(|e| anyhow::anyhow!("the history's pipelines: {e}"))?;
        // Sized for nothing yet; the first pass calls `ensure` with the frame
        // it actually got.
        let neural = match neural_weights_from_args()? {
            Some(w) => Some((
                NeuralPipeline::new(&ctx)
                    .map_err(|e| anyhow::anyhow!("the neural denoiser's pipelines: {e}"))?,
                NeuralDenoiser::new(&ctx, &w, 0, 0),
            )),
            None => None,
        };
        let filter = pixel_filter_from_args();
        let restir = restir_from_args();
        if let Some((m, sp, r)) = restir {
            eprintln!("court  gpu: ReSTIR DI on — {m} candidates, {sp} spatial pass(es) at {r} px");
        }

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
            placements: Vec::new(),
            face_count: 0,
            prev_placements: None,
            ball_places: 0,
            prev_ball_places: 0,
            max_depth,
            filter,
            restir,
            env: stage.environment().clone(),
            sun: stage.sun(),
            resident: None,
            uploaded: None,
            history,
            neural,
            denoise: denoise_from_args(),
            budget: budget_from_args(),
            said_budget: false,
            target: None,
            passes: 0,
            size: (0, 0),
            last_view: None,
            last_camera: None,
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
    fn at(&mut self, snap: &Snapshot, stage: &render::Scene) -> (GpuScene, Vec<Placement>) {
        let mut scene = self.statics.clone();
        // The statics own faces 0..statics.faces.len() and never move, so
        // they get no placement and their ids stay `InstanceMotion::STATIC`.
        let mut places: Vec<Placement> = Vec::new();
        for at in stage.ball_placements(snap) {
            for part in &self.ball {
                let start = scene.faces.len() as u32;
                scene = scene.merge(part.placed(&at));
                places.push(Placement {
                    faces: (start, scene.faces.len() as u32),
                    to_world: at.clone(),
                });
            }
        }
        self.ball_places = places.len();
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
                let start = scene.faces.len() as u32;
                scene = scene.merge(placed);
                places.push(Placement {
                    faces: (start, scene.faces.len() as u32),
                    to_world: to_world.clone(),
                });
            }
        }
        scene.lights = self.lights.clone();
        (scene, places)
    }

    /// What moved since the pass the history is currently in, packed for the
    /// reprojection.
    ///
    /// One instance slot per thing that actually moved — a ball part, a net
    /// segment — carrying `prev_T · cur_T⁻¹`, and the id table pointing every
    /// face of that instance at its slot. Everything else, which is the whole
    /// court, stays [`InstanceMotion::STATIC`] and costs a lane in the table.
    ///
    /// `None` when there is nothing to say: no previous pass, a frame whose
    /// instances are not the ones the previous pass had (a net that grew a
    /// segment is a different scene, and the ids are no longer comparable),
    /// or a second pass of a frame nothing moved in.
    fn motion(&self) -> (Option<InstanceMotion>, usize) {
        let Some(prev) = self.prev_placements.as_ref() else {
            return (None, 0);
        };
        let cur = &self.placements;
        if self.face_count == 0 {
            return (None, 0);
        }
        // The instance lists line up frame to frame while the net keeps the
        // same number of cords. When it does not — a cord went degenerate and
        // was dropped — the balls still line up, and they are the part of the
        // picture the eye is following. Reprojecting them and leaving the net
        // to the per-pixel clamp keeps a history that used to be thrown away
        // whole. The ball parts are the head of both lists, in the order
        // `at` merged them.
        let paired = if prev.len() == cur.len() {
            cur.len()
        } else if self.prev_ball_places == self.ball_places && self.ball_places > 0 {
            self.ball_places
        } else {
            return (None, 0);
        };
        let mut ids = vec![InstanceMotion::STATIC; self.face_count as usize];
        let mut mats: Vec<[f32; 12]> = Vec::new();
        for (p, c) in prev.iter().take(paired).zip(cur.iter().take(paired)) {
            if p.faces != c.faces {
                return (None, 0);
            }
            let Some(inv) = c.to_world.inverse() else {
                continue;
            };
            let m = row_major(&p.to_world.then(&inv));
            if is_identity(&m) {
                continue;
            }
            let slot = mats.len() as u32;
            for id in c.faces.0..c.faces.1.min(self.face_count) {
                ids[id as usize] = slot;
            }
            mats.push(m);
        }
        if mats.is_empty() {
            return (None, 0);
        }
        let n = mats.len();
        (Some(InstanceMotion::new(&ids, &mats)), n)
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
    /// There is no keep mask and no scissor on this tier any more. What tells
    /// the history what is still true is the pass itself: the previous
    /// camera, and [`Stage::motion`]'s per-instance `prev_T · cur_T⁻¹`. Every
    /// pass is a full-frame pass, and every pixel decides for itself.
    ///
    /// What comes back is the texture the picture is now in, on the viewport's
    /// own device. Nothing was read back to make it.
    pub fn accumulate(
        &mut self,
        stage: &render::Scene,
        snap: &Snapshot,
        frame_id: u64,
        camera: &Camera,
        size: (u32, u32),
        samples: u32,
    ) -> anyhow::Result<Arc<wgpu::Texture>> {
        let n = (size.0 as u64) * (size.1 as u64);
        anyhow::ensure!(n > 0, "an empty picture");
        let assembled = Instant::now();
        if self.scene.as_ref().is_none_or(|(id, _)| *id != frame_id) {
            let (scene, places) = self.at(snap, stage);
            self.face_count = scene.faces.len() as u32;
            self.placements = places;
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
        // What moved since the pass the history is in — the balls, the net,
        // and whether the eye did.
        let (motion, moving) = self.motion();
        let camera_moved = self.last_camera != Some(*camera);
        // The reprojection is worth its two dispatches when something moved.
        // A still camera over a still frame — which is what `--shot` is, pass
        // after pass — reprojects a view onto itself for nothing, and asking
        // for it would put every pixel through a depth and normal gate it can
        // only lose by.
        let prev_view = self
            .last_view
            .filter(|(s, _)| *s == size && (camera_moved || motion.is_some()))
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
        // climbing `frame_index` to move the jitter and the RNG. The whole
        // frame every time — there is no box and no mask left on this tier.
        let mut accumulated = Duration::ZERO;
        let asked = samples.max(1);

        // Where this pass's samples go, decided before a ray is traced. The
        // budget reads the guide planes and the raw sample the *previous*
        // pass left resident, and the motion table this pass is about to
        // reproject with — so a ball that is about to move has already asked
        // for the samples by the time the trace starts.
        //
        // The pass folds `asked` samples per pixel on average, exactly as the
        // uniform loop does, and takes `asked * budget.rounds` rounds to place
        // them: a pixel cannot be given four times its share out of one round,
        // and the rounds are where the range comes from. Each round is still a
        // full-frame trace, because the integrator takes one sample per
        // invocation and skipping a pixel inside it is not this crate's line
        // to write — so what `--budget` buys today is *placement* at the cost
        // of trace dispatches, and it goes free the day the trace can skip.
        let budget = self.budget.map(|b| SampleBudget {
            rays_per_frame: (size.0 as f32) * (size.1 as f32) * asked as f32,
            rounds: asked * b.rounds.max(1),
            ..b
        });
        let rounds = budget.as_ref().map(|b| b.rounds).unwrap_or(asked);
        if let Some(b) = budget.as_ref() {
            if !self.said_budget {
                self.said_budget = true;
                eprintln!(
                    "court  gpu: the samples are directed — bias {:.2}, {} rounds, \
                     {:.0} samples a pass, every pixel served within {} frames",
                    b.bias, b.rounds, b.rays_per_frame, b.floor_k,
                );
            }
            self.pipeline
                .budget_frame(
                    &self.ctx,
                    &self.history,
                    res,
                    &cam,
                    b,
                    motion.as_ref(),
                    self.passes,
                )
                .map_err(|e| anyhow::anyhow!("the budget: {e}"))?;
        }
        for k in 0..rounds {
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
            // …and the same sun, which with `sky 1` is the only thing the
            // clerestory openings have to let in.
            state.set_sun(self.sun.as_ref());
            // ReSTIR, if asked for. The reservoirs live in the resident
            // scene and carry across passes on their own; every sample of
            // the pass gets its own generation, temporal reuse and spatial
            // round, because each is an independent sample of the frame.
            if let Some((m, sp, r)) = self.restir {
                state.set_restir(m, sp, r);
            }
            match budget.as_ref() {
                Some(b) => self
                    .pipeline
                    .accumulate_resident_round(
                        &self.ctx,
                        &self.history,
                        res,
                        &cam,
                        state,
                        &[],
                        if k == 0 { prev_view.as_ref() } else { None },
                        &denoise,
                        if k == 0 { motion.as_ref() } else { None },
                        b,
                        k,
                        self.passes,
                    )
                    .map_err(|e| anyhow::anyhow!("the tracer: {e}"))?,
                None => self
                    .pipeline
                    .accumulate_resident_temporal(
                        &self.ctx,
                        &self.history,
                        res,
                        &cam,
                        state,
                        // No keep mask: every pixel keeps what it has until
                        // the reprojection cannot find it or the clamp
                        // shortens it.
                        &[],
                        // Only the first sample of the pass reprojects. After
                        // it the history is already in this pass's view, and
                        // the camera does not move between the samples of one
                        // pass.
                        if k == 0 { prev_view.as_ref() } else { None },
                        &denoise,
                        if k == 0 { motion.as_ref() } else { None },
                    )
                    .map_err(|e| anyhow::anyhow!("the tracer: {e}"))?,
            }
            accumulated += box_started.elapsed();
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
        match self.neural.as_mut() {
            Some((pipe, net)) => self
                .pipeline
                .denoise_and_resolve_resident_neural(
                    &self.ctx,
                    &self.history,
                    pipe,
                    net,
                    res,
                    &denoise,
                    &view,
                )
                .map_err(|e| anyhow::anyhow!("the neural denoiser: {e}"))?,
            None => self
                .pipeline
                .denoise_and_resolve_resident(&self.ctx, &self.history, res, &denoise, &view)
                .map_err(|e| anyhow::anyhow!("the denoiser: {e}"))?,
        }

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
                 {:.1} ms tracing ({} sample{}, {} moving instance{}, {:.1} ms tracing and \
                 accumulating, {:.1} ms denoising once)",
                size.0,
                size.1,
                assembly.as_secs_f64() * 1e3,
                upload.as_secs_f64() * 1e3,
                traced.elapsed().as_secs_f64() * 1e3,
                samples.max(1),
                if samples.max(1) == 1 { "" } else { "s" },
                moving,
                if moving == 1 { "" } else { "s" },
                accumulated.as_secs_f64() * 1e3,
                denoise_time.as_secs_f64() * 1e3,
            );
        }
        self.last_view = Some((size, cam));
        self.last_camera = Some(*camera);
        // What the *next* pass differences against. The history is now in
        // this pass's poses, whatever the frame does after it.
        self.prev_placements = Some(self.placements.clone());
        self.prev_ball_places = self.ball_places;
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
