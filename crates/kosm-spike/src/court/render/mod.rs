//! The court, lit: vcad's BRep path tracer.
//!
//! There is no renderer here. The picture is `vcad-kernel-raytrace`'s
//! `pathtrace` — the same integrator `vcad-render --photoreal` uses — pointed
//! at the level. What this module does is assemble its `Scene`: evaluate the
//! authored document's roots to vcad solids, build one BVH per root over the
//! *untessellated* BRep, resolve each root's material name to a `Pbr`, and put
//! the balls where phyz says they are. So the rim's silhouette is the ring of
//! rod segments the CAD says it is, at any resolution, and the picture and the
//! physics are reading the same file.
//!
//! Millimetres. vcad is a CAD kernel and its solids are in millimetres; phyz
//! is in metres. The whole picture is built in vcad's units, and every phyz
//! quantity that enters it is multiplied by [`PER_M`] — the only unit
//! conversion in the picture, and the only place either unit is named.
//!
//! Geometry the level does not author, the gym gets in code: four walls, a
//! ceiling, a floor beyond the slab, and the rows of light panels that are the
//! only light there is. The moment the level grows roots with material `wall`
//! or `ceiling`, this stops — the room is then the level's business, and only
//! the panels stay.

pub mod instances;
mod materials;

use std::collections::HashMap;
use std::sync::Arc;

use vcad_kernel::Solid;
use vcad_kernel_math::{Point3, Transform, Vec3};
use vcad_kernel_raytrace::pathtrace::{self, AreaLight, Environment, Ground, Object, PathTraceOptions, Pbr};
use vcad_kernel_raytrace::Bvh;

pub use vcad_kernel_raytrace::pathtrace::{Camera, Film};

use super::parts::PlacedSolid;
use super::{Court, CourtScene};
use crate::scene::MM;
use phyz_math::{Mat3, Vec3 as PVec3};

/// Metres (phyz) to millimetres (vcad). The only unit conversion in the picture.
pub const PER_M: f64 = 1.0 / MM;

/// Everything in the court that moves, at one instant: each ball's centre
/// (metres) and world → body rotation, and whatever else the court carries
/// for the picture. A frame of a recording is one of these; the static half
/// of the picture never needs the `Court` itself.
#[derive(Clone)]
pub struct Snapshot {
    pub balls: Vec<(PVec3, Mat3)>,
    pub extras: Vec<PlacedSolid>,
}

impl Snapshot {
    pub fn of(court: &Court) -> Self {
        Self {
            balls: (0..court.bodies()).map(|k| (court.centre(k), court.rotation(k))).collect(),
            extras: court.extras.clone(),
        }
    }
}

/// One traceable thing: a BVH, what it is made of, and where it sits.
///
/// The solid the BVH was built over is kept alongside it. The CPU tracer
/// never looks at it — a BVH is all it wants — but a GPU scene is packed
/// from the BRep itself, so a renderer that is not the CPU one needs the
/// geometry and not just the index over it.
struct Placed {
    solid: Arc<Solid>,
    bvh: Arc<Bvh>,
    pbr: Pbr,
    to_world: Transform,
}

impl Placed {
    fn object(&self) -> Object {
        Object::placed(self.bvh.clone(), self.pbr, self.to_world.clone())
    }
}

/// The court's picture: everything that does not move, plus the recipe for
/// everything that does.
pub struct Scene {
    /// The level's roots and the gym, already placed.
    statics: Vec<Placed>,
    /// The ball's own appearance, centred on the origin: its solid and its
    /// seams, each with a material, drawn once per ball at that ball's pose.
    ball: Vec<(String, Arc<Solid>, Arc<Bvh>, Pbr, Transform)>,
    /// BVHs for `Court::extras`, kept across frames so a net that only moves
    /// is not rebuilt. Keyed by the solid's identity.
    extras: HashMap<usize, Arc<Bvh>>,
    lights: Vec<AreaLight>,
    env: Environment,
    ground: Option<Ground>,
    /// The document, for resolving an extra's material name.
    doc: vcad_ir::Document,
}

impl Scene {
    /// The static picture, built once: the level's geometry and the gym's light.
    pub fn new(scene: &CourtScene) -> anyhow::Result<Self> {
        let a = &scene.authored;
        let doc = a.document.clone();

        // Each root, walked to the placed primitives it is a union of, rather
        // than evaluated to one solid. Only a genuinely boolean subtree is
        // evaluated, and only that subtree; see [`instances`].
        let mut prims = instances::Prims::default();
        let mut bvhs: HashMap<usize, Arc<Bvh>> = HashMap::new();
        let mut statics = Vec::new();
        let mut ball: Vec<(String, Arc<Solid>, Arc<Bvh>, Pbr, Transform)> = Vec::new();
        let mut authored_room = false;
        for root in &doc.roots {
            let name = root.material.as_str();
            authored_room |= matches!(name, "wall" | "ceiling");
            let placed = instances::instances(&doc, root.root, &mut prims)?;
            anyhow::ensure!(!placed.is_empty(), "the court's `{name}` root evaluated to no solid");
            let pbr = materials::pbr(&doc, name);
            let mut traceable = 0usize;
            for inst in placed {
                // one BVH per distinct solid, shared by every instance of it
                let bvh = bvhs
                    .entry(Arc::as_ptr(&inst.solid) as usize)
                    .or_insert_with(|| Arc::new(build_bvh(&inst.solid)))
                    .clone();
                if bvh.root().is_none() {
                    continue;
                }
                traceable += 1;
                // a `ball` root (and its `ball-seams`) is not part of the court:
                // it is the ball's own appearance, drawn once per ball at that
                // ball's pose
                if matches!(name, "ball" | "ball-seams" | "seam") {
                    ball.push((name.to_owned(), inst.solid, bvh, pbr, inst.to_world));
                } else {
                    statics.push(Placed { solid: inst.solid, bvh, pbr, to_world: inst.to_world });
                }
            }
            anyhow::ensure!(traceable > 0, "the court's `{name}` root has no traceable geometry");
        }

        // the gym: only until the level authors it
        let h = a.parameter("gym_h_mm")?;
        let margin = a.parameter("gym_margin_mm")?;
        let (wx, wy) = (0.5 * a.parameter("court_x_mm")? + margin, 0.5 * a.parameter("court_y_mm")? + margin);
        let floor_z = -a.parameter_or("court_t_mm", 40.0);
        let mut ground = None;
        if !authored_room {
            let wall = materials::pbr(&doc, "wall");
            let t = 100.0;
            let slab = |sx: f64, sy: f64, sz: f64, at: (f64, f64, f64), pbr: Pbr| {
                let solid = Arc::new(Solid::cube(sx, sy, sz));
                Placed {
                    bvh: Arc::new(build_bvh(&solid)),
                    solid,
                    pbr,
                    to_world: Transform::translation(at.0, at.1, at.2),
                }
            };
            statics.push(slab(t, 2.0 * wy + 2.0 * t, h + t, (wx, -wy - t, floor_z), wall));
            statics.push(slab(t, 2.0 * wy + 2.0 * t, h + t, (-wx - t, -wy - t, floor_z), wall));
            statics.push(slab(2.0 * wx, t, h + t, (-wx, wy, floor_z), wall));
            statics.push(slab(2.0 * wx, t, h + t, (-wx, -wy - t, floor_z), wall));
            statics.push(slab(2.0 * wx, 2.0 * wy, t, (-wx, -wy, h), materials::pbr(&doc, "ceiling")));
            // the floor beyond the slab, which the slab sits on
            ground = Some(Ground { z: floor_z, material: wall, shadow_catcher: false });
        }

        // the light panels, face down, in rows under the ceiling
        let (rows, cols) = (a.parameter("light_rows")?.max(1.0) as usize, a.parameter("light_cols")?.max(1.0) as usize);
        let (lw, ll) = (a.parameter("light_w_mm")?, a.parameter("light_l_mm")?);
        let radiance = a.parameter_or("light_radiance", 18.0) as f32;
        let mut lights = Vec::new();
        for i in 0..cols {
            for j in 0..rows {
                let x = ((i as f64 + 0.5) / cols as f64 * 2.0 - 1.0) * wx;
                let y = ((j as f64 + 0.5) / rows as f64 * 2.0 - 1.0) * wy;
                lights.push(AreaLight {
                    center: Point3::new(x, y, h - 5.0),
                    // u × v points down: the panel emits at the floor
                    u: Vec3::new(0.0, 0.5 * ll, 0.0),
                    v: Vec3::new(0.5 * lw, 0.0, 0.0),
                    emission: [radiance; 3],
                });
            }
        }

        // a ball root is the ball; without one, a sphere of the right size
        if ball.is_empty() {
            let solid = Arc::new(Solid::sphere(scene.ball_r * PER_M, 64));
            ball.push((
                "ball".to_owned(),
                solid.clone(),
                Arc::new(build_bvh(&solid)),
                materials::pbr(&doc, "ball"),
                Transform::identity(),
            ));
        }

        let env = a.parameter_or("env_radiance", 0.05) as f32;
        Ok(Self {
            statics,
            ball,
            extras: HashMap::new(),
            lights,
            env: Environment::constant([env; 3]),
            ground,
            doc,
        })
    }

    /// The picture at the court's current state: the static half, the balls
    /// where phyz has them, and whatever else the court is carrying.
    ///
    /// Everything that moves crosses the unit boundary here: a ball's centre
    /// is phyz metres, and `PER_M` is what makes it a vcad millimetre.
    pub fn at(&mut self, court: &Court) -> pathtrace::Scene {
        self.at_snapshot(&Snapshot::of(court))
    }

    /// The picture at a recorded instant; see [`Snapshot`].
    pub fn at_snapshot(&mut self, snap: &Snapshot) -> pathtrace::Scene {
        let mut objects: Vec<Object> = self.statics.iter().map(Placed::object).collect();
        for (centre, rot) in &snap.balls {
            let c = *centre * PER_M;
            // `rotation` is world → body; an object → world placement is its transpose
            let r = rot.transpose();
            let at = rigid(&r, c.x, c.y, c.z);
            for (_, _, bvh, pbr, local) in &self.ball {
                objects.push(Object::placed(bvh.clone(), *pbr, Transform { matrix: at.matrix * local.matrix }));
            }
        }
        // BVHs for the extras, kept from frame to frame by the solid's
        // identity — and dropped when a solid stops being handed to us, so a
        // net that mints a new cylinder now and then does not grow this forever.
        let mut seen: HashMap<usize, Arc<Bvh>> = HashMap::with_capacity(snap.extras.len());
        for extra in &snap.extras {
            let key = Arc::as_ptr(&extra.solid) as usize;
            let bvh = match seen.get(&key) {
                Some(b) => b.clone(),
                None => {
                    let b = self
                        .extras
                        .remove(&key)
                        .unwrap_or_else(|| Arc::new(build_bvh(&extra.solid)));
                    seen.insert(key, b.clone());
                    b
                }
            };
            if bvh.root().is_none() {
                continue;
            }
            objects.push(Object::placed(bvh, materials::pbr(&self.doc, &extra.material), extra.to_world.clone()));
        }
        self.extras = seen;
        pathtrace::Scene {
            objects,
            lights: self.lights.clone(),
            env: self.env.clone(),
            ground: self.ground,
        }
    }

    // ---- the same picture, for a renderer that packs BReps -----------------
    //
    // The CPU tracer wants BVHs and gets them above. A GPU scene is packed
    // from the BRep instead, and packs each solid once and then says where
    // its instances are — so what it needs from here is the geometry, the
    // material and the placement, and never a BVH. These hand that over
    // without a second copy of the assembly logic: the same statics, the same
    // ball parts, the same extras, the same lights.

    /// The level's geometry and the gym's, each solid with its material and
    /// where it sits. Built once and never moves.
    pub fn static_parts(&self) -> impl Iterator<Item = (&Solid, Pbr, &Transform)> {
        self.statics.iter().map(|p| (p.solid.as_ref(), p.pbr, &p.to_world))
    }

    /// The ball's own parts — its solid and its seams — centred on the origin,
    /// one copy of each to be placed at every ball's pose, each with the root
    /// material that named it.
    pub fn ball_parts(&self) -> impl Iterator<Item = (&str, &Solid, Pbr, &Transform)> {
        self.ball.iter().map(|(name, s, _, pbr, at)| (name.as_str(), s.as_ref(), *pbr, at))
    }

    /// Where each ball is at this instant, as an object → world placement in
    /// millimetres — one per ball, to be applied to every part of
    /// [`Self::ball_parts`].
    pub fn ball_placements(&self, snap: &Snapshot) -> Vec<Transform> {
        snap.balls
            .iter()
            .map(|(centre, rot)| {
                let c = *centre * PER_M;
                let r = rot.transpose();
                rigid(&r, c.x, c.y, c.z)
            })
            .collect()
    }

    /// Whatever else the court is carrying this frame — the net — as solid,
    /// material and placement.
    pub fn extra_parts<'a>(&'a self, snap: &'a Snapshot) -> impl Iterator<Item = (&'a Solid, Pbr, &'a Transform)> {
        snap.extras
            .iter()
            .map(move |e| (e.solid.as_ref(), materials::pbr(&self.doc, &e.material), &e.to_world))
    }

    /// The gym's light panels: the only lights there are.
    pub fn lights(&self) -> &[AreaLight] {
        &self.lights
    }

    /// How many traceable objects the static half has, for a sanity check.
    pub fn static_count(&self) -> usize {
        self.statics.len()
    }

    /// How many light panels the level asked for.
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Where the light panels are, in millimetres. A reprojecting renderer
    /// needs them to know where a moving ball's shadow lands.
    pub fn light_centres(&self) -> Vec<Point3> {
        self.lights.iter().map(|l| l.center).collect()
    }

    /// The ball's own bounding radius in millimetres, from its solid.
    pub fn ball_radius_mm(&self) -> f64 {
        self.ball
            .iter()
            .filter_map(|(_, _, bvh, _, _)| bvh.bounds())
            .map(|b| {
                let d = b.max - b.min;
                0.5 * (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
            })
            .fold(0.0, f64::max)
    }

    /// A world-space bounding sphere for one extra, in millimetres: its BVH's
    /// bounds carried through its placement. Builds (and caches) the BVH the
    /// same way drawing it would.
    pub fn extra_sphere(&mut self, extra: &PlacedSolid) -> Option<(Point3, f64)> {
        let key = Arc::as_ptr(&extra.solid) as usize;
        let bvh = self
            .extras
            .entry(key)
            .or_insert_with(|| Arc::new(build_bvh(&extra.solid)))
            .clone();
        let b = bvh.bounds()?;
        let c = Point3::new(
            0.5 * (b.min.x + b.max.x),
            0.5 * (b.min.y + b.max.y),
            0.5 * (b.min.z + b.max.z),
        );
        let d = b.max - b.min;
        let r = 0.5 * (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
        Some((extra.to_world.apply_point(&c), r))
    }
}

/// A BVH over a solid: its analytic BRep if it has one, its tessellation if
/// not — the same fallback `vcad-render --photoreal` takes.
fn build_bvh(solid: &Solid) -> Bvh {
    match solid.as_brep() {
        Some(brep) => Bvh::build(brep),
        None => {
            let mut mesh = solid.to_mesh(0);
            vcad_kernel::vcad_kernel_tessellate::render_bake_default(&mut mesh);
            Bvh::build_mesh(&mesh)
        }
    }
}

/// A rigid object → world transform from a body → world rotation and a
/// translation in millimetres.
fn rigid(r: &phyz_math::Mat3, x: f64, y: f64, z: f64) -> Transform {
    Transform {
        matrix: tang::Mat4::new(
            r[(0, 0)], r[(0, 1)], r[(0, 2)], x, //
            r[(1, 0)], r[(1, 1)], r[(1, 2)], y, //
            r[(2, 0)], r[(2, 1)], r[(2, 2)], z, //
            0.0, 0.0, 0.0, 1.0,
        ),
    }
}

/// The camera the level asks for, in millimetres.
pub fn camera(scene: &CourtScene) -> anyhow::Result<Camera> {
    let a = &scene.authored;
    let eye = Point3::new(a.parameter("cam_x_mm")?, a.parameter("cam_y_mm")?, a.parameter("cam_z_mm")?);
    let target = Point3::new(a.parameter("cam_at_x_mm")?, a.parameter("cam_at_y_mm")?, a.parameter("cam_at_z_mm")?);
    let mut cam = Camera::look_at(eye, target, Vec3::z(), a.parameter_or("cam_vfov_deg", 42.0));
    // a real aperture: the radius of the iris, and the plane it is sharp on
    cam.aperture = a.parameter_or("cam_aperture_mm", 0.0).max(0.0);
    let focus = a.parameter_or("cam_focus_mm", 0.0);
    if focus > 0.0 {
        cam.focus_dist = focus;
    }
    Ok(cam)
}

/// Integrator settings the level asks for, at a given sample count.
pub fn options(scene: &CourtScene, spp: usize, seed: u64) -> PathTraceOptions {
    let a = &scene.authored;
    PathTraceOptions {
        spp: spp.max(1) as u32,
        max_depth: a.parameter_or("max_depth", 6.0).max(1.0) as u32,
        show_background: true,
        seed,
        denoise: a.parameter_or("denoise", 1.0) > 0.5,
        ..Default::default()
    }
}

/// One render. The picture is `vcad-kernel-raytrace`'s; this only names it.
pub fn render(picture: &pathtrace::Scene, cam: &Camera, width: u32, height: u32, opts: &PathTraceOptions) -> Film {
    pathtrace::render(picture, cam, width, height, opts)
}

/// Sum `src` into `dst`, so a frame can be the average of its sub-frames.
pub fn accumulate(dst: &mut Film, src: &Film) {
    for (d, s) in dst.rgb.iter_mut().zip(&src.rgb) {
        *d += *s;
    }
    for (d, s) in dst.alpha.iter_mut().zip(&src.alpha) {
        *d += *s;
    }
}

/// Tonemap a film to an image, dividing by however many sub-frames went into it.
pub fn to_image(film: &Film, exposure: f64, n: usize) -> image::RgbaImage {
    let px = film.to_srgb8(exposure as f32 / n.max(1) as f32, false);
    image::RgbaImage::from_raw(film.width, film.height, px).expect("film is width × height × 4")
}
