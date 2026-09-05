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
struct Placed {
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
    ball: Vec<(Arc<Bvh>, Pbr)>,
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
        let evaluated = vcad_eval::evaluate_document(&doc, &vcad_eval::EvalOptions { skip_clash_detection: true, ..Default::default() })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        anyhow::ensure!(
            evaluated.parts.len() == doc.roots.len(),
            "the court evaluated to {} parts for {} roots",
            evaluated.parts.len(),
            doc.roots.len()
        );

        let mut statics = Vec::new();
        let mut ball: Vec<(Arc<Bvh>, Pbr)> = Vec::new();
        let mut authored_room = false;
        for (part, root) in evaluated.parts.iter().zip(&doc.roots) {
            let name = root.material.as_str();
            authored_room |= matches!(name, "wall" | "ceiling");
            let Some(solid) = part.solid.as_ref() else {
                anyhow::bail!("the court's `{name}` root evaluated to no solid");
            };
            let bvh = build_bvh(solid);
            if bvh.root().is_none() {
                anyhow::bail!("the court's `{name}` root has no traceable geometry");
            }
            let bvh = Arc::new(bvh);
            // a `ball` root (and its `ball-seams`) is not part of the court: it
            // is the ball's own appearance, drawn once per ball at that ball's pose
            if matches!(name, "ball" | "ball-seams" | "seam") {
                ball.push((bvh, materials::pbr(&doc, name)));
                continue;
            }
            statics.push(Placed { bvh, pbr: materials::pbr(&doc, name), to_world: Transform::identity() });
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
            let slab = |sx: f64, sy: f64, sz: f64, at: (f64, f64, f64), pbr: Pbr| Placed {
                bvh: Arc::new(build_bvh(&Solid::cube(sx, sy, sz))),
                pbr,
                to_world: Transform::translation(at.0, at.1, at.2),
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
            ball.push((
                Arc::new(build_bvh(&Solid::sphere(scene.ball_r * PER_M, 64))),
                materials::pbr(&doc, "ball"),
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
            for (bvh, pbr) in &self.ball {
                objects.push(Object::placed(bvh.clone(), *pbr, rigid(&r, c.x, c.y, c.z)));
            }
        }
        for extra in &snap.extras {
            let key = Arc::as_ptr(&extra.solid) as usize;
            let bvh = self
                .extras
                .entry(key)
                .or_insert_with(|| Arc::new(build_bvh(&extra.solid)))
                .clone();
            if bvh.root().is_none() {
                continue;
            }
            objects.push(Object::placed(bvh, materials::pbr(&self.doc, &extra.material), extra.to_world.clone()));
        }
        pathtrace::Scene {
            objects,
            lights: self.lights.clone(),
            env: self.env.clone(),
            ground: self.ground,
        }
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
            .filter_map(|(bvh, _)| bvh.bounds())
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
