//! The cove, lit.
//!
//! The court's renderer, pointed at a beach. Same shape as
//! [`kosm::brep`]: evaluate the authored document's roots to the
//! placed primitives they are a union of, build one BVH per distinct solid,
//! resolve each surface's name to a flat [`materials::pbr`], and put the
//! things that move where the simulation says they are. The picture and the
//! physics read one file.
//!
//! Three things are the cove's own.
//!
//! **The geometry is not all vcad.** The sea is not in the document — it is a
//! [`HeightField`] with an authored swell, because a surface that will one
//! day be the tide's is a field of heights and never was a solid. But
//! `kosm_render::Scene` is generic over a *single* geometry type, so a
//! picture that is both B-reps and water needs one type that is both:
//! [`CoveGeom`]. Dispatch is static, the BVH is built per object, and the
//! whole cost is a branch per primitive test.
//!
//! **The being is a capsule, and a capsule is nobody's primitive.** Neither
//! vcad's kernel nor `kosm-render`'s analytic set has one, and the obvious
//! construction — a cylinder unioned with a sphere at each end — is a
//! boolean between surfaces that meet *tangentially* along a circle, which
//! is the case a BRep kernel is worst at. So the being is tessellated
//! directly, with its exact analytic normals on every vertex
//! ([`capsule_mesh`]). The shape is the capsule the physics steps and the
//! score is read off, not an ellipsoid that resembles one.
//!
//! **The door swings.** It is the level's own root, drawn at a hinge angle
//! rather than baked into the ground, which is what makes it a body and not
//! a texture.
//!
//! Millimetres. vcad's solids are in millimetres and phyz is in metres; the
//! whole picture is assembled in millimetres and every simulation quantity
//! that enters it is multiplied by [`PER_M`], which is the only unit
//! conversion in here.



use std::collections::HashMap;
use std::sync::Arc;

use kosm_render::caustics::{self, CausticMap, CausticOptions};
use kosm_render::geometry::Geometry;
use kosm_render::heightfield::HeightField;
use kosm_render::math::{Aabb, Point3, Transform, Vec3, transform_aabb};
use kosm_render::pathtrace::{
    self, Camera, Environment, Film, GradientEnv, Ground, Object, PathTraceOptions, Pbr, Sun,
};
use kosm_render::{Bvh, Hit, Ray, TriMesh};
use phyz_math::{Mat3, Vec3 as PVec3};
use vcad_kernel::Solid;
use vcad_kernel_math::Transform as VTransform;
use vcad_kernel_raytrace::BrepGeom;

use super::CoveScene;
use kosm::brep::instances::{self, Prims};
use super::materials;
use kosm::scene::MM;

/// Metres (phyz) to millimetres (vcad). The only unit conversion in the picture.
pub const PER_M: f64 = 1.0 / MM;

// ---- the one geometry ------------------------------------------------------

/// The cove is B-reps and water, and a [`pathtrace::Scene`] is generic over
/// one geometry, so this is the one.
pub enum CoveGeom {
    /// The level's solids: analytic BRep faces, or triangles when a boolean
    /// degraded and there is no BRep left to trace.
    Brep(BrepGeom),
    /// The sea.
    Water(HeightField),
}

impl Geometry for CoveGeom {
    fn len(&self) -> usize {
        match self {
            CoveGeom::Brep(g) => g.len(),
            CoveGeom::Water(w) => w.len(),
        }
    }
    fn bounds(&self, i: usize) -> Aabb {
        match self {
            CoveGeom::Brep(g) => g.bounds(i),
            CoveGeom::Water(w) => w.bounds(i),
        }
    }
    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        match self {
            CoveGeom::Brep(g) => g.intersect(ray, i, t_min, t_max),
            CoveGeom::Water(w) => w.intersect(ray, i, t_min, t_max),
        }
    }
    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) {
        match self {
            CoveGeom::Brep(g) => g.intersect_all(ray, i, out),
            CoveGeom::Water(w) => w.intersect_all(ray, i, out),
        }
    }
    fn occludes(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> bool {
        match self {
            CoveGeom::Brep(g) => g.occludes(ray, i, t_min, t_max),
            CoveGeom::Water(w) => w.occludes(ray, i, t_min, t_max),
        }
    }
}

/// What the integrator traces here: a [`CoveGeom`] BVH, a material, a placement.
pub type CoveObject = Object<CoveGeom>;
/// The cove's picture, ready for [`kosm_render::pathtrace::render`].
pub type Picture = pathtrace::Scene<CoveGeom>;

// ---- what moves ------------------------------------------------------------

/// The state of the cove the picture needs: where the being is, and how far
/// the door has swung.
///
/// This is the renderer's whole input, and it is deliberately not step 3's
/// `Snapshot`: the picture wants a pose, and the simulation happens to be one
/// way of getting one. A solved pose, a hand-placed pose and a live snapshot
/// all arrive here as the same three numbers, which is what lets the offline
/// frame exist before the being can walk.
#[derive(Clone, Debug)]
pub struct Placement {
    /// The being's centre in world metres, and its body → world rotation.
    /// The body's axes are right, facing, up.
    pub being: (PVec3, Mat3),
    /// The hero's solids, placed — every part of the figure and the glass in
    /// its hand, at the transforms one snapshot of the simulation puts them
    /// at. [`None`] is the capsule, which is what the offline still and the
    /// solvability sweep are written against.
    ///
    /// The solids themselves are [`Scene`]'s and are built once; what changes
    /// per frame is the fourth field of each of these, so a walking hero costs
    /// one 4×4 a part and not one tessellation. An `Arc<[_]>` rather than a
    /// `Vec`, because a `Placement` is cloned per pass and compared against
    /// the pose the caustic map was traced at.
    pub hero: Option<Arc<[HeroPart]>>,
    /// How far the door has swung, radians. Zero is shut; positive opens it
    /// out of the cliff into the cove.
    pub door_angle: f64,
    /// The rune's score at this pose, in `[0, 1]`.
    ///
    /// The picture carries it because the *rim* carries it: the keyhole's ring
    /// glows `glow_floor + glow_gain · score` and there is no other channel for
    /// it — no text, no HUD. It rides on the placement rather than on the
    /// [`Scene`] because it changes at the rune thread's rate, not the level's,
    /// and because a still is then a pose and a number and needs no simulation.
    pub score: f64,
    /// Where the glint is, if the player has been stuck long enough to earn
    /// one: the centre of a small emissive sphere sitting on the sand, in world
    /// metres. [`None`] is the ordinary case and draws nothing.
    pub glint: Option<PVec3>,
}

impl Placement {
    /// The being standing on the sand at `(x, y)`, facing the door, leaning
    /// `tilt` radians about its own left-right axis; the door shut.
    ///
    /// The lean is the puzzle's second knob — a capsule's focus moves with
    /// it — so it belongs in the pose and not in the camera.
    pub fn standing(scene: &CoveScene, x: f64, y: f64, tilt: f64) -> Self {
        let centre = PVec3::new(x, y, scene.sand_z_at(x, y) + scene.being_h / 2.0);
        let door = scene.door_frame().origin;
        let mut facing = PVec3::new(door.x - x, door.y - y, 0.0);
        if facing.norm() < 1e-9 {
            facing = PVec3::new(0.0, 1.0, 0.0);
        }
        let facing = facing.normalize();
        let up = PVec3::new(0.0, 0.0, 1.0);
        // right, facing, up is right-handed with facing × up = right
        let right = facing.cross(up);
        // lean about `right`: the being pitches forward, and its up leans
        // with it, which is what a capsule leaning is
        let (s, c) = tilt.sin_cos();
        let lean_up = up * c - facing * s;
        let lean_facing = facing * c + up * s;
        Self {
            being: (centre, columns(right, lean_facing, lean_up)),
            hero: None,
            door_angle: 0.0,
            score: 0.0,
            glint: None,
        }
    }

    /// The same pose, with the hero's solids placed on it.
    pub fn with_hero(mut self, hero: Option<Arc<[HeroPart]>>) -> Self {
        self.hero = hero;
        self
    }

    /// Whether this placement draws the figure rather than the capsule.
    pub fn is_hero(&self) -> bool {
        self.hero.as_ref().is_some_and(|h| !h.is_empty())
    }

    /// The same pose, with the rune's score on it.
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// The same pose, with the glint where the hint put it.
    pub fn with_glint(mut self, glint: Option<PVec3>) -> Self {
        self.glint = glint;
        self
    }

    /// Where the being is looking: its body +y, in world.
    pub fn facing(&self) -> PVec3 {
        let r = self.being.1;
        PVec3::new(r[(0, 1)], r[(1, 1)], r[(2, 1)])
    }
}

// ---- the hero -------------------------------------------------------------

/// One of the hero's solids, placed for one frame.
///
/// The name is `sims/rune/hero/figure.rs`'s body name — `hood`, `boot_r`,
/// `strap` — or `lens` and `lens_ring` for the glass in the hand. It is
/// carried because a part that is drawn in the wrong place is a bug you can
/// only see, and a name in a test is how you see it in a number instead.
#[derive(Clone)]
pub struct HeroPart {
    pub name: String,
    pub bvh: Arc<Bvh<CoveGeom>>,
    pub to_world: Transform,
    pub pbr: Pbr,
}

// A BVH has no `Debug` and would be no use as one; what a reader of a
// `Placement` wants to know about a part is its name and where it ended up.
impl std::fmt::Debug for HeroPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.to_world.matrix.c3;
        f.debug_struct("HeroPart").field("name", &self.name).field("at", &[c.x, c.y, c.z]).finish()
    }
}

/// One of the hero's solids as it was **authored**: which of the body's links
/// carries it, and where it sits in that link's own frame.
///
/// This is the whole of "build once, re-place per frame". `figure()` is
/// evaluated exactly once, at [`Scene::new`], and every primitive its bodies
/// are a union of gets one BVH; what a walking hero costs after that is one
/// 4×4 per primitive per frame.
struct HeroSolid {
    name: String,
    bvh: Arc<Bvh<CoveGeom>>,
    pbr: Pbr,
    /// The cove's surface name this part is painted with — `cloak`, `boot`,
    /// `brass` — kept so the raster tier can resolve it to the same library
    /// entry `pbr` was derived from.
    material: String,
    /// The solid itself, for a tier that needs triangles rather than a BVH.
    /// `None` for the parts this file tessellates itself.
    solid: Option<Arc<Solid>>,
    /// Which link of `kosm::player::BodySpec::hero` this rides on: an index
    /// into `Snapshot::parts`, which is in the spec's link order.
    link: usize,
    /// Primitive → the link's own frame, in **hero-local millimetres**.
    local: Transform,
}

/// Hero-local millimetres as the body's own frame: `+y` forward becomes `+x`,
/// `+x` right becomes `−y`, `z` is `z`.
///
/// The rotation half of [`kosm::player::body::hero_local`], without its
/// millimetres — because the picture is assembled in millimetres and the
/// thousand cancels. That cancellation is the only reason this matrix and not
/// that function is what the placement uses; they are the same map.
fn hero_axes() -> Mat3 {
    Mat3::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
}

/// How far outside link `i`'s own flesh a body-local point is, metres.
///
/// The **signed** distance to the nearest lump, so a part that overlaps two
/// links goes to the one it is deepest inside rather than to the one whose
/// pivot happens to be nearer. That distinction is not academic: the cloak's
/// skirt cone has its centroid 117 mm from the pelvis ball's centre and 87 mm
/// from the chest's, so a nearest-centre rule would hang the skirt off the
/// torso and swing it when the hero leaned.
fn lump_depth(link: &kosm::player::Link, q: PVec3) -> f64 {
    link.lumps
        .iter()
        .map(|l| {
            let d = l.to - l.from;
            let t = if d.norm_squared() > 1e-18 { ((q - l.from).dot(&d) / d.norm_squared()).clamp(0.0, 1.0) } else { 0.0 };
            (q - (l.from + d * t)).norm() - l.radius
        })
        .fold(f64::INFINITY, f64::min)
}

/// A solid of the hero's, traced with **smoothed** vertex normals.
///
/// [`geometry_of`] is the cove's and shades a boolean's tessellation flat,
/// which is right for a cliff cut out of a slab and wrong for a hood: the
/// figure's cowl is a union of spheres and a torus and a hood with facets on
/// it is a hood nobody believes. `hero/stage.rs::smooth_normals` is the rule
/// the hero's own stills are drawn with — welded by position, creased at 46°
/// — so the live figure and the doorstep plate shade the same way.
fn hero_geometry(solid: &Solid) -> CoveGeom {
    if let Some(brep) = solid.as_brep() {
        let brep = Arc::new(brep.clone());
        let faces = brep.topology.faces.iter().map(|(id, _)| id).collect();
        return CoveGeom::Brep(BrepGeom::BRep { brep, faces });
    }
    let mut mesh = solid.to_mesh(0);
    vcad_kernel::vcad_kernel_tessellate::render_bake_default(&mut mesh);
    let n = mesh.vertices.len() / 3;
    let positions: Vec<Point3> = (0..n)
        .map(|i| Point3::new(mesh.vertices[i * 3] as f64, mesh.vertices[i * 3 + 1] as f64, mesh.vertices[i * 3 + 2] as f64))
        .collect();
    let normals = super::hero::stage::smooth_normals(&positions, &mesh.indices);
    CoveGeom::Brep(BrepGeom::Mesh(TriMesh::new(positions, normals, &mesh.indices)))
}

/// A body → world rotation from its three axes, as columns.
fn columns(x: PVec3, y: PVec3, z: PVec3) -> Mat3 {
    Mat3::new(x.x, y.x, z.x, x.y, y.y, z.y, x.z, y.z, z.z)
}

// ---- the picture -----------------------------------------------------------

/// One traceable thing: a BVH, what it is made of, and where it sits.
struct Placed {
    bvh: Arc<Bvh<CoveGeom>>,
    pbr: Pbr,
    to_world: Transform,
}

impl Placed {
    fn object(&self) -> CoveObject {
        Object::placed(self.bvh.clone(), self.pbr, self.to_world.clone())
    }
}

/// The cove's picture: everything that does not move, plus the recipe for
/// everything that does.
pub struct Scene {
    /// The `ground` root's primitives and the sea, already placed.
    statics: Vec<Placed>,
    /// The door's primitives, in the frame the document put them in. The
    /// hinge is applied on top of these, per frame.
    door: Vec<Placed>,
    /// The hinge: the world point the door turns about, in millimetres.
    hinge: Point3,
    /// The keyhole's rim, in the door's own shut frame, so the hinge carries
    /// it. Its material is rebuilt per pose from the score; only its geometry
    /// is here.
    rim: Arc<Bvh<CoveGeom>>,
    /// What the rim glows at a score of zero, and what the score buys.
    glow_floor: f64,
    glow_gain: f64,
    /// The glint's sphere, centred on the origin, and its radiance.
    glint: Arc<Bvh<CoveGeom>>,
    glint_glow: f64,
    /// The being, centred on the origin with its axis along +z.
    being: Placed,
    /// The hero's costume, built once, each part in the frame of the link
    /// that carries it. Empty when the level could not evaluate the figure.
    hero: Vec<HeroSolid>,
    /// The glass in the hand and the ring around it, in the lens's own frame:
    /// centred on the origin with the optical axis along +z, which is what
    /// `hero/kit.rs` cuts them in.
    held: Vec<HeroSolid>,
    env: Environment,
    sun: Sun,
    ground: Option<Ground>,
    /// How many photons the rune's caustic is traced with. Zero is off.
    photons: usize,
    /// The gather radius, millimetres, at [`Self::gather_photons`].
    gather: f64,
    /// The photon count [`Self::gather`] was authored for.
    ///
    /// A density estimate's noise inside the disc falls as `1/sqrt(N·r²)`, so
    /// a map shot with a tenth of the photons wants a radius `sqrt(10)` wider
    /// to read as smooth rather than as confetti. The authored radius is the
    /// scale the *rune* is read at and it belongs to the still's budget; a
    /// live map at `caustic_photons_live` gets it scaled by
    /// `sqrt(gather_photons / photons)`, so the offline frame is bit-identical
    /// and the walking picture has a caustic instead of sparkle.
    gather_photons: f64,
    /// Solids a caller has put in the cove that the cove knows nothing about.
    /// See [`Scene::push_part`]. Empty for `kosm run rune`.
    extras: Vec<Placed>,
    /// Whether the being is drawn at all. See [`Scene::set_being_visible`].
    show_being: bool,
    /// Whether the *camera's* being disperses. The caustic pass's always does.
    ///
    /// See [`materials::being_achromatic`]: at one sample a pixel a pass the
    /// hero-wavelength draw is what makes the live body confetti, and the
    /// dispersion the level is about lives in the photon map.
    body_dispersion: bool,
}

impl Scene {
    /// The static picture, built once: the level's geometry, the sea, and the
    /// daylight.
    pub fn new(scene: &CoveScene) -> anyhow::Result<Self> {
        let a = &scene.authored;
        let doc = &a.document;

        let mut prims = Prims::default();
        let mut bvhs: HashMap<usize, Arc<Bvh<CoveGeom>>> = HashMap::new();
        let mut statics = Vec::new();
        let mut door = Vec::new();
        for root in &doc.roots {
            let placed = instances::instances(doc, root.root, &mut prims)?;
            anyhow::ensure!(!placed.is_empty(), "the cove's `{}` root evaluated to no solid", root.material);
            for inst in placed {
                let bvh = bvhs
                    .entry(Arc::as_ptr(&inst.solid) as usize)
                    .or_insert_with(|| Arc::new(Bvh::build(geometry_of(&inst.solid))))
                    .clone();
                let Some(local) = bvh.bounds() else { continue };
                let to_world = placement(&inst.to_world);
                if root.material == "door" || root.material == materials::DOOR {
                    door.push(Placed { bvh, pbr: materials::pbr(doc, materials::DOOR), to_world });
                } else {
                    // one root, three surfaces; see `ground_material`
                    let name = ground_material(&transform_aabb(&local, &to_world), scene);
                    statics.push(Placed { bvh, pbr: materials::pbr(doc, name), to_world });
                }
            }
        }
        anyhow::ensure!(!door.is_empty(), "the cove needs a `door` root to draw");
        anyhow::ensure!(!statics.is_empty(), "the cove has no ground to draw");

        // the sea, from the waterline out
        let sea = sea_field(scene, a);
        statics.push(Placed {
            bvh: Arc::new(Bvh::build(CoveGeom::Water(sea))),
            pbr: materials::pbr(doc, "water"),
            to_world: Transform::identity(),
        });

        let being = Placed {
            bvh: Arc::new(Bvh::build(CoveGeom::Brep(BrepGeom::Mesh(capsule_mesh(
                scene.being_r * PER_M,
                scene.being_h * PER_M,
                a.parameter_or("being_segments", 96.0).max(8.0) as usize,
                a.parameter_or("being_rings", 32.0).max(3.0) as usize,
            ))))),
            pbr: materials::being(scene.n_d),
            to_world: Transform::identity(),
        };

        // The hinge: the door's -x edge, in the plane of the cliff face.
        let hinge = Point3::new(
            (scene.door_x - scene.door_w / 2.0) * PER_M,
            scene.cliff_face_y() * PER_M,
            0.0,
        );

        // The keyhole's ring, on the door's face and a few millimetres proud of
        // it. It is built in the door's *shut* world frame, exactly as the
        // door's own parts are, so the one hinge transform carries both and the
        // rim can never drift off the stone it is cut into.
        let rim = Arc::new(Bvh::build(CoveGeom::Brep(BrepGeom::Mesh(annulus_mesh(
            Point3::new(
                scene.door_x * PER_M,
                (scene.cliff_face_y() - RIM_PROUD_MM * MM) * PER_M,
                (scene.door_sill() + scene.aperture_z) * PER_M,
            ),
            scene.aperture_r * PER_M,
            (scene.aperture_r + scene.rim_w) * PER_M,
            a.parameter_or("rim_segments", 96.0).max(8.0) as usize,
        )))));
        // A capsule whose height is two radii is the sphere, exactly.
        let glint_r = scene.glint_r * PER_M;
        let glint = Arc::new(Bvh::build(CoveGeom::Brep(BrepGeom::Mesh(capsule_mesh(
            glint_r,
            2.0 * glint_r,
            a.parameter_or("glint_segments", 32.0).max(8.0) as usize,
            a.parameter_or("glint_rings", 12.0).max(3.0) as usize,
        )))));

        let (env, sun) = daylight(scene);
        // The ocean past the sea's own lattice: a plane at the waterline, set
        // a hand's breadth under it so the swell never fights it, which is
        // the horizon and nothing else.
        let ground = Some(Ground {
            z: (scene.sea_z - a.parameter_or("sea_floor_mm", 150.0) * MM) * PER_M,
            material: materials::pbr(doc, "water"),
            shadow_catcher: false,
        });
        // The figure, evaluated once. A level that cannot build it says so and
        // carries on with the capsule rather than failing to open a window.
        let (hero, held) = match hero_solids(doc, scene) {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("cove   the hero could not be built ({e}); the capsule stands in");
                (Vec::new(), Vec::new())
            }
        };

        Ok(Self {
            statics,
            door,
            hinge,
            rim,
            glow_floor: scene.glow_floor,
            glow_gain: scene.glow_gain,
            glint,
            glint_glow: scene.glint_glow,
            being,
            hero,
            held,
            env,
            sun,
            ground,
            photons: a.parameter_or("caustic_photons", 600_000.0).max(0.0) as usize,
            // The pass's own default is a two-hundredth of the refractor —
            // four millimetres of a being, which at this photon count is
            // sparkle and not a caustic. Half the aperture is the scale the
            // rune is actually read at, so it is the scale to gather at.
            gather: a.parameter_or("caustic_radius_mm", scene.aperture_r * PER_M / 2.0),
            gather_photons: a.parameter_or("caustic_radius_photons", 600_000.0).max(1.0),
            extras: Vec::new(),
            show_being: true,
            body_dispersion: true,
        })
    }

    /// Whether the camera's being carries its dispersion curve.
    ///
    /// On by default, which is the offline frame. A tier that traces one raw
    /// sample a pixel a pass turns it off (see
    /// [`materials::being_achromatic`]) and loses nothing the level is about:
    /// [`Self::caustic_map`] shoots its photons through the dispersive glass
    /// whatever this says.
    pub fn set_body_dispersion(&mut self, on: bool) {
        self.body_dispersion = on;
    }

    /// Draw one more thing in every picture this scene makes: a BVH over this
    /// picture's geometry, a material, and an object → world placement in
    /// millimetres.
    ///
    /// Additive and orthogonal to everything above it — the cove itself never
    /// calls this. A character, a prop, a second refractor: anything a caller
    /// holds as a placed BVH can be put in the cove's own light without the
    /// cove having to know what it is. A `pbr` that transmits joins the caustic
    /// pass's aim for nothing, because [`kosm_render::caustics`] looks at the
    /// picture's materials and at nothing else — so a lens pushed in here
    /// throws a rune exactly the way the being does.
    ///
    /// [`geometry_of`] turns a placed vcad solid into the geometry this wants;
    /// [`mesh_geometry`] does the same for a caller that has tessellated
    /// something itself.
    pub fn push_object(&mut self, bvh: Arc<Bvh<CoveGeom>>, pbr: Pbr, to_world: Transform) {
        self.extras.push(Placed { bvh, pbr, to_world });
    }

    /// Whether the being is in the picture. On by default, which is the cove.
    ///
    /// A still of something *else* standing in this cove wants the level, the
    /// light and the door and not the capsule — and, because the being is a
    /// dielectric, leaving it in would also spend half the rune's photons on a
    /// body that is not in shot.
    pub fn set_being_visible(&mut self, on: bool) {
        self.show_being = on;
    }

    /// The gather radius the caustic is read at, millimetres.
    ///
    /// The authored default is half the keyhole, which is the scale a being's
    /// broad, aberrated focus is read at. A refractor that focuses tighter than
    /// that — a figured lens rather than a body — wants a smaller disc or the
    /// estimate averages its own spot away.
    pub fn set_gather(&mut self, radius_mm: f64) {
        self.gather = radius_mm.max(1e-6);
        self.gather_photons = self.photons.max(1) as f64;
    }

    /// How many photons the rune is traced with. Zero is off.
    pub fn set_photons(&mut self, photons: usize) {
        self.photons = photons;
    }

    /// The gather radius a map of `photons` photons is asked for, in
    /// millimetres. See [`Self::gather_photons`].
    fn gather_for(&self, photons: usize) -> f64 {
        let widen = (self.gather_photons / (photons.max(1) as f64)).sqrt();
        self.gather * widen.clamp(1.0, MAX_WIDENING)
    }

    /// The hero's solids at one snapshot of the body: every costume part on
    /// the link that carries it, and the glass wherever the arm actually got
    /// it to.
    ///
    /// The whole per-frame cost of drawing a figure. A part authored at
    /// `p_mm` in the hero's own millimetres, on a link whose pivot is `pivot`
    /// (body-local metres) and whose frame the simulation has put at
    /// `(pos, R)` (world metres, `R` body → world), is drawn at
    ///
    /// ```text
    /// world_mm = R · P · p_mm + 1000 · (pos − R · pivot)
    /// ```
    ///
    /// where `P` is [`hero_axes`]. The thousand that turns hero-millimetres
    /// into body-metres and the thousand that turns world-metres back into
    /// the picture's millimetres cancel in the linear part and survive only
    /// in the translation, which is why this is one 4×4 and not two.
    pub fn hero_at(
        &self,
        parts: &[kosm::player::Part],
        pivots: &[PVec3],
        held: Option<kosm::player::Pose>,
    ) -> Option<Arc<[HeroPart]>> {
        if self.hero.is_empty() || parts.is_empty() {
            return None;
        }
        let axes = hero_axes();
        let mut out: Vec<HeroPart> = Vec::with_capacity(self.hero.len() + self.held.len());
        for solid in &self.hero {
            let (Some(part), Some(pivot)) = (parts.get(solid.link), pivots.get(solid.link)) else { continue };
            let (pos, r) = (part.pose.pos, part.pose.rot);
            let t = (pos - r.mul_vec(*pivot)) * PER_M;
            let frame = rigid(&r.mul_mat(&axes), t.x, t.y, t.z);
            out.push(HeroPart {
                name: solid.name.clone(),
                bvh: solid.bvh.clone(),
                pbr: solid.pbr,
                to_world: Transform::from_matrix(frame.matrix * solid.local.matrix),
            });
        }
        // The glass is not on a link. It is where `Body::held` says it is,
        // which is not where the arm was asked to put it, because an arm has
        // mass — and that is the pose the scorer reads too.
        if let Some(pose) = held {
            let c = pose.pos * PER_M;
            let frame = rigid(&pose.rot, c.x, c.y, c.z);
            for solid in &self.held {
                out.push(HeroPart {
                    name: solid.name.clone(),
                    bvh: solid.bvh.clone(),
                    pbr: solid.pbr,
                    to_world: Transform::from_matrix(frame.matrix * solid.local.matrix),
                });
            }
        }
        (!out.is_empty()).then(|| out.into())
    }

    /// The link pivots the hero's placement needs, body-local metres. Built
    /// once by the caller and handed to [`Scene::hero_at`] each frame.
    pub fn hero_pivots() -> Vec<PVec3> {
        kosm::player::BodySpec::hero(&super::being::hero_skeleton(&super::hero::figure::Rig::DEFAULT))
            .links
            .iter()
            .map(|l| l.pivot)
            .collect()
    }

    /// The picture at one pose: the static half, the being where it stands,
    /// the door at its hinge angle.
    pub fn at(&mut self, p: &Placement) -> Picture {
        let pbr = if self.body_dispersion {
            self.being.pbr
        } else {
            materials::achromatic(self.being.pbr)
        };
        self.picture_at(p, pbr)
    }

    /// The picture at one pose with the being made of `pbr`.
    fn picture_at(&self, p: &Placement, being_pbr: Pbr) -> Picture {
        let mut objects: Vec<CoveObject> = self.statics.iter().map(Placed::object).collect();
        let swing = self.swing(p.door_angle);
        for part in &self.door {
            objects.push(Object::placed(
                part.bvh.clone(),
                part.pbr,
                Transform::from_matrix(swing.matrix * part.to_world.matrix),
            ));
        }
        // The rim rides the same swing the door does, and its radiance is the
        // score. Emissive and not a light: see [`materials::rim`].
        objects.push(Object::placed(
            self.rim.clone(),
            materials::rim(self.glow_floor, self.glow_gain, p.score),
            swing,
        ));
        if let Some(g) = p.glint {
            let g = g * PER_M;
            objects.push(Object::placed(
                self.glint.clone(),
                materials::glint(self.glint_glow),
                Transform::translation(g.x, g.y, g.z),
            ));
        }
        objects.extend(self.extras.iter().map(Placed::object));
        // The body: the figure when there is one, and the capsule when there
        // is not. Never both — the capsule is the hero's *optical* proxy and
        // a figure standing inside a lens the size of itself is worse than
        // either half of that sentence.
        if let Some(hero) = p.hero.as_ref().filter(|h| !h.is_empty()) {
            if self.show_being {
                for part in hero.iter() {
                    objects.push(Object::placed(part.bvh.clone(), part.pbr, part.to_world.clone()));
                }
            }
        } else if self.show_being {
            let (centre, rot) = p.being;
            let c = centre * PER_M;
            objects.push(Object::placed(self.being.bvh.clone(), being_pbr, rigid(&rot, c.x, c.y, c.z)));
        }
        Picture {
            objects,
            lights: Vec::new(),
            env: self.env.clone(),
            sun: Some(self.sun),
            ground: self.ground,
            splats: None,
        }
    }

    /// The rune, as light: photons from the sun through the being, deposited
    /// wherever they land — which is the door, when the pose is the solution.
    ///
    /// Step 4 is what *scores* this map; the picture only has to show it. The
    /// being is the level's one transmissive surface (see
    /// [`materials`]), so the caustic pass aims its whole photon budget at it
    /// with nothing here to say so.
    pub fn caustic_map(&mut self, p: &Placement) -> CausticMap {
        if self.photons == 0 {
            return CausticMap::empty();
        }
        // Always the dispersive glass, whatever the camera's being is made of:
        // the rune's spectral rim is the photon map's, and it is the one place
        // in this level where the dispersion is the point.
        let picture = self.picture_at(p, self.being.pbr);
        caustics::trace(
            &picture,
            &CausticOptions {
                photons: self.photons,
                radius: Some(self.gather_for(self.photons)),
                ..Default::default()
            },
        )
    }

    /// The sun, for a caller that wants to place something along it.
    pub fn sun(&self) -> Sun {
        self.sun
    }

    /// How many traceable objects the static half has, for a sanity check.
    pub fn static_count(&self) -> usize {
        self.statics.len()
    }

    /// The door's hinge turn: about the vertical through its hung edge.
    /// Positive `angle` swings the door out of the cliff into the cove, which
    /// is -y, and a turn about +z takes the far edge the other way — so the
    /// matrix turns by `-angle`.
    fn swing(&self, angle: f64) -> Transform {
        let (h, r) = (self.hinge, Transform::rotation_z(-angle));
        Transform::from_matrix(
            tang::Mat4::translation(h.x, h.y, h.z) * r.matrix * tang::Mat4::translation(-h.x, -h.y, -h.z),
        )
    }
}

/// The hero's solids: the costume on its links, and the glass in its hand.
///
/// Evaluated once. Every body of `hero/figure.rs` is walked into the placed
/// primitives its union is made of, each distinct solid gets one BVH, and
/// each primitive is assigned to **the link whose flesh it is deepest
/// inside** ([`lump_depth`]) — so a bone drawn between the hip and the knee
/// rides the thigh, the shin below it rides the shin, and the cowl rides the
/// neck. Nothing here is a table of names: the assignment is geometric,
/// because the pivots the figure is drawn from and the pivots the body is
/// built from are the same `Rig::pivots()`.
///
/// The lens and its ring come back separately: they hang off
/// [`kosm::player::Body::held`], which is where the arm actually got the
/// glass to rather than where it was asked to put it.
fn hero_solids(doc: &vcad_ir::Document, scene: &CoveScene) -> anyhow::Result<(Vec<HeroSolid>, Vec<HeroSolid>)> {
    use super::hero::{figure, kit};
    use kosm::build::Params;

    // The same rig the body is built from — `being::hero_skeleton(Rig::DEFAULT)`
    // — so a part's rest pose and its link's pivot are the same number.
    let built = figure::figure(&Params::default())?;
    let spec = kosm::player::BodySpec::hero(&super::being::hero_skeleton(&figure::Rig::DEFAULT));

    let mut prims = Prims::default();
    let mut bvhs: HashMap<usize, Arc<Bvh<CoveGeom>>> = HashMap::new();
    let mut parts = Vec::new();
    for body in &built.bodies {
        let pbr = materials::pbr(&built.document, &body.material);
        for inst in instances::instances(&built.document, body.root, &mut prims)? {
            let bvh = bvhs
                .entry(Arc::as_ptr(&inst.solid) as usize)
                .or_insert_with(|| Arc::new(Bvh::build(hero_geometry(&inst.solid))))
                .clone();
            let Some(local) = bvh.bounds() else { continue };
            let to_world = placement(&inst.to_world);
            // where this primitive sits in the body's own frame, metres
            let box_mm = transform_aabb(&local, &to_world);
            let mid = Point3::new(
                0.5 * (box_mm.min.x + box_mm.max.x),
                0.5 * (box_mm.min.y + box_mm.max.y),
                0.5 * (box_mm.min.z + box_mm.max.z),
            );
            let q = kosm::player::body::hero_local(mid.x, mid.y, mid.z);
            let link = (0..spec.links.len())
                .min_by(|&a, &b| {
                    lump_depth(&spec.links[a], q)
                        .partial_cmp(&lump_depth(&spec.links[b], q))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            parts.push(HeroSolid {
                name: body.name.clone(),
                bvh,
                pbr,
                material: body.material.clone(),
                solid: Some(inst.solid.clone()),
                link,
                local: to_world,
            });
        }
    }
    anyhow::ensure!(!parts.is_empty(), "the hero evaluated to no solid");

    // The glass. `lens_mesh` is exact spherical caps with analytic normals,
    // which is the same trade the being's capsule makes and for the same
    // reason: the boolean is two spheres two and a half metres across meeting
    // in a disc a hundred and ten millimetres wide.
    let a = &scene.authored;
    let lens = Arc::new(Bvh::build(mesh_geometry(kit::lens_mesh(
        a.parameter_or("lens_segments", 96.0).max(8.0) as usize,
        a.parameter_or("lens_rings", 24.0).max(2.0) as usize,
    ))));
    let mut held = vec![HeroSolid {
        name: "lens".into(),
        bvh: lens,
        pbr: materials::lens_glass(scene.n_d),
        material: "N-BK7".into(),
        // The lens is `kit::lens_mesh`'s triangles and not a solid at all —
        // the boolean is two spheres two and a half metres across meeting in
        // a disc a hundred and ten millimetres wide. The raster tier asks
        // `kit::lens_mesh` for the same triangles.
        solid: None,
        link: 0,
        local: Transform::identity(),
    }];
    // …and its brass ring, out of the same `Built` the doorstep stills use.
    // Decoration, and it earns the line: a rimless disc of glass in a fist is
    // a hole in the picture, and the ring is what says the hero is holding
    // something rather than standing behind it.
    if let Ok(hardware) = kit::hardware(&Params::default()) {
        for body in &hardware.bodies {
            let pbr = materials::pbr(doc, &body.material);
            for inst in instances::instances(&hardware.document, body.root, &mut prims)? {
                let bvh = Arc::new(Bvh::build(hero_geometry(&inst.solid)));
                held.push(HeroSolid {
                    name: body.name.clone(),
                    bvh,
                    pbr,
                    material: body.material.clone(),
                    solid: Some(inst.solid.clone()),
                    link: 0,
                    local: placement(&inst.to_world),
                });
            }
        }
    }
    Ok((parts, held))
}

/// Which of the cove's surfaces one primitive of the `ground` root is.
///
/// The document unions the beach, the cliff, the boulders, the headlands and
/// the reef into one solid because the bake wants one inside; the picture wants
/// two colours out of it. The instance walk hands back the primitives that
/// union was made of, so the split is geometric and is stated here once: the
/// sand — beach and seabed, which are one slab — is the only primitive that
/// covers the cove in *both* directions, and everything standing on it is rock.
///
/// Both halves of that test are load-bearing now that the slab runs on past the
/// waterline. The headlands are cut like the beach and so are just as long in y
/// as it is; what separates them is that they are five metres wide in x, not
/// forty. The cliff is the mirror image — the width, not the length. So the
/// rule is the conjunction, and a primitive has to be the whole floor to be
/// sand. Millimetres, because these are world bounds of a vcad solid.
fn ground_material(world: &Aabb, scene: &CoveScene) -> &'static str {
    let half = 0.5 * scene.cove * PER_M;
    let (w, l) = (world.max.x - world.min.x, world.max.y - world.min.y);
    if w > half && l > half { "sand" } else { "rock" }
}

/// The level's daylight: a low afternoon sun and the sky it hangs in.
///
/// The sun is the key and the sky is the fill, and the ratio between them is
/// the whole look. Too much sky and the being's shadow disappears into a
/// wash; too little and the cove goes contrasty and photographic instead of
/// flat and bright. The defaults put roughly as much irradiance on the sand
/// from the sky as from the sun at this elevation, which leaves a shadow that
/// is soft, blue, and unmistakably there.
pub fn daylight(scene: &CoveScene) -> (Environment, Sun) {
    let a = &scene.authored;
    let sky = a.parameter_or("sky_intensity", 0.42) as f32;
    let irr = a.parameter_or("sun_irradiance", 6.2) as f32;
    let d = scene.sun_dir();
    let radius = a.parameter_or("sun_angular_radius", 0.02);
    let sun = Sun::new(
        Vec3::new(d.x, d.y, d.z),
        radius,
        // low afternoon light: warm, and warmer the lower it is
        [irr, 0.77 * irr, 0.46 * irr],
    );
    let env = if !sky_is_model(a) {
        // `sky = 0`: the two-colour gradient the level hung under before
        // there was a model. Kept so a picture can be taken both ways and the
        // difference looked at rather than argued about.
        Environment::Gradient(GradientEnv {
            zenith: [0.14, 0.30, 0.62],
            horizon: [0.38, 0.58, 0.85],
            // what a downward ray outside the cove finds: the sand it came off
            ground: [0.42, 0.36, 0.24],
            intensity: sky,
        })
    } else {
        Environment::Sky(sky_env(scene))
    };
    (env, sun)
}

/// Whether the level hangs under the analytic sky or the old gradient.
///
/// The level's `sky` knob says, and `--sky gradient` overrides it. A process
/// global and not a parameter because a `CoveScene` is built from
/// `Params::default()` at a dozen call sites, none of which is handed a flag,
/// and threading one through all of them to answer a comparison question
/// would be a worse change than this is.
static SKY_MODEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// `--sky gradient` / `--sky preetham`, once, before anything is drawn.
pub fn set_sky_model(on: bool) {
    SKY_MODEL.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

fn sky_is_model(a: &kosm::build::Built) -> bool {
    match SKY_MODEL.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => a.parameter_or("sky", 1.0) >= 0.5,
    }
}

/// The gradient's own mean radiance over the upper hemisphere, at
/// `sky_intensity = 1`.
///
/// [`SkyEnv`] is normalised to *its* mean radiance, so multiplying the knob by
/// this is what makes `sky = 0` and `sky = 1` the same exposure — the two look
/// different, which is the point, but the meter does not have to chase a stop
/// between them and neither does the level author.
const GRADIENT_FILL: f32 = 0.389;

/// The cove's clear sky, as the model both tiers read.
///
/// Preetham, at the level's own sun. `sky_turbidity` is the haze — two and a
/// half is a clean afternoon over water, four is the same beach with the wind
/// off the land — and `ground_albedo` scales the sand's own colour, which is
/// what a ray leaving the cove downward actually finds.
pub fn sky_env(scene: &CoveScene) -> kosm_render::env::SkyEnv {
    let a = &scene.authored;
    let d = scene.sun_dir();
    let g = a.parameter_or("ground_albedo", 1.0) as f32;
    kosm_render::env::SkyEnv::new(
        Vec3::new(d.x, d.y, d.z),
        a.parameter_or("sky_turbidity", 2.5) as f32,
        [0.42 * g, 0.36 * g, 0.24 * g],
        a.parameter_or("sky_intensity", 0.42) as f32 * GRADIENT_FILL,
        a.parameter_or("sun_angular_radius", 0.02) as f32,
    )
}

/// The haze, as both tiers apply it.
///
/// `air_density` is the extinction at sea level per metre and `air_scale_m`
/// the height it falls by `e` over. The default is deliberately gentle: over a
/// forty-metre cove the far headland reads a shade hazier than the sand
/// underfoot and nothing else changes, which is what aerial perspective looks
/// like at this scale. It is a **participating-medium approximation** that
/// the reference tracer does not trace — `kosm_render::post` says so at
/// length — and it is applied identically on both tiers so the settle blend
/// has nothing to fade between.
pub fn air(scene: &CoveScene) -> kosm_render::post::Aerial {
    let a = &scene.authored;
    kosm_render::post::Aerial {
        density: a.parameter_or("air_density", 0.0045) as f32,
        scale_h: a.parameter_or("air_scale_m", 60.0) as f32,
    }
}

/// The film both tiers put the light through: exposure, the lens's `cos⁴`,
/// bloom, ACES, sRGB. `exposure` is the level's number already multiplied by
/// whatever the meter measured.
pub fn post(scene: &CoveScene, exposure: f32) -> kosm_render::post::Post {
    let a = &scene.authored;
    kosm_render::post::Post {
        exposure,
        vignette: a.parameter_or("vignette", 0.35) as f32,
        bloom_threshold: a.parameter_or("bloom_threshold", 1.05) as f32,
        bloom_strength: a.parameter_or("bloom", 0.10) as f32,
        bloom_radius_px: a.parameter_or("bloom_radius_px", 7.0) as f32,
        aerial: air(scene),
        sky: sky_is_model(a).then(|| sky_env(scene)),
        // The cove is traced in vcad's millimetres; the haze is stated per
        // metre. `PER_M` is the one place the two units meet.
        units_per_metre: PER_M as f32,
    }
}

/// The sea: a lattice of heights at `sea_z`, from the waterline out to sea.
///
/// The swell is authored — two sine waves crossing at a shallow angle, five
/// centimetres of amplitude between them over wavelengths of seven and three
/// metres — because this slice's sea is a surface to look at and to stop the
/// being at, and nothing else. When the tide becomes the clock these heights
/// come from the far field instead and this function is where they arrive.
fn sea_field(scene: &CoveScene, a: &kosm::build::Built) -> HeightField {
    let (a1, l1) = (a.parameter_or("swell_a_mm", 30.0), a.parameter_or("swell_l_mm", 7000.0));
    let (a2, l2) = (a.parameter_or("swell_b_mm", 20.0), a.parameter_or("swell_m_mm", 3000.0));
    let cell = a.parameter_or("swell_cell_mm", 250.0).max(10.0);
    // Twice the cove across and a cove deep: wide enough that the camera never
    // finds its edge, and past it the ocean is the ground plane.
    let (x0, x1) = (-scene.cove * PER_M, scene.cove * PER_M);
    let y1 = scene.waterline() * PER_M;
    let y0 = y1 - scene.cove * PER_M;
    let nx = (((x1 - x0) / cell).round() as usize + 1).max(2);
    let ny = (((y1 - y0) / cell).round() as usize + 1).max(2);
    let (k1, k2) = (std::f64::consts::TAU / l1, std::f64::consts::TAU / l2);
    // the two crests run at shallow angles either side of the shore normal,
    // so the swell walks in rather than arriving as one flat corrugation
    let (c1, s1) = 0.20f64.sin_cos();
    let (c2, s2) = (-0.55f64).sin_cos();
    let mut heights = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        let y = y0 + j as f64 * cell;
        for i in 0..nx {
            let x = x0 + i as f64 * cell;
            heights.push(a1 * (k1 * (x * s1 + y * c1)).sin() + a2 * (k2 * (x * s2 + y * c2) + 1.7).sin());
        }
    }
    HeightField::new(nx, ny, Point3::new(x0, y0, scene.sea_z * PER_M), (cell, cell), heights)
}

/// A capsule of radius `r` and total height `h`, centred on the origin with
/// its axis along +z, as triangles with exact normals.
///
/// A latitude-longitude sphere cut at its equator and pulled apart by the
/// cylindrical span: the two cap rings at `phi = 0` are the ends of the
/// barrel, so the barrel needs no separate construction and the surface is
/// closed by the same index pattern throughout. The normals are the analytic
/// ones — a capsule's normal is `(cos φ cos θ, cos φ sin θ, sin φ)` wherever
/// you are on it, poles and barrel alike — so the shading is exact even
/// though the silhouette is not, which is the trade a tessellated dielectric
/// makes. Degenerate triangles at the two poles are dropped by [`TriMesh`].
fn capsule_mesh(r: f64, h: f64, segments: usize, rings: usize) -> TriMesh {
    let half = (h / 2.0 - r).max(0.0);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    // rings of latitude: the lower cap from the south pole up to its equator,
    // then the upper cap from its equator to the north pole
    let mut lat: Vec<(f64, f64)> = Vec::new();
    for v in 0..=rings {
        let phi = -std::f64::consts::FRAC_PI_2 * (1.0 - v as f64 / rings as f64);
        lat.push((phi, -half));
    }
    for v in 0..=rings {
        let phi = std::f64::consts::FRAC_PI_2 * (v as f64 / rings as f64);
        lat.push((phi, half));
    }
    for &(phi, z0) in &lat {
        let (sp, cp) = phi.sin_cos();
        for i in 0..segments {
            let th = std::f64::consts::TAU * i as f64 / segments as f64;
            let (st, ct) = th.sin_cos();
            let n = Vec3::new(cp * ct, cp * st, sp);
            positions.push(Point3::new(r * n.x, r * n.y, r * n.z + z0));
            normals.push(n);
        }
    }
    let mut indices: Vec<u32> = Vec::new();
    for band in 0..lat.len() - 1 {
        let (a0, b0) = ((band * segments) as u32, ((band + 1) * segments) as u32);
        for i in 0..segments as u32 {
            let j = (i + 1) % segments as u32;
            indices.extend_from_slice(&[a0 + i, a0 + j, b0 + j]);
            indices.extend_from_slice(&[a0 + i, b0 + j, b0 + i]);
        }
    }
    TriMesh::new(positions, normals, &indices)
}

/// How far out of the door's face the rim sits, millimetres.
///
/// Small enough that it reads as cut into the stone rather than hung on it,
/// large enough that no ray ever has to choose between two surfaces at the same
/// depth. At the sun's twenty-nine degrees off the door's normal it shades
/// under three millimetres of a hundred-and-twenty-millimetre keyhole, which is
/// why the rune's own photons do not notice it is there.
const RIM_PROUD_MM: f64 = 5.0;

/// A flat ring in the plane `y = centre.y`, facing -y: the keyhole's rim.
///
/// Millimetres, and world-placed rather than centred on the origin, because it
/// belongs to the door's shut frame and is swung by the door's own hinge. Two
/// rings of vertices and one quad between each pair; the normal is the door's,
/// which is exact for a flat annulus and is all a self-lit surface needs.
fn annulus_mesh(centre: Point3, r_in: f64, r_out: f64, segments: usize) -> TriMesh {
    let n = Vec3::new(0.0, -1.0, 0.0);
    let mut positions = Vec::with_capacity(2 * segments);
    let mut normals = Vec::with_capacity(2 * segments);
    for i in 0..segments {
        let th = std::f64::consts::TAU * i as f64 / segments as f64;
        let (s, c) = th.sin_cos();
        for r in [r_in, r_out] {
            positions.push(Point3::new(centre.x + r * c, centre.y, centre.z + r * s));
            normals.push(n);
        }
    }
    let mut indices: Vec<u32> = Vec::with_capacity(6 * segments);
    for i in 0..segments as u32 {
        let j = (i + 1) % segments as u32;
        let (a0, a1, b0, b1) = (2 * i, 2 * i + 1, 2 * j, 2 * j + 1);
        indices.extend_from_slice(&[a0, a1, b1]);
        indices.extend_from_slice(&[a0, b1, b0]);
    }
    TriMesh::new(positions, normals, &indices)
}

/// The geometry a solid is traced as: its analytic BRep if it has one, its
/// tessellation if a boolean took the BRep away — the same fallback the court
/// takes, and the same one `vcad-render --photoreal` takes.
/// A caller's own triangles, as this picture's geometry.
///
/// The way in for a shape the kernel will not hand over as a *trimmed* B-rep —
/// the being's capsule is one, and so is a lens cut as the intersection of two
/// spheres. Give the mesh exact analytic normals and the shading is exact even
/// where the silhouette is a polygon, which is the same trade [`capsule_mesh`]
/// makes.
pub fn mesh_geometry(mesh: TriMesh) -> CoveGeom {
    CoveGeom::Brep(BrepGeom::Mesh(mesh))
}

pub fn geometry_of(solid: &Solid) -> CoveGeom {
    if let Some(brep) = solid.as_brep() {
        let brep = Arc::new(brep.clone());
        let faces = brep.topology.faces.iter().map(|(id, _)| id).collect();
        return CoveGeom::Brep(BrepGeom::BRep { brep, faces });
    }
    let mut mesh = solid.to_mesh(0);
    vcad_kernel::vcad_kernel_tessellate::render_bake_default(&mut mesh);
    let n = mesh.vertices.len() / 3;
    let positions = (0..n)
        .map(|i| Point3::new(mesh.vertices[i * 3] as f64, mesh.vertices[i * 3 + 1] as f64, mesh.vertices[i * 3 + 2] as f64))
        .collect();
    let normals = if mesh.normals.len() == mesh.vertices.len() {
        (0..n).map(|i| Vec3::new(mesh.normals[i * 3] as f64, mesh.normals[i * 3 + 1] as f64, mesh.normals[i * 3 + 2] as f64)).collect()
    } else {
        Vec::new()
    };
    CoveGeom::Brep(BrepGeom::Mesh(TriMesh::new(positions, normals, &mesh.indices)))
}

/// vcad's placement, as the tracer's. The two are the same 4×4 under
/// different names.
fn placement(t: &VTransform) -> Transform {
    Transform { matrix: t.matrix }
}

/// A rigid object → world transform from a body → world rotation and a
/// translation in millimetres.
fn rigid(r: &Mat3, x: f64, y: f64, z: f64) -> Transform {
    Transform {
        matrix: tang::Mat4::new(
            r[(0, 0)], r[(0, 1)], r[(0, 2)], x, //
            r[(1, 0)], r[(1, 1)], r[(1, 2)], y, //
            r[(2, 0)], r[(2, 1)], r[(2, 2)], z, //
            0.0, 0.0, 0.0, 1.0,
        ),
    }
}

// ---- the camera and the frame ----------------------------------------------

/// Third person, over the shoulder — except at the door, where over the
/// shoulder is inside the cliff. Millimetres.
///
/// The far framing is the design's: three metres behind the being along its
/// facing and one and a half above, stepped a shoulder's width to the right so
/// the being sits off centre and the door it is walking at is not behind its
/// own head. It is right everywhere on the beach and wrong at exactly the place
/// the game ends. The solved pose stands the being four hundred millimetres off
/// the cliff face; three metres behind it, the eye is nearly in the stone, the
/// door fills the frame, and the being's own body sits over the keyhole it is
/// solving. A camera that hides the puzzle at the moment it is solved is not a
/// camera.
///
/// So there is a second framing, the doorstep: two metres out from the *face*
/// rather than from the being, two and a fifth up, a step to the being's right,
/// and aimed at the aperture itself. From there the door is seen from above and
/// to one side, the being leans into frame, and the keyhole and its rim stay
/// clear of it. The two are blended by how close the being is to the face —
/// `t = clamp((2 − d)/2, 0, 1)` — so nothing snaps: walking the last two metres
/// to the door raises the camera and swings it round as you go.
///
/// Two things are then true whatever the pose. The eye is never behind the
/// cliff's face plane, because it is clamped to a body's radius and half a
/// metre in front of it; and the level's own `cam_x_mm`/`cam_y_mm`/`cam_z_mm`
/// still win outright, because a fixed camera is an author's decision and not
/// a rule this function gets to overrule. The rest of the knobs are
/// `cam_back_mm`, `cam_up_mm`, `cam_side_mm`, `cam_ahead_mm`, `cam_vfov_deg`
/// and, for the doorstep, `cam_door_back_mm`, `cam_door_up_mm`,
/// `cam_door_side_mm`, `cam_door_reach_m` and `cam_face_clear_mm`.
pub fn camera(scene: &CoveScene, p: &Placement) -> Camera {
    let a = &scene.authored;
    let (centre, _) = p.being;
    let c = centre * PER_M;
    let f = p.facing();
    // the being's own right, which is where "over the shoulder" is measured.
    // `f × ẑ` is horizontal whatever the being's lean, which is what a camera
    // that steps sideways wants.
    let r = f.cross(&PVec3::new(0.0, 0.0, 1.0));
    let r = if r.norm() > 1e-9 { r.normalize() } else { PVec3::new(1.0, 0.0, 0.0) };

    // the far framing: over the shoulder, following
    let back = a.parameter_or("cam_back_mm", 3000.0);
    let up = a.parameter_or("cam_up_mm", 1500.0);
    let side = a.parameter_or("cam_side_mm", 2000.0);
    let ahead = a.parameter_or("cam_ahead_mm", 3500.0);
    let far_eye = [c.x - f.x * back + r.x * side, c.y - f.y * back + r.y * side, c.z + up];
    let far_target = [c.x + f.x * ahead, c.y + f.y * ahead, c.z];

    // the doorstep framing: off the face, up, and round to the shoulder side
    let face = scene.cliff_face_y();
    let d_back = a.parameter_or("cam_door_back_mm", 1500.0);
    let d_up = a.parameter_or("cam_door_up_mm", 1500.0);
    let d_side = a.parameter_or("cam_door_side_mm", 2400.0);
    let (ex, ey) = (c.x + r.x * d_side, (face * PER_M - d_back).min(c.y));
    let near_eye = [ex, ey, scene.sand_z_at(ex * MM, ey * MM) * PER_M + d_up];
    let ap = scene.door_frame().origin * PER_M;
    let near_target = [ap.x, ap.y, ap.z];

    // how much of the doorstep this pose wants
    let reach = a.parameter_or("cam_door_reach_m", 2.0).max(1e-3);
    let t = ((reach - (face - centre.y)) / reach).clamp(0.0, 1.0);
    let mix = |lo: f64, hi: f64| lo + (hi - lo) * t;

    // and the two things that are not negotiable: the eye stays out of the rock,
    // and it stays out of the sea. Wading, the being's centre drops toward the
    // waterline and the far framing's `cam_up_mm` drops with it; a metre of
    // that and the eye would be under the swell, looking at the inside of an
    // opaque teal surface. So the eye is floored a hand's breadth above the
    // flat waterline — the being in the water is seen from over it.
    let clear = (face - scene.being_r) * PER_M - a.parameter_or("cam_face_clear_mm", 500.0);
    let afloat = (scene.sea_z * PER_M) + a.parameter_or("cam_sea_clear_mm", 400.0);
    let eye = Point3::new(
        a.parameter_or("cam_x_mm", mix(far_eye[0], near_eye[0])),
        a.parameter_or("cam_y_mm", mix(far_eye[1], near_eye[1]).min(clear)),
        a.parameter_or("cam_z_mm", mix(far_eye[2], near_eye[2]).max(afloat)),
    );
    // The *aim* turns to the keyhole faster than the eye walks round to it.
    // Both on the same `t` and the last twenty per cent of the blend leaves the
    // door a fifth of a frame off centre with the cliff filling the rest, which
    // is a picture of the wrong thing; `√t` spends that fifth on the aim, where
    // it costs nothing and buys the composition.
    let ta = t.sqrt();
    let aim = |lo: f64, hi: f64| lo + (hi - lo) * ta;
    let target = Point3::new(
        aim(far_target[0], near_target[0]),
        aim(far_target[1], near_target[1]),
        aim(far_target[2], near_target[2]),
    );
    Camera::look_at(eye, target, Vec3::new(0.0, 0.0, 1.0), a.parameter_or("cam_vfov_deg", 50.0))
}

/// Integrator settings the level asks for, at a given sample count.
///
/// `max_depth` is higher than the court's: every path that carries the rune
/// enters the being and leaves it before it has touched anything, so a budget
/// that is fine for a gym full of opaque surfaces spends itself inside the
/// glass here.
pub fn options(scene: &CoveScene, spp: usize, seed: u64) -> PathTraceOptions {
    let a = &scene.authored;
    PathTraceOptions {
        spp: spp.max(1) as u32,
        max_depth: a.parameter_or("max_depth", 12.0).max(1.0) as u32,
        show_background: true,
        seed,
        denoise: a.parameter_or("denoise", 1.0) > 0.5,
        ..Default::default()
    }
}

/// Tonemap a film to an image, with no camera to hang the lens terms on.
///
/// The haze and the vignette both need the rig — one wants the eye's height
/// and the pixel's own view ray, the other the field of view — so a caller
/// with only a film gets the exposure and the tonemap, which is what this
/// always did. [`to_image_through`] is the one the cove's stills take.
pub fn to_image(film: &Film, exposure: f64) -> image::RgbaImage {
    let px = film.to_srgb8(exposure as f32, false);
    image::RgbaImage::from_raw(film.width, film.height, px).expect("film is width × height × 4")
}

/// The same, through the level's own film: haze, exposure, `cos⁴`, bloom,
/// ACES, sRGB — the chain `kosm-view`'s raster tier applies in a shader, so a
/// still taken on either tier is the same picture.
pub fn to_image_through(
    film: &Film,
    scene: &CoveScene,
    cam: &Camera,
    exposure: f64,
) -> image::RgbaImage {
    let px = post(scene, exposure as f32).apply(film, cam, false);
    image::RgbaImage::from_raw(film.width, film.height, px).expect("film is width × height × 4")
}

/// One frame of the cove at one pose, through the CPU integrator.
pub fn frame(
    scene: &CoveScene,
    placement: &Placement,
    size: (u32, u32),
    spp: usize,
) -> anyhow::Result<image::RgbaImage> {
    let mut picture = Scene::new(scene)?;
    frame_in(&mut picture, scene, placement, size, spp)
}

/// The same, on a [`Scene`] the caller has already built.
///
/// `sims/rune/mod.rs`'s still needs one: to draw the hero at a solved
/// [`super::rune::HeroPose`] it has to ask [`Scene::hero_at`] where the
/// costume goes, and that is a method on a built scene. One statement of the
/// render, two ways in.
pub fn frame_in(
    picture: &mut Scene,
    scene: &CoveScene,
    placement: &Placement,
    size: (u32, u32),
    spp: usize,
) -> anyhow::Result<image::RgbaImage> {
    let rune = picture.caustic_map(placement);
    let at = picture.at(placement);
    let cam = camera(scene, placement);
    let opts = options(scene, spp, STILL_SEED);
    let film = pathtrace::render_with_caustics(&at, &cam, size.0, size.1, &opts, Some(&rune));
    Ok(to_image_through(&film, scene, &cam, scene.authored.parameter_or("exposure", 0.7)))
}

/// The most [`Scene::gather_for`] may widen the authored gather radius.
///
/// The `1/sqrt(N)` law would ask for three and a half times the still's radius
/// at the live tier's fiftieth of its photons, and measured at two hundred
/// passes the last of that widening buys almost nothing — the caustic's
/// Laplacian falls by seven per cent between twice and three and a half times
/// — while it plainly costs the thing the rune is *for*: at two hundred
/// millimetres the disc is a fifth of the patch and the spectral rim is
/// averaged into the sand. Twice is where the trade turns.
const MAX_WIDENING: f64 = 2.0;

/// The still's seed. Fixed, so two runs of `kosm run rune` differ only where the
/// level does.
const STILL_SEED: u64 = 0xc0_be_51_11;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_capsule_is_closed_and_its_normals_are_unit() {
        let m = capsule_mesh(350.0, 1400.0, 24, 8);
        assert!(!m.triangles().is_empty());
        assert_eq!(m.normals().len(), m.positions().len());
        for n in m.normals() {
            assert!((n.norm() - 1.0).abs() < 1e-12);
        }
        // every vertex is exactly a radius from the axis segment
        for p in m.positions() {
            let z = p.z.clamp(-350.0, 350.0);
            let d = ((p.x * p.x + p.y * p.y) + (p.z - z) * (p.z - z)).sqrt();
            assert!((d - 350.0).abs() < 1e-9, "{p:?} is {d} from the axis");
        }
    }

    /// The offline frame's radius is the authored one, to the float, and the
    /// live tier's is widened — bounded. The first half is the promise that
    /// nothing here moved `kosm run rune`'s picture.
    #[test]
    fn the_gather_widens_for_a_smaller_map_and_never_for_the_still() {
        let scene = CoveScene::bundled().expect("the bundled cove");
        let picture = Scene::new(&scene).expect("the cove's picture");
        let authored = picture.gather;
        assert_eq!(picture.gather_for(picture.photons), authored, "the still's own budget");
        assert_eq!(picture.gather_for(picture.photons * 2), authored, "and it never narrows");
        let live = picture.gather_for(picture.photons / 50);
        assert!(live > authored, "a fiftieth of the photons wants a wider disc: {live}");
        assert!(live <= authored * MAX_WIDENING + 1e-9, "and not an unbounded one: {live}");
        // …and the camera's being is the dispersive one until it is told not
        // to be, which is what keeps `kosm run rune` spectral.
        assert!(picture.being.pbr.is_dispersive());
        assert!(!materials::achromatic(picture.being.pbr).is_dispersive());
    }

    /// One primitive of the `ground` root is the floor and every other one is
    /// rock. The rule is a footprint test (see [`ground_material`]) and the
    /// footprints moved when the cove was closed: the sand slab now runs
    /// fourteen metres past the waterline, and the headlands cut out of the
    /// same plane are exactly as long as it is. So the check is that there is
    /// still precisely *one* sand — the beach and its seabed — and that the
    /// cliff, the boulders, the headlands and the reef are all rock.
    #[test]
    fn the_ground_is_one_beach_and_the_rest_of_it_is_rock() -> anyhow::Result<()> {
        let scene = CoveScene::bundled()?;
        let picture = Scene::new(&scene)?;
        let sand = materials::pbr(&scene.authored.document, "sand").base_color;
        let rock = materials::pbr(&scene.authored.document, "rock").base_color;
        // the sea is in `statics` too, and it is neither
        let water = materials::pbr(&scene.authored.document, "water").base_color;
        let (mut sands, mut rocks) = (0, 0);
        for p in &picture.statics {
            match p.pbr.base_color {
                c if c == sand => sands += 1,
                c if c == rock => rocks += 1,
                c if c == water => {}
                c => panic!("the ground grew a surface that is neither sand nor rock: {c:?}"),
            }
        }
        assert_eq!(sands, 1, "the beach and its seabed are one slab, so exactly one primitive is sand");
        // three boulders, a sea stack, the cliff, six headland steps and the reef
        assert!(rocks >= 10, "only {rocks} of the cove's primitives came out as rock");
        Ok(())
    }

    /// The picture never dips under the sea. Wading, the being's centre falls
    /// toward the waterline and the far framing's eye falls with it; the floor
    /// is what keeps the camera over the water looking down at a being in it
    /// rather than inside an opaque teal surface looking at nothing.
    #[test]
    fn the_eye_stays_over_the_water() {
        let scene = CoveScene::bundled().expect("the bundled cove");
        let floor = (scene.sea_z + scene.authored.parameter_or("cam_sea_clear_mm", 400.0) * MM) * PER_M;
        // out along the seabed, until the being is under
        for k in 0..20 {
            let y = scene.waterline() - scene.seabed * k as f64 / 19.0;
            let p = Placement::standing(&scene, 0.0, y, 0.0);
            let cam = camera(&scene, &p);
            assert!(cam.eye.z >= floor - 1e-9, "wading at y = {y:+.1} m the eye is at {:.0} mm, under the waterline's floor at {floor:.0}", cam.eye.z);
        }
    }

    #[test]
    fn a_standing_being_faces_the_door_and_stands_on_the_sand() {
        let scene = CoveScene::bundled().expect("the bundled cove");
        let p = Placement::standing(&scene, scene.spawn_x, scene.spawn_y, 0.0);
        let (c, _) = p.being;
        assert!((c.z - (scene.sand_z_at(c.x, c.y) + scene.being_h / 2.0)).abs() < 1e-12);
        let to_door = scene.door_frame().origin - c;
        let f = p.facing();
        assert!(f.z.abs() < 1e-12, "the facing is horizontal");
        assert!(f.dot(&PVec3::new(to_door.x, to_door.y, 0.0)) > 0.0, "and it points at the door");
    }
}



// ---- the raster tier's view of the same level -------------------------------
//
// Everything below is **additive**: it hands the same solids this file already
// traces to `kosm_view::raster` as triangles, and changes nothing about the
// picture the tracer makes. Two tiers, one level — see
// `docs/plans/2026-09-10-bake-and-raster-design.md`.

/// What one tessellated part of the cove is, so the raster tier knows how to
/// move it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The beach, the cliff, the boulders, the headlands, the reef: fixed.
    Ground,
    /// The door leaf, which swings on the hinge.
    Door,
    /// The keyhole's rim, which rides the same hinge and glows with the score.
    Rim,
    /// The glint, placed on the sand wherever the hint puts it.
    Glint,
    /// The capsule, when the level is played with one.
    Being,
    /// A part of the hero's costume, riding link `link` of the body.
    Hero { link: usize },
    /// The glass in the hand and its brass ring, in the lens's own frame.
    Held,
}

/// One tessellated part, in the frame [`RasterMesh::to_world`] maps out of.
///
/// Millimetres — the picture's units — because everything else in this file
/// is, and because the one conversion belongs at the boundary rather than
/// scattered through the walk. `kosm_view::raster::Mesh::from_mm` is the
/// boundary.
pub struct RasterMesh {
    /// The document's own body name, or the level's word for it.
    pub name: String,
    /// The cove's surface name — `sand`, `rock`, `stone`, `cloak` — which
    /// resolves through [`materials::gpu_cove`] to one library entry.
    pub material: String,
    pub positions: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub indices: Vec<u32>,
    /// The part's own placement inside its role's frame, row-major
    /// millimetres. For [`Role::Ground`] this is the whole placement; for a
    /// hero part it is the primitive's frame *inside* the link.
    pub to_world: [[f64; 4]; 4],
    pub role: Role,
}

/// The cove, tessellated once, for the raster tier.
///
/// The same walk [`Scene::new`] does — the document's roots into placed
/// primitives, `ground_material` splitting the one ground solid into sand and
/// rock, the door picked out by its root — with `to_mesh` where that builds a
/// BVH. The normals are **smoothed** throughout, with `hero/stage.rs`'s own
/// welding rule, because a raster tier has no analytic surface to fall back on
/// and a facetted cliff is a low-poly mountain.
///
/// The rim, the glint and the capsule come from the same [`annulus_mesh`] and
/// [`capsule_mesh`] the tracer draws, so the two tiers are the same shapes.
pub fn raster_meshes(scene: &CoveScene) -> anyhow::Result<Vec<RasterMesh>> {
    let a = &scene.authored;
    let doc = &a.document;
    let mut prims = Prims::default();
    let mut out = Vec::new();
    let segments = a.parameter_or("raster_segments", RASTER_SEGMENTS as f64).max(3.0) as u32;

    for root in &doc.roots {
        for inst in instances::instances(doc, root.root, &mut prims)? {
            let (positions, normals, indices) = tessellate(&inst.solid, segments);
            if indices.is_empty() {
                continue;
            }
            let to_world = placement(&inst.to_world);
            let (role, material) = if root.material == "door" || root.material == materials::DOOR {
                (Role::Door, materials::DOOR.to_owned())
            } else {
                let local = aabb_of(&positions);
                let world = transform_aabb(&local, &to_world);
                (Role::Ground, ground_material(&world, scene).to_owned())
            };
            out.push(RasterMesh {
                name: root.material.clone(),
                material,
                positions,
                normals,
                indices,
                to_world: rows(&to_world),
                role,
            });
        }
    }
    anyhow::ensure!(
        out.iter().any(|m| m.role == Role::Door),
        "the cove needs a `door` root to draw"
    );

    // The keyhole's rim, in the door's shut world frame — the hinge carries
    // both, exactly as it does for the tracer.
    out.push(mesh_part(
        "rim",
        "stone",
        annulus_mesh(
            Point3::new(
                scene.door_x * PER_M,
                (scene.cliff_face_y() - RIM_PROUD_MM * MM) * PER_M,
                (scene.door_sill() + scene.aperture_z) * PER_M,
            ),
            scene.aperture_r * PER_M,
            (scene.aperture_r + scene.rim_w) * PER_M,
            a.parameter_or("rim_segments", 96.0).max(8.0) as usize,
        ),
        Role::Rim,
    ));

    let glint_r = scene.glint_r * PER_M;
    out.push(mesh_part(
        "glint",
        "sand",
        capsule_mesh(glint_r, 2.0 * glint_r, 32, 12),
        Role::Glint,
    ));
    out.push(mesh_part(
        "being",
        "N-BK7",
        capsule_mesh(
            scene.being_r * PER_M,
            scene.being_h * PER_M,
            a.parameter_or("being_segments", 96.0).max(8.0) as usize,
            a.parameter_or("being_rings", 32.0).max(3.0) as usize,
        ),
        Role::Being,
    ));

    // The figure and the glass. `hero_solids` already decides which link
    // carries which primitive; this repeats the walk for the triangles and
    // takes the same answer, so the raster hero and the traced hero are one
    // costume on one skeleton.
    if let Ok((parts, held)) = hero_solids(doc, scene) {
        for (solids, role) in [(&parts, None), (&held, Some(Role::Held))] {
            for s in solids {
                let (positions, normals, indices) = match s.solid.as_ref() {
                    Some(solid) => tessellate(solid, segments),
                    // the lens: `kit::lens_mesh`'s exact spherical caps, the
                    // same triangles with the same analytic normals the
                    // tracer refracts through
                    None => {
                        let mesh = super::hero::kit::lens_mesh(
                            a.parameter_or("lens_segments", 96.0).max(8.0) as usize,
                            a.parameter_or("lens_rings", 24.0).max(2.0) as usize,
                        );
                        (
                            mesh.positions().iter().map(|p| [p.x, p.y, p.z]).collect(),
                            mesh.normals().iter().map(|n| [n.x, n.y, n.z]).collect(),
                            mesh.triangles().iter().flat_map(|t| t.iter().copied()).collect(),
                        )
                    }
                };
                if indices.is_empty() {
                    continue;
                }
                out.push(RasterMesh {
                    name: s.name.clone(),
                    material: s.material.clone(),
                    positions,
                    normals,
                    indices,
                    to_world: rows(&s.local),
                    role: role.unwrap_or(Role::Hero { link: s.link }),
                });
            }
        }
    }
    Ok(out)
}

/// A part built from a mesh this file already makes, centred in its own frame.
fn mesh_part(name: &str, material: &str, mesh: TriMesh, role: Role) -> RasterMesh {
    let positions = mesh.positions().iter().map(|p| [p.x, p.y, p.z]).collect();
    let normals = mesh.normals().iter().map(|n| [n.x, n.y, n.z]).collect();
    let indices = mesh.triangles().iter().flat_map(|t| t.iter().copied()).collect();
    RasterMesh {
        name: name.to_owned(),
        material: material.to_owned(),
        positions,
        normals,
        indices,
        to_world: rows(&Transform::identity()),
        role,
    }
}

/// How finely a B-rep is tessellated for the raster tier.
///
/// **`to_mesh(0)` is not "the default", it is the floor.**
/// `TessellationParams::from_segments(0)` clamps to three segments round a
/// circle and four bands of latitude, which is a sphere drawn as an
/// octahedron. Nothing in the tracer ever noticed, because
/// [`hero_geometry`] and [`geometry_of`] hand a B-rep to the integrator as a
/// *B-rep* — the rays intersect the analytic sphere and the picture is round
/// however coarse a mesh of it would have been. A rasterizer has no such
/// fallback: the silhouette it draws is the triangles it is given.
///
/// Forty-eight is where the cove stops changing: the hero's cowl is a
/// four-hundred-millimetre sphere seen from a metre and a half, which is
/// about seven hundred pixels across at 1280 wide, and forty-eight segments
/// put a facet under fifteen of them — under the crease angle
/// `stage::smooth_normals` welds across, so the shading is smooth and only
/// the silhouette is polygonal at all. It costs triangles and nothing else:
/// the tier draws the cove once and the buffers never change.
const RASTER_SEGMENTS: u32 = 48;

/// A solid's triangles with smoothed normals, millimetres.
fn tessellate(solid: &Solid, segments: u32) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<u32>) {
    let mut mesh = solid.to_mesh(segments);
    vcad_kernel::vcad_kernel_tessellate::render_bake_default(&mut mesh);
    let n = mesh.vertices.len() / 3;
    let pts: Vec<Point3> = (0..n)
        .map(|i| {
            Point3::new(
                mesh.vertices[i * 3] as f64,
                mesh.vertices[i * 3 + 1] as f64,
                mesh.vertices[i * 3 + 2] as f64,
            )
        })
        .collect();
    // `render_bake` is vcad's own "prepare for rendering" pass and it leaves
    // crease-aware vertex normals behind; they are the ones the tracer's mesh
    // path ([`geometry_of`]) shades with, so taking them here is what keeps
    // the two tiers on one surface. The welded fallback is for a solid that
    // arrived as bare triangles with none.
    let normals = if mesh.normals.len() == mesh.vertices.len() {
        (0..n)
            .map(|i| {
                [
                    mesh.normals[i * 3] as f64,
                    mesh.normals[i * 3 + 1] as f64,
                    mesh.normals[i * 3 + 2] as f64,
                ]
            })
            .collect()
    } else {
        super::hero::stage::smooth_normals(&pts, &mesh.indices)
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect()
    };
    (pts.iter().map(|p| [p.x, p.y, p.z]).collect(), normals, mesh.indices.clone())
}

/// A `Transform` as row-major rows, which is how the raster tier reads a vcad
/// placement.
fn rows(t: &Transform) -> [[f64; 4]; 4] {
    let m = &t.matrix;
    [
        [m.c0.x, m.c1.x, m.c2.x, m.c3.x],
        [m.c0.y, m.c1.y, m.c2.y, m.c3.y],
        [m.c0.z, m.c1.z, m.c2.z, m.c3.z],
        [m.c0.w, m.c1.w, m.c2.w, m.c3.w],
    ]
}

fn aabb_of(positions: &[[f64; 3]]) -> Aabb {
    let mut b = Aabb::empty();
    for p in positions {
        b.include_point(&Point3::new(p[0], p[1], p[2]));
    }
    b
}

/// The door's hinge turn at `angle`, as the raster tier's rows.
///
/// The tracer's [`Scene::swing`] without a built scene in hand: the same
/// rotation about the vertical through the hung edge, so the two tiers swing
/// one door.
pub fn door_swing(scene: &CoveScene, angle: f64) -> [[f64; 4]; 4] {
    let h = Point3::new(
        (scene.door_x - scene.door_w / 2.0) * PER_M,
        scene.cliff_face_y() * PER_M,
        0.0,
    );
    let r = Transform::rotation_z(-angle);
    rows(&Transform::from_matrix(
        tang::Mat4::translation(h.x, h.y, h.z) * r.matrix * tang::Mat4::translation(-h.x, -h.y, -h.z),
    ))
}

/// The world frame link `link` puts its solids in, millimetres.
///
/// The translation half of [`Scene::hero_at`], stated for a caller that has
/// the parts and the pivots but not a built scene: `world_mm = R·P·p_mm +
/// 1000·(pos − R·pivot)`, where `P` is [`hero_axes`].
pub fn hero_frame(part: &kosm::player::Part, pivot: PVec3) -> [[f64; 4]; 4] {
    let (pos, r) = (part.pose.pos, part.pose.rot);
    let t = (pos - r.mul_vec(pivot)) * PER_M;
    rows(&rigid(&r.mul_mat(&hero_axes()), t.x, t.y, t.z))
}

/// The frame the glass in the hand is in, millimetres.
pub fn held_frame(pose: &kosm::player::Pose) -> [[f64; 4]; 4] {
    let c = pose.pos * PER_M;
    rows(&rigid(&pose.rot, c.x, c.y, c.z))
}

/// The being's own frame, millimetres, from a `Placement`'s body → world pair.
pub fn being_frame(centre: PVec3, rot: &Mat3) -> [[f64; 4]; 4] {
    let c = centre * PER_M;
    rows(&rigid(rot, c.x, c.y, c.z))
}

/// A translation, millimetres — where the glint goes.
pub fn at_frame(p: PVec3) -> [[f64; 4]; 4] {
    let c = p * PER_M;
    rows(&Transform::translation(c.x, c.y, c.z))
}
