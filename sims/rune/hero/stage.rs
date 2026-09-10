//! The doorstep, as a stage: sand, a cliff, the cove's door, and its daylight.
//!
//! A character sheet does not want a forty-metre cove. It wants the one
//! square of it the player will look at — the sand at the foot of the door —
//! lit by exactly the light the cove is lit by, so that a still made here is
//! a still of the level and not of a studio.
//!
//! So the geometry is small and local and the *light is copied*. Every number
//! under "the daylight" below is `sims/rune/render.rs`'s `daylight` and
//! `sims/rune/scene.rs`'s `sun_az_deg`/`sun_el_deg`, and the cove's four
//! colours under [`palette`] are `sims/rune/materials.rs`'s. They are copied
//! rather than called because `render.rs` and `materials.rs` belong to the
//! cove and the cove is a forty-metre bake; a character stage that had to
//! build one to borrow a colour would not be a character stage.
//!
//! The **substances** are not copied. Every arm of [`palette`] starts from
//! `kosm::material`'s `pbr()` for the name and overrides only what the art
//! direction owns, exactly as `sims/rune/materials.rs` does — and the hero's
//! own costume is in that library too, so the colour a body is *authored*
//! with (`Body::substance`) and the colour this file paints it are one
//! number, checked by a test at the foot of the file.
//!
//! **The frame.** Millimetres, z up, and the origin is the *door's sill* —
//! the point on the sand at the middle of the shut door's face. So:
//!
//! - the door's face is the plane `y = 0`, the stone runs +y into the cliff;
//! - the sand is the plane `z = SLOPE · y`, which is zero at the door and
//!   falls away toward the sea at −y, exactly the cove's `beach_slope`;
//! - the keyhole is at `(0, 0, APERTURE_Z)`, which is the cove's
//!   `aperture_z_mm` above the sill.
//!
//! That is the cove's own geometry with its origin moved, so a pose solved
//! here drops into `sims/rune` by adding `(door_x, cliff_face_y, door_sill)`.

use std::collections::HashMap;
use std::sync::Arc;

use kosm::brep::instances::{self, Prims};
use kosm::build::{Built, Params, build};
use kosm_render::math::{Point3, Transform, Vec3};
use kosm_render::pathtrace::{
    self, Camera, Environment, Film, GradientEnv, Object, PathTraceOptions, Pbr, Sun,
};
use kosm_render::{Bvh, TriMesh};
use vcad_kernel::Solid;
use vcad_kernel_math::Transform as VTransform;
use vcad_kernel_raytrace::BrepGeom;

// ---- the cove's numbers, copied ------------------------------------------

/// The beach's grade. `sims/rune/scene.rs`, `beach_slope`.
pub const SLOPE: f64 = 0.06;
/// The door: `door_w_mm`, `door_h_mm`, `door_t_mm`.
pub const DOOR_W: f64 = 1800.0;
pub const DOOR_H: f64 = 2600.0;
pub const DOOR_T: f64 = 200.0;
/// The keyhole: `aperture_r_mm` at `aperture_z_mm` above the sill.
pub const APERTURE_R: f64 = 120.0;
pub const APERTURE_Z: f64 = 400.0;
/// The rim ring around it: `rim_w_mm`, and how far proud of the face it sits
/// (`render.rs`'s `RIM_PROUD_MM`).
pub const RIM_W: f64 = 30.0;
pub const RIM_PROUD: f64 = 5.0;
/// Toward the sun: `sun_az_deg` about z from +x toward +y, `sun_el_deg` up.
pub const SUN_AZ_DEG: f64 = 250.0;
pub const SUN_EL_DEG: f64 = 22.0;
/// The d-line index of N-BK7, `n_d`.
pub const N_D: f64 = 1.5168;

/// The unit vector **toward** the sun, and the one the light **travels**
/// along, which is its negative.
///
/// The whole of the doorstep's staging hangs off the second of these: a lens
/// throws its focus along the chief ray through its own centre, so where the
/// hero may stand is decided by this vector and the keyhole's height and by
/// nothing else. See [`super::doorstep`].
pub fn sun_dir() -> Vec3 {
    let (az, el) = (SUN_AZ_DEG.to_radians(), SUN_EL_DEG.to_radians());
    let (sa, ca) = az.sin_cos();
    let (se, ce) = el.sin_cos();
    Vec3::new(ce * ca, ce * sa, se)
}

/// The direction sunlight travels: down the beach and into the cliff face.
pub fn sun_ray() -> Vec3 {
    -sun_dir()
}

/// The top of the sand at a point on the beach, in the stage's frame.
pub fn sand_z(y: f64) -> f64 {
    SLOPE * y
}

/// The keyhole's centre.
pub fn keyhole() -> Point3 {
    Point3::new(0.0, 0.0, APERTURE_Z)
}

// ---- the level ------------------------------------------------------------

/// The stage: one slab of sand, the cliff either side of the door and over
/// it, the door itself, and the keyhole with its rim.
///
/// Every root is a union of primitives and there is not one boolean in it, so
/// [`instances`] hands back placed spheres and cubes and no `vcad_eval` runs.
/// That is not thrift for its own sake: a primitive keeps its analytic BRep,
/// and an analytic BRep is what the tracer wants.
pub fn stage(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        let span = b.param("stage_span_mm", 26000.0); // how much beach there is
        let slope = b.param("stage_slope", SLOPE);
        let door_w = b.param("door_w_mm", DOOR_W);
        let door_h = b.param("door_h_mm", DOOR_H);
        let door_t = b.param("door_t_mm", DOOR_T);
        let ap_r = b.param("aperture_r_mm", APERTURE_R);
        let ap_z = b.param("aperture_z_mm", APERTURE_Z);
        let rim_w = b.param("rim_w_mm", RIM_W);
        let cliff_t = b.param("cliff_t_mm", 4000.0);
        let cliff_h = b.param("cliff_h_mm", 5200.0);

        // ---- the sand ----------------------------------------------------
        // A slab, modelled flat with its top face through the origin, tilted
        // about x, then dropped onto the sand plane at its own mid-y. The
        // cove's `on_sand` trick, and the reason the beach is one primitive
        // rather than a mesh: `rotate_x` of a cube is still a cube.
        let deg = slope.atan().to_degrees();
        let thick = 2000.0;
        let mid = -0.35 * span; // the beach reaches further seaward than inland
        let length = span * (1.0 + slope * slope).sqrt();
        b.body("sand").material("sand").add(
            b.boxed(span, length, thick)
                .at(0.0, 0.0, -0.5 * thick)
                .rotate_x(deg)
                .at(0.0, mid, slope * mid),
        );

        // ---- the cliff ----------------------------------------------------
        // Three slabs around the door's opening, their faces flush with it —
        // the cove sets its door in a recess exactly this deep, so the face of
        // the stone and the face of the door are one plane and the door reads
        // by colour alone.
        let jamb = 0.5 * (span - door_w);
        let cliff = b.body("cliff");
        cliff.material("rock");
        for side in [-1.0, 1.0] {
            cliff.add(b.box_at(
                [side * 0.5 * door_w, side * (0.5 * door_w + jamb)],
                [0.0, cliff_t],
                [-1500.0, cliff_h],
            ));
        }
        cliff.add(b.box_at([-0.5 * door_w, 0.5 * door_w], [0.0, cliff_t], [door_h, cliff_h]));

        // ---- the door ------------------------------------------------------
        b.body("door")
            .material("stone")
            .add(b.boxed(door_w, door_t, door_h).at(0.0, 0.5 * door_t, 0.5 * door_h));

        // ---- the keyhole ---------------------------------------------------
        // The cove's aperture is not geometry: it is the disc on the face the
        // score is read over. Here it is drawn, because a still has to show
        // the player what the lens is aimed at — a shallow dark disc set into
        // the stone, and the rim ring around it, proud of the face by the same
        // five millimetres `render.rs` gives it.
        b.body("keyhole").material("keyhole").add(
            b.cylinder(ap_r, 40.0).rotate_x(90.0).at(0.0, 40.0, ap_z),
        );
        // The ring: a torus lying in the door's face. Its minor radius is half
        // the ring's width, so it stands `rim_w/2` proud where the cove's
        // annulus stands `RIM_PROUD`; near enough, and a torus is a primitive.
        b.body("rim").material("rim").add(
            b.torus(ap_r + 0.5 * rim_w, 0.5 * rim_w)
                .rotate_x(90.0)
                .at(0.0, RIM_PROUD, ap_z),
        );
    })
}

// ---- the daylight ---------------------------------------------------------

/// The cove's sky and the cove's sun, copied out of `sims/rune/render.rs`'s
/// `daylight` to the last digit.
///
/// The ratio between them is the look: roughly as much irradiance on the sand
/// from the sky as from the sun at this elevation, which leaves a shadow that
/// is soft, blue and unmistakably there. A character lit any other way is a
/// character from another game.
pub fn daylight() -> (Environment, Sun) {
    let sky = 0.42f32;
    let env = Environment::Gradient(GradientEnv {
        zenith: [0.14, 0.30, 0.62],
        horizon: [0.38, 0.58, 0.85],
        ground: [0.42, 0.36, 0.24],
        intensity: sky,
    });
    let irr = 6.2f32;
    let d = sun_dir();
    let sun = Sun::new(Vec3::new(d.x, d.y, d.z), 0.02, [irr, 0.77 * irr, 0.46 * irr]);
    (env, sun)
}

/// What the film is developed at. `sims/rune/scene.rs`'s `exposure`.
pub const EXPOSURE: f64 = 0.7;

// ---- the palette ----------------------------------------------------------

/// A surface's name, resolved to a flat PBR.
///
/// The cove's four are copied from `sims/rune/materials.rs`; the rest are the
/// hero's own and are chosen against those four. Linear colours, never sRGB.
///
/// **The substance is the library's and the look is the level's** — the rule
/// `sims/rune/materials.rs` states, kept here. Every arm below starts from
/// `kosm::material`'s `pbr()` for the name and overrides exactly the fields
/// the art direction owns: the albedo, the roughness, the weak dielectric
/// highlight. Nothing here restates a density and nothing here adds a lobe.
///
/// The costume names — `cloak`, `cream`, `skin`, `blush`, `ink`, `boot` — are
/// entries in the library in their own right, so the colour below and the
/// colour a `Body::substance` was authored with are the same number; the test
/// at the foot of this file is what keeps them that way. The two the hero
/// *re*-colours are `leather` and `brass`, whose library entries are a
/// general leather and a general brass and whose place in this picture is a
/// russet satchel and a warm buckle.
///
/// One colour and one roughness each, no clearcoat, no sheen, and **no
/// subsurface**: the library's `skin` scatters, as skin does, and this look
/// does not. The picture's interest is the shapes, the sun and the glass.
pub fn palette(name: &str) -> Pbr {
    let lib = |name: &str| {
        kosm::material::named(name).unwrap_or_else(|| panic!("`{name}` is not in kosm::material")).pbr()
    };
    // A flat surface over whatever the library says the substance is: a
    // diffuse lobe, a weak dielectric highlight, nothing layered, no walk.
    let flat = |substance: &str, base: [f32; 3], roughness: f32| Pbr {
        base_color: base,
        roughness,
        specular: 0.25,
        subsurface: 0.0,
        ..lib(substance)
    };
    match name {
        // ---- the cove's, copied ------------------------------------------
        "sand" => flat("dry sand", [0.85, 0.54, 0.22], 0.9),
        "rock" => flat("granite", [0.13, 0.19, 0.33], 0.9),
        "stone" => flat("granite", [0.24, 0.20, 0.16], 0.85),

        // ---- the door's furniture ----------------------------------------
        // The keyhole is a hole: darker than any stone, so it reads as depth
        // and not as a coin stuck to the door.
        "keyhole" => flat("granite", [0.03, 0.025, 0.02], 0.9),
        // The rim glows. `materials::rim(glow_floor = 0.8, gain, score = 0)`
        // is the keyhole nobody has solved yet — findable, not loud — and
        // that is the state every one of these stills is in.
        "rim" => Pbr {
            base_color: [0.06, 0.05, 0.04],
            roughness: 0.6,
            specular: 0.2,
            emissive: [0.8, 0.688, 0.464],
            ..Default::default()
        },

        // ---- the hero ------------------------------------------------------
        // The cloak has to read against *three* backdrops: warm pale sand, a
        // cool blue-grey cliff, and the dark warm stone of the door it stands
        // in front of. Warm anything disappears into the sand; anything dark
        // disappears into the door. So: a saturated mid-value cyan-teal, the
        // one hue nothing else in the cove owns — the sea is far darker and
        // greener — with the value sitting squarely between the sand above it
        // and the stone behind it.
        //
        // These six are the library's own numbers, unchanged. They are here
        // as arms rather than as a fall-through so that the one file an
        // agent reads to change the costume is still this one.
        "cloak" | "cream" | "skin" | "blush" | "ink" | "boot" => {
            Pbr { specular: 0.25, subsurface: 0.0, ..lib(name) }
        }
        // The satchel and its strap: russet leather, the one warm accent, on
        // the side that tells the hero's left from its right.
        "leather" => flat("leather", [0.30, 0.105, 0.045], 0.65),
        // Brass buckle and the lens's ring: warm metal, rough enough that the
        // low sun leaves a smear and not a star.
        "brass" => Pbr { roughness: 0.28, ..lib("brass") },

        // ---- the kit --------------------------------------------------------
        // The mirror. A real metal, so the base colour is F0 rather than an
        // albedo — and **rougher than the library's fresh silver**, which is
        // 0.02 and is what a mirror is on the day it is silvered.
        //
        // The satin is for the picture and it is honest: a hand mirror carried
        // in a satchel on a beach is satin, not a laser flat. It is also the
        // difference between a patch and a speckle. A 0.02 lobe is a
        // hundredth of a degree wide, so the only camera path that ever
        // reaches the sun through it is one that hits the sun's own disc
        // exactly, and the door lights up in single pixels; at 0.08 the lobe
        // is a few degrees, next-event estimation at the mirror connects to
        // the sun on nearly every bounce that gets there, and what lands is a
        // soft-edged patch half a metre across.
        "silver" => Pbr { roughness: 0.05, ..lib("silver") },
        // The glass the lens is made of: N-BK7 with its Sellmeier pair, which
        // is what makes it throw a caustic that disperses — and what makes
        // `caustics::is_caustic_refractor` find it and spend the photon
        // budget on it and nothing else.
        "glass" => glass(),
        // The prism's is not the same glass. See `kit::PRISM_GLASS`.
        "flint" => flint(),
        _ => flat("clay", [0.55, 0.55, 0.55], 0.8),
    }
}

/// N-BK7: the being's glass, without the being's metre of iron in it.
///
/// `transmission: 1.0` and not thin-walled is exactly the pair
/// [`kosm_render::caustics::is_caustic_refractor`] looks for, so declaring
/// this *is* declaring what the photon pass is aimed at. The attenuation is
/// dropped because the thickest thing here is a hundred and fifty millimetres
/// of prism, where a metre-scale Beer-Lambert tint is invisible.
pub fn glass() -> Pbr {
    Pbr {
        base_color: [1.0, 1.0, 1.0],
        roughness: 0.0,
        transmission: 1.0,
        ior: N_D as f32,
        specular: 1.0,
        sellmeier: Some(kosm_render::spectrum::BK7_SELLMEIER),
        ..Default::default()
    }
}

/// Lead crystal: the prism's glass, and **not** the lens's.
///
/// The library's `lead crystal` is `n_d = 1.60` at an Abbe number of 33,
/// against N-BK7's 1.5168 at 64. The index barely matters; the Abbe number is
/// everything, because it *is* the dispersion — a flint splits the visible
/// band 2.4 times as wide as a crown does, and 2.4× is the difference between
/// a spectrum that needs eleven metres of beach to separate and one that
/// needs four. See [`super::kit::PRISM_GLASS`] for the arithmetic and
/// [`super::tools`] for the measurement.
///
/// No Sellmeier pair, and that is not a shortcut: the renderer reconstructs a
/// one-term Cauchy from `ior` and `abbe`
/// ([`kosm_render::spectrum::cauchy_index`]) whenever a glass does not carry
/// coefficients, and the library has an Abbe number for lead crystal and no
/// coefficients. One degree of freedom, honestly used.
pub fn flint() -> Pbr {
    let lead = kosm::material::named("lead crystal").expect("lead crystal is in the library");
    Pbr { base_color: [1.0, 1.0, 1.0], roughness: 0.0, specular: 1.0, ..lead.pbr() }
}

// ---- assembling a picture --------------------------------------------------

/// What the tracer traces here: vcad's analytic faces, or triangles when a
/// boolean took the BRep away. The cove needs a second geometry for its sea;
/// the doorstep has no sea, so one is enough.
pub type Geom = BrepGeom;
/// A picture of the doorstep, ready for [`kosm_render::pathtrace::render`].
pub type Picture = pathtrace::Scene<Geom>;

/// One BVH per distinct solid, shared by every instance of it.
///
/// A hero has two boots of one radius and a dozen spheres of another; a
/// turntable draws the same hero three times. Building the acceleration
/// structure once per *solid* rather than once per placement is what makes
/// that free.
#[derive(Default)]
pub struct Cast {
    prims: Prims,
    bvhs: HashMap<usize, Arc<Bvh<Geom>>>,
    objects: Vec<Object<Geom>>,
}

impl Cast {
    /// Put every body of `built` into the picture, placed by `place`.
    ///
    /// `place(body_name)` is the body → world transform, or `None` to leave
    /// that body out — which is how one document serves four stills: the
    /// prism is in `tools.png` and not on the doorstep, the hero is on the
    /// doorstep and not in `tools.png`, and neither needs a second build.
    pub fn add(
        &mut self,
        built: &Built,
        place: &dyn Fn(&str) -> Option<Transform>,
    ) -> anyhow::Result<()> {
        for body in &built.bodies {
            let Some(to_world) = place(&body.name) else { continue };
            let pbr = palette(&body.material);
            for inst in instances::instances(&built.document, body.root, &mut self.prims)? {
                let bvh = self
                    .bvhs
                    .entry(Arc::as_ptr(&inst.solid) as usize)
                    .or_insert_with(|| Arc::new(Bvh::build(geometry_of(&inst.solid))))
                    .clone();
                let local = placement(&inst.to_world);
                self.objects.push(Object::placed(
                    bvh,
                    pbr,
                    Transform::from_matrix(to_world.matrix * local.matrix),
                ));
            }
        }
        Ok(())
    }

    /// Put a mesh into the picture directly, under a material name.
    ///
    /// The kit's lens is two spherical caps a hundred and ten millimetres
    /// across cut out of spheres two and a half *metres* across, and the
    /// prism is three planes; both are exact as triangles with analytic
    /// normals and neither is a boolean a CAD kernel should be asked to do at
    /// that ratio. Same trade `render.rs` makes for the being's capsule.
    pub fn add_mesh(&mut self, mesh: TriMesh, material: &str, to_world: Transform) {
        let bvh = Arc::new(Bvh::build(BrepGeom::Mesh(mesh)));
        self.objects.push(Object::placed(bvh, palette(material), to_world));
    }

    /// How many traceable objects have been placed.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// The picture, under the cove's daylight.
    pub fn picture(self) -> Picture {
        let (env, sun) = daylight();
        Picture {
            objects: self.objects,
            lights: Vec::new(),
            env,
            sun: Some(sun),
            ground: None,
            splats: None,
        }
    }
}

/// The geometry a solid is traced as: its analytic BRep if it has one, its
/// tessellation if a boolean took the BRep away. `sims/rune/render.rs`'s
/// `geometry_of`, which is `vcad-render --photoreal`'s.
fn geometry_of(solid: &Solid) -> Geom {
    if let Some(brep) = solid.as_brep() {
        let brep = Arc::new(brep.clone());
        let faces = brep.topology.faces.iter().map(|(id, _)| id).collect();
        return BrepGeom::BRep { brep, faces };
    }
    let mut mesh = solid.to_mesh(0);
    vcad_kernel::vcad_kernel_tessellate::render_bake_default(&mut mesh);
    let n = mesh.vertices.len() / 3;
    let positions: Vec<Point3> = (0..n)
        .map(|i| {
            Point3::new(
                mesh.vertices[i * 3] as f64,
                mesh.vertices[i * 3 + 1] as f64,
                mesh.vertices[i * 3 + 2] as f64,
            )
        })
        .collect();
    // The kernel's own normals are per *face* — it emits a corner per corner
    // and a facet normal on each — so they are not used even when they are
    // there. See [`smooth_normals`].
    let normals = smooth_normals(&positions, &mesh.indices);
    BrepGeom::Mesh(TriMesh::new(positions, normals, &mesh.indices))
}

/// Vertex normals for a tessellation that arrived without any.
///
/// A boolean hands back a solid with no BRep left, and a solid with no BRep
/// is traced as triangles. Triangles with no normals shade *flat*, and a
/// hood with facets on it is a hood nobody believes — it was the one visible
/// difference between the cowl this figure wears and a ball of cloth.
///
/// So the faces are averaged into their corners. Two refinements make that
/// safe on a CAD tessellation rather than merely convenient:
///
/// - **Welding.** A kernel emits a vertex per corner, so the six triangles
///   round a point share no index at all. They are gathered by position, at a
///   micron, which is far below any feature and far above any round-off.
/// - **Creases.** A face is only allowed into a corner's average if it faces
///   within [`CREASE`] of that corner's own face. So the round of the cowl
///   comes out smooth and the sharp lip where the cut meets it stays sharp,
///   which is exactly the distinction a CAD surface carries and a naive
///   average destroys.
fn smooth_normals(positions: &[Point3], indices: &[u32]) -> Vec<Vec3> {
    let key = |p: &Point3| {
        let q = |v: f64| (v * 1e3).round() as i64;
        (q(p.x), q(p.y), q(p.z))
    };
    let tris: Vec<[usize; 3]> = indices
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .filter(|t| t.iter().all(|&i| i < positions.len()))
        .collect();
    let face: Vec<Vec3> = tris
        .iter()
        .map(|t| {
            let (a, b, c) = (positions[t[0]], positions[t[1]], positions[t[2]]);
            let n = (b - a).cross(c - a);
            if n.norm() > 0.0 { n.normalize() } else { Vec3::new(0.0, 0.0, 1.0) }
        })
        .collect();
    // every face that touches a welded position, and its normal
    let mut at: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (f, t) in tris.iter().enumerate() {
        for &i in t {
            at.entry(key(&positions[i])).or_default().push(f);
        }
    }
    // each corner takes the faces round its own position that agree with the
    // face it belongs to
    let mut out = vec![Vec3::new(0.0, 0.0, 0.0); positions.len()];
    let min_cos = CREASE.to_radians().cos();
    for (f, t) in tris.iter().enumerate() {
        for &i in t {
            let mut sum = Vec3::new(0.0, 0.0, 0.0);
            for &g in at.get(&key(&positions[i])).map(Vec::as_slice).unwrap_or_default() {
                if face[g].dot(face[f]) >= min_cos {
                    sum += face[g];
                }
            }
            out[i] = if sum.norm() > 1e-12 { sum.normalize() } else { face[f] };
        }
    }
    out
}

/// How far apart two faces may point and still be smoothed together.
const CREASE: f64 = 46.0;

/// vcad's placement, as the tracer's: the same 4×4 under two names.
fn placement(t: &VTransform) -> Transform {
    Transform { matrix: t.matrix }
}

// ---- frames ----------------------------------------------------------------

/// A rotation taking +z to `dir`, with world +z kept as up wherever it can
/// be. Used to stand a solid built along its own axis up along the sun.
pub fn align_z(dir: Vec3) -> Transform {
    let f = dir.normalize();
    let up = if f.z.abs() > 0.999 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 0.0, 1.0) };
    let r = up.cross(f).normalize();
    let u = f.cross(r);
    Transform {
        matrix: tang::Mat4::new(
            r.x, u.x, f.x, 0.0, //
            r.y, u.y, f.y, 0.0, //
            r.z, u.z, f.z, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ),
    }
}

/// A yaw about z, then a translation: a thing standing on the sand facing
/// somewhere. Millimetres and radians.
pub fn stand(at: Point3, yaw: f64) -> Transform {
    let (s, c) = yaw.sin_cos();
    Transform {
        matrix: tang::Mat4::new(
            c, -s, 0.0, at.x, //
            s, c, 0.0, at.y, //
            0.0, 0.0, 1.0, at.z, //
            0.0, 0.0, 0.0, 1.0,
        ),
    }
}

// ---- the film ---------------------------------------------------------------

/// The integrator's settings. `max_depth` is the cove's twelve, for the cove's
/// reason: a path that carries a caustic enters glass and leaves it before it
/// has touched anything.
pub fn options(spp: usize, seed: u64) -> PathTraceOptions {
    PathTraceOptions {
        spp: spp.max(1) as u32,
        max_depth: 12,
        show_background: true,
        seed,
        denoise: true,
        ..Default::default()
    }
}

/// The same, for a frame whose subject is a **rare path**.
///
/// Two of the integrator's defaults are exactly wrong for the mirror in
/// `tools.png`, and both took a render apiece to find.
///
/// - `adaptive: true` makes `spp` a ceiling and lets a pixel stop once its
///   own variance says it has settled. A patch of sun bounced off a 250 mm
///   disc reaches a pixel on the shaded door through a diffuse bounce that
///   finds the mirror about one sample in forty — so the pixel looks smooth,
///   dark and converged for the first thirty-nine, stops, and the patch never
///   arrives. Raising the budget from 256 to 1024 changed the render *time*
///   by nothing at all, which is the tell.
/// - `firefly_clamp: Some(12.0)` truncates any indirect estimate past twelve
///   radiance units. A one-in-forty path carries forty times the mean by
///   construction, so the few samples that did find the mirror were shaved to
///   a twelfth of what they were worth. The relative clamp does the same job
///   against the pixel's own running mean, which is what a spike-hunting
///   frame wants: fireflies still die, a genuinely bright pixel does not.
///
/// Everything else is [`options`]'s.
pub fn options_rare(spp: usize, seed: u64) -> PathTraceOptions {
    PathTraceOptions {
        adaptive: false,
        firefly_clamp: None,
        firefly_clamp_relative: Some(24.0),
        // …and two more à-trous passes, because what the rare path leaves
        // behind after all that is not noise in the ordinary sense but a
        // mottle at the scale of a few pixels, which is exactly what a wider
        // edge-aware footprint is for.
        denoise_iters: 7,
        ..options(spp, seed)
    }
}

/// Tonemap a film to an image, at the cove's exposure.
pub fn to_image(film: &Film) -> image::RgbaImage {
    let px = film.to_srgb8(EXPOSURE as f32, false);
    image::RgbaImage::from_raw(film.width, film.height, px).expect("film is width × height × 4")
}

/// A camera looking from `eye` at `target`, z up.
pub fn look(eye: Point3, target: Point3, vfov_deg: f64) -> Camera {
    Camera::look_at(eye, target, Vec3::new(0.0, 0.0, 1.0), vfov_deg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stage is the cove's geometry with its origin moved, and the light
    /// is the cove's light. Both halves of that are load-bearing: a character
    /// sheet lit differently from the level is a character sheet for a
    /// different game.
    #[test]
    fn the_stage_stands_where_the_cove_says_and_is_lit_the_way_it_is() -> anyhow::Result<()> {
        let built = stage(&Params::default())?;
        assert_eq!(built.param("door_w_mm"), Some(DOOR_W));
        assert_eq!(built.param("aperture_z_mm"), Some(APERTURE_Z));
        // the sand falls away from the door toward the sea
        assert!(sand_z(-2000.0) < 0.0 && sand_z(0.0) == 0.0);
        // the sun is behind the doorstep and above it, and its light travels
        // into the face of the door, which faces -y
        let d = sun_dir();
        assert!(d.z > 0.0, "the sun is up");
        assert!(sun_ray().y > 0.0, "and its light runs into the cliff face");
        // one body per named surface, and every one of them resolves
        let names: Vec<&str> = built.bodies.iter().map(|b| b.material.as_str()).collect();
        for n in ["sand", "rock", "stone", "keyhole", "rim"] {
            assert!(names.contains(&n), "the stage lost its {n}");
        }
        // the rim is a light and nothing the caustic pass will aim at
        let rim = palette("rim");
        assert!(rim.emissive[0] > 0.0 && rim.transmission == 0.0);
        // and the glass is exactly what it will aim at — both of them
        let g = palette("glass");
        assert!(g.transmission > 0.0 && !g.thin_walled && g.sellmeier.is_some());
        let f = palette("flint");
        assert!(f.transmission > 0.0 && !f.thin_walled, "the prism has to be a refractor");
        assert!(f.sellmeier.is_none() && f.abbe > 0.0, "the flint disperses off its Abbe number");
        assert!(f.abbe < g.abbe.max(64.17), "the flint has to split wider than the crown");
        Ok(())
    }

    /// The costume is the library's, to the bit.
    ///
    /// The hero's parts are authored with `Body::substance`, so the name a
    /// body carries is a library name and the colour this file paints it has
    /// to be the colour that entry holds — otherwise a figure would answer
    /// one thing when asked what it is made of and look like another. The
    /// six below are the cove's costume; `leather` and `brass` are excluded
    /// on purpose, because those two the level does re-colour.
    #[test]
    fn the_costume_this_file_paints_is_the_costume_the_library_holds() {
        for name in ["cloak", "cream", "skin", "blush", "ink", "boot"] {
            let m = kosm::material::named(name).unwrap_or_else(|| panic!("`{name}` left the library"));
            let p = palette(name);
            assert_eq!(p.base_color, m.pbr().base_color, "`{name}` is two colours");
            assert_eq!(p.roughness, m.roughness as f32, "`{name}` is two roughnesses");
            // flat, whatever the substance does: skin scatters and this look
            // does not
            assert_eq!(p.subsurface, 0.0, "`{name}` brought a subsurface walk with it");
            assert_eq!(p.transmission, 0.0, "`{name}` is not a window");
            assert!(m.density > 0.0, "`{name}` has no density to hand a rig");
        }
        // and the two that are re-coloured really are re-coloured, so that
        // the exclusion above is a statement and not an oversight
        assert_ne!(palette("leather").base_color, kosm::material::named("leather").unwrap().pbr().base_color);
    }
}
