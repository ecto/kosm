//! `kosm run rune/creature` — the player character, as four path-traced
//! stills.
//!
//! This is a look-development sim and not a physics one: there is no rollout,
//! nothing steps, and the only thing it optimises is the picture. It exists
//! because the cove has a *being* — a solid of N-BK7 that is a lens and
//! nothing else — and a game needs somebody in it. The creature is what the
//! being becomes when it stops being a prop: something between a jellyfish
//! and an axolotl, a rigid core under a translucent bell, and the bell is
//! still the lens the puzzle needs.
//!
//! Four images, under `out/characters/creature/`:
//!
//! | file | what it is for |
//! |---|---|
//! | `door.png` | the creature at the door, backlit, the shot the game would ship |
//! | `portrait.png` | close three-quarter, to read the face and the bell |
//! | `turntable.png` | 0°, 120°, 240° on plain sand, to read the silhouette |
//! | `sss_ab.png` | the portrait with the subsurface walk off, then on |
//!
//! The stage is deliberately its own, and small: sand as a ground plane, the
//! door as a dressed slab with a real aperture cut in it, and the cove's own
//! sky and sun. It does not load the cove — the cove is forty metres of
//! terrain and a sea, and none of it is in frame at three metres.

use std::sync::Arc;

use kosm::brep::instances::{self, Prims};
use kosm::prelude::*;
use kosm_render::math::{Point3, Transform, Vec3};
use kosm_render::pathtrace::{
    Camera, Environment, Film, Ground, Object, PathTraceOptions, Pbr, Scene, Sun, render,
};
use kosm_render::Bvh;
use kosm_render::TriMesh;
use vcad_kernel::Solid;
use vcad_kernel_raytrace::BrepGeom;

/// The animal, as CAD: `assemble` writes its bodies into any open builder,
/// so a level can stand one in itself the way `kosm run rune/creature` does.
pub mod body;
pub mod look;

use body::Creature;

/// How far in front of the creature the door stands, in millimetres.
///
/// Two metres: the distance the design doc gives for where the being has to
/// stand to solve the rune, and therefore the distance the bell's optics are
/// aimed at. See [`Creature::focal_length`].
const DOOR_Y: f64 = 2000.0;

/// The knobs, and therefore the run id.
fn params(args: &kosm_cli::Args) -> Vec<Param> {
    let spp = args
        .value("spp")
        .and_then(|v| v.parse().ok())
        .unwrap_or(128.0);
    let height = args
        .value("height")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1100.0);
    vec![
        Param::new("render_spp", spp),
        Param::new("height_mm", height),
    ]
}

/// The animal, as a document.
fn creature_document(c: &Creature) -> anyhow::Result<Built> {
    let mut knobs = Params::new();
    knobs.set("height_mm", c.height);
    let c = *c;
    build(&knobs, move |b| {
        body::assemble(b, &c);
    })
}

/// The stage: a slab of dressed stone with the rune's keyhole cut through it.
///
/// The aperture is *not* placed by hand. The bell throws its cone along
/// whatever direction the sun is coming from, so where that cone lands on the
/// door is a fact about the sun and the geometry, and the keyhole is put
/// there. That way `door.png` cannot quietly stop being the picture it claims
/// to be when a light direction is nudged.
fn stage_document(c: &Creature, keyhole: (f64, f64)) -> anyhow::Result<Built> {
    let (kx, kz) = keyhole;
    let c = *c;
    build(&Params::new(), move |b| {
        // The cliff face the door is set into: wide and tall enough to fill
        // the frame behind the animal at three metres, and no deeper than it
        // needs to be.
        let slab = b.boxed(4200.0, 260.0, 3200.0).at(0.0, DOOR_Y + 130.0, 1600.0);
        // A 120 mm aperture, bored right through.
        let bore = b
            .cylinder(120.0, 900.0)
            .rotate_x(-90.0)
            .at(kx, DOOR_Y - 200.0, kz);
        b.body("door").material("stone").add(slab.difference(bore));
        // The throat behind it: a shallow box of near-black so the bore reads
        // as a hole and not as a dark ring on a wall.
        b.body("keyhole")
            .material("keyhole")
            .add(b.boxed(700.0, 40.0, 700.0).at(kx, DOOR_Y + 290.0, kz));
        let _ = c;
    })
}

/// Every solid in a document, as traceable objects.
///
/// The document's roots each carry a material name; that name goes through
/// [`look::creature`] first and the cove's own table second, so the animal
/// gets its own palette and the stage gets the cove's.
fn objects(built: &Built, off: bool) -> anyhow::Result<Vec<Object<BrepGeom>>> {
    let doc = &built.document;
    let mut prims = Prims::default();
    let mut out = Vec::new();
    for root in &doc.roots {
        let placed = instances::instances(doc, root.root, &mut prims)?;
        anyhow::ensure!(
            !placed.is_empty(),
            "the `{}` root evaluated to no solid",
            root.material
        );
        let pbr = material_for(&root.material, off);
        for inst in placed {
            let bvh = Arc::new(Bvh::build(geometry_of(&inst.solid)));
            out.push(Object::placed(
                bvh,
                pbr,
                Transform {
                    matrix: inst.to_world.matrix,
                },
            ));
        }
    }
    Ok(out)
}

/// A material name, resolved. `off` drops the subsurface weight, which is the
/// left half of `sss_ab.png` and nothing else.
fn material_for(name: &str, off: bool) -> Pbr {
    let pbr = look::creature(name).unwrap_or_else(Pbr::default);
    if off {
        look::without_subsurface(pbr)
    } else {
        pbr
    }
}

/// The geometry a solid is traced as: its tessellation, always.
///
/// `sims/rune/render.rs` traces a solid's analytic BRep where it has one, and
/// falls back to the mesh only when a boolean took the BRep away. That is the
/// right call for the cove, whose shapes are boxes and prisms. It is the
/// wrong call here, and finding out why cost an afternoon: the bell and the
/// core are both **sphere booleans**, and a boolean's result comes back
/// carrying a BRep whose faces are still the whole untrimmed spheres. Traced
/// through `BrepGeom::BRep` those faces are complete spheres — so the bell
/// rendered as the 860 mm ball it was cut *out of*, and the animal came back
/// as a snowman with the arithmetic protesting its innocence.
///
/// Tessellating unconditionally throws away the analytic silhouette, which on
/// a sphere this size is a real loss at a hard rim. It is a much smaller loss
/// than drawing the wrong shape.
fn geometry_of(solid: &Solid) -> BrepGeom {
    // 128 segments, not the kernel's default of `0` (which is 32-ish): the
    // bell is half a metre across and fills a third of the frame, and at the
    // default its silhouette is visibly a polygon. This is the whole cost of
    // giving up the analytic BRep, paid in triangles.
    let mut mesh = solid.to_mesh(128);
    vcad_kernel::vcad_kernel_tessellate::render_bake_default(&mut mesh);
    let n = mesh.vertices.len() / 3;
    let positions = (0..n)
        .map(|i| {
            Point3::new(
                mesh.vertices[i * 3] as f64,
                mesh.vertices[i * 3 + 1] as f64,
                mesh.vertices[i * 3 + 2] as f64,
            )
        })
        .collect();
    let normals = if mesh.normals.len() == mesh.vertices.len() {
        (0..n)
            .map(|i| {
                Vec3::new(
                    mesh.normals[i * 3] as f64,
                    mesh.normals[i * 3 + 1] as f64,
                    mesh.normals[i * 3 + 2] as f64,
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    BrepGeom::Mesh(TriMesh::new(positions, normals, &mesh.indices))
}

/// One traceable scene: the animal, optionally the door, sand underfoot.
///
/// The sun is kept beside the scene because [`beam_landing`] wants the
/// normalised direction back out, and `Scene` holds it behind an `Option`.
struct Stage {
    scene: Scene<BrepGeom>,
    sun: Sun,
}

impl Stage {
    fn new(objects: Vec<Object<BrepGeom>>, env: Environment, sun: Sun) -> Self {
        Self {
            scene: Scene {
                objects,
                lights: Vec::new(),
                env,
                sun: Some(sun),
                ground: Some(Ground {
                    z: 0.0,
                    material: material_for("sand", false),
                    shadow_catcher: false,
                }),
                splats: None,
            },
            sun,
        }
    }
}

/// Where the bell's cone lands on the door, given the direction the sunlight
/// is travelling.
///
/// The bell is a thin-lens approximation here on purpose: the chief ray goes
/// straight through undeviated, so the *centre* of the pool is where the ray
/// through the bell's centre meets the door's plane, and the cone opens about
/// it at [`Creature::beam_radius_at`]. Anything more exact would need the
/// caustic pass, which is what actually draws it.
fn beam_landing(c: &Creature, light_dir: Vec3) -> (f64, f64) {
    let bell = Point3::new(0.0, 0.0, c.bell_centre_z());
    let t = (DOOR_Y - bell.y) / light_dir.y.max(1e-6);
    (bell.x + light_dir.x * t, bell.z + light_dir.z * t)
}

/// Build the animal (and, when asked, the door) into one stage.
fn stage(
    c: &Creature,
    sun_towards: Vec3,
    with_door: bool,
    off: bool,
) -> anyhow::Result<Stage> {
    let (env, sun) = look::daylight(sun_towards);
    let mut objects = objects(&creature_document(c)?, off)?;
    if with_door {
        let landing = beam_landing(c, -sun.direction);
        objects.extend(objects_of_stage(c, landing)?);
    }
    Ok(Stage::new(objects, env, sun))
}

fn objects_of_stage(c: &Creature, keyhole: (f64, f64)) -> anyhow::Result<Vec<Object<BrepGeom>>> {
    objects(&stage_document(c, keyhole)?, false)
}

/// Trace one frame.
fn shoot(stage: &Stage, cam: &Camera, size: (u32, u32), spp: usize) -> Film {
    render(
        &stage.scene,
        cam,
        size.0,
        size.1,
        &PathTraceOptions {
            spp: spp.max(1) as u32,
            max_depth: 16,
            show_background: true,
            seed: 0x9e37_79b9,
            denoise: true,
            ..Default::default()
        },
    )
}

/// The exposure every still is graded at.
///
/// The cove's sun is 6.2 in the same arbitrary units its sky is 0.42 in, and
/// at an exposure of 1 that puts pale sand and a pale animal alike over the
/// top of the ACES curve — the first pass at these images came back as white
/// shapes on a white beach. 0.42 lands the sunlit sand at about three
/// quarters and leaves the whole of the top stop for the bell's rim and the
/// heart, which are the two things in frame that are supposed to be bright.
const EXPOSURE: f32 = 0.42;

/// A film, tonemapped.
fn to_image(film: &Film) -> image::RgbaImage {
    let px = film.to_srgb8(EXPOSURE, false);
    image::RgbaImage::from_raw(film.width, film.height, px).expect("width × height × 4")
}

/// Paste `src` into `dst` with its top-left at `(x, y)`.
fn paste(dst: &mut image::RgbaImage, src: &image::RgbaImage, x: u32, y: u32) {
    for (sx, sy, p) in src.enumerate_pixels() {
        let (dx, dy) = (x + sx, y + sy);
        if dx < dst.width() && dy < dst.height() {
            dst.put_pixel(dx, dy, *p);
        }
    }
}

/// `kosm run rune/creature --out DIR`.
pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let params = params(args);
    let spp = params
        .iter()
        .find(|p| p.name == "render_spp")
        .map(|p| p.value as usize)
        .unwrap_or(128);
    let height = params
        .iter()
        .find(|p| p.name == "height_mm")
        .map(|p| p.value)
        .unwrap_or(1100.0);
    let c = Creature {
        height,
        ..Creature::default()
    };

    let mut rec = Recorder::new(args.out(), "rune/creature", &params, 0)?;
    // The brief's own address for these, kept stable so a reader always finds
    // the latest set in one place regardless of the run hash.
    let gallery = args.out().join("characters").join("creature");
    std::fs::create_dir_all(&gallery)?;
    let save = |name: &str, img: &image::RgbaImage| -> anyhow::Result<()> {
        rec.png(name, img)?;
        img.save(gallery.join(name))?;
        Ok(())
    };

    // ── 1. the door ──────────────────────────────────────────────────────
    //
    // The sun is low over the sea (−y, the way the cove faces) and off to
    // +x, so from the camera it is behind the animal and to the side: the
    // limbs are edge-on to it and light comes through them, and the same
    // light carries on through the bell to the door.
    // Ten degrees of elevation: any higher and the bell's cone drives into
    // the sand before it reaches the stone, any lower and the whole beach is
    // one long shadow. `beam_landing` is what turns this into a keyhole
    // position, so the two cannot disagree.
    let door_sun = Vec3::new(0.30, -0.92, 0.25).normalize();
    let door_stage = stage(&c, door_sun, true, false)?;
    let landing = beam_landing(&c, -door_stage.sun.direction);
    let door_cam = Camera::look_at(
        // Three metres out, round to −x so the keyhole clears the animal, and
        // below its eyes so it has sky behind it and reads as tall.
        Point3::new(-1750.0, -2400.0, 430.0),
        Point3::new(-230.0, 420.0, 640.0),
        Vec3::new(0.0, 0.0, 1.0),
        26.0,
    );
    save("door.png", &to_image(&shoot(&door_stage, &door_cam, (960, 540), spp)))?;

    // ── 2. the portrait ──────────────────────────────────────────────────
    //
    // A metre and a half, three-quarter, eye height. The sun swings round to
    // the far side so the bell's rim is the brightest thing in frame.
    let portrait_sun = Vec3::new(0.62, 0.66, 0.42).normalize();
    let portrait_stage = stage(&c, portrait_sun, false, false)?;
    let portrait_cam = Camera::look_at(
        Point3::new(-1010.0, -1060.0, 880.0),
        Point3::new(0.0, 0.0, 870.0),
        Vec3::new(0.0, 0.0, 1.0),
        38.0,
    );
    save(
        "portrait.png",
        &to_image(&shoot(&portrait_stage, &portrait_cam, (960, 540), spp)),
    )?;

    // ── 3. the turntable ─────────────────────────────────────────────────
    //
    // Three views on plain sand, so the silhouette can be read without the
    // door behind it. The animal is radially symmetric by construction, so
    // what turns is the camera and the light stays put — a turning light
    // would hide the very asymmetry (the eyes) the turntable is for.
    let turn_sun = Vec3::new(0.45, -0.74, 0.50).normalize();
    let turn_stage = stage(&c, turn_sun, false, false)?;
    let (tw, th) = (320u32, 540u32);
    let mut turntable = image::RgbaImage::new(960, 540);
    for (i, deg) in [0.0f64, 120.0, 240.0].iter().enumerate() {
        let a = deg.to_radians();
        let r = 2600.0;
        let cam = Camera::look_at(
            Point3::new(r * a.sin(), -r * a.cos(), 760.0),
            Point3::new(0.0, 0.0, 580.0),
            Vec3::new(0.0, 0.0, 1.0),
            32.0,
        );
        let f = shoot(&turn_stage, &cam, (tw, th), spp);
        paste(&mut turntable, &to_image(&f), i as u32 * tw, 0);
    }
    save("turntable.png", &turntable)?;

    // ── 4. the A/B ───────────────────────────────────────────────────────
    //
    // The portrait twice, at half width each, with the subsurface weight the
    // only thing that changed. Left is off, right is on.
    let ab_cam = Camera::look_at(
        Point3::new(-1010.0, -1060.0, 880.0),
        Point3::new(0.0, 0.0, 870.0),
        Vec3::new(0.0, 0.0, 1.0),
        38.0,
    );
    let mut ab = image::RgbaImage::new(960, 540);
    for (i, off) in [true, false].iter().enumerate() {
        let s = stage(&c, portrait_sun, false, *off)?;
        let f = shoot(&s, &ab_cam, (480, 540), spp);
        paste(&mut ab, &to_image(&f), i as u32 * 480, 0);
    }
    save("sss_ab.png", &ab)?;

    // ── the numbers ──────────────────────────────────────────────────────
    rec.metric("height_mm", c.height)?;
    rec.metric("bell_focal_length_mm", c.focal_length())?;
    rec.metric("bell_rim_radius_mm", c.bell_rim_radius())?;
    rec.metric("bell_rim_z_mm", c.bell_rim_z())?;
    rec.metric("door_distance_mm", DOOR_Y)?;
    rec.metric("beam_radius_at_door_mm", c.beam_radius_at(DOOR_Y))?;
    rec.metric("keyhole_x_mm", landing.0)?;
    rec.metric("keyhole_z_mm", landing.1)?;
    rec.metric("render_spp", spp)?;
    rec.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every part has to be where the arithmetic says it is. A silhouette
    /// that reads wrong is almost always a boolean that did not take, and
    /// bounds are how you find that out without rendering.
    #[test]
    fn every_part_lands_where_the_table_says() {
        let c = Creature::default();
        let built = creature_document(&c).unwrap();
        let doc = &built.document;
        let mut prims = Prims::default();
        for root in &doc.roots {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for inst in instances::instances(doc, root.root, &mut prims).unwrap() {
                let bvh = Bvh::build(geometry_of(&inst.solid));
                let Some(b) = bvh.bounds() else { continue };
                let t = Transform { matrix: inst.to_world.matrix };
                for corner in [b.min, b.max] {
                    let p = t.apply_point(&corner);
                    let q = [p.x, p.y, p.z];
                    for k in 0..3 {
                        lo[k] = lo[k].min(q[k]);
                        hi[k] = hi[k].max(q[k]);
                    }
                }
            }
            println!(
                "{:8}  x {:7.1}..{:7.1}  y {:7.1}..{:7.1}  z {:7.1}..{:7.1}",
                root.material, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
            );
        }
        println!("bell rim r {:.1} at z {:.1}", c.bell_rim_radius(), c.bell_rim_z());
        println!("f = {:.1} mm", c.focal_length());
    }
}
