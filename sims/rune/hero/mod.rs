//! `kosm run rune/hero` — the adventurer, as four path-traced stills.
//!
//! A character sheet for Rune's player, made the only way an agent can make
//! one: build it, light it with the level's own sun, trace it, read the png,
//! change a number, trace it again.
//!
//! ```text
//! kosm run rune/hero                       all four, 128 spp
//! kosm run rune/hero --only door --spp 16  one of them, fast
//! kosm run rune/hero --photons 200000      a cheaper caustic pass
//! HERO_SCAN=1 kosm run rune/hero --only tools --spp 4
//! ```
//!
//! That last one is the debugging loop this file was built with: it prints
//! the caustic map's irradiance on a grid of the sand, brightest first, so an
//! agent can tell a caustic that is *missing* from one that is merely out of
//! frame — the two failures look identical in a png and are nothing alike.
//!
//! Four files under `out/characters/hero/`:
//!
//! | file            | what it is |
//! |-----------------|------------|
//! | `door.png`      | the hero at the doorstep, holding the lens up to the sun, its caustic on the door beside the keyhole |
//! | `portrait.png`  | a close three-quarter, for the costume: hood, collar, satchel |
//! | `tools.png`     | the kit on the sand — the lens's focus, the prism's spectrum, the mirror's sun patch, all real light |
//! | `turntable.png` | 0°, 120°, 240°, three panels in one frame |
//!
//! # Where the hero may stand, and why it is not two and a half metres
//!
//! An ideal lens focuses a parallel bundle at the point where the *chief ray*
//! — the one through its own centre, which is undeviated — crosses the focal
//! plane. Tilting the lens moves that point along the chief ray and nowhere
//! else. So the sun's image lands on the line that leaves the lens's centre
//! along the sunbeam, whatever the lens is doing, and staging this shot is
//! one equation:
//!
//! ```text
//! z_lens + (−d_z/d_y)·y_lens = APERTURE_Z          d = the sunbeam, 22° down
//! ```
//!
//! with the hero's feet on the sand (`z = SLOPE·y`) and the lens as high as
//! its arms reach. Solve it ([`doorstep`]) at the cove's sun and a 1.1 m
//! adventurer's reach — shoulder 600 mm, arm 420 mm — and the answer is
//! **1.30 m from the door**, not 2.5 m. It cannot be 2.5 m: at 2.5 m the
//! lens would have to be held 1.35 m above the sill, which is 250 mm over the
//! crown of a hero 1.1 m tall. Two and a half metres of doorstep needs either
//! a sun below about 13° or a taller adventurer, and the cove's sun is 22°.
//! The number the brief asked for is in the *lens* instead, where it belongs:
//! f = 2.5 m, so at a 1.32 m throw the cone has closed to a 104 mm ellipse —
//! bright, still converging, and legible beside a 240 mm keyhole.

use std::fs;
use std::path::Path;
use std::time::Instant;

use kosm::build::{Built, Params};
use kosm_render::caustics::{self, CausticMap, CausticOptions};
use kosm_render::math::{Point3, Transform, Vec3};
use kosm_render::pathtrace;

pub mod figure;
pub mod kit;
pub mod stage;

use figure::Rig;
use stage::{Cast, Picture};

/// The still's seed. Fixed, so two runs differ only where the hero does.
const SEED: u64 = 0x48_45_52_4f; // "HERO"

/// Every frame is this size: a landscape plate an agent can read at a glance.
const SIZE: (u32, u32) = (960, 540);

// ---- the pose --------------------------------------------------------------

/// A hero standing somewhere on the sand, facing somewhere.
pub struct Stance {
    /// The knobs the figure is built from — the pose lives in these.
    pub params: Params,
    /// The rig those knobs describe.
    pub rig: Rig,
    /// Where the boots are, on the sand.
    pub feet: Point3,
    /// Which way it faces: a turn about z, radians, from facing +y.
    pub yaw: f64,
}

impl Stance {
    /// The hero's own frame, as a placement.
    pub fn to_world(&self) -> Transform {
        stage::stand(self.feet, self.yaw)
    }

    /// A point of the hero's, in the world.
    pub fn at(&self, local: [f64; 3]) -> Point3 {
        let (s, c) = self.yaw.sin_cos();
        Point3::new(
            self.feet.x + local[0] * c - local[1] * s,
            self.feet.y + local[0] * s + local[1] * c,
            self.feet.z + local[2],
        )
    }
}

/// The knobs for a pose: two hand targets, a head tilt, and whether the
/// satchel is worn.
fn pose(right: [f64; 3], left: [f64; 3], head_tilt: f64) -> Params {
    let mut p = Params::new();
    p.set("hand_r_x_mm", right[0]);
    p.set("hand_r_y_mm", right[1]);
    p.set("hand_r_z_mm", right[2]);
    p.set("hand_l_x_mm", left[0]);
    p.set("hand_l_y_mm", left[1]);
    p.set("hand_l_z_mm", left[2]);
    p.set("head_tilt_deg", head_tilt);
    p
}

/// How high the hero holds the lens: the arm's elevation over the shoulder,
/// and how much of its straight reach it uses.
///
/// Sixty-six degrees is very nearly the elevation that maximises how far back
/// the hero can stand and still put the sun on the keyhole — the standing
/// distance is `(shoulder + L sinφ + k L cosφ − keyhole)/(slope + k)` and
/// that peaks where `tanφ = 1/k` — and a hair off straight, because a
/// straight arm is a mannequin's arm.
const ARM_ELEVATION_DEG: f64 = 58.0;
const ARM_EXTENSION: f64 = 0.97;

/// Where the hero must stand for the lens it is holding to put the sun in the
/// keyhole, and which way it must face.
///
/// Two constraints and two unknowns. The lens has to be on the sunbeam
/// through the keyhole — that is the `x` and `z` equations below — and the
/// hero has to be facing the keyhole, which is what decides where its hands
/// are relative to its feet. Facing depends on position and position on
/// facing, so it is iterated; five turns is far past convergence for a yaw
/// that only ever moves twenty degrees.
/// Where on the door's face the sun is put.
///
/// Not the keyhole itself, and that is a *rendering* fact rather than a
/// staging preference: the keyhole is a dark recess with a self-lit ring
/// round it, and a caustic dropped into it lands on an albedo of 0.03 under
/// something already glowing. A hand's breadth to the side and a little above
/// it, the same light lands on plain sunlit stone and reads as what it is —
/// and it reads as the puzzle, too, because the whole game is the difference
/// between *near* the keyhole and *in* it.
///
/// The offset is in `x` and not in `z` on purpose. Height is expensive: the
/// standing distance is `(reach − aim_z)/(slope + k)`, so every millimetre the
/// aim rises costs two millimetres of doorstep, and sideways costs nothing at
/// all.
const AIM: [f64; 3] = [340.0, 0.0, 400.0];

pub fn doorstep() -> Stance {
    let rig = Rig::DEFAULT;
    let d = stage::sun_ray();
    let (kx, kz) = (d.x / d.y, -d.z / d.y);
    let phi = ARM_ELEVATION_DEG.to_radians();
    // How far out the arms are. The cant below turns the grip in the hero's
    // own frame, which pushes one hand further from its shoulder than the
    // other, so the extension is backed off until the *far* hand is inside
    // the arm rather than assuming the near one settles it.
    let mut extension = ARM_EXTENSION;
    let mut hands = ([0.0; 3], [0.0; 3]);
    // A first pass with the arms straight out at `phi`, to get a yaw to turn
    // the grip into; the loop below then re-solves the stance from the hands
    // the grip actually asks for.
    let mut ly = ARM_EXTENSION * rig.reach() * phi.cos();
    let mut lz = rig.shoulder_z + ARM_EXTENSION * rig.reach() * phi.sin();

    let (mut yaw, mut feet) = (0.0f64, Point3::origin());
    for _ in 0..5 {
        let (s, c) = yaw.sin_cos();
        // z: the beam that leaves the lens has to arrive at the keyhole's height
        let fy = (AIM[2] - lz - kz * ly * c) / (stage::SLOPE + kz);
        let y_lens = fy + ly * c;
        // x: and at the keyhole's x, which the sun's azimuth walks it toward
        let fx = AIM[0] + kx * y_lens + ly * s;
        feet = Point3::new(fx, fy, stage::sand_z(fy));
        yaw = (fx - AIM[0]).atan2(AIM[1] - fy);
    }

    // The lens is held *canted* off the sunbeam, and that is a free choice:
    // a thin lens images a parallel bundle where the undeviated chief ray
    // through its own centre crosses the focal plane, so turning it moves the
    // focus **along the sunbeam** and never off it. The caustic lands on the
    // same spot at any cant; all that changes is that the throw is measured
    // against `f/cos θ` instead of `f`, so the patch is a little wider and a
    // little softer.
    //
    // What the cant buys is the picture. Square to the sun, the lens is
    // square to the one direction the camera cannot be — the camera has to be
    // off the beam or it is photographing its own shadow — and it shows as a
    // gold line. Canted toward the camera it shows as a disc.
    let axis = cant(stage::sun_dir(), LENS_CANT_DEG.to_radians());
    // Both hands on the rim, so the glass is held and not floated: the rim's
    // horizontal diameter, in the hero's own frame.
    // The two points of the rim nearest the hero's own left and right: its
    // x-axis, dropped into the plane of the glass. Any other pair on that
    // circle asks one arm for more than the other, and the arm it asks is
    // 420 mm long.
    let (s, c) = yaw.sin_cos();
    let hero_x = Vec3::new(c, s, 0.0);
    let u = (hero_x - axis * axis.dot(hero_x)).normalize();
    // …in the hero's own frame, where the pose knobs live
    let grip = 0.5 * kit::LENS_D * Vec3::new(u.x * c + u.y * s, -u.x * s + u.y * c, u.z);
    for _ in 0..24 {
        let out = extension * rig.reach();
        ly = out * phi.cos();
        lz = rig.shoulder_z + out * phi.sin();
        hands = (
            [grip.x, ly + grip.y, lz + grip.z],
            [-grip.x, ly - grip.y, lz - grip.z],
        );
        let far = [hands.0, hands.1]
            .iter()
            .zip([1.0, -1.0])
            .map(|(h, side)| {
                let s = rig.shoulder(side);
                ((h[0] - s[0]).powi(2) + (h[1] - s[1]).powi(2) + (h[2] - s[2]).powi(2)).sqrt()
            })
            .fold(0.0f64, f64::max);
        if far <= 0.97 * rig.reach() {
            break;
        }
        extension -= 0.01;
    }
    // The hands moved; the stance moves with them.
    for _ in 0..5 {
        let (s, c) = yaw.sin_cos();
        let fy = (AIM[2] - lz - kz * ly * c) / (stage::SLOPE + kz);
        let y_lens = fy + ly * c;
        let fx = AIM[0] + kx * y_lens + ly * s;
        feet = Point3::new(fx, fy, stage::sand_z(fy));
        yaw = (fx - AIM[0]).atan2(AIM[1] - fy);
    }

    // Looking up at it. Not the whole way — a Mii's head on a 63° neck is a
    // Mii falling over backwards — but far enough that the gaze clears the
    // brow and lands on the glass.
    let look = (lz - rig.shoulder_z).atan2(ly).to_degrees();
    let params = pose(hands.0, hands.1, (0.55 * look).min(32.0));
    Stance { rig: Rig::of(&params), params, feet, yaw }
}

/// Where the lens is, given the stance that is holding it: between the hands.
/// The sun's direction turned `by` about the vertical, toward the camera.
///
/// Positive is counter-clockwise seen from above, which at the cove's sun
/// azimuth of 250° is the way the doorstep camera lies.
pub fn cant(dir: Vec3, by: f64) -> Vec3 {
    let (s, c) = by.sin_cos();
    Vec3::new(dir.x * c - dir.y * s, dir.x * s + dir.y * c, dir.z).normalize()
}

/// How far off the sunbeam the lens is held. See [`doorstep`].
const LENS_CANT_DEG: f64 = 42.0;

pub fn held_lens(stance: &Stance) -> Point3 {
    let (r, l) = (stance.rig.hand_right, stance.rig.hand_left);
    stance.at([0.5 * (r[0] + l[0]), 0.5 * (r[1] + l[1]), 0.5 * (r[2] + l[2])])
}

// ---- putting a picture together --------------------------------------------

/// Everything the four stills share: the stage, the hero and the hardware,
/// built once each.
struct Studio {
    stage: Built,
    hardware: Built,
}

impl Studio {
    fn new() -> anyhow::Result<Self> {
        Ok(Self { stage: stage::stage(&Params::default())?, hardware: kit::hardware(&Params::default())? })
    }

    /// The stage, whole.
    fn set(&self, cast: &mut Cast) -> anyhow::Result<()> {
        cast.add(&self.stage, &|_| Some(Transform::identity()))
    }

    /// A hero at a stance.
    fn hero(&self, cast: &mut Cast, stance: &Stance) -> anyhow::Result<()> {
        let built = figure::figure(&stance.params)?;
        let to_world = stance.to_world();
        cast.add(&built, &|_| Some(to_world.clone()))
    }

    /// The lens: glass and ring, its axis along `axis`, centred at `at`.
    fn lens(&self, cast: &mut Cast, at: Point3, axis: Vec3) -> anyhow::Result<()> {
        let frame = Transform::from_matrix(
            tang::Mat4::translation(at.x, at.y, at.z) * stage::align_z(axis).matrix,
        );
        cast.add_mesh(kit::lens_mesh(128, 10), "glass", frame.clone());
        cast.add(&self.hardware, &|name| (name == "lens_ring").then(|| frame.clone()))
    }

    /// The prism, placed by the beam it bends, with the stone stop that makes
    /// its spectrum a spectrum standing in front of it.
    fn prism(&self, cast: &mut Cast, frame: Transform) -> anyhow::Result<()> {
        cast.add_mesh(kit::prism_mesh(), "glass", frame.clone());
        if std::env::var("HERO_NOSTOP").is_ok() {
            return Ok(());
        }
        cast.add(&self.hardware, &|name| (name == "stop").then(|| frame.clone()))
    }

    /// The mirror, placed by the beam it turns.
    fn mirror(&self, cast: &mut Cast, frame: Transform) -> anyhow::Result<()> {
        cast.add(&self.hardware, &|name| {
            name.starts_with("mirror").then(|| frame.clone())
        })
    }
}

/// Trace the caustic pass over a picture: the lens's focus and the prism's
/// spectrum, and nothing else, because they are the only transmissive solids
/// in it.
fn rune_light(picture: &Picture, photons: usize, radius: f64) -> CausticMap {
    if photons == 0 {
        return CausticMap::empty();
    }
    caustics::trace(
        picture,
        &CausticOptions { photons, radius: Some(radius), seed: SEED, ..Default::default() },
    )
}

/// Render one frame and say what it cost.
fn shoot(
    picture: &Picture,
    camera: &pathtrace::Camera,
    size: (u32, u32),
    spp: usize,
    map: Option<&CausticMap>,
) -> image::RgbaImage {
    let film = pathtrace::render_with_caustics(picture, camera, size.0, size.1, &stage::options(spp, SEED), map);
    stage::to_image(&film)
}

// ---- the four stills --------------------------------------------------------

/// Where `door.png` is shot from: an azimuth about the hero measured the
/// world's way, from +x toward +y, a distance and a height.
const CAMERA_AZIMUTH_DEG: f64 = 332.0;
const CAMERA_BACK: f64 = 3450.0;
const CAMERA_UP: f64 = 950.0;
const CAMERA_VFOV_DEG: f64 = 30.0;

/// `door.png`: the hero at the doorstep with the lens up in both hands, and
/// the sun coming through it onto the stone beside the keyhole.
fn door(studio: &Studio, dir: &Path, spp: usize, photons: usize) -> anyhow::Result<()> {
    let stance = doorstep();
    let lens = held_lens(&stance);
    let mut cast = Cast::default();
    studio.set(&mut cast)?;
    studio.hero(&mut cast, &stance)?;
    studio.lens(&mut cast, lens, cant(stage::sun_dir(), LENS_CANT_DEG.to_radians()))?;
    let picture = cast.picture();

    let t0 = Instant::now();
    let map = rune_light(&picture, photons, 15.0);
    let traced = t0.elapsed().as_secs_f64();

    // Behind the hero and a little to its right, low, looking past it at the
    // door. The sun is directly behind the hero — the puzzle put it there —
    // so this is the one arc of camera positions where three things are true
    // at once: the door's face is square enough to us to show the caustic,
    // the lens is turned enough toward us to read as a disc rather than an
    // edge, and the hero is lit rather than a silhouette.
    let orbit = CAMERA_AZIMUTH_DEG.to_radians();
    let eye = Point3::new(
        stance.feet.x + CAMERA_BACK * orbit.cos(),
        stance.feet.y + CAMERA_BACK * orbit.sin(),
        stance.feet.z + CAMERA_UP,
    );
    // Aimed at the middle of the story: half way between the glass in the
    // hero's hands and the spot of sun it is putting on the door.
    // Between the glass and the mark it is putting on the stone, and lower
    // than either: the hero's boots are the bottom of the frame and a
    // character sheet that crops its own feet is not one.
    let mix = |a: f64, b: f64| 0.5 * a + 0.5 * b;
    let target = Point3::new(mix(lens.x, AIM[0]), mix(lens.y, AIM[1]), 0.62 * AIM[2] + 0.2 * lens.z);
    let camera = stage::look(eye, target, CAMERA_VFOV_DEG);

    let t1 = Instant::now();
    let img = shoot(&picture, &camera, SIZE, spp, Some(&map));
    let path = dir.join("door.png");
    img.save(&path)?;
    let throw = (lens - stage::keyhole()).norm();
    println!(
        "hero door: standing {:.2} m off the face at ({:+.0}, {:+.0}) mm, facing {:+.1}°, lens {:.0} mm up and {:.0} mm from the keyhole → a {:.0} mm patch; {} photons in {:.1} s, {}×{} at {spp} spp in {:.1} s → {}",
        -stance.feet.y / 1000.0,
        stance.feet.x,
        stance.feet.y,
        stance.yaw.to_degrees(),
        lens.z - stage::sand_z(lens.y),
        throw,
        kit::patch_at(throw),
        map.len(),
        traced,
        SIZE.0,
        SIZE.1,
        t1.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(())
}

/// `portrait.png`: a close three-quarter, lit from the side, for the costume.
///
/// A different stance from the doorstep on purpose. At the door the sun is
/// directly behind the hero — that is what the puzzle demands — and a
/// backlit costume is a silhouette. So the hero is turned a quarter out of
/// the beam here, which puts the key on its hood and its satchel and leaves
/// the sky to fill the rest.
fn portrait(studio: &Studio, dir: &Path, spp: usize, photons: usize) -> anyhow::Result<()> {
    let feet_y = -3400.0;
    let feet = Point3::new(1600.0, feet_y, stage::sand_z(feet_y));
    // facing azimuth 214°: three-quarters toward the camera, a quarter into
    // the sun at 250°, so the lit side is the side that is turned to us
    let yaw = (214.0f64 - 90.0).to_radians();
    // The lens carried at its side in the right hand, the left hand resting
    // on the satchel: two things to look at, and both of them are props.
    let params = pose([310.0, 300.0, 530.0], [-300.0, 60.0, 420.0], 5.0);
    let stance = Stance { rig: Rig::of(&params), params, feet, yaw };

    let mut cast = Cast::default();
    studio.set(&mut cast)?;
    studio.hero(&mut cast, &stance)?;
    // the lens hanging from the hand, its face turned to the camera
    // Carried by the rim, not floated beside it: the hand goes on the rim and
    // the glass hangs below it, which means the centre is a radius *down the
    // lens's own plane* from the hand — not straight down the world, because
    // the lens is tilted and its rim is tilted with it.
    let held = stance.at([310.0, 300.0, 530.0]);
    // turned to the camera, which is what a hand carrying a lens does with it
    let axis = Vec3::new(-0.62, -0.66, 0.42).normalize();
    let up_in_plane = (Vec3::new(0.0, 0.0, 1.0) - axis * axis.z).normalize();
    let lens = held - up_in_plane * (0.5 * kit::LENS_D);
    studio.lens(&mut cast, lens, axis)?;
    let picture = cast.picture();
    let map = rune_light(&picture, photons / 3, 25.0);

    // 1.5 m out on the hero's front-left, at the height of its collar: close
    // enough that the head fills the frame the way a portrait lens would.
    let eye = stance.at([-1330.0, 1470.0, 800.0]);
    let camera = stage::look(eye, stance.at([0.0, 40.0, 560.0]), 35.0);
    let t0 = Instant::now();
    shoot(&picture, &camera, SIZE, spp, Some(&map)).save(dir.join("portrait.png"))?;
    println!(
        "hero portrait: {:.2} m from the collar, facing {:.0}° with the sun at {:.0}°; {spp} spp in {:.1} s → {}",
        (eye - stance.at([0.0, 40.0, 560.0])).norm() / 1000.0,
        90.0 + yaw.to_degrees(),
        stage::SUN_AZ_DEG,
        t0.elapsed().as_secs_f64(),
        dir.join("portrait.png").display()
    );
    Ok(())
}

/// `tools.png`: the kit set down on the sand, and the three things it does to
/// the light.
///
/// Each tool is *aimed*, not placed: the lens sits high enough on the boulder
/// that its cone has closed to a hot ellipse by the time it reaches the sand,
/// the prism is turned about the sunbeam until its exit grazes and the
/// spectrum stretches out instead of dropping at its foot ([`kit::aim`]), and
/// the mirror is set to the one attitude that sends the sun back down onto
/// the beach rather than into the sky. The numbers each of them lands at are
/// printed, so the picture and the arithmetic can be checked against each
/// other.
/// How high the lens is propped for `tools.png`.
///
/// The one number the whole picture turns on. The sun descends `0.427` of a
/// millimetre against the beach for every millimetre it travels, so a lens
/// `h` above the sand throws `h/0.427` before it lands — and the cone has
/// closed to `220 · (1 − throw/f)` when it gets there. At 780 mm that is a
/// 1.83 m throw and a 59 mm spot: fourteen times the sun, which is a spot
/// that burns rather than a bright patch that does not read.
const LENS_HIGH: f64 = 520.0;

/// How far the prism throws its spectrum.
///
/// Two things fight over this number and neither of them wins outright.
///
/// **Colour** wants it long. The band spreads 1.6°, so through a 26 mm slit
/// red does not clear violet until `26/tan 1.6° = 930 mm`.
///
/// **Brightness** wants it short — and brightness is the harder constraint,
/// because a spectrum has to be seen against sand that is already in full
/// sun. A slit passes a slit's worth of light and the spread only ever
/// dilutes it: at 930 mm the patch is twice the slit and the irradiance is
/// about a fifth of what the sand already has from the sun and the sky, which
/// is a stain and not a rainbow. At 250 mm the spread is 7 mm on a 26 mm
/// slit, so the ends of the band are coloured and its middle is white — and
/// what lands is about as bright again as the sand, which reads.
///
/// Brightness wins, and the picture says so. Measured on the sand with
/// [`kosm_render::caustics::CausticMap::irradiance`] — which is what the
/// `HERO_SCAN=1` loop in this file is for — the band comes out at 2.2 at
/// 250 mm, 1.5 at 340 mm and 0.75 at 780 mm, against a sand that is already
/// getting about 3 from the sun and the sky together. So 250 mm: the spread
/// is 6.9 mm on a 26 mm slit, a quarter of a band-width, and what lands is a
/// bright line with a red-orange end to it lying along the edge of the stop's
/// own shadow.
///
/// **It is not a rainbow, and it cannot be one here.** Separating 1.6° across
/// a 26 mm slit takes 930 mm, and 930 mm of spreading leaves the light too
/// thin to see against sunlit sand — the two requirements are in direct
/// opposition and the sun is on the wrong side of the trade. A rainbow needs
/// the same slit throwing onto something *dark*, which on this beach means
/// waiting for the door: stone at an albedo of 0.24 rather than sand at 0.85,
/// and the cove's own shadow to land it in.
const PRISM_THROW: f64 = 250.0;

/// Which way the mirror throws its patch of sun: a world bearing, from +x
/// toward +y. See where it is used.
const MIRROR_BEARING_DEG: f64 = 20.0;

fn tools(studio: &Studio, dir: &Path, spp: usize, photons: usize) -> anyhow::Result<()> {
    let d = stage::sun_ray();
    // The kit is laid out in the *sun's* frame, not the world's, because
    // everything in this picture is about where the light goes: `along` is
    // the way it travels over the sand, and `across` is square to that. Put a
    // tool down in these two and its caustic is where it can be seen.
    let along = Vec3::new(d.x, d.y, 0.0).normalize();
    let across = Vec3::new(along.y, -along.x, 0.0);
    let hub = Point3::new(2400.0, -4200.0, 0.0);
    let put = |right: f64, ahead: f64, up: f64| {
        let p = hub + across * right + along * ahead;
        Point3::new(p.x, p.y, stage::sand_z(p.y) + up)
    };

    let mut cast = Cast::default();
    studio.set(&mut cast)?;

    // ---- the lens: leaning on a boulder, high enough to focus ------------
    // A lens of f = 2.5 m lying flat on the sand throws nothing: the cone has
    // barely begun to close by the time it reaches the ground a hand's
    // breadth away. Height is what makes a focus — the beam comes down
    // `(sin el + slope · …)` for every millimetre it travels, so a lens 620 mm
    // up throws a metre and a half and the cone has closed to 92 mm, which is
    // six times the sun and reads as a hot white spot. That is why there is a
    // boulder here: not scenery, a tripod.
    let lens = put(-330.0, -1250.0, LENS_HIGH);
    // The boulder is placed *touching* the glass, and where that is is
    // arithmetic: a sphere of radius `R` whose centre is `Δz` below the lens
    // is `sqrt(R² − Δz²)` wide there, so the rim leans on it at that plus the
    // lens's own 110 mm and not a millimetre further. A tool floating a hand's
    // breadth off its prop is the one thing that would give the whole picture
    // away as a render.
    let rock_r: f64 = 420.0;
    let rock_z: f64 = 120.0;
    let lean = (rock_r * rock_r - (LENS_HIGH - rock_z) * (LENS_HIGH - rock_z)).max(0.0).sqrt() + 0.5 * kit::LENS_D;
    let rock = put(-330.0 - lean, -1190.0, rock_z);
    let prism = put(210.0, 620.0, 170.0);
    let mirror = put(430.0, -520.0, 250.0);
    let delta = kit::min_deviation(stage::N_D);
    let (spin, out) = kit::aim(prism, d, delta, PRISM_THROW);
    let prism_spot = kit::land_on_sand(prism, out).expect("the spectrum comes down somewhere");
    let props = kosm::build::build(&Params::default(), move |b| {
        b.body("boulder").material("rock").add(
            b.sphere(rock_r)
                .at(rock.x, rock.y, rock.z)
                // and two stones, which is what the other two are propped on
                .union(b.sphere(112.0).at(prism.x + 30.0, prism.y - 50.0, prism.z - 150.0))
                .union(b.sphere(136.0).at(mirror.x + 20.0, mirror.y - 70.0, mirror.z - 232.0)),
        );
    })?;
    cast.add(&props, &|_| Some(Transform::identity()))?;
    studio.lens(&mut cast, lens, stage::sun_dir())?;
    let lens_spot = kit::land_on_sand(lens, d).expect("the sun comes down somewhere");
    let throw = (lens_spot - lens).norm();

    // ---- the prism ---------------------------------------------------------
    // A spectrum is not bright. A slit passes a slit's worth of sun and the
    // prism spreads it thinner still, so what lands is *about* as bright as
    // the sand it lands on — and light that is as bright as its background is
    // light nobody can see. The one place on a beach where a spectrum reads
    // is a shadow, and there is a metre of boulder standing in the sun a step
    // away casting one.
    //
    // So the prism is not placed and then aimed; it is *moved until its aim
    // lands in the shade*. Sliding it sideways slides the landing point with
    // it one for one, so a handful of turns converges from anywhere.
    studio.prism(&mut cast, kit::prism_frame(prism, d, out))?;

    // ---- the mirror: turned until the sun comes back down ------------------
    // A mirror lying face up on a beach sends the sun into the sky; the only
    // attitude that puts it back on the sand is nearly upright, which is why
    // this one is propped against a stone. Aimed across the light rather than
    // along it, so its patch lands where the camera is and its own face is
    // turned toward us instead of away.
    // Aimed at 20° — across the sun's own bearing of 70° and well off it —
    // for two reasons at once. The patch lands out in the open where the frame
    // can hold it, and the normal that turns `into` into `out` is the bisector
    // of the two, which at that angle points back at the camera: a mirror
    // aimed anywhere else in this scene shows the picture its dark back.
    let bounce = kit::descend(mirror, MIRROR_BEARING_DEG.to_radians(), 820.0)
        .expect("the mirror can always be aimed at the sand");
    studio.mirror(&mut cast, kit::mirror_frame(mirror, d, bounce))?;
    let mirror_spot = kit::land_on_sand(mirror, bounce).expect("the patch comes down somewhere");

    let picture = cast.picture();
    let t0 = Instant::now();
    let map = rune_light(&picture, photons, 20.0);
    let traced = t0.elapsed().as_secs_f64();

    // Low and close, up-sun of the kit and a third of a turn off the beam, so
    // the three patches lie out ahead in the frame and the tools still cast
    // their shadows across it instead of straight away from us.
    // Square to the beam, not up it. A camera up-sun sees exactly what the sun
    // sees — every shadow hidden behind the thing that casts it, including the
    // shade this picture needs — so it is swung round until the light runs
    // *across* the frame: each tool on the left of its own patch, every shadow
    // visible, and the sand between them doing the work of showing where the
    // light went.
    let eye = {
        let p = hub + across * 2300.0 + along * -520.0;
        Point3::new(p.x, p.y, stage::sand_z(p.y) + 640.0)
    };
    let look_at = {
        let p = hub + along * -190.0;
        Point3::new(p.x, p.y, stage::sand_z(p.y) + 280.0)
    };
    let camera = stage::look(eye, look_at, 42.0);
    // What the pass actually put on the sand, at each of the three places the
    // arithmetic says it should be. A caustic that is in the map but not in
    // the picture is a framing bug; one that is in neither is an aiming bug,
    // and these three numbers are what tells them apart.
    let n = Vec3::new(0.0, -stage::SLOPE, 1.0).normalize();
    let watts = |p: Point3| {
        let e = map.irradiance(p + n * 0.5, n);
        (e[0] + e[1] + e[2]) as f64 / 3.0
    };
    println!(
        "hero tools: prism at ({:+.0}, {:+.0}, {:+.0}), exit ({:+.3}, {:+.3}, {:+.3})",
        prism.x, prism.y, prism.z, out.x, out.y, out.z
    );
    println!(
        "hero tools irradiance: lens spot {:.2}, prism spot {:.2}, mirror spot {:.2}, bare sand {:.2}",
        watts(lens_spot),
        watts(prism_spot),
        watts(mirror_spot),
        watts(Point3::new(hub.x - 1500.0, hub.y, stage::sand_z(hub.y)))
    );
    if std::env::var("HERO_SCAN").is_ok() {
        let mut best: Vec<(f64, f64, f64)> = Vec::new();
        let step = 60.0;
        for i in -60..60 {
            for j in -60..60 {
                let (x, y) = (hub.x + i as f64 * step, hub.y + j as f64 * step);
                let p = Point3::new(x, y, stage::sand_z(y));
                let e = map.irradiance(p + n * 0.5, n);
                let w = (e[0] + e[1] + e[2]) as f64 / 3.0;
                if w > 0.02 {
                    best.push((w, x, y));
                }
            }
        }
        best.sort_by(|a, b| b.0.total_cmp(&a.0));
        best.truncate(20);
        for (w, x, y) in best {
            println!("  scan {w:8.1} at ({x:+.0}, {y:+.0})");
        }
    }
    let t1 = Instant::now();
    shoot(&picture, &camera, SIZE, spp, Some(&map)).save(dir.join("tools.png"))?;
    println!(
        "hero tools: lens {:.0} mm up throws {:.0} mm to a {:.0} mm patch at ({:+.0}, {:+.0}); prism turned {:.0}° lands {:.0} mm away at ({:+.0}, {:+.0}); mirror patch {:.0} mm away at ({:+.0}, {:+.0}); {} photons in {:.1} s, {spp} spp in {:.1} s → {}",
        lens.z - stage::sand_z(lens.y),
        throw,
        kit::patch_at(throw),
        lens_spot.x,
        lens_spot.y,
        spin.to_degrees(),
        (prism_spot - prism).norm(),
        prism_spot.x,
        prism_spot.y,
        (mirror_spot - mirror).norm(),
        mirror_spot.x,
        mirror_spot.y,
        map.len(),
        traced,
        t1.elapsed().as_secs_f64(),
        dir.join("tools.png").display()
    );
    Ok(())
}

/// `turntable.png`: 0°, 120°, 240°, three panels side by side in one plate.
///
/// The camera orbits and the hero does not, so the three panels are three
/// views of one pose and any asymmetry in the figure is the figure's own.
fn turntable(studio: &Studio, dir: &Path, spp: usize) -> anyhow::Result<()> {
    let feet_y = -3400.0;
    let feet = Point3::new(-5200.0, feet_y, stage::sand_z(feet_y));
    let params = pose([252.0, 108.0, 330.0], [-252.0, 108.0, 330.0], 2.0);
    let stance = Stance { rig: Rig::of(&params), params, feet, yaw: 0.0 };
    let mut cast = Cast::default();
    studio.set(&mut cast)?;
    studio.hero(&mut cast, &stance)?;
    let picture = cast.picture();

    let panel = (SIZE.0 / 3, SIZE.1);
    let mut plate = image::RgbaImage::new(SIZE.0, SIZE.1);
    let t0 = Instant::now();
    for (i, turn) in [0.0f64, 120.0, 240.0].iter().enumerate() {
        // The orbit starts in front of the hero and to the sunward side, so
        // the first panel is the one with the face in it.
        let a = (turn + 20.0).to_radians();
        let (s, c) = a.sin_cos();
        let radius = 2150.0;
        let eye = Point3::new(
            feet.x - s * radius,
            feet.y - c * radius,
            feet.z + 760.0,
        );
        let camera = stage::look(eye, Point3::new(feet.x, feet.y, feet.z + 580.0), 34.0);
        let film = pathtrace::render(&picture, &camera, panel.0, panel.1, &stage::options(spp, SEED));
        let view = stage::to_image(&film);
        for y in 0..panel.1 {
            for x in 0..panel.0 {
                plate.put_pixel(i as u32 * panel.0 + x, y, *view.get_pixel(x, y));
            }
        }
    }
    plate.save(dir.join("turntable.png"))?;
    println!(
        "hero turntable: 3 × {}×{} at {spp} spp in {:.1} s → {}",
        panel.0,
        panel.1,
        t0.elapsed().as_secs_f64(),
        dir.join("turntable.png").display()
    );
    Ok(())
}

// ---- the sim ----------------------------------------------------------------

/// `kosm run rune/hero`.
pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let dir = args.out().join("characters").join("hero");
    fs::create_dir_all(&dir)?;
    let spp = args
        .value("spp")
        .and_then(|v| v.parse().ok())
        .or_else(|| std::env::var("KOSM_SPP").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(128usize);
    let photons = args.value("photons").and_then(|v| v.parse().ok()).unwrap_or(1_500_000usize);
    let only = args.value("only");
    let want = |name: &str| only.is_none_or(|o| o == name);

    let studio = Studio::new()?;
    let (r, a, d) = kit::lens_numbers();
    println!(
        "hero kit: a {:.0} mm lens of two R = {:.1} mm caps {:.2} mm apart, {:.2} mm thick, f = {:.0} mm at n_d = {}; a {:.0} mm prism deviating {:.1}° and spreading {:.2}°; a {:.0} mm mirror at roughness 0.02",
        kit::LENS_D,
        r,
        2.0 * a,
        d,
        kit::focal(r, 0.5 * kit::LENS_D, stage::N_D),
        stage::N_D,
        kit::PRISM_SIDE,
        kit::min_deviation(stage::N_D).to_degrees(),
        kit::dispersion(kit::index_at(400.0), kit::index_at(700.0)).to_degrees(),
        kit::MIRROR_D,
    );

    let t0 = Instant::now();
    if want("door") {
        door(&studio, &dir, spp, photons)?;
    }
    if want("portrait") {
        portrait(&studio, &dir, spp, photons)?;
    }
    if want("tools") {
        tools(&studio, &dir, spp, photons)?;
    }
    if want("turntable") {
        turntable(&studio, &dir, spp)?;
    }
    println!("hero: {:.1} s in all → {}", t0.elapsed().as_secs_f64(), dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The doorstep is solved, and the solution is what it claims: the sun
    /// through the lens's centre arrives at [`AIM`], a hand's breadth beside
    /// the keyhole.
    #[test]
    fn the_doorstep_puts_the_sun_in_the_keyhole() {
        let stance = doorstep();
        let lens = held_lens(&stance);
        let hit = kit::land_on_sand(lens, stage::sun_ray());
        // it does *not* reach the sand — it hits the door first, which is the
        // whole point; so march it to the face instead
        let d = stage::sun_ray();
        let t = -lens.y / d.y;
        let at_face = lens + d * t;
        assert!(
            (at_face.x - AIM[0]).abs() < 1.0,
            "the beam lands {:.1} mm off its mark in x",
            at_face.x - AIM[0]
        );
        assert!(
            (at_face.z - AIM[2]).abs() < 1.0,
            "the beam lands {:.1} mm off its mark in z",
            at_face.z - AIM[2]
        );
        // and its mark is beside the keyhole, on the door: near enough to
        // read as aimed at it, far enough not to be swallowed by it
        let off = (Point3::new(AIM[0], AIM[1], AIM[2]) - stage::keyhole()).norm();
        assert!(off > stage::APERTURE_R + stage::RIM_W, "the caustic is inside the keyhole");
        assert!(off < 0.5 * stage::DOOR_W, "the caustic is off the door");
        if let Some(p) = hit {
            assert!(p.y > 0.0, "the beam would only meet the sand behind the cliff");
        }
        // the hero is on the sand, in front of the door, facing it
        assert!(stance.feet.y < 0.0 && stance.feet.y > -2000.0, "{:?}", stance.feet);
        assert!((stance.feet.z - stage::sand_z(stance.feet.y)).abs() < 1e-9);
        // it faces what it is aiming at
        let to_aim = Vec3::new(AIM[0] - stance.feet.x, AIM[1] - stance.feet.y, 0.0).normalize();
        let facing = Vec3::new(-stance.yaw.sin(), stance.yaw.cos(), 0.0);
        assert!(facing.dot(to_aim) > 0.999, "the hero is not facing its own mark");
        // the lens is within the hero's reach, not floating in front of it
        let shoulder = stance.at(stance.rig.shoulder(1.0));
        assert!((shoulder - stance.at(stance.rig.hand_right)).norm() <= stance.rig.reach() + 1e-9);
    }

    /// Two and a half metres is not staged away — it is impossible, and this
    /// is the arithmetic that says so, so that nobody re-opens it.
    #[test]
    fn two_and_a_half_metres_would_need_a_lens_over_the_heros_head() {
        let rig = Rig::DEFAULT;
        let d = stage::sun_ray();
        let kz = -d.z / d.y;
        // to land on the keyhole from 2.5 m out, the lens must be this high
        // over the sand under it
        let y = -2500.0;
        let need = stage::APERTURE_Z + kz * (-y) + stage::SLOPE * y - stage::sand_z(y);
        let can = rig.shoulder_z + rig.reach();
        assert!(need > can + 200.0, "the reach would be {need:.0} mm and the hero has {can:.0}");
        assert!(need > rig.height, "and it is over its head by {:.0} mm", need - rig.height);
    }

    /// The stills are the four files the character sheet is, and the studio
    /// builds without a cove.
    #[test]
    fn the_studio_builds_the_stage_the_hero_and_the_hardware() -> anyhow::Result<()> {
        let studio = Studio::new()?;
        let mut cast = Cast::default();
        studio.set(&mut cast)?;
        let before = cast.len();
        assert!(before >= 5, "the stage came out as {before} objects");
        studio.hero(&mut cast, &doorstep())?;
        assert!(cast.len() > before + 15, "the hero is more than a handful of parts");
        studio.lens(&mut cast, Point3::new(0.0, -1000.0, 900.0), stage::sun_dir())?;
        // …and the picture the whole of that makes has exactly one sun
        let picture = cast.picture();
        assert!(picture.sun.is_some());
        assert!(picture.lights.is_empty(), "the cove's light is the sky and the sun");
        Ok(())
    }
}
