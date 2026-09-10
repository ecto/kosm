//! `kosm run rune/hero` — the adventurer, as six path-traced stills.
//!
//! A character sheet for Rune's player, made the only way an agent can make
//! one: build it, light it with the level's own sun, trace it, read the png,
//! change a number, trace it again.
//!
//! ```text
//! kosm run rune/hero                       all six, 128 spp
//! kosm run rune/hero --only door --spp 16  one of them, fast
//! kosm run rune/hero --photons 200000      a cheaper caustic pass
//! HERO_SCAN=1 kosm run rune/hero --only tools --spp 4
//! ```
//!
//! That last one is the debugging loop this file was built with, and it does
//! two jobs. It prints the caustic map's irradiance over the door's face, per
//! channel, with the band's own red-to-blue centroid separation measured off
//! it — a caustic that is *missing*, one that is merely out of frame, and one
//! that is there but white all look identical in a png and are nothing alike.
//! And it prints where every tool and every mark of light is as a bearing off
//! the camera's own axis, with the shadow coordinate that says whether it is
//! standing in the sun. Framing four tools and three marks in one frame is
//! the hard part of `tools.png`, and seven numbers answer it far faster than
//! seven renders.
//!
//! Six files under `out/characters/hero/`:
//!
//! | file             | what it is |
//! |------------------|------------|
//! | `door.png`       | the hero at the doorstep, holding the lens up to the sun, its caustic on the door beside the keyhole |
//! | `door_face.png`  | the same pose from the sunward side, where the face reads and the glass shows the sun |
//! | `portrait.png`   | a close three-quarter, for the costume: hood, collar, satchel, strap |
//! | `tools.png`      | the kit at the doorstep — the lens's burn, the prism's rainbow, the mirror's patch, all real light |
//! | `turntable.png`  | 0°, 120°, 240°, three panels in one frame |
//! | `sheet.png`      | the character sheet: six views, 1920 × 1080 |
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
//! its arms reach. Solve it ([`doorstep`]) at the cove's sun and a 1.11 m
//! adventurer's reach — shoulder 588 mm, arm 404 mm — and the answer is about
//! **1.2 m from the door**, not 2.5 m. It cannot be 2.5 m: at 2.5 m the lens
//! would have to be held 1.48 m above the sill, which is 370 mm over the
//! crown of a hero 1.11 m tall. Two and a half metres of doorstep needs
//! either a sun below about 13° or a taller adventurer, and the cove's sun is
//! 22°. The number the brief asked for is in the *lens* instead, where it
//! belongs: f = 2.5 m, so at a 1.3 m throw the cone has closed to a hundred
//! millimetre ellipse — bright, still converging, and legible beside a 240 mm
//! keyhole.
//!
//! # The face and the caustic cannot be in one frame
//!
//! The puzzle puts the sun *behind* the hero: the lens has to be on the beam
//! that reaches the keyhole, so the hero stands between the sun and the door
//! and looks into the cliff. Its face is therefore turned away from the sun
//! in every pose that solves the puzzle, and the door — which carries the
//! caustic — is behind its head.
//!
//! That is not stageable away, so it is not staged away. `door.png` is shot
//! from behind the shoulder, where the caustic reads and the face does not;
//! `door_face.png` is the same figure in the same pose from in front, where
//! the face reads on sky and sand bounce, the glass carries the sun's own
//! reflection, and the caustic is behind the camera. Two frames, one truth.

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

/// Every frame but the sheet is this size: a landscape plate an agent can
/// read at a glance.
const SIZE: (u32, u32) = (960, 540);

/// The sheet is the deliverable, so it is the one that is full size.
const SHEET: (u32, u32) = (1920, 1080);

// ---- the pose --------------------------------------------------------------

/// A hero standing somewhere on the sand, facing somewhere.
pub struct Stance {
    /// The knobs the figure is built from — the pose lives in these.
    pub params: Params,
    /// The rig those knobs describe.
    pub rig: Rig,
    /// Where the boots are, on the sand.
    pub feet: Point3,
    /// Which way it faces: a turn about z, radians, from facing +y. Its
    /// world bearing, measured from +x toward +y, is `90° + yaw`.
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

    /// Which way it is looking, in the world.
    pub fn facing(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), self.yaw.cos(), 0.0)
    }
}

/// How a body is held: the two hand targets, the head, the torso and the feet.
///
/// Nine numbers, and every one of them is a `Param` in the run hash. `Pose`
/// exists rather than a nine-argument function because three of the six
/// stills share a pose and differ only in where the camera is, and a struct
/// with defaults is how that stays one definition.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub right: [f64; 3],
    pub left: [f64; 3],
    /// Chin up, degrees.
    pub head_tilt: f64,
    /// Torso forward over the hips, degrees.
    pub torso_lean: f64,
    /// Right foot ahead of the left, millimetres.
    pub stride: f64,
    /// Hips back over the rear foot, millimetres. Negative is back.
    pub pelvis_y: f64,
}

impl Pose {
    /// Arms down and a little out, weight even. The turntable's pose.
    pub const REST: Pose = Pose {
        right: [268.0, 96.0, 344.0],
        left: [-268.0, 96.0, 344.0],
        head_tilt: 2.0,
        torso_lean: 0.0,
        stride: 0.0,
        pelvis_y: 0.0,
    };

    /// A relaxed stand: the weight on one leg, a little lean, hands clear of
    /// the cloak. What the sheet's four turnaround panels are shot in — a
    /// figure standing to attention is a figure nobody believes.
    pub const STAND: Pose = Pose {
        right: [286.0, 128.0, 352.0],
        left: [-274.0, 84.0, 332.0],
        head_tilt: 3.0,
        torso_lean: 3.0,
        stride: 62.0,
        pelvis_y: -22.0,
    };

    fn params(&self) -> Params {
        let mut p = Params::new();
        p.set("hand_r_x_mm", self.right[0]);
        p.set("hand_r_y_mm", self.right[1]);
        p.set("hand_r_z_mm", self.right[2]);
        p.set("hand_l_x_mm", self.left[0]);
        p.set("hand_l_y_mm", self.left[1]);
        p.set("hand_l_z_mm", self.left[2]);
        p.set("head_tilt_deg", self.head_tilt);
        p.set("torso_lean_deg", self.torso_lean);
        p.set("stride_mm", self.stride);
        p.set("pelvis_y_mm", self.pelvis_y);
        p
    }

    /// A stance: this pose, standing on the sand at `feet`, facing `yaw`.
    pub fn stand(&self, feet: Point3, yaw: f64) -> Stance {
        let params = self.params();
        Stance { rig: Rig::of(&params), params, feet, yaw }
    }
}

/// How the hero holds the lens up: where the gripping hand goes, as a
/// direction from its own shoulder, and how much of the arm's straight reach
/// it uses.
///
/// **One hand, and out to the side.** The last pass had both hands on the rim
/// and the glass on the centre line, and when the head grew to 456 mm across
/// the lens ended up *inside the skull* — the caustic pass found the
/// refractor, aimed a million and a half photons at it, and every one of them
/// hit hair before it hit glass, so the map came back empty and the door was
/// bare. A metre-ten figure with a 404 mm arm and a head that wide cannot
/// hold anything clear of its own face at arm's length in front of it. It can
/// hold it up and out, which is what somebody sighting through a lens does
/// anyway.
///
/// `PSI` is the swing out to the hero's right, `PHI` the lift, both measured
/// at the shoulder. Together they put the glass 300 mm clear of the hood at
/// its nearest point, which is checked in
/// [`tests::the_lens_is_held_clear_of_the_heros_own_head`].
const ARM_SWING_DEG: f64 = 28.0;
const ARM_ELEVATION_DEG: f64 = 46.0;
const ARM_EXTENSION: f64 = 0.97;

/// Where the hero's free hand goes: up and out on the other side, elbow
/// clear of the ribs. Two raised arms is a figure doing something; one raised
/// arm and one hanging is a figure holding a shopping bag.
const FREE_HAND: [f64; 3] = [-330.0, 190.0, 700.0];

/// Which way the glass lies from the fist that is holding its rim: up and
/// outboard, projected into the plane of the lens.
///
/// A hand on a rim puts the glass a radius away in the *lens's own plane*,
/// not a radius up the world — the lens is canted and its rim is canted with
/// it — so this hint is projected before it is used.
const GRIP_HINT: [f64; 3] = [0.55, 0.15, 0.82];

/// How far the hero leans into the door, degrees.
///
/// It buys three things at once, which is why it is worth a joint. The
/// silhouette gets a diagonal instead of a stack of circles; the weight goes
/// onto the back foot, because the hips have gone back under it; and the
/// glass comes forward out of the hood's own shadow, so the caustic that
/// leaves it is thrown from lit cloth rather than from the dark under the
/// cowl.
const DOOR_LEAN_DEG: f64 = 13.0;
/// The doorstep's footwork: the right foot forward, the hips back over the
/// left. Together with the lean, that is a figure pushing up at a door.
const DOOR_STRIDE: f64 = 76.0;
const DOOR_PELVIS_Y: f64 = -26.0;

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

/// A hero at the doorstep, and the glass it is holding.
pub struct Doorstep {
    pub stance: Stance,
    /// The centre of the lens, in the world.
    pub lens: Point3,
    /// Its optical axis: the sunbeam, canted toward the camera.
    pub axis: Vec3,
}

/// Where the hero must stand for the lens it is holding to put the sun in the
/// keyhole, and which way it must face.
///
/// Two constraints and two unknowns. The lens has to be on the sunbeam
/// through the keyhole — that is the `x` and `z` equations below — and the
/// hero has to be facing the keyhole, which is what decides where its hand
/// is relative to its feet. Facing depends on position and position on
/// facing, so it is iterated; six turns is far past convergence for a yaw
/// that only ever moves thirty degrees.
///
/// The two equations, with the lens at hero-local `(lx, ly, lz)` and the feet
/// at `(fx, fy)` facing `yaw`:
///
/// ```text
/// lens_y = fy + lx sin yaw + ly cos yaw          lens_z = slope·fy + lz
/// lens_z + k_z · lens_y = aim_z                  lens_x − k_x · lens_y = aim_x
/// ```
///
/// The first solves for `fy` and the second for `fx`. The last pass's version
/// dropped the `lx sin yaw` and `lx cos yaw` terms because the lens was on
/// the hero's centre line; it is not any more, and they are back.
///
/// The torso lean enters through the *shoulder*, and only there. A lean drops
/// the shoulder and pushes it forward, which moves where a given arm angle
/// puts the glass; it does not move the hands, because the hands are the
/// knobs. That is what makes the lean free: the caustic lands in the same
/// place leaned or level, and only the figure changes.
pub fn doorstep() -> Doorstep {
    let base = Pose { torso_lean: DOOR_LEAN_DEG, stride: DOOR_STRIDE, pelvis_y: DOOR_PELVIS_Y, ..Pose::REST };
    let rig = Rig::of(&base.params());
    let d = stage::sun_ray();
    let (kx, kz) = (d.x / d.y, -d.z / d.y);
    let (psi, phi) = (ARM_SWING_DEG.to_radians(), ARM_ELEVATION_DEG.to_radians());
    let shoulder = rig.shoulder(1.0);
    let out = ARM_EXTENSION * rig.reach();
    // The gripping hand, as a direction from its own shoulder.
    let hand = [
        shoulder[0] + out * psi.sin() * phi.cos(),
        shoulder[1] + out * psi.cos() * phi.cos(),
        shoulder[2] + out * phi.sin(),
    ];

    // Iterate: the yaw decides the lens's plane in the hero's own frame,
    // which decides where the glass sits off the fist, which decides the
    // stance, which decides the yaw.
    let axis = |yaw: f64| cant(stage::sun_dir(), LENS_CANT_DEG.to_radians() * (1.0 + 0.0 * yaw));
    let (mut yaw, mut feet, mut lens_local) = (0.0f64, Point3::origin(), hand);
    for _ in 0..6 {
        let (s, c) = yaw.sin_cos();
        // the optical axis in the hero's own frame
        let a = axis(yaw);
        let a_local = Vec3::new(a.x * c + a.y * s, -a.x * s + a.y * c, a.z);
        let hint = Vec3::new(GRIP_HINT[0], GRIP_HINT[1], GRIP_HINT[2]);
        let w = (hint - a_local * a_local.dot(hint)).normalize();
        lens_local = [
            hand[0] + 0.5 * kit::LENS_D * w.x,
            hand[1] + 0.5 * kit::LENS_D * w.y,
            hand[2] + 0.5 * kit::LENS_D * w.z,
        ];
        let (lx, ly, lz) = (lens_local[0], lens_local[1], lens_local[2]);
        let fy = (AIM[2] - lz - kz * (lx * s + ly * c)) / (stage::SLOPE + kz);
        let lens_y = fy + lx * s + ly * c;
        let fx = AIM[0] + kx * lens_y - lx * c + ly * s;
        feet = Point3::new(fx, fy, stage::sand_z(fy));
        yaw = (fx - AIM[0]).atan2(AIM[1] - fy);
    }

    // Looking up at it. Not the whole way — a Mii's head on a 63° neck is a
    // Mii falling over backwards — but far enough that the gaze clears the
    // brow and lands on the glass. The lean has already tipped the head
    // forward, so the tilt has that to undo before it starts.
    let neck = rig.neck();
    let look = (lens_local[2] - neck[2]).atan2(
        ((lens_local[0]).powi(2) + (lens_local[1] - neck[1]).powi(2)).sqrt(),
    );
    let pose = Pose {
        right: hand,
        left: FREE_HAND,
        head_tilt: (0.62 * look.to_degrees() + DOOR_LEAN_DEG).min(40.0),
        ..base
    };
    let stance = pose.stand(feet, yaw);
    let lens = stance.at(lens_local);
    Doorstep { stance, lens, axis: axis(yaw) }
}

/// The sun's direction turned `by` about the vertical, toward the camera.
///
/// Positive is counter-clockwise seen from above, which at the cove's sun
/// azimuth of 250° is the way the doorstep camera lies.
pub fn cant(dir: Vec3, by: f64) -> Vec3 {
    let (s, c) = by.sin_cos();
    Vec3::new(dir.x * c - dir.y * s, dir.x * s + dir.y * c, dir.z).normalize()
}

/// How far off the sunbeam the lens is held. See [`doorstep`].
///
/// Sixty degrees, and the number is spent on the *cameras*. A thin lens
/// images a parallel bundle wherever its undeviated chief ray crosses the
/// focal plane, so a cant moves the caustic not at all — all it costs is
/// `cos θ` of the collected sun and a wider, softer patch, and all it buys is
/// which way the disc is turned. At 42° the glass was face-on to `door.png`
/// and exactly edge-on to `door_face.png`, which is a gold hairline; at 60°
/// it is fourteen degrees off face-on to one and seventy to the other, so
/// both frames get a disc instead of one getting a disc and one getting a
/// wire.
const LENS_CANT_DEG: f64 = 60.0;

// ---- putting a picture together --------------------------------------------

/// Everything the six stills share: the stage, the hero and the hardware,
/// built once each.
struct Studio {
    stage: Built,
    hardware: Built,
    /// The sheet's own ground: one flat slab, two hundred metres across.
    ///
    /// The cove's beach is a 26 m box on a 6 % grade, which is right for the
    /// cove and wrong for a turnaround: its far edge *is* the horizon, so the
    /// horizon is 0.78 m up on the inland side and 0.78 m down on the seaward
    /// one, and a camera orbiting the hero walks that line up and down the
    /// frame. Six panels with the horizon in six places read as six
    /// afternoons. Flat and far puts it in the same place in all of them.
    flat: Built,
}

impl Studio {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            stage: stage::stage(&Params::default())?,
            hardware: kit::hardware(&Params::default())?,
            flat: kosm::build::build(&Params::default(), |b| {
                b.body("sand").material("sand").add(
                    b.boxed(200_000.0, 200_000.0, 4000.0).at(0.0, 0.0, -2000.0),
                );
            })?,
        })
    }

    /// The stage, whole.
    fn set(&self, cast: &mut Cast) -> anyhow::Result<()> {
        cast.add(&self.stage, &|_| Some(Transform::identity()))
    }

    /// The sand and nothing else: the sheet's backdrop, which is plain sand
    /// and a sky and no landmark at all, so the six panels differ only by
    /// where the camera is.
    fn sand(&self, cast: &mut Cast) -> anyhow::Result<()> {
        cast.add(&self.flat, &|_| Some(Transform::identity()))
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
        cast.add_mesh(kit::prism_mesh(), "flint", frame.clone());
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
    shoot_with(picture, camera, size, &stage::options(spp, SEED), map)
}

/// The same, with the integrator's settings spelled out. `tools.png` needs
/// [`stage::options_rare`]; nothing else does.
fn shoot_with(
    picture: &Picture,
    camera: &pathtrace::Camera,
    size: (u32, u32),
    opts: &pathtrace::PathTraceOptions,
    map: Option<&CausticMap>,
) -> image::RgbaImage {
    let film = pathtrace::render_with_caustics(picture, camera, size.0, size.1, opts, map);
    stage::to_image(&film)
}

/// Copy one panel into a plate at a grid cell.
fn paste(plate: &mut image::RgbaImage, panel: &image::RgbaImage, at: (u32, u32)) {
    for y in 0..panel.height() {
        for x in 0..panel.width() {
            plate.put_pixel(at.0 + x, at.1 + y, *panel.get_pixel(x, y));
        }
    }
}

// ---- the doorstep, from both sides ------------------------------------------

/// Where `door.png` is shot from: an azimuth about the hero measured the
/// world's way, from +x toward +y, a distance and a height.
const CAMERA_AZIMUTH_DEG: f64 = 316.0;
const CAMERA_BACK: f64 = 2700.0;
const CAMERA_UP: f64 = 800.0;
const CAMERA_VFOV_DEG: f64 = 35.0;

/// The doorstep picture: the stage, the hero in its solved stance, the lens
/// in its hands, and the caustic that comes off it. Built once and shot
/// twice, because tracing a million and a half photons for each of two
/// cameras onto the same geometry would be tracing it twice for nothing.
fn doorstep_picture(studio: &Studio, photons: usize) -> anyhow::Result<(Doorstep, Picture, CausticMap, f64)> {
    let step = doorstep();
    let mut cast = Cast::default();
    studio.set(&mut cast)?;
    studio.hero(&mut cast, &step.stance)?;
    studio.lens(&mut cast, step.lens, step.axis)?;
    let picture = cast.picture();
    let t0 = Instant::now();
    let map = rune_light(&picture, photons, 15.0);
    Ok((step, picture, map, t0.elapsed().as_secs_f64()))
}

/// `door.png`: the hero at the doorstep with the lens up in both hands, and
/// the sun coming through it onto the stone beside the keyhole.
fn door(
    dir: &Path,
    step: &Doorstep,
    picture: &Picture,
    map: &CausticMap,
    traced: f64,
    spp: usize,
) -> anyhow::Result<()> {
    let (stance, lens) = (&step.stance, step.lens);
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
    // Between the glass and the mark it is putting on the stone, and lower
    // than either: the hero's boots are the bottom of the frame and a
    // character sheet that crops its own feet is not one.
    let mix = |a: f64, b: f64| 0.62 * a + 0.38 * b;
    let target = Point3::new(mix(lens.x, AIM[0]), mix(lens.y, AIM[1]), 0.45 * AIM[2] + 0.35 * lens.z);
    let camera = stage::look(eye, target, CAMERA_VFOV_DEG);

    let t1 = Instant::now();
    let img = shoot(picture, &camera, SIZE, spp, Some(map));
    let path = dir.join("door.png");
    img.save(&path)?;
    let throw = (lens - stage::keyhole()).norm();
    println!(
        "hero door: standing {:.2} m off the face at ({:+.0}, {:+.0}) mm, facing {:+.1}°, leaning {:.0}°, lens {:.0} mm up and {:.0} mm from the keyhole → a {:.0} mm patch; {} photons in {:.1} s, {}×{} at {spp} spp in {:.1} s → {}",
        -stance.feet.y / 1000.0,
        stance.feet.x,
        stance.feet.y,
        stance.yaw.to_degrees(),
        stance.rig.torso_lean,
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

/// Where `door_face.png` stands: how far round from dead ahead, how far out,
/// how high the eye is and what it looks at — the last two as fractions of
/// the hero's own height.
///
/// **The side is forced and the angle is nearly forced.** The hero faces the
/// cliff, so its left is four metres of stone and only its right is open
/// beach; and there are only about 1.2 m of doorstep, so the camera has to
/// swing round until it clears `y = 0`. Thirty-four degrees off the front at
/// two metres is the widest three-quarter that fits, and the height is what
/// gets the sightline over the raised arm — from any lower, the elbow that is
/// holding the lens is exactly between the camera and the face.
const FACE_TURN_DEG: f64 = 26.0;
const FACE_DIST: f64 = 1650.0;
const FACE_EYE: f64 = 0.74;
const FACE_LOOK: f64 = 0.55;
const FACE_VFOV_DEG: f64 = 42.0;
/// How near the cliff a camera may get. The cliff's face is the plane `y = 0`
/// and there is four metres of stone behind it, so a camera that crosses this
/// line is a camera inside a rock.
const FACE_CLEARANCE: f64 = -420.0;

/// `door_face.png`: the same hero, the same pose, from in front.
///
/// The sun is behind the hero at the doorstep and cannot be anywhere else, so
/// this face is lit by the sky and by a beach at an albedo of 0.85 — which at
/// a 22° sun is a great deal of warm bounce, and is exactly the light a
/// chin-up face catches best. What the camera is here for is the two things
/// `door.png` cannot show: the face, and the sun's own image sitting in the
/// glass.
///
/// The distance is walked in rather than fixed, because the doorstep is not
/// deep enough to promise two metres: the camera steps forward until it is
/// clear of the cliff, and stops.
fn door_face(
    dir: &Path,
    step: &Doorstep,
    picture: &Picture,
    map: &CausticMap,
    spp: usize,
) -> anyhow::Result<()> {
    let (stance, lens) = (&step.stance, step.lens);
    let h = stance.rig.height;
    let f = stance.facing();
    // …turned toward the hero's own right, which is the only side of it that
    // is not four metres of cliff.
    let (s, c) = (-FACE_TURN_DEG).to_radians().sin_cos();
    let dir_h = Vec3::new(f.x * c - f.y * s, f.x * s + f.y * c, 0.0);
    let mut dist = FACE_DIST;
    let eye = loop {
        let eye = Point3::new(
            stance.feet.x + dir_h.x * dist,
            stance.feet.y + dir_h.y * dist,
            stance.feet.z + FACE_EYE * h,
        );
        if eye.y <= FACE_CLEARANCE || dist <= 900.0 {
            break eye;
        }
        dist -= 40.0;
    };
    let look = Point3::new(
        0.86 * stance.feet.x + 0.14 * lens.x,
        0.86 * stance.feet.y + 0.14 * lens.y,
        stance.feet.z + FACE_LOOK * h,
    );
    let camera = stage::look(eye, look, FACE_VFOV_DEG);

    let t0 = Instant::now();
    shoot(picture, &camera, SIZE, spp, Some(map)).save(dir.join("door_face.png"))?;
    println!(
        "hero door_face: {FACE_TURN_DEG:.0}° off the hero's front at {:.2} m, eye ({:+.0}, {:+.0}, {:.0}); {spp} spp in {:.1} s → {}",
        dist / 1000.0,
        eye.x,
        eye.y,
        eye.z,
        t0.elapsed().as_secs_f64(),
        dir.join("door_face.png").display()
    );
    Ok(())
}

// ---- the portrait -----------------------------------------------------------

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
    // facing bearing 292°: three-quarters toward the camera and a little out
    // of the sun at 250°, so the key rakes across the hood and the strap
    let yaw = (292.0f64 - 90.0).to_radians();
    // The lens carried at its side in the right hand, the left hand resting
    // on the satchel: two things to look at, and both of them are props.
    let pose = Pose {
        right: [318.0, 306.0, 512.0],
        left: [-296.0, 74.0, 408.0],
        head_tilt: 6.0,
        torso_lean: 4.0,
        stride: 58.0,
        pelvis_y: -20.0,
    };
    let stance = pose.stand(feet, yaw);

    let mut cast = Cast::default();
    studio.set(&mut cast)?;
    studio.hero(&mut cast, &stance)?;
    // Carried by the rim, not floated beside it: the hand goes on the rim and
    // the glass hangs below it, which means the centre is a radius *down the
    // lens's own plane* from the hand — not straight down the world, because
    // the lens is tilted and its rim is tilted with it.
    let held = stance.at(pose.right);
    let axis = Vec3::new(-0.62, -0.66, 0.42).normalize();
    let up_in_plane = (Vec3::new(0.0, 0.0, 1.0) - axis * axis.z).normalize();
    let lens = held - up_in_plane * (0.5 * kit::LENS_D);
    studio.lens(&mut cast, lens, axis)?;
    let picture = cast.picture();
    let map = rune_light(&picture, photons / 3, 25.0);

    // 1.6 m out on the hero's front-left, at the height of its collar: close
    // enough that the head fills the frame the way a portrait lens would.
    let eye = stance.at([-1240.0, 1500.0, 790.0]);
    let look_at = stance.at([0.0, 40.0, 600.0]);
    let camera = stage::look(eye, look_at, 35.0);
    let t0 = Instant::now();
    shoot(&picture, &camera, SIZE, spp, Some(&map)).save(dir.join("portrait.png"))?;
    println!(
        "hero portrait: {:.2} m from the collar, facing {:.0}° with the sun at {:.0}°; {spp} spp in {:.1} s → {}",
        (eye - look_at).norm() / 1000.0,
        90.0 + yaw.to_degrees(),
        stage::SUN_AZ_DEG,
        t0.elapsed().as_secs_f64(),
        dir.join("portrait.png").display()
    );
    Ok(())
}

// ---- the tools --------------------------------------------------------------

/// `tools.png`: the kit at the doorstep, and the three things it does to the
/// light.
///
/// # Why this picture is staged at the door and not out on the sand
///
/// The last pass laid the kit on open beach and measured what landed: a lens
/// spot that read, a mirror patch that read, and a spectrum that did not,
/// because a spectrum through a slit is `slit/(slit + throw·Δ)` of bare sun
/// and sunlit sand at an albedo of 0.85 is brighter than that at any throw
/// long enough to separate the colours. The two requirements are in direct
/// opposition and no amount of aiming resolves them.
///
/// What resolves them is a **darker screen**, and that means two changes.
///
/// - **A shadow.** The cove's door is stone at an albedo of 0.24 and it faces
///   the sun, so a standing stone is put up-sun of it to lay a strip of its
///   face in shade. A low sun is unhelpful here — a stone of height `H` at a
///   horizontal `L` up-sun shades only up to `H − L·tan 22°` — so the stone
///   is over two metres, which on a beach at the mouth of a sealed door is
///   not an odd thing to find.
/// - **A wider slit and a flint prism.** Both are in [`kit`]: lead crystal
///   spreads the band 3.9° where N-BK7 spreads it 1.6°, so a 300 mm band
///   needs four metres of throw rather than eleven; and the slit goes from
///   26 mm to 60, which trades resolution for light at exactly one for one
///   and is the trade this picture needs.
///
/// The **lens** cannot join them and no staging fixes that: its caustic
/// travels along the sunbeam, so it can only ever land where the sun already
/// does. So it burns on the one dark thing that *is* in the sun — a flat
/// slab of the same stone, out in the foreground, where the burn is the
/// brightest pixel in the frame by a factor of nine.
///
/// Every number below is printed at the end of the run, and `HERO_SCAN=1`
/// prints the map itself over the door's face.
///
/// Where the prism's band lands on the door's face, and where the mirror puts
/// its patch. Both inside the shade stone's shadow; the constants below are
/// what the shadow is sized around.
const BAND_AT: [f64; 3] = [-420.0, 0.0, 760.0];
const PATCH_AT: [f64; 3] = [-640.0, 0.0, 1180.0];

/// How far the prism stands from the mark it makes.
///
/// Three metres and a fifth at lead crystal's 3.9° through a 60 mm slit is a
/// 307 mm band at about a fifth of bare sun. Longer would be better on both
/// the length and nothing else — and the *cluster* is what stops it, not the
/// arithmetic. [`kosm_render::caustics::trace`] aims its photons at a disc
/// covering every refractor in the scene, so the budget that reaches the
/// glass goes as `glass area / π·extent²`: put the lens in the foreground and
/// the prism five metres away and 99.9 % of a million and a half photons fly
/// between them and land on nothing. Keeping the two tools inside two metres
/// of each other is worth more to this picture than another hundred
/// millimetres of rainbow.
const PRISM_THROW: f64 = 3200.0;
/// How high the prism sits above the sand. The spin on the cone is solved
/// from this, because a prism buried in the beach throws nothing.
const PRISM_UP: f64 = 620.0;
/// Which side of the sun-line the prism stands on: `+1` or `−1`. See where
/// the spin is solved.
const PRISM_HAND: f64 = 1.0;
/// How far the mirror stands from its patch.
///
/// Close, and closer than it looks like it needs to be, because the mirror's
/// patch is the one mark in this picture the *path tracer* has to find rather
/// than the photon pass. A metal bends no photon that `caustics` will keep —
/// the pass deposits only what a transmissive solid refracted — so the patch
/// exists only as a camera→door→mirror→sun path, and the door's diffuse
/// bounce finds the mirror with a probability equal to the solid angle the
/// mirror subtends there. At two metres a 250 mm disc is 0.35 % of a
/// hemisphere and the patch is speckle; at 1.2 m it is 1.1 %, which at 256
/// samples is three hits a pixel and a patch the denoiser can hold.
const MIRROR_THROW: f64 = 900.0;

/// The shade stone: how far up-sun of the door's face its top edge is, and
/// how high on the face its shadow therefore reaches.
///
/// **Far, and that is not for the shadow's sake.** A low sun makes an
/// unhelpful shadow — a stone of height `H` at a horizontal `L` up-sun shades
/// the face only up to `H − L·tan 22°` — so every metre of distance costs
/// four hundred millimetres of stone, and the obvious move is to stand it
/// close. The obvious move puts it **in the prism's beam**: the prism throws
/// from three metres out at forty-six degrees off the sun, and near the door
/// that beam and the sun-line the stone sits on have converged to the same
/// place. The measured symptom is unmistakable and was measured — the band
/// went from 374 mm with a red end and a blue end to 60 mm of red, because
/// the stone had eaten everything the glass bent furthest.
///
/// So the stone stands two metres up-sun, where the two lines are still a
/// metre and a half apart, and is 2.6 m tall to make up for it.
const SHADE_BACK: f64 = 1100.0;
const SHADE_SHADOW_TOP: f64 = 1350.0;
const SHADE_W: f64 = 850.0;
const SHADE_T: f64 = 300.0;

/// How high the lens is propped for its burn, above the slab it burns on.
///
/// The one number that picture turns on. The beam drops `sin 22° = 0.375` of
/// a millimetre for every millimetre it travels, so a lens `h` over its
/// target throws `h/0.375`, and the cone has closed to `220·(1 − throw/f)`
/// when it arrives. At 640 mm that is a 1.71 m throw and a 70 mm spot: ten
/// times bare sun, which is a burn and not a bright patch.
const LENS_HIGH: f64 = 640.0;

fn tools(studio: &Studio, dir: &Path, spp: usize, photons: usize) -> anyhow::Result<()> {
    let d = stage::sun_ray();
    let along = Vec3::new(d.x, d.y, 0.0).normalize();
    let across = Vec3::new(along.y, -along.x, 0.0);
    let (delta, spread) = kit::prism_deviation();

    let band = Point3::new(BAND_AT[0], BAND_AT[1], BAND_AT[2]);
    let patch = Point3::new(PATCH_AT[0], PATCH_AT[1], PATCH_AT[2]);
    let face_n = Vec3::new(0.0, -1.0, 0.0);

    // ---- the shade stone --------------------------------------------------
    // Its top edge is on the sunbeam that grazes the door at `SHADE_SHADOW_TOP`,
    // `SHADE_BACK` millimetres up-sun of the face — which is one line, because
    // the beam is a straight line and the sand is a plane.
    let t_back = SHADE_BACK / (d.x * d.x + d.y * d.y).sqrt();
    let lip = Point3::new(BAND_AT[0], 0.0, SHADE_SHADOW_TOP) - d * t_back;
    let shade_h = lip.z - stage::sand_z(lip.y);
    let shade_bearing = along.y.atan2(along.x).to_degrees() - 90.0;

    // ---- the prism --------------------------------------------------------
    // The exit lies on a cone at `delta` about the sunbeam, so the one degree
    // of freedom is where on that cone. It is spent on *height*: the spin is
    // bisected until the prism, `PRISM_THROW` back along the exit, stands
    // `PRISM_UP` off the sand. That is the only constraint the prism has —
    // everything else about where it ends up is a consequence.
    let height_at = |spin: f64| {
        let (p, _) = kit::place_for(band, d, delta, spin, PRISM_THROW);
        p.z - stage::sand_z(p.y)
    };
    // The exit rises 24° above the horizon at spin 0 and falls to 68° below
    // it at spin π. The prism is a throw back *along* that exit, so its
    // height runs the other way — underground at spin 0, four metres up at
    // spin π — and rises monotonically in between, which is what a bisection
    // needs. (Getting that sense backwards is not a subtle failure: it hangs
    // the prism four and a half metres in the air and the spectrum lands
    // behind the cliff.)
    //
    // The **sign** of the spin is the other half of the choice, and it is
    // free: `−s` is `+s` mirrored in the vertical plane through the sunbeam,
    // so it puts the prism on the other side of the sun-line at the same
    // height and the same throw. It is worth a constant because the two sides
    // frame completely differently — on one, the prism stands directly behind
    // the shade stone from every camera that can see the shadow it makes.
    let (mut lo, mut hi) = (0.0f64, std::f64::consts::PI);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if height_at(PRISM_HAND * mid) < PRISM_UP { lo = mid } else { hi = mid }
    }
    let spin = PRISM_HAND * 0.5 * (lo + hi);
    let (prism, out) = kit::place_for(band, d, delta, spin, PRISM_THROW);

    // ---- the mirror -------------------------------------------------------
    // No cone: a mirror can send the sun anywhere, so it is simply put down
    // where the frame wants it and turned until it hits its mark.
    // How far across the sunbeam a point is from the shade stone's own centre
    // line. `across · d = 0` by construction, so this number is **invariant
    // along the sunbeam**: a point is in the stone's shadow exactly when its
    // `shade_u` is inside the stone's half width and it is down-sun of it,
    // and that is one dot product rather than a ray cast.
    //
    // Every tool that has to be *in the sun* is placed by it.
    let shade_u = |p: Point3| across.dot(p - lip);

    let mirror = {
        // On the far side of the door from the prism, and that is not taste:
        // the prism's beam is three metres of thin air on the up-sun side and
        // an early version put the mirror's 250 mm disc straight through it,
        // which cut the blue half out of the spectrum and left a red smear
        // that measured 60 mm.
        //
        // Then it spent three renders in the **shade stone's own shadow**,
        // which is the failure this whole picture is prone to and the reason
        // `shade_u` exists: the stone is there to darken a strip of door, the
        // strip is a wedge of shadow a metre and a half wide reaching four
        // metres down the beach, and a mirror standing in it reflects a
        // patch of nothing at all. So the mirror is pushed along `across`
        // until its whole disc is clear of that wedge, and no further —
        // every millimetre past it lengthens the throw and dims the patch.
        let clear = 0.5 * SHADE_W + 0.5 * kit::MIRROR_D + 180.0 - shade_u(patch);
        // …and stood back up-sun as well, which costs nothing in `shade_u`
        // (that coordinate is invariant along the beam) and buys the one
        // thing that decides how bright the patch is: the angle it meets the
        // door at. Straight across, the bounce arrives at seventy degrees to
        // the stone and smears its 250 mm of sun over a square metre.
        let p = patch + across * clear.max(MIRROR_THROW) - along * 260.0;
        Point3::new(p.x, p.y, stage::sand_z(p.y) + 520.0)
    };
    let bounce = (patch - mirror).normalize();

    // ---- the lens ---------------------------------------------------------
    // A dark slab of the door's own stone lying on the sand, and the lens
    // propped over it: the burn is on the slab, `LENS_HIGH/sin 22°` down-sun
    // of the glass.
    //
    // It is *near the prism* and that is deliberate — see [`PRISM_THROW`].
    // It also cannot be anywhere in shade: a lens's caustic travels down the
    // sunbeam, so it lands only where the sun already lands, and the way to
    // make a burn read is not to darken its surroundings but to shrink its
    // own patch. At 640 mm over the slab it is 70 mm across, which is ten
    // times bare sun, and on stone at an albedo of 0.24 that is the
    // brightest thing in the frame.
    let burn = {
        let p = Point3::new(BAND_AT[0], 0.0, 0.0) - along * 1450.0 + across * 620.0;
        Point3::new(p.x, p.y, stage::sand_z(p.y) + 150.0)
    };
    let lens_throw = LENS_HIGH / (-d.z);
    let lens = burn - d * lens_throw;
    // The prop: a low flat slab under the burn, and a second one the lens
    // leans against, its top edge exactly at the glass's rim so the two
    // touch. A tool floating a hand's breadth off its prop is the one thing
    // that would give the whole picture away as a render.
    let prop = lens - d.cross(across).normalize() * 0.0 - Vec3::new(0.0, 0.0, 0.5 * kit::LENS_D);

    let mut cast = Cast::default();
    studio.set(&mut cast)?;
    let props = kosm::build::build(&Params::default(), move |b| {
        // the shade stone, stood in the sun's own frame so its face is square
        // to the beam and its shadow is a rectangle rather than a lozenge
        // The cliff's rock, not the door's stone: it is a piece of the cove
        // that has fallen out of the cliff, it reads cool against the warm
        // props and the warm door, and a 1.9 m slab in the middle of the
        // frame that matches everything around it is a hole in the picture.
        b.body("shade").material("rock").add(
            b.box_at([-0.5 * SHADE_W, 0.5 * SHADE_W], [0.0, SHADE_T], [-400.0, shade_h])
                .rotate_z(shade_bearing)
                .at(lip.x, lip.y, stage::sand_z(lip.y)),
        );
        let rock = b.body("props");
        rock.material("stone");
        // the burn's slab, and the block the lens leans on
        // The slab the burn lands on. Its **top is `burn.z` exactly**: the
        // caustic map is gathered inside a 14 mm radius, so a slab whose top
        // is twenty millimetres under the point the arithmetic quotes reads
        // as a caustic that is not there.
        rock.add(
            b.box_at([-640.0, 640.0], [-1000.0, 1000.0], [-600.0, 0.0])
                .rotate_z(shade_bearing)
                .at(burn.x, burn.y, burn.z),
        );
        rock.add(
            b.box_at([-320.0, 320.0], [-200.0, 200.0], [-1200.0, 0.0])
                .rotate_z(shade_bearing)
                .at(prop.x, prop.y, prop.z),
        );
        // and a stone apiece under the prism and the mirror
        rock.add(b.sphere(150.0).at(prism.x + 40.0, prism.y - 60.0, prism.z - 190.0));
        rock.add(b.sphere(180.0).at(mirror.x + 30.0, mirror.y - 80.0, mirror.z - 290.0));
    })?;
    cast.add(&props, &|_| Some(Transform::identity()))?;

    studio.lens(&mut cast, lens, stage::sun_dir())?;
    studio.prism(&mut cast, kit::prism_frame(prism, d, out))?;
    studio.mirror(&mut cast, kit::mirror_frame(mirror, d, bounce))?;

    let picture = cast.picture();
    let t0 = Instant::now();
    // Four times the budget the other stills get, because this one is asking
    // a photon map for *colour* and not merely for a bright patch: a band
    // whose red end and blue end have to be told apart needs enough photons
    // in each end of it to average, and the emission disc that covers two
    // tools two metres apart throws most of them away. Four million is a
    // couple of seconds.
    let map = rune_light(&picture, photons.max(600_000) * 4, 14.0);
    let traced = t0.elapsed().as_secs_f64();

    // ---- what actually landed ---------------------------------------------
    let cos_face = out.dot(-face_n).abs();
    println!(
        "hero tools: prism at ({:+.0}, {:+.0}, {:.0}) throwing {:.2} m at {:+.1}° elevation; δ = {:.1}°, spread {:.2}°, slit {:.0} mm → a {:.0} mm band at {:.2} of bare sun",
        prism.x,
        prism.y,
        prism.z,
        PRISM_THROW / 1000.0,
        out.z.asin().to_degrees(),
        delta.to_degrees(),
        spread.to_degrees(),
        kit::PRISM_SLIT,
        kit::band_length(kit::PRISM_SLIT, PRISM_THROW, cos_face),
        kit::band_brightness(kit::PRISM_SLIT, PRISM_THROW),
    );
    println!(
        "hero tools: shade stone {:.0} mm tall {:.2} m up-sun, shadow to z = {:.0} mm; lens {:.0} mm up throws {:.0} mm to a {:.0} mm spot ({:.0}× bare sun); mirror {:.2} m from its patch",
        shade_h,
        SHADE_BACK / 1000.0,
        SHADE_SHADOW_TOP,
        LENS_HIGH,
        lens_throw,
        kit::patch_at(lens_throw),
        (kit::LENS_D / kit::patch_at(lens_throw)).powi(2),
        MIRROR_THROW / 1000.0,
    );
    let watts = |p: Point3, n: Vec3| {
        let e = map.irradiance(p + n * 0.5, n);
        (e[0] + e[1] + e[2]) as f64 / 3.0
    };
    let sand_n = Vec3::new(0.0, -stage::SLOPE, 1.0).normalize();
    println!(
        "hero tools irradiance: band {:.2}, patch {:.2}, burn {:.2}, bare door {:.2}",
        watts(band, face_n),
        watts(patch, face_n),
        watts(burn, sand_n),
        watts(Point3::new(600.0, 0.0, 760.0), face_n),
    );
    let sep = scan_face(&map, band, face_n, std::env::var("HERO_SCAN").is_ok());
    println!(
        "hero tools spectrum: the band is {:.0} mm across, red-to-blue centroids {:.0} mm apart, ends {:.2} and {:.2} in R/B",
        sep.extent, sep.separation, sep.red_end, sep.blue_end
    );

    // Low and out on the sunward side, looking along the door's face rather
    // than square at it: square on, the shade stone stands in front of its own
    // shadow and hides the band it made. From the side the stone is on the
    // left of the frame, its shadow runs off to the right of it, and the three
    // marks of light lie in a row across the stone in shade.
    let eye = {
        let p = Point3::new(BAND_AT[0], 0.0, 0.0) + across * 4400.0 - along * 1500.0;
        Point3::new(p.x, p.y, stage::sand_z(p.y) + 1280.0)
    };
    let look_at = Point3::new(-1000.0, -820.0, 760.0);
    let camera = stage::look(eye, look_at, 44.0);
    if std::env::var("HERO_SCAN").is_ok() {
        // Where everything is, as the camera sees it: a bearing off the view
        // axis and a distance. Framing four tools and three marks of light in
        // one frame is the hard part of this picture, and it is far quicker
        // to read as seven numbers than as seven renders.
        let axis = (look_at - eye).normalize();
        let bear = |p: Point3| {
            let v = p - eye;
            let off = (v.y.atan2(v.x) - axis.y.atan2(axis.x)).to_degrees();
            ((off + 540.0) % 360.0 - 180.0, v.norm() / 1000.0)
        };
        for (name, p) in [
            ("band", band),
            ("patch", patch),
            ("burn", burn),
            ("shade", Point3::new(lip.x, lip.y, 0.0)),
            ("prism", prism),
            ("mirror", mirror),
            ("lens", lens),
        ] {
            let (off, dist) = bear(p);
            let u = shade_u(p);
            let lit = if u.abs() > 0.5 * SHADE_W { "sun " } else { "SHADE" };
            println!("  frame {name:7} {off:+6.1}° off axis at {dist:.2} m, shade_u {u:+7.0} ({lit})");
        }
    }

    // Three times the samples the other stills get and none of the
    // integrator's usual thrift — see [`stage::options_rare`] for why the
    // mirror's patch needs both.
    let rare = spp.max(256) * 3;
    let t1 = Instant::now();
    shoot_with(&picture, &camera, SIZE, &stage::options_rare(rare, SEED), Some(&map))
        .save(dir.join("tools.png"))?;
    println!(
        "hero tools: {} photons in {:.1} s, {rare} spp (non-adaptive) in {:.1} s → {}",
        map.len(),
        traced,
        t1.elapsed().as_secs_f64(),
        dir.join("tools.png").display()
    );
    Ok(())
}

/// What a scan of the caustic map over the door's face found.
struct Spectrum {
    /// How far the lit cells reach, millimetres.
    extent: f64,
    /// How far apart the red channel's centroid and the blue channel's are.
    /// This is the number that separates a rainbow from a white bar: a bar
    /// has both centroids in the same place.
    separation: f64,
    /// The red-to-blue ratio at the red end of the band, and the blue-to-red
    /// ratio at the blue end. Both above one is a spectrum.
    red_end: f64,
    blue_end: f64,
}

/// Walk a grid of the door's face and measure what the photon pass put on it.
///
/// The colour separation is the measurement the picture is judged on before
/// anything is rendered, because a band that is present and white and a band
/// that is present and split look the same in a thumbnail and cost the same
/// number of photons. Weight each cell by its own red and its own blue,
/// take the two centroids, and the distance between them is the spectrum.
fn scan_face(map: &CausticMap, about: Point3, n: Vec3, verbose: bool) -> Spectrum {
    let (half, step) = (900.0f64, 12.0f64);
    let mut cells: Vec<(f64, f64, [f32; 3])> = Vec::new();
    let mut peak = 0.0f64;
    let k = (half / step) as i64;
    for i in -k..=k {
        for j in -k..=k {
            let (x, z) = (about.x + i as f64 * step, about.z + j as f64 * step);
            let e = map.irradiance(Point3::new(x, about.y, z) + n * 0.5, n);
            let w = (e[0] + e[1] + e[2]) as f64 / 3.0;
            peak = peak.max(w);
            if w > 1e-4 {
                cells.push((x, z, e));
            }
        }
    }
    let floor = 0.10 * peak;
    cells.retain(|(_, _, e)| (e[0] + e[1] + e[2]) as f64 / 3.0 > floor);
    if cells.is_empty() {
        return Spectrum { extent: 0.0, separation: 0.0, red_end: 0.0, blue_end: 0.0 };
    }
    let centroid = |c: usize| {
        let w: f64 = cells.iter().map(|(_, _, e)| e[c] as f64).sum();
        let x: f64 = cells.iter().map(|(x, _, e)| x * e[c] as f64).sum::<f64>() / w.max(1e-12);
        let z: f64 = cells.iter().map(|(_, z, e)| z * e[c] as f64).sum::<f64>() / w.max(1e-12);
        (x, z)
    };
    let (rx, rz) = centroid(0);
    let (bx, bz) = centroid(2);
    let separation = ((rx - bx).powi(2) + (rz - bz).powi(2)).sqrt();
    // the band's own axis, taken as the line between the two centroids when
    // there is one, and as the longest chord when there is not
    let axis = if separation > 1e-6 {
        ((rx - bx) / separation, (rz - bz) / separation)
    } else {
        (1.0, 0.0)
    };
    let s: Vec<f64> = cells.iter().map(|(x, z, _)| x * axis.0 + z * axis.1).collect();
    let (lo, hi) = s.iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    // the ends: the tenth of the band furthest each way along that axis
    let end = |take_high: bool| {
        let cut = if take_high { hi - 0.15 * (hi - lo) } else { lo + 0.15 * (hi - lo) };
        let mut sum = [0.0f64; 3];
        for (i, (_, _, e)) in cells.iter().enumerate() {
            let keep = if take_high { s[i] >= cut } else { s[i] <= cut };
            if keep {
                for c in 0..3 {
                    sum[c] += e[c] as f64;
                }
            }
        }
        sum
    };
    let red = end(true);
    let blue = end(false);
    if verbose {
        let mut best: Vec<(f64, f64, f64, [f32; 3])> = cells
            .iter()
            .map(|(x, z, e)| ((e[0] + e[1] + e[2]) as f64 / 3.0, *x, *z, *e))
            .collect();
        best.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("  scan: {} lit cells over {:.0} mm, floor {floor:.3}", cells.len(), hi - lo);
        for (w, x, z, e) in best.into_iter().take(24) {
            println!("  scan {w:8.3} at ({x:+7.0}, {z:7.0})  rgb {:6.3} {:6.3} {:6.3}", e[0], e[1], e[2]);
        }
        println!("  scan red centroid ({rx:+.0}, {rz:.0}), blue centroid ({bx:+.0}, {bz:.0})");
    }
    // Clamped, because a band that separates *completely* divides by a blue
    // that is exactly zero and prints fourteen digits of nothing.
    let ratio = |a: f64, b: f64| (a / b.max(1e-4 * a.max(1e-12))).min(999.0);
    Spectrum {
        extent: hi - lo,
        separation,
        red_end: ratio(red[0], red[2]),
        blue_end: ratio(blue[2], blue[0]),
    }
}

// ---- the turntable and the sheet --------------------------------------------

/// `turntable.png`: 0°, 120°, 240°, three panels side by side in one plate.
///
/// The camera orbits and the hero does not, so the three panels are three
/// views of one pose and any asymmetry in the figure is the figure's own.
fn turntable(studio: &Studio, dir: &Path, spp: usize) -> anyhow::Result<()> {
    let feet_y = -3400.0;
    let feet = Point3::new(-5200.0, feet_y, stage::sand_z(feet_y));
    let stance = Pose::STAND.stand(feet, 0.0);
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
        let radius = 2250.0;
        let eye = Point3::new(feet.x - s * radius, feet.y - c * radius, feet.z + 780.0);
        let camera = stage::look(eye, Point3::new(feet.x, feet.y, feet.z + 600.0), 34.0);
        let film = pathtrace::render(&picture, &camera, panel.0, panel.1, &stage::options(spp, SEED));
        paste(&mut plate, &stage::to_image(&film), (i as u32 * panel.0, 0));
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

/// One cell of the character sheet.
struct View {
    name: &'static str,
    /// Where the camera is, as a turn about the hero from dead ahead: 0 is
    /// the front, 180 the back, +90 the hero's own left.
    turn: f64,
    dist: f64,
    /// The eye's height and what it looks at, both as a fraction of the
    /// hero's height, so the panels stay framed if the figure's proportions
    /// move again.
    eye: f64,
    look: f64,
    vfov: f64,
    /// Whether the hero is holding the lens up in this panel.
    holding: bool,
}

/// Which way the sheet's hero faces, as a world bearing.
///
/// The sun is at 250°, so a hero facing 305° takes it 55° off its own front:
/// far enough round that the front panel has a lit side and a shaded side and
/// a nose of shadow between them, and near enough that the three-quarter has
/// the key on the cheek it turns to us. Every panel is that one light — a
/// turnaround whose panels are lit differently is a turnaround of six
/// characters.
const SHEET_FACING_DEG: f64 = 305.0;

fn sheet_views() -> Vec<View> {
    // The four turnaround panels are the *same* camera at four bearings, to
    // the digit: a turnaround whose panels are framed differently is four
    // drawings of four characters. The face panel and the kit panel are the
    // two that are allowed to differ, and they differ only in distance.
    let round = |name, turn| View { name, turn, dist: 2500.0, eye: 0.56, look: 0.50, vfov: 34.0, holding: false };
    vec![
        round("front", 0.0),
        round("three-quarter", 40.0),
        round("back", 180.0),
        round("side", 90.0),
        View { name: "face", turn: 22.0, dist: 1600.0, eye: 0.88, look: 0.80, vfov: 30.0, holding: false },
        View { name: "kit", turn: -30.0, dist: 2500.0, eye: 0.56, look: 0.50, vfov: 34.0, holding: true },
    ]
}

/// Where the hero holds the lens for the sheet's last panel: up and out on
/// its right, turned so the disc faces the camera.
const SHEET_LENS_HAND: [f64; 3] = [386.0, 214.0, 604.0];

/// One panel of the sheet, rendered.
///
/// `with_hero` is what the silhouette test turns off: the same camera over
/// the same sand with nobody in it, so "is the hero in this panel" is a pixel
/// difference and not a guess about colour.
fn sheet_panel(
    studio: &Studio,
    view: &View,
    size: (u32, u32),
    spp: usize,
    with_hero: bool,
) -> anyhow::Result<image::RgbaImage> {
    // On the **middle of the sand slab**, and that is worth a comment: the
    // beach is a finite box, so its far edge is the horizon, and a hero
    // standing off-centre has that edge nearer on one side than the other.
    // Orbit a camera round it and the horizon line walks up and down the
    // frame from panel to panel, which on a contact sheet reads as six
    // different afternoons. At the slab's centre every bearing sees the same
    // thirteen metres of sand.
    let feet = Point3::origin();
    let yaw = (SHEET_FACING_DEG - 90.0).to_radians();
    let pose = if view.holding {
        Pose { right: SHEET_LENS_HAND, head_tilt: 12.0, ..Pose::STAND }
    } else {
        Pose::STAND
    };
    let stance = pose.stand(feet, yaw);

    let mut cast = Cast::default();
    studio.sand(&mut cast)?;
    if with_hero {
        studio.hero(&mut cast, &stance)?;
        if view.holding {
            let held = stance.at(SHEET_LENS_HAND);
            // turned to the camera, so the sheet's last panel shows the disc
            let bearing = (SHEET_FACING_DEG + view.turn).to_radians();
            let axis = Vec3::new(bearing.cos(), bearing.sin(), 0.30).normalize();
            // The glass rises *above* the fist rather than hanging below it:
            // a magnifier held up to look through, which is the one verb this
            // game has, and which keeps the disc clear of the belt.
            let up_in_plane = (Vec3::new(0.0, 0.0, 1.0) - axis * axis.z).normalize();
            studio.lens(&mut cast, held + up_in_plane * (0.5 * kit::LENS_D), axis)?;
        }
    }
    let picture = cast.picture();

    let h = stance.rig.height;
    let bearing = (SHEET_FACING_DEG + view.turn).to_radians();
    let eye = Point3::new(
        feet.x + bearing.cos() * view.dist,
        feet.y + bearing.sin() * view.dist,
        feet.z + view.eye * h,
    );
    let camera = stage::look(eye, Point3::new(feet.x, feet.y, feet.z + view.look * h), view.vfov);
    let film = pathtrace::render(&picture, &camera, size.0, size.1, &stage::options(spp, SEED));
    Ok(stage::to_image(&film))
}

/// `sheet.png`: the character sheet — six views on plain sand under one sun.
///
/// Three across and two down, 640 × 540 each. Front, three-quarter, back and
/// side are the turnaround; the fifth is the face at portrait distance; the
/// sixth is the hero holding the lens, because a character who carries the
/// game's one verb ought to be shown carrying it.
fn sheet(studio: &Studio, dir: &Path, spp: usize) -> anyhow::Result<()> {
    let views = sheet_views();
    let panel = (SHEET.0 / 3, SHEET.1 / 2);
    let mut plate = image::RgbaImage::new(SHEET.0, SHEET.1);
    let t0 = Instant::now();
    for (i, view) in views.iter().enumerate() {
        let img = sheet_panel(studio, view, panel, spp, true)?;
        let (col, row) = (i as u32 % 3, i as u32 / 3);
        paste(&mut plate, &img, (col * panel.0, row * panel.1));
    }
    plate.save(dir.join("sheet.png"))?;
    println!(
        "hero sheet: {} at {}×{}, {spp} spp in {:.1} s → {}",
        views.iter().map(|v| v.name).collect::<Vec<_>>().join(", "),
        panel.0,
        panel.1,
        t0.elapsed().as_secs_f64(),
        dir.join("sheet.png").display()
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
    let (delta, spread) = kit::prism_deviation();
    let (n_d, abbe) = kit::prism_glass();
    println!(
        "hero kit: a {:.0} mm lens of two R = {:.1} mm N-BK7 caps {:.2} mm apart, {:.2} mm thick, f = {:.0} mm at n_d = {}; a {:.0} mm {} prism (n_d = {n_d}, V = {abbe}) deviating {:.1}° and spreading {:.2}° through a {:.0} mm slit; a {:.0} mm mirror at roughness 0.02",
        kit::LENS_D,
        r,
        2.0 * a,
        d,
        kit::focal(r, 0.5 * kit::LENS_D, stage::N_D),
        stage::N_D,
        kit::PRISM_SIDE,
        kit::PRISM_GLASS,
        delta.to_degrees(),
        spread.to_degrees(),
        kit::PRISM_SLIT,
        kit::MIRROR_D,
    );

    let t0 = Instant::now();
    if want("door") || want("door_face") {
        let (step, picture, map, traced) = doorstep_picture(&studio, photons)?;
        if want("door") {
            door(&dir, &step, &picture, &map, traced, spp)?;
        }
        if want("door_face") {
            door_face(&dir, &step, &picture, &map, spp)?;
        }
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
    if want("sheet") {
        sheet(&studio, &dir, spp)?;
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
        let step = doorstep();
        let (stance, lens) = (&step.stance, step.lens);
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
        assert!(stance.facing().dot(to_aim) > 0.999, "the hero is not facing its own mark");
        // the lens is within the hero's reach from its *leaned* shoulder,
        // which is the shoulder the arm is actually drawn from
        for side in [1.0, -1.0] {
            let shoulder = stance.at(stance.rig.shoulder(side));
            let hand = stance.at(stance.rig.hand(side));
            assert!((shoulder - hand).norm() <= stance.rig.reach() + 1e-9, "the arm is short");
        }
        // and it is leaning into the door with its weight back
        assert!(stance.rig.torso_lean > 8.0, "the hero is standing to attention");
        assert!(stance.rig.pelvis_y < 0.0, "the weight is not on the back foot");
    }

    /// The glass is outside the hero's own head, and by a margin.
    ///
    /// This is the failure that cost a whole render and looked like nothing:
    /// the head went from 410 mm across to 456, the lens stayed on the centre
    /// line at arm's length, and it ended up *inside the skull*. The caustic
    /// pass duly found the refractor, aimed a million and a half photons at
    /// it, and deposited **zero** — every one of them hit hood before it hit
    /// glass — so `door.png` came back with a bare door and no error
    /// anywhere. A hundred millimetres of clearance is cheap insurance.
    #[test]
    fn the_lens_is_held_clear_of_the_heros_own_head() {
        let step = doorstep();
        let r = &step.stance.rig;
        // in the hero's own frame: the head, the hood's crown, and the glass
        let neck = r.neck();
        let tip = |p: [f64; 3]| {
            let (s, c) = r.head_tilt.to_radians().sin_cos();
            let (y, z) = (p[1], p[2]);
            [p[0], neck[1] + y * c - z * s, neck[2] + y * s + z * c]
        };
        let dz = r.head_z - r.neck_z;
        let head = tip([0.0, 0.0, dz]);
        let crown = tip([0.0, -0.18 * r.head_r, dz + 0.06 * r.head_r]);
        // the lens, back out of the world and into the hero's frame
        let (s, c) = step.stance.yaw.sin_cos();
        let v = step.lens - step.stance.feet;
        let local = [v.x * c + v.y * s, -v.x * s + v.y * c, v.z];
        let gap = |centre: [f64; 3], radius: f64| {
            let d = ((local[0] - centre[0]).powi(2)
                + (local[1] - centre[1]).powi(2)
                + (local[2] - centre[2]).powi(2))
            .sqrt();
            d - 0.5 * kit::LENS_D - radius
        };
        assert!(gap(head, r.head_r) > 100.0, "the glass is {:.0} mm off the skull", gap(head, r.head_r));
        assert!(
            gap(crown, r.head_r + 28.0) > 100.0,
            "the glass is {:.0} mm off the hood",
            gap(crown, r.head_r + 28.0)
        );
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

    /// The stills are the files the character sheet is, and the studio builds
    /// without a cove.
    #[test]
    fn the_studio_builds_the_stage_the_hero_and_the_hardware() -> anyhow::Result<()> {
        let studio = Studio::new()?;
        let mut cast = Cast::default();
        studio.set(&mut cast)?;
        let before = cast.len();
        assert!(before >= 5, "the stage came out as {before} objects");
        studio.hero(&mut cast, &doorstep().stance)?;
        assert!(cast.len() > before + 15, "the hero is more than a handful of parts");
        studio.lens(&mut cast, Point3::new(0.0, -1000.0, 900.0), stage::sun_dir())?;
        // …and the picture the whole of that makes has exactly one sun
        let picture = cast.picture();
        assert!(picture.sun.is_some());
        assert!(picture.lights.is_empty(), "the cove's light is the sky and the sun");
        // the sheet's backdrop is sand and nothing else: no door, no cliff,
        // no keyhole to give away where the turnaround was shot
        let mut plain = Cast::default();
        studio.sand(&mut plain)?;
        assert_eq!(plain.len(), 1, "the sheet's backdrop is one slab of sand");
        Ok(())
    }

    /// Every one of the sheet's six panels has a hero in it.
    ///
    /// Not "has some non-sand pixels": the same camera over the same beach is
    /// rendered twice, once with the figure and once without, and the panel
    /// passes when enough pixels differ. That is a silhouette test and it
    /// cannot be fooled by a colour that happens to look like cloth — which
    /// matters most for the panel it is really aimed at, the back, where a
    /// hero facing away from the camera at 1.1 m tall in a 640-wide frame is
    /// the easiest thing in this file to lose off the edge.
    ///
    /// Tiny and cheap: 128 × 108 at 2 spp is a thousandth of the sheet's own
    /// cost and answers exactly the question asked.
    #[test]
    fn the_sheet_has_the_hero_in_all_six_panels() -> anyhow::Result<()> {
        let studio = Studio::new()?;
        let size = (128u32, 108u32);
        let total = (size.0 * size.1) as f64;
        for view in sheet_views() {
            let with = sheet_panel(&studio, &view, size, 2, true)?;
            let without = sheet_panel(&studio, &view, size, 2, false)?;
            let hero = with
                .pixels()
                .zip(without.pixels())
                .filter(|(a, b)| {
                    (0..3).any(|c| (a.0[c] as i32 - b.0[c] as i32).abs() > 10)
                })
                .count() as f64;
            let share = hero / total;
            assert!(
                share > 0.05,
                "`{}` is only {:.1}% hero — it has fallen out of its own frame",
                view.name,
                100.0 * share
            );
            assert!(
                share < 0.75,
                "`{}` is {:.1}% hero — the camera is inside the costume",
                view.name,
                100.0 * share
            );
        }
        Ok(())
    }
}
