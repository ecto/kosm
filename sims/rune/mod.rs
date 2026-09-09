//! The rune: the cove, a being of glass, and one door that opens to light.
//!
//! The first slice of Rune. [`scene`] is the level — sand rising out of the
//! sea to a cliff, a few rocks, and a door set into the cliff face — and this
//! module is the level read as numbers: every knob resolved once, at the
//! boundary, from the document's millimetres into the metres and radians
//! everything downstream works in.
//!
//! It is built the way the skatepark is. The document evaluates to roots, the
//! roots bake to a `kosm-scan` map directory ([`bake`]), and a body stands on
//! the baked field through the same SDF contact path the K1's feet use
//! ([`sim::step_on_sdf`]). What the cove adds over the park is that its ground
//! is a *plane*: an exact signed distance to a plane is linear, and trilinear
//! interpolation of a linear function is exact, so a 100 mm cell costs the
//! beach nothing. That is what [`tests`] checks first, before a being or a
//! caustic is put on it.
//!
//! ```text
//! kosm run rune           bake, solve, sweep, and one still under out/
//! kosm run rune --view    the cove in a window (needs --features view)
//! ```
//!
//! Metres and z up in here; millimetres only in the document. The waterline is
//! `y = -cove/2` at `sea_z`, the sand rises toward +y at `beach_slope`, and the
//! cliff face — the plane the door and its aperture live in — faces -y.

use std::fs;
use std::path::Path;

use kosm_scan::stl;
use phyz_math::Vec3;

use kosm::build::{Built, Params};
use kosm::scene::MM;
use crate::skatepark::{self, BakeOpts, Part};

pub mod bake;
pub mod being;
pub mod hint;
pub mod materials;
pub mod render;
pub mod rune;
pub mod scene;
pub mod sim;
/// The cove in a window, and the CPU tier behind it. `--view`.
#[cfg(feature = "view")]
pub mod game;
#[cfg(test)]
mod rune_tests;
#[cfg(test)]
mod tests;

/// The cove's knobs, resolved to simulation units: metres, radians, seconds.
pub struct CoveScene {
    pub authored: Built,
    /// The level is a `cove` square centred on x = 0, sea along -y.
    pub cove: f64,
    pub beach_slope: f64,
    pub sea_z: f64,
    /// How far past the waterline the sand carries on, at the same grade: the
    /// seabed the being wades over. The cove is closed along -y by the reef at
    /// the far end of it, not by the sand running out.
    pub seabed: f64,
    /// The inner face of the headlands that close +/-x. Rock beyond it.
    pub headland_x: f64,
    /// The shore break: how fast the water runs shoreward, m/s. The one knob
    /// the sea has, and the one that decides how deep the being can wade.
    pub surf: f64,
    /// The cliff above the sand at its foot, and how deep it is along y.
    pub cliff_h: f64,
    pub cliff_t: f64,
    pub door_w: f64,
    pub door_h: f64,
    pub door_t: f64,
    pub door_x: f64,
    /// The keyhole: a disc on the door's face, `aperture_z` above the sand
    /// there. Not geometry — the face is solid stone.
    pub aperture_r: f64,
    pub aperture_z: f64,
    /// The score that opens the door.
    pub open_frac: f64,
    /// The hint, as light. The aperture's rim is an emissive ring on the door
    /// face whose radiance is `glow_floor + glow_gain · score`: the floor is
    /// what makes the keyhole findable from across the beach at a score of
    /// zero, and the gain is what makes holding it obvious. `rim_w` is how
    /// wide the ring is outside `aperture_r`.
    pub glow_floor: f64,
    pub glow_gain: f64,
    pub rim_w: f64,
    /// The glint: after `glint_after` seconds without the score rising, a
    /// small emissive sphere of radius `glint_r` appears on the sand
    /// `glint_step` along the horizontal part of the hint's gradient, and
    /// shines at `glint_glow`.
    pub glint_after: f64,
    pub glint_step: f64,
    pub glint_r: f64,
    pub glint_glow: f64,
    /// Toward the sun: azimuth about z from +x toward +y, elevation above the
    /// horizon. Radians.
    pub sun_az: f64,
    pub sun_el: f64,
    pub being_r: f64,
    pub being_h: f64,
    pub n_d: f64,
    pub walk_mps: f64,
    pub spawn_x: f64,
    pub spawn_y: f64,
    /// The solved pose, as the document last recorded it: metres and radians.
    pub solution_x: f64,
    pub solution_y: f64,
    pub solution_tilt: f64,
    pub cell: f64,
    pub pad: f64,
}

impl CoveScene {
    /// The cove at its authored defaults.
    pub fn bundled() -> anyhow::Result<Self> {
        Self::of(scene::scene(&Params::default())?)
    }

    /// The cove's knobs, read off whatever level was built. The skatepark's
    /// `SkateparkScene::of` is the same idea: the level is a Rust function,
    /// and this is the one place its millimetres become metres.
    pub fn of(a: Built) -> anyhow::Result<Self> {
        let s = Self {
            cove: a.millimetres("cove_mm")?,
            beach_slope: a.parameter("beach_slope")?,
            sea_z: a.millimetres("sea_z_mm")?,
            seabed: a.parameter_or("seabed_mm", 14000.0) * MM,
            headland_x: a.parameter_or("headland_x_mm", 15000.0) * MM,
            surf: a.parameter_or("surf_mps", 4.0),
            cliff_h: a.millimetres("cliff_h_mm")?,
            cliff_t: a.millimetres("cliff_t_mm")?,
            door_w: a.millimetres("door_w_mm")?,
            door_h: a.millimetres("door_h_mm")?,
            door_t: a.millimetres("door_t_mm")?,
            door_x: a.millimetres("door_x_mm")?,
            aperture_r: a.millimetres("aperture_r_mm")?,
            aperture_z: a.millimetres("aperture_z_mm")?,
            open_frac: a.parameter_or("open_frac", 0.3),
            glow_floor: a.parameter_or("glow_floor", 0.8),
            glow_gain: a.parameter_or("glow_gain", 4.0),
            rim_w: a.parameter_or("rim_w_mm", 30.0) * MM,
            glint_after: a.parameter_or("glint_after_s", 30.0),
            glint_step: a.parameter_or("glint_step_m", 1.5),
            glint_r: a.parameter_or("glint_r_mm", 60.0) * MM,
            glint_glow: a.parameter_or("glint_glow", 6.0),
            sun_az: a.parameter("sun_az_deg")?.to_radians(),
            sun_el: a.parameter("sun_el_deg")?.to_radians(),
            being_r: a.millimetres("being_r_mm")?,
            being_h: a.millimetres("being_h_mm")?,
            n_d: a.parameter("n_d")?,
            walk_mps: a.parameter_or("walk_mps", 1.4),
            spawn_x: a.millimetres("spawn_x_mm")?,
            spawn_y: a.millimetres("spawn_y_mm")?,
            solution_x: a.parameter_or("solution_x_mm", 0.0) * MM,
            solution_y: a.parameter_or("solution_y_mm", 0.0) * MM,
            solution_tilt: a.parameter_or("solution_tilt_deg", 0.0).to_radians(),
            cell: a.millimetres("sdf_cell_mm")?,
            pad: a.millimetres("sdf_pad_mm")?,
            authored: a,
        };
        anyhow::ensure!(s.cove > 0.0 && s.cell > 0.0 && s.pad >= 0.0, "the cove needs a positive size and cell and a non-negative pad");
        anyhow::ensure!(s.beach_slope > 0.0, "the beach must rise out of the sea");
        anyhow::ensure!(s.seabed > 0.0, "the sea needs a floor: seabed_mm carries the sand past the waterline");
        anyhow::ensure!(s.headland_x > 0.0 && s.headland_x < s.cove / 2.0, "the headlands must stand inside the cove square");
        anyhow::ensure!(s.being_h > 2.0 * s.being_r, "being_h_mm is the capsule's whole height, so it must clear two radii");
        anyhow::ensure!(s.aperture_z + s.aperture_r < s.door_h, "the aperture must be inside the door");
        anyhow::ensure!(s.door_w.min(s.door_h) > 0.0 && s.door_t > 0.0, "the door needs a size");
        Ok(s)
    }

    /// The waterline: the sea meets the sand at this y.
    pub fn waterline(&self) -> f64 {
        -self.cove / 2.0
    }

    /// The seaward edge of the seabed: where the sand, and with it the baked
    /// field, finally stops. The reef stands just inside it.
    pub fn seabed_y(&self) -> f64 {
        self.waterline() - self.seabed
    }

    /// The top of the seabed at that edge — the deepest ground in the cove, and
    /// the floor the bake is sampled down from.
    pub fn seabed_z(&self) -> f64 {
        self.sand_z_at(0.0, self.seabed_y())
    }

    /// The floor of the sampled volume: a metre under the deepest ground there
    /// is. What is under *that* is [`being::Cove`]'s net and nothing else.
    ///
    /// [`being::Cove`]: super::being::Cove
    pub fn floor(&self) -> f64 {
        self.seabed_z().min(self.sea_z) - 1.0
    }

    /// The cliff's face — the plane the door and its aperture live in.
    pub fn cliff_face_y(&self) -> f64 {
        self.cove / 2.0 - self.cliff_t
    }

    /// The top of the sand at a point on the beach. The sand is a plane that
    /// meets the sea at the waterline and rises toward +y, so x does not enter
    /// it; it is in the signature because the caller has a point, not a y, and
    /// because the day the beach is not a plane this is the one place to say so.
    pub fn sand_z_at(&self, _x: f64, y: f64) -> f64 {
        self.sea_z + self.beach_slope * (y - self.waterline())
    }

    /// The upward unit normal of the sand.
    pub fn sand_normal(&self) -> Vec3 {
        Vec3::new(0.0, -self.beach_slope, 1.0) / (1.0 + self.beach_slope * self.beach_slope).sqrt()
    }

    /// The sand at the foot of the door: the sill the door stands on.
    pub fn door_sill(&self) -> f64 {
        self.sand_z_at(self.door_x, self.cliff_face_y())
    }

    /// The top of the cliff, above the sand at its own foot.
    pub fn cliff_top(&self) -> f64 {
        self.sand_z_at(0.0, self.cove / 2.0) + self.cliff_h
    }

    /// Where the being starts, its centre standing on the sand.
    pub fn spawn(&self) -> Vec3 {
        Vec3::new(self.spawn_x, self.spawn_y, self.sand_z_at(self.spawn_x, self.spawn_y) + self.being_h / 2.0)
    }

    /// The unit vector *toward* the sun. Azimuth is measured about z from +x
    /// toward +y and elevation up from the horizon, so 200° and 22° is a sun
    /// low over the sea off the -x, -y corner and the light it sends travels
    /// into the cliff face.
    pub fn sun_dir(&self) -> Vec3 {
        let (sa, ca) = self.sun_az.sin_cos();
        let (se, ce) = self.sun_el.sin_cos();
        Vec3::new(ce * ca, ce * sa, se)
    }

    /// The door's face as the rune reads it.
    pub fn door_frame(&self) -> DoorFrame {
        DoorFrame {
            origin: Vec3::new(self.door_x, self.cliff_face_y(), self.door_sill() + self.aperture_z),
            normal: Vec3::new(0.0, -1.0, 0.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 0.0, 1.0),
            radius: self.aperture_r,
        }
    }

    /// The playable footprint: the cove square and the seabed under the water,
    /// from the field's floor to the sand at the cliff.
    pub fn extent(&self) -> (Vec3, Vec3) {
        let h = self.cove / 2.0;
        (Vec3::new(-h, self.seabed_y(), self.floor()), Vec3::new(h, self.cliff_face_y(), self.door_sill()))
    }

    /// The volume the field is sampled over: the cove square *and the seabed*,
    /// from the field's floor to the top of the cliff and the pad. Past it
    /// there is no floor, which is the rule the park has past its padding —
    /// which is why the headlands, the cliff and the reef stand well inside it.
    pub fn volume(&self) -> (Vec3, Vec3) {
        let h = self.cove / 2.0;
        (Vec3::new(-h, self.seabed_y(), self.floor()), Vec3::new(h, h, self.cliff_top() + self.pad))
    }

    pub fn opts(&self) -> BakeOpts {
        BakeOpts { cell: self.cell, pad: self.pad, volume: Some(self.volume()), extent: Some(self.extent()) }
    }

    /// The level's roots in metres. Everything the sun and the being can touch
    /// is ground the bake must see; the door is the one part that moves, so it
    /// is a body later and a hole in the field now.
    pub fn parts(&self) -> anyhow::Result<Vec<Part>> {
        let mut parts = skatepark::parts_of(&self.authored, &|_| true)?;
        for part in &mut parts {
            part.collide = part.name != "door";
        }
        anyhow::ensure!(parts.iter().any(|p| p.name == "door"), "the cove needs a `door` root");
        anyhow::ensure!(parts.iter().any(|p| p.collides()), "the cove has no ground");
        Ok(parts)
    }
}

/// The door's face: the aperture's centre, the normal pointing out into the
/// cove, and the in-face axes the score is measured in. Metres.
#[derive(Clone, Copy, Debug)]
pub struct DoorFrame {
    /// The aperture's centre, in world metres.
    pub origin: Vec3,
    /// Out of the face, into the cove: -y.
    pub normal: Vec3,
    /// In the face, +x.
    pub right: Vec3,
    /// In the face, +z.
    pub up: Vec3,
    /// The aperture's radius.
    pub radius: f64,
}

impl DoorFrame {
    /// A point on the face, from in-face coordinates.
    pub fn at(&self, right: f64, up: f64) -> Vec3 {
        self.origin + self.right * right + self.up * up
    }
}

/// `kosm run rune` — evaluate, draw, bake, roll, solve, sweep, report.
///
/// `--view` opens the window instead (needs `--features view`); everything
/// below is the headless loop an agent closes: files out, numbers on stdout.
pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    #[cfg(feature = "view")]
    if args.view {
        kosm_view::init();
        return game::run(args);
    }
    let out = args.out();
    let scene = CoveScene::bundled()?;
    for w in &scene.authored.warnings {
        eprintln!("cove warning: {w}");
    }
    let dir = out.join("cove");
    fs::create_dir_all(&dir)?;
    let parts = scene.parts()?;
    let tris: Vec<[Vec3; 3]> = parts.iter().flat_map(|p| p.tris.clone()).collect();
    stl::write_binary_stl(&dir.join("cove.stl"), &tris).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let svg = vcad_render::render_svg_str(&scene.authored.document.to_json()?, 2.0).map_err(|e| anyhow::anyhow!(e))?;
    fs::write(dir.join("cove.svg"), svg)?;
    println!(
        "cove: {} roots, {} tris → {} and {}; the sand runs from z = {:.2} m at the waterline to {:.2} m at the cliff, the door's sill is {:.2} m and its aperture is at ({:.2}, {:.2}, {:.2}) m",
        parts.len(),
        tris.len(),
        dir.join("cove.stl").display(),
        dir.join("cove.svg").display(),
        scene.sea_z,
        scene.sand_z_at(0.0, scene.cove / 2.0),
        scene.door_sill(),
        scene.door_frame().origin.x,
        scene.door_frame().origin.y,
        scene.door_frame().origin.z,
    );
    let sun = scene.sun_dir();
    println!("cove sun: toward ({:+.3}, {:+.3}, {:+.3}), {:.0}° azimuth and {:.0}° up", sun.x, sun.y, sun.z, scene.sun_az.to_degrees(), scene.sun_el.to_degrees());
    bake::run(&scene, out)?;
    // The solve prints its answer and writes `out/solved/rune.params`; the
    // level carries the same three numbers as the defaults of its
    // `solution_*` knobs, so a solve that agrees with them changes nothing
    // and a solve that does not says so on stdout.
    rune::solve_and_record(&scene, 100_000, out)?;
    if scene.solution_x != 0.0 || scene.solution_y != 0.0 {
        let climbs = hint::solvable(&scene, 6, 200, &dir.join("solvable.txt"))?;
        let opened = climbs.iter().filter(|c| c.solved).count();
        println!("cove solvable: {opened} of {} spawns open the door -> {}", climbs.len(), dir.join("solvable.txt").display());
    }
    still(&scene, &dir)
}

/// The wide shot, for an agent who cannot open a window.
///
/// [`still`] frames the *solution*, which since step 8 means the doorstep
/// camera: two metres of cliff and a being's shoulder. That is the right
/// picture of the puzzle and it is no picture at all of the level, and the
/// level now has edges — the headlands, the seabed, the reef — that exist
/// precisely so the player cannot leave and that appear nowhere in a close-up.
/// So `kosm run rune` writes a second frame, from the spawn, at a quarter of
/// the still's samples: the far framing, over the shoulder, with the sand
/// running up to the cliff between two headlands of rock.
fn wide(scene: &CoveScene, dir: &Path) -> anyhow::Result<()> {
    let a = &scene.authored;
    let placement = render::Placement::standing(scene, scene.spawn_x, scene.spawn_y, 0.0);
    let (w, h) = (a.parameter_or("render_w", 960.0) as u32, a.parameter_or("render_h", 540.0) as u32);
    let spp = (a.parameter_or("render_spp", 64.0) as usize / 4).max(1);
    let path = dir.join("spawn.png");
    render::frame(scene, &placement, (w, h), spp)?.save(&path)?;
    println!("cove spawn: the being where the player finds it, at ({:+.2}, {:+.2}) m; {w}×{h} at {spp} spp → {}", scene.spawn_x, scene.spawn_y, path.display());
    Ok(())
}

/// One picture of the cove: the being at the solved pose if there is one, at
/// its spawn if there is not, standing on the sand and facing the door.
///
/// The solution is a solved knob and starts at zero, so "not solved yet" is
/// exactly "both knobs are zero" — and on a level that has not been solved,
/// what this draws is the being where the player finds it.
fn still(scene: &CoveScene, dir: &Path) -> anyhow::Result<()> {
    let a = &scene.authored;
    let solved = scene.solution_x != 0.0 || scene.solution_y != 0.0;
    let (x, y, tilt) = if solved {
        (scene.solution_x, scene.solution_y, scene.solution_tilt)
    } else {
        (scene.spawn_x, scene.spawn_y, 0.0)
    };
    let placement = render::Placement::standing(scene, x, y, tilt);
    let (w, h) = (a.parameter_or("render_w", 960.0) as u32, a.parameter_or("render_h", 540.0) as u32);
    let spp = std::env::var("KOSM_SPP").ok().and_then(|v| v.parse().ok()).unwrap_or(a.parameter_or("render_spp", 64.0) as usize);
    let path = dir.join("frame.png");
    let t0 = std::time::Instant::now();
    render::frame(scene, &placement, (w, h), spp)?.save(&path)?;
    println!(
        "cove frame: the being {} at ({:+.2}, {:+.2}) m, leaning {:.1}°, facing the door; {w}×{h} at {spp} spp → {} in {:.1} s",
        if solved { "at the solved pose" } else { "at its spawn (nothing solved yet)" },
        x,
        y,
        tilt.to_degrees(),
        path.display(),
        t0.elapsed().as_secs_f64()
    );
    wide(scene, dir)
}
