//! What the tide left, and who else lives here.
//!
//! [`super::scene`] is the cove's *geology* — the beach, the beds of the
//! cliff, the headlands, the reef, the door and its frame. This file is the
//! layer over it that says the place has a history: pebbles graded toward the
//! waterline and banked under the cliff, shells and driftwood above the tide
//! line, kelp on the reef, marram on the dune foot and the headland steps, two
//! tide pools, and the cove's two other inhabitants standing in it.
//!
//! Millimetres and degrees, vcad's, exactly as the level is.
//!
//! ## Everything here is an instance
//!
//! [`kosm::brep::instances`] keys a primitive by its kind and its dimensions
//! *bit for bit*, so ninety pebbles authored as `b.sphere(60.0)` are one solid
//! and ninety placements, and forty-four tufts of twelve blades are one cone
//! and five hundred and twenty-eight placements. Both tiers get that for free:
//! the tracer builds one BVH per distinct solid and places it, and the raster
//! tier reads the same walk. What breaks it is a `scale` or a `difference` per
//! prop — either sends the subtree to `vcad_eval` and gives it a solid of its
//! own — so the scatter varies **size by choosing among a handful of prototype
//! radii** and orientation by rotation, and never by scaling.
//!
//! ## Nothing here is baked
//!
//! Every root this file declares is [`decorative`](kosm::build::Body::decorative)
//! and is left out of the collision set by [`super::CoveScene::parts`]: the
//! signed distance field is the `ground` root and nothing else. A pebble a
//! tenth of the bake's 100 mm cell cannot be represented in that field
//! anyway, and a tuft of grass you cannot walk through is a worse lie than one
//! you can. What *is* in the field — the boulders at the cliff's foot, the
//! tide pools' basins and their stone lips — is in `scene.rs`, in the `ground`
//! solid, because the bake wants one inside.
//!
//! ## And nothing here is in the light path
//!
//! The rune is the sun through the hero's lens into the keyhole, and the
//! design's one hard rule is that nothing may shadow it. Two facts make that
//! checkable rather than hopeful. The sun travels shoreward and *down*, so
//! every point on the segment from the sun to the lens is at `y` less than the
//! lens's own — nothing standing between the hero and the cliff can shadow it,
//! at any height. And the segment from the lens to the keyhole runs from
//! 1.5 m above the sand at the hero's hand down to 0.4 m at the keyhole, all
//! of it within a metre of the door's centre line. So [`Site::clear`] keeps
//! the apron in front of the door empty, and that one rule is the whole of the
//! guarantee.

use kosm::build::Builder;

/// The cove's geometry, as the scatter needs it. Millimetres.
///
/// Handed in rather than read from a [`super::CoveScene`], because the scatter
/// runs *inside* the build closure — the level is being authored, so there is
/// no `Built` to ask yet.
#[derive(Clone, Copy, Debug)]
pub struct Site {
    /// Half the cove square.
    pub half: f64,
    pub sea_z: f64,
    pub slope: f64,
    /// The cliff's face, and the door on it.
    pub face: f64,
    pub door_x: f64,
    pub door_w: f64,
    /// The inner face of the headlands, and how the steps march out.
    pub headland_x: f64,
    pub headland_step: f64,
    pub headland_rise: f64,
    pub headland_steps: usize,
    /// The seaward edge of the seabed, and where the reef stands.
    pub sand_y0: f64,
    pub reef_y: f64,
    /// The fall line the marble test rolls down: nothing goes near it.
    pub spawn_x: f64,
    /// The seed every draw below is a fixed function of.
    pub seed: u64,
}

impl Site {
    /// The top of the sand at `y`.
    pub fn z(&self, y: f64) -> f64 {
        self.sea_z + self.slope * (y + self.half)
    }

    /// The waterline.
    pub fn waterline(&self) -> f64 {
        -self.half
    }

    /// Whether a prop may stand at `(x, y)`.
    ///
    /// Four exclusions, and each one is a test somewhere else in the level:
    ///
    /// - **the door's apron.** A cylinder `APRON` across in front of the
    ///   door's foot. The lens-to-keyhole segment lives inside it and nothing
    ///   else may.
    /// - **the fall line.** `sims/rune/tests.rs` rolls a marble down `x =
    ///   spawn_x` and holds it to ten millimetres of cross-line drift; the
    ///   props are not in the field, but the *pools* are, and one rule is
    ///   better than two.
    /// - **the headlands.** Past their inner face is rock, not sand.
    /// - **the ends.** A prop hanging off the seaward edge of the seabed, or
    ///   buried in the cliff.
    pub fn clear(&self, x: f64, y: f64) -> bool {
        let apron = (x - self.door_x).hypot(y - self.face) > APRON;
        let fall_line = (x - self.spawn_x).abs() > FALL_LINE;
        let inside = x.abs() < self.headland_x - 1200.0;
        let ends = y > self.sand_y0 + 2500.0 && y < self.face - 700.0;
        apron && fall_line && inside && ends
    }
}

/// How far in front of the door's foot nothing stands.
///
/// Three metres. The hero solves the rune 1.14 m off the face with the glass
/// about 1.5 m up; the segment from that glass to the keyhole is inside a
/// metre of the door's centre line the whole way, and three metres is that
/// with a two-metre margin. It is also, not by accident, about where the
/// doorstep camera's eye sits.
pub const APRON: f64 = 3000.0;

/// How wide a corridor the marble's fall line keeps to itself.
pub const FALL_LINE: f64 = 2000.0;

// ── the tide pools ──────────────────────────────────────────────────────────

/// One basin in the sand: where it is, how wide the water is, how deep the
/// basin is cut, and the radius of the stone lip round it. Millimetres.
#[derive(Clone, Copy, Debug)]
pub struct Pool {
    pub x: f64,
    pub y: f64,
    /// The radius of the sphere the basin is cut with. The water's own radius
    /// is [`Pool::water_r`].
    pub tool_r: f64,
    pub depth: f64,
    pub lip_r: f64,
}

impl Pool {
    /// The water's surface radius: a sphere of radius `R` sunk `d` below a
    /// plane meets it in a circle of radius `sqrt(2Rd − d²)`.
    pub fn water_r(&self) -> f64 {
        (2.0 * self.tool_r * self.depth - self.depth * self.depth).max(0.0).sqrt()
    }

    /// How full it is: the water sits this far above the basin's floor.
    pub fn fill(&self) -> f64 {
        0.62 * self.depth
    }
}

/// The cove's two tide pools, from whichever source of knobs the caller has.
///
/// `scene.rs` passes the builder's, so they are the document's; `render.rs`
/// passes the built document's, so the water disc it draws in each one is over
/// the basin `scene.rs` cut. One table, two readers — the same shape
/// [`super::automaton::layout`] has, and for the same reason.
///
/// Both sit just above the waterline, where the sand is a hand's breadth up:
/// cut 150 mm into it and the basin's floor is *below* sea level, which is
/// what makes a tide pool a tide pool rather than a puddle.
pub fn pools(knob: &dyn Fn(&str, f64) -> f64) -> Vec<Pool> {
    let tool_r = knob("pool_tool_r_mm", 3000.0);
    let depth = knob("pool_depth_mm", 150.0);
    let lip_r = knob("pool_lip_r_mm", 1120.0);
    vec![
        Pool { x: knob("pool_a_x_mm", -4500.0), y: knob("pool_a_y_mm", -18600.0), tool_r, depth, lip_r },
        Pool { x: knob("pool_b_x_mm", 6800.0), y: knob("pool_b_y_mm", -17400.0), tool_r, depth, lip_r },
    ]
}

// ── the scatter ─────────────────────────────────────────────────────────────

/// Everything the tide left, as decorative roots on the open document.
///
/// One root per kind, so the picture can paint each with its own substance and
/// so a reader of `parts.json` can see at a glance what is scattered and how
/// much of it there is.
pub fn scatter(b: &Builder, s: &Site) {
    pebbles(b, s);
    shells(b, s);
    driftwood(b, s);
    kelp(b, s);
    marram(b, s);
}

/// Pebbles: four sizes of sphere, more than half buried, graded.
///
/// The grading is the point. A beach is not a uniform sprinkle — the swash
/// sorts it, so the shingle banks up at the top of the wave's reach and again
/// where the cliff's own debris comes down. So a little under half of them are
/// drawn about the waterline, a little under a third under the cliff, and the
/// rest over the open sand.
fn pebbles(b: &Builder, s: &Site) {
    let n = b.param("pebbles", 96.0).max(0.0) as usize;
    // Four sizes off one knob, so the beach grades when the knob moves and
    // so all four are the same solid four times over rather than four solids.
    let r = b.param("pebble_r_mm", 62.0);
    let sizes = [r, 0.66 * r, 1.42 * r, 1.90 * r];
    let body = b.body("pebbles");
    body.material("rock").decorative();
    let mut drawn = 0usize;
    for i in 0..n * 3 {
        if drawn == n {
            break;
        }
        let (u, v, w) = (draw(s.seed, i, 0), draw(s.seed, i, 1), draw(s.seed, i, 2));
        let x = -s.headland_x + 2.0 * s.headland_x * u;
        let y = match v {
            // the swash line, and the berm just above it
            v if v < 0.44 => s.waterline() - 2200.0 + 6600.0 * tri(w),
            // the cliff's own debris, banked at its foot
            v if v < 0.72 => s.face - 6000.0 + 4600.0 * w,
            // and the open sand between them
            _ => s.waterline() + 4000.0 + (s.face - 5000.0 - s.waterline() - 4000.0) * w,
        };
        if !s.clear(x, y) {
            continue;
        }
        let r = sizes[(draw(s.seed, i, 3) * sizes.len() as f64) as usize % sizes.len()];
        // Buried past its equator, so what shows is a dome and not a ball
        // sitting on top of the sand like a dropped marble.
        body.add(b.sphere(r).at(x, y, s.z(y) - (0.42 + 0.22 * draw(s.seed, i, 4)) * r));
        drawn += 1;
    }
}

/// Scallops: a fan of tapered ribs, which is what a scallop is.
///
/// One prototype — nine ribs on a `circular_pattern`, so the whole shell is
/// nine instances of one cone — laid flat and turned by a drawn yaw. Nacre,
/// and the one surface in the cove that keeps its thin film: the library's
/// `shell (nacre)` carries four hundred nanometres of aragonite platelet and
/// that interference *is* the iridescence, which on something 150 mm across
/// costs the flat look nothing and buys the one glint on the sand that is not
/// the hint.
fn shells(b: &Builder, s: &Site) {
    let n = b.param("shells", 6.0).max(0.0) as usize;
    let r = b.param("shell_r_mm", 150.0);
    let ribs = b.param("shell_ribs", 9.0).max(3.0) as u32;
    // A rib: a cone laid along +y, thin at the hinge and wide at the rim.
    let shell = b
        .cone(0.10 * r, 0.20 * r, r)
        .rotate_x(-90.0)
        .circular_pattern([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], ribs, 150.0)
        // the fan is drawn from 0° to 150°, so this centres it on +y
        .rotate_z(-75.0);
    let body = b.body("shells");
    body.material("shell").decorative();
    let mut drawn = 0usize;
    for i in 0..n * 4 {
        if drawn == n {
            break;
        }
        let (u, v) = (draw(s.seed, i, 10), draw(s.seed, i, 11));
        let x = -s.headland_x + 2.0 * s.headland_x * u;
        // above the swash, where a shell is left rather than rolled
        let y = s.waterline() + 1500.0 + 9000.0 * v;
        if !s.clear(x, y) {
            continue;
        }
        body.add(
            shell
                .clone()
                .rotate_z(360.0 * draw(s.seed, i, 12))
                .at(x, y, s.z(y) + 0.05 * r),
        );
        drawn += 1;
    }
}

/// Driftwood: three bent trunks above the tide line.
///
/// One prototype of three capped rods hinged a few degrees off each other —
/// a stick that has been in the sea is never straight — turned and dropped.
fn driftwood(b: &Builder, s: &Site) {
    let n = b.param("driftwood", 3.0).max(0.0) as usize;
    let r = b.param("driftwood_r_mm", 88.0);
    let seg = b.param("driftwood_seg_mm", 820.0);
    // `rod_z` is a capped cylinder on the origin along z; `rotate_y(90)` lays
    // it along +x, and the three angles either side of that are the bend.
    let log = b
        .rod_z(r, seg)
        .rotate_y(83.0)
        .at(-0.94 * seg, 0.0, 0.10 * seg)
        .union(b.rod_z(r, seg).rotate_y(90.0))
        .union(b.rod_z(r, seg).rotate_y(99.0).at(0.94 * seg, 0.0, -0.14 * seg));
    let body = b.body("driftwood");
    body.material("driftwood").decorative();
    let mut drawn = 0usize;
    for i in 0..n * 6 {
        if drawn == n {
            break;
        }
        let (u, v) = (draw(s.seed, i, 20), draw(s.seed, i, 21));
        let x = -s.headland_x + 2.0 * s.headland_x * u;
        let y = s.waterline() + 4500.0 + 9500.0 * v;
        if !s.clear(x, y) {
            continue;
        }
        body.add(
            log.clone()
                .rotate_z(360.0 * draw(s.seed, i, 22))
                .at(x, y, s.z(y) + 0.35 * r),
        );
        drawn += 1;
    }
}

/// Kelp on the reef: strands of tapered segments, drooping seaward off the
/// boulders that break the surface.
fn kelp(b: &Builder, s: &Site) {
    let n = b.param("kelp", 8.0).max(0.0) as usize;
    let r = b.param("kelp_r_mm", 46.0);
    let seg = b.param("kelp_seg_mm", 620.0);
    // Four segments, each leaning further over than the last: a blade that
    // stands out of the rock and lies down on the water.
    let strand = b
        .cone(r, 0.75 * r, seg)
        .union(b.cone(0.75 * r, 0.5 * r, seg).rotate_y(26.0).at(0.13 * seg, 0.0, 0.98 * seg))
        .union(b.cone(0.5 * r, 0.3 * r, seg).rotate_y(58.0).at(0.55 * seg, 0.0, 1.80 * seg))
        .union(b.cone(0.3 * r, 0.12 * r, seg).rotate_y(84.0).at(1.32 * seg, 0.0, 2.22 * seg));
    let body = b.body("kelp");
    body.material("kelp").decorative();
    for i in 0..n {
        let t = i as f64 / n.max(1) as f64;
        // spread along the reef, off its shoreward face
        let x = -0.82 * s.headland_x + 1.64 * s.headland_x * t + 900.0 * (draw(s.seed, i, 30) - 0.5);
        let y = s.reef_y + 1400.0 + 900.0 * draw(s.seed, i, 31);
        body.add(
            strand
                .clone()
                .rotate_z(360.0 * draw(s.seed, i, 32))
                .at(x, y, s.z(y) + 200.0 * draw(s.seed, i, 33)),
        );
    }
}

/// Marram: tufts of twelve blades, on the dune foot under the cliff and on the
/// headland steps.
///
/// A blade is a cone tapered nearly to a point, tilted off vertical and turned
/// all the way round by a `circular_pattern` — twelve instances of one cone,
/// and every tuft is that same prototype again at a drawn yaw. The design's
/// note says the cove was chosen so that grass would be nobody's problem yet;
/// this is the smallest grass that reads as grass in a flat palette: a
/// silhouette of thin tapers, one colour, no texture.
fn marram(b: &Builder, s: &Site) {
    let n = b.param("tufts", 52.0).max(0.0) as usize;
    let blades = b.param("blades", 12.0).max(3.0) as u32;
    let h = b.param("blade_h_mm", 520.0);
    let tuft = |lean: f64, count: u32| {
        b.cone(0.034 * h, 0.005 * h, h).rotate_y(lean).circular_pattern(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            count,
            360.0,
        )
    };
    // Two prototypes, so a run of tufts is not one silhouette repeated: a
    // tight upright one and a wider splayed one.
    let tight = tuft(18.0, blades);
    let splayed = tuft(31.0, blades + 2);
    let body = b.body("marram");
    body.material("marram").decorative();

    // The dune foot: a band along the cliff, thinning as it comes away.
    let mut drawn = 0usize;
    let dune = (n * 2) / 3;
    for i in 0..dune * 4 {
        if drawn == dune {
            break;
        }
        let (u, v) = (draw(s.seed, i, 40), draw(s.seed, i, 41));
        let x = -s.headland_x + 2.0 * s.headland_x * u;
        let y = s.face - 900.0 - 3600.0 * v * v;
        if !s.clear(x, y) {
            continue;
        }
        let proto = if draw(s.seed, i, 42) < 0.5 { &tight } else { &splayed };
        body.add(proto.clone().rotate_z(360.0 * draw(s.seed, i, 43)).at(x, y, s.z(y) - 0.06 * h));
        drawn += 1;
    }

    // And the headland steps: a tuft or two on every tread, which is what
    // says the rock up there is old.
    let per_step = ((n - dune) / s.headland_steps.max(1)).max(1);
    for step in 0..s.headland_steps {
        let inner = s.headland_x + step as f64 * s.headland_step;
        let lift = (step as f64 + 1.0) * s.headland_rise;
        for k in 0..per_step {
            let i = 100 + step * 16 + k;
            for side in [-1.0f64, 1.0] {
                let x = side * (inner + 300.0 + 900.0 * draw(s.seed, i, 50));
                let y = s.waterline() + 4000.0 + 22000.0 * draw(s.seed, i, 51);
                let proto = if draw(s.seed, i, 52) < 0.5 { &tight } else { &splayed };
                body.add(
                    proto
                        .clone()
                        .rotate_z(360.0 * draw(s.seed, i, 53))
                        .at(x, y, s.z(y) + lift - 0.06 * h),
                );
            }
        }
    }
}

// ── the other two ───────────────────────────────────────────────────────────

/// The cove's inhabitants, standing where the level put them.
///
/// Both are figures the repository already has, dropped into this document by
/// [`kosm::build::Builder::bodies_since`] and placed: the automaton of
/// [`super::automaton`] beside the door, and the creature of
/// [`super::creature`] in the nearer tide pool. Static — nothing steps them
/// yet — and decorative, so neither is in the baked field.
///
/// **The automaton comes without its glass.** See
/// [`super::automaton::assemble`]: a second transmissive body in this level
/// would join the caustic pass's aim and spend the rune's photons on a
/// bystander. What it holds is the brass bezel of the instrument, lowered to
/// its hip, which is the pose of somebody waiting rather than solving —
/// and the pose that keeps it out of the door's swing and out of the light.
pub fn inhabitants(b: &Builder, s: &Site, pools: &[Pool]) {
    use super::automaton;
    use super::creature::body::{self as creature, Creature};

    // ---- the automaton, beside the door --------------------------------
    let ax = b.param("automaton_x_mm", -2700.0);
    let ay = b.param("automaton_y_mm", 15300.0);
    let ayaw = b.param("automaton_yaw_deg", 24.0);
    let mut l = automaton::layout(&|_: &str, default: f64| default);
    // Idle: the ring down at the hip and level, which drops the arms through
    // the two-link IK that put them overhead. Nothing else about the doll
    // changes, because nothing else about a doll standing still would.
    l.lens_y = 0.34 * l.h;
    l.lens_z = 0.42 * l.h;
    l.lens_pitch = 12.0;
    let mark = b.bodies();
    automaton::assemble(b, &l, 2450.0, false);
    for part in b.bodies_since(mark) {
        part.decorative().rotate_z(ayaw).at(ax, ay, s.z(ay));
    }

    // ---- the creature, in the near tide pool ---------------------------
    if let Some(pool) = pools.first() {
        let c = Creature::default();
        let cyaw = b.param("creature_yaw_deg", 155.0);
        // Sitting in the water: the animal's origin is the sand under it, so
        // dropping it by a third of the basin's depth settles it in.
        let cz = s.z(pool.y) - pool.depth + 0.30 * pool.depth;
        let mark = b.bodies();
        creature::assemble(b, &c);
        for part in b.bodies_since(mark) {
            part.decorative().rotate_z(cyaw).at(pool.x, pool.y, cz);
        }
    }
}

// ── the draw ────────────────────────────────────────────────────────────────

/// A number in `[0, 1)` that is a fixed function of the seed and two indices.
///
/// Not a stream: the reef above it makes the same argument. A draw that
/// depends on how many draws came before it is a draw that moves when a knob
/// changes the count, and the bake's hash is over the document — so every
/// number here is addressed, and the same seed gives the same beach in every
/// run and in every process.
///
/// The mixer is MurmurHash3's finalizer over a Weyl-spaced key, which is what
/// `splitmix64` is; fifty-three bits of it become the mantissa.
fn draw(seed: u64, i: usize, k: u64) -> f64 {
    let mut x = seed
        ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ k.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A triangular draw on `[0, 1)`, peaked in the middle: `(u + v)/2` for two
/// uniforms, written as one so the caller spends one draw on it.
fn tri(u: f64) -> f64 {
    if u < 0.5 { (0.5 * u).sqrt() } else { 1.0 - (0.5 * (1.0 - u)).sqrt() }
}
