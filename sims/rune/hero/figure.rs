//! The adventurer: one parametric assembly of smooth solids.
//!
//! Switch Sports proportions on a 1.1 m frame — a big head, a small body,
//! stubby rounded limbs, flat saturated colour, no texture and no hair. The
//! silhouette does all the work, which is exactly the brief a path tracer
//! given clean CAD and a soft sky answers well.
//!
//! **Not a character from another game.** The costume is a hooded cyan-teal
//! cloak over cream trousers, a cream collar, dark boots and a russet satchel
//! on the left hip. The colour is chosen in [`super::stage::palette`] against
//! the three things it has to stand in front of, and the hood is a cowl
//! rather than a peak, so nothing here is a tunic and nothing is a cap.
//!
//! **The frame.** Hero-local millimetres: the origin is on the ground between
//! the boots, +y is the way it faces, +z is up, and +x is therefore its own
//! right. One [`super::stage::stand`] puts it on the sand facing anywhere.
//!
//! **How it is posed.** Every limb is a capsule between two named joints, and
//! the joints are computed, not authored: the hands are the knobs
//! (`hand_r_x_mm` and its five friends) and the elbows fall out of a two-link
//! solve ([`joint`]) with a hint that keeps them pointing down and out. So a
//! pose is six numbers and a head tilt, and the day this figure is rigged for
//! real those same six numbers become an IK target. The pivots are named in
//! [`Rig`] and are what an articulation would hinge on:
//!
//! ```text
//! ankle  (±foot_x, 0, ankle_z)      hip     (±hip_x, 0, hip_z)
//! knee   solved, bulging +y          shoulder (±shoulder_x, 0, shoulder_z)
//! elbow  solved, bulging −y/out      neck    (0, 0, neck_z)
//! ```

use kosm::build::{Builder, Built, Params, Shape, build};

/// The figure's proportions and its pose, in millimetres and degrees.
///
/// One struct, read either from a [`Builder`]'s knobs (so every one of them
/// is a `Param` in the run hash) or from a [`Params`] (so the caller can work
/// out where the hands ended up without building anything). [`Rig::read`] is
/// the single definition of what each knob is called and what it defaults to.
#[derive(Clone, Copy, Debug)]
pub struct Rig {
    /// Overall height, top of the head to the sand. Everything below is
    /// authored against it rather than derived from it, because a Mii is not
    /// a scaled human — but it is what the assembly is checked against.
    pub height: f64,
    pub head_r: f64,
    pub head_z: f64,
    pub neck_z: f64,
    pub shoulder_x: f64,
    pub shoulder_z: f64,
    pub chest_r: f64,
    pub chest_z: f64,
    pub hip_x: f64,
    pub hip_z: f64,
    pub skirt_r: f64,
    pub skirt_z: f64,
    pub foot_x: f64,
    pub ankle_z: f64,
    pub leg_r: f64,
    pub boot_r: f64,
    pub upper_arm: f64,
    pub forearm: f64,
    pub arm_r: f64,
    pub hand_r: f64,
    /// Where the right and left hands are, hero-local. The pose lives here.
    pub hand_right: [f64; 3],
    pub hand_left: [f64; 3],
    /// How far the head is tipped back to look up, degrees. Positive is up.
    pub head_tilt: f64,
    /// Whether the satchel and its strap are drawn. One knob because a
    /// turntable wants them and a silhouette test does not.
    pub satchel: bool,
}

impl Rig {
    /// The default figure: 1.1 m, arms down, looking level.
    pub const DEFAULT: Rig = Rig {
        height: 1100.0,
        head_r: 205.0,
        head_z: 890.0,
        neck_z: 665.0,
        shoulder_x: 168.0,
        shoulder_z: 600.0,
        chest_r: 175.0,
        chest_z: 470.0,
        hip_x: 88.0,
        hip_z: 320.0,
        skirt_r: 252.0,
        skirt_z: 290.0,
        foot_x: 96.0,
        ankle_z: 92.0,
        leg_r: 54.0,
        boot_r: 84.0,
        upper_arm: 215.0,
        forearm: 205.0,
        arm_r: 62.0,
        hand_r: 76.0,
        hand_right: [250.0, 60.0, 380.0],
        hand_left: [-250.0, 60.0, 380.0],
        head_tilt: 0.0,
        satchel: true,
    };

    /// Read the rig through a knob lookup — a `Builder`'s or a `Params`'.
    ///
    /// One function, so a knob cannot mean one thing to the CAD and another
    /// to the code that places the hero on the sand.
    pub fn read(knob: &dyn Fn(&str, f64) -> f64) -> Rig {
        let d = Rig::DEFAULT;
        Rig {
            height: knob("hero_h_mm", d.height),
            head_r: knob("head_r_mm", d.head_r),
            head_z: knob("head_z_mm", d.head_z),
            neck_z: knob("neck_z_mm", d.neck_z),
            shoulder_x: knob("shoulder_x_mm", d.shoulder_x),
            shoulder_z: knob("shoulder_z_mm", d.shoulder_z),
            chest_r: knob("chest_r_mm", d.chest_r),
            chest_z: knob("chest_z_mm", d.chest_z),
            hip_x: knob("hip_x_mm", d.hip_x),
            hip_z: knob("hip_z_mm", d.hip_z),
            skirt_r: knob("skirt_r_mm", d.skirt_r),
            skirt_z: knob("skirt_z_mm", d.skirt_z),
            foot_x: knob("foot_x_mm", d.foot_x),
            ankle_z: knob("ankle_z_mm", d.ankle_z),
            leg_r: knob("leg_r_mm", d.leg_r),
            boot_r: knob("boot_r_mm", d.boot_r),
            upper_arm: knob("upper_arm_mm", d.upper_arm),
            forearm: knob("forearm_mm", d.forearm),
            arm_r: knob("arm_r_mm", d.arm_r),
            hand_r: knob("hand_r_mm", d.hand_r),
            hand_right: [
                knob("hand_r_x_mm", d.hand_right[0]),
                knob("hand_r_y_mm", d.hand_right[1]),
                knob("hand_r_z_mm", d.hand_right[2]),
            ],
            hand_left: [
                knob("hand_l_x_mm", d.hand_left[0]),
                knob("hand_l_y_mm", d.hand_left[1]),
                knob("hand_l_z_mm", d.hand_left[2]),
            ],
            head_tilt: knob("head_tilt_deg", d.head_tilt),
            satchel: knob("satchel", if d.satchel { 1.0 } else { 0.0 }) > 0.5,
        }
    }

    /// The rig a set of knobs describes, without building anything.
    pub fn of(params: &Params) -> Rig {
        Rig::read(&|name, default| params.get(name).unwrap_or(default))
    }

    /// The shoulder joint on a side: +1 is the hero's right.
    pub fn shoulder(&self, side: f64) -> [f64; 3] {
        [side * self.shoulder_x, 0.0, self.shoulder_z]
    }

    /// How far a hand can get from its shoulder: the two links, straight.
    pub fn reach(&self) -> f64 {
        self.upper_arm + self.forearm
    }

    /// The hand target on a side.
    pub fn hand(&self, side: f64) -> [f64; 3] {
        if side > 0.0 { self.hand_right } else { self.hand_left }
    }
}

/// The hero, at whatever pose the knobs describe.
pub fn figure(params: &Params) -> anyhow::Result<Built> {
    build(params, assemble)
}

/// Emit the whole figure into a builder.
///
/// Every part is its own named body with its own material, so the picture can
/// re-colour or drop any one of them and a later rig can hang a joint off any
/// one of them.
fn assemble(b: &Builder) {
    let r = Rig::read(&|name, default| b.param(name, default));

    // ---- the legs and the boots -----------------------------------------
    // Hip → knee → ankle. The knee is solved and told to bulge forward (+y),
    // which is the way a knee bends; the boot is a ball with a toe.
    for (side, tag) in [(1.0, "r"), (-1.0, "l")] {
        let hip = [side * r.hip_x, 0.0, r.hip_z];
        let ankle = [side * r.foot_x, 0.0, r.ankle_z];
        let thigh = 0.55 * (r.hip_z - r.ankle_z);
        let shin = r.hip_z - r.ankle_z - thigh + 20.0;
        let knee = joint(hip, ankle, thigh, shin, [0.0, 1.0, 0.0]);
        b.body(&format!("leg_{tag}"))
            .material("cream")
            .add(bone(b, hip, knee, r.leg_r).union(bone(b, knee, ankle, r.leg_r * 0.92)));
        // The boot: a ball at the ankle and a rounded toe in front of it, so
        // the foot has a direction without a single hard edge on it.
        b.body(&format!("boot_{tag}")).material("boot").add(
            b.sphere(r.boot_r)
                .at(ankle[0], ankle[1], r.boot_r * 0.98)
                .union(b.sphere(r.boot_r * 0.78).at(ankle[0], ankle[1] + 96.0, r.boot_r * 0.72))
                .union(b.sphere(r.boot_r * 0.86).at(ankle[0], ankle[1] + 48.0, r.boot_r * 0.9)),
        );
    }

    // ---- the cloak -------------------------------------------------------
    // A flared skirt and a capsule chest, unioned in one body because they
    // are one garment. The cone's wide end is at the bottom: vcad's cone has
    // its base at z = 0, so it is placed at the hem and rises to the waist.
    let waist_r = 0.71 * r.skirt_r;
    b.body("cloak").material("cloak").add(
        b.cone(r.skirt_r, waist_r, r.chest_z - r.skirt_z + 30.0)
            .at(0.0, 0.0, r.skirt_z)
            // the chest: a capsule from the waist to the shoulders
            .union(b.cylinder(r.chest_r, r.shoulder_z - r.chest_z).at(0.0, 0.0, r.chest_z))
            .union(b.sphere(r.chest_r).at(0.0, 0.0, r.chest_z))
            .union(b.sphere(r.chest_r).at(0.0, 0.0, r.shoulder_z)),
    );

    // The belt: a ring around the cloak at the waist, dark, so the figure has
    // a middle. The cone's radius there, read off the taper.
    let belt_z = r.skirt_z + 0.62 * (r.chest_z - r.skirt_z);
    let belt_r = r.skirt_r + (waist_r - r.skirt_r) * (belt_z - r.skirt_z) / (r.chest_z - r.skirt_z + 30.0);
    b.body("belt")
        .material("boot")
        .add(b.torus(belt_r, 22.0).at(0.0, 0.0, belt_z));

    // ---- the arms --------------------------------------------------------
    // Shoulder → elbow → hand, the elbow solved and told to bulge back and
    // outward so the arms never fold through the chest.
    for (side, tag) in [(1.0, "r"), (-1.0, "l")] {
        let shoulder = r.shoulder(side);
        let hand = r.hand(side);
        let elbow = joint(shoulder, hand, r.upper_arm, r.forearm, [side * 0.8, -0.6, -0.2]);
        b.body(&format!("arm_{tag}"))
            .material("cloak")
            .add(bone(b, shoulder, elbow, r.arm_r).union(bone(b, elbow, hand, r.arm_r * 0.9)));
        // The cuff, where the sleeve ends and the hand begins.
        b.body(&format!("cuff_{tag}"))
            .material("cream")
            .add(b.sphere(r.arm_r * 1.04).at(
                hand[0] - 0.28 * (hand[0] - elbow[0]),
                hand[1] - 0.28 * (hand[1] - elbow[1]),
                hand[2] - 0.28 * (hand[2] - elbow[2]),
            ));
        b.body(&format!("hand_{tag}"))
            .material("skin")
            .add(b.sphere(r.hand_r).at(hand[0], hand[1], hand[2]));
    }

    // ---- the collar ------------------------------------------------------
    // A cream ring where the cloak meets the neck: the brightest thing on the
    // figure, and the reason the eye goes to the head.
    b.body("collar")
        .material("cream")
        .add(b.torus(0.86 * r.chest_r, 46.0).at(0.0, 0.0, r.shoulder_z + 32.0));

    // ---- the head, the hood and the face ----------------------------------
    // All four bodies are built about the *neck pivot* at the origin, tipped
    // by `head_tilt_deg`, and then lifted to the neck. That is what a neck
    // joint is, and it is the one place in the figure where a rotation is
    // authored rather than solved.
    let tip = |shape: Shape| shape.rotate_x(r.head_tilt).at(0.0, 0.0, r.neck_z);
    let dz = r.head_z - r.neck_z;

    b.body("neck")
        .material("skin")
        .add(tip(b.cylinder(0.32 * r.head_r, dz).at(0.0, 0.0, -20.0)));
    b.body("head").material("skin").add(tip(b.sphere(r.head_r).at(0.0, 0.0, dz)));

    // The eyes: two dots, set on the front of the head and barely proud of
    // it. No mouth, no brows, no hair — a Mii's whole face is its eyes.
    let eye_r = 0.17 * r.head_r;
    for side in [1.0, -1.0] {
        let n = normalize([side * 0.33, 0.93, 0.10]);
        let s = r.head_r - 0.62 * eye_r;
        let tag = if side > 0.0 { "r" } else { "l" };
        b.body(&format!("eye_{tag}")).material("ink").add(tip(
            b.sphere(eye_r).at(n[0] * s, n[1] * s, dz + n[2] * s),
        ));
    }
    // A mouth, and only a dot of one. Two eyes and nothing else is a mask;
    // three marks is a face, and three is where this stops.
    let mouth = normalize([0.0, 0.94, -0.34]);
    // sat *on* the surface rather than under it: a dot placed a whole radius
    // inside a 205 mm head is a dot nobody ever sees
    let ms = r.head_r - 0.22 * (0.42 * eye_r);
    b.body("mouth").material("ink").add(tip(
        b.sphere(0.42 * eye_r).at(mouth[0] * ms, mouth[1] * ms, dz + mouth[2] * ms),
    ));

    // The hood: a sphere a little larger than the head, pushed back and up so
    // the face is out in the open and the cowl covers the crown and the nape.
    // A sphere and not a cone, because a peak would read as somebody else's
    // hat; the drape at the back is a second, smaller ball.
    // The hood, in three analytic pieces and no boolean.
    //
    // The obvious construction — a bigger ball behind the head — was tried
    // and reads as a balloon, because two spheres can only ever meet in a
    // circle wider than the face: the head's exposed cap is a hemisphere or
    // it is nothing. The obvious repair, cutting the face out of the ball
    // with a `difference`, was tried too, and it is *right* and it is
    // **faceted**: vcad's boolean hands back a BRep of planar faces, so a
    // hood that had been a smooth sphere comes back as a geodesic dome, and
    // no amount of re-normalling fixes a silhouette.
    //
    // So the cowl is a ball for the crown, a fat ring of rolled cloth around
    // the opening, and a fold of drape at the nape — three primitives, every
    // one of them analytic and exactly round at any zoom. The ring is what
    // does the work the cut would have done: it stands inside the ball's
    // mouth and brings the opening in from the head's own equator to
    // something a face fits in.
    //
    // The mouth is where the two spheres meet, and that circle is exact: for
    // centres `d` apart it is the plane `y = (d² + head² − hood²)/2d` at
    // radius `sqrt(head² − y²)`. Everything below is laid on it.
    let hood_r = r.head_r + 36.0;
    let hood_y = -0.72 * r.head_r;
    let lift = 0.20 * r.head_r;
    let d = -hood_y;
    let mouth_y = (d * d + r.head_r * r.head_r - hood_r * hood_r) / (2.0 * d);
    let mouth_r = (r.head_r * r.head_r - mouth_y * mouth_y).max(1.0).sqrt();
    let roll = 0.25 * r.head_r; // the rolled edge's tube
    b.body("hood").material("cloak").add(tip(
        b.sphere(hood_r)
            .at(0.0, hood_y, dz + lift)
            // the roll around the opening: it is what a face is framed by
            .union(b.torus(mouth_r - 0.12 * roll, roll).rotate_x(90.0).at(0.0, mouth_y, dz + lift))
            // and a fold of cloth at the nape
            .union(b.sphere(0.44 * r.head_r).at(0.0, hood_y - 0.52 * r.head_r, dz - 0.30 * r.head_r)),
    ));
    // The lining: cream piping laid *on the head*, at the circle where the
    // face disappears into the cowl. On the head and not on the roll,
    // because anything drawn between the two is inside one of them — the
    // head is 205 mm of solid and the roll is 60 mm of tube, and the gap
    // between them is a millimetre of nothing.
    let lining_r = 0.87 * r.head_r;
    let lining_y = (r.head_r * r.head_r - lining_r * lining_r).max(1.0).sqrt();
    b.body("hood_lining").material("cream").add(tip(
        b.torus(lining_r, 0.20 * roll).rotate_x(90.0).at(0.0, lining_y, dz),
    ));

    // ---- the satchel -------------------------------------------------------
    // On the left hip, over the cloak, with a strap over the right shoulder.
    // It is the one warm colour on the figure and the one hard-edged box, so
    // it does two jobs: it breaks the silhouette and it says "adventurer".
    if r.satchel {
        // Outside the skirt, not inside it. The cloak flares to `skirt_r` at
        // the hem and a bag tucked within that radius is a bag nobody ever
        // sees; this one hangs clear of the cloth on the hero's left, low
        // enough to break the flare's silhouette.
        let bag = [-0.86 * r.skirt_r - 46.0, -34.0, r.hip_z + 44.0];
        b.body("satchel").material("leather").add(
            b.boxed(214.0, 108.0, 168.0)
                .at(bag[0], bag[1], bag[2])
                .union(b.boxed(226.0, 120.0, 62.0).at(bag[0], bag[1] - 6.0, bag[2] + 70.0)),
        );
        b.body("buckle")
            .material("brass")
            .add(b.cylinder(28.0, 20.0).rotate_x(90.0).at(bag[0], bag[1] - 66.0, bag[2] + 34.0));
        // The strap runs over the far shoulder and across the chest, which is
        // where it is seen from the front and from either three-quarter.
        b.body("strap").material("cream").add(
            bone(
                b,
                [0.80 * r.shoulder_x, -0.30 * r.chest_r, r.shoulder_z + 30.0],
                [0.10 * r.chest_r, -0.98 * r.chest_r, r.chest_z + 40.0],
                22.0,
            )
            .union(bone(
                b,
                [0.10 * r.chest_r, -0.98 * r.chest_r, r.chest_z + 40.0],
                [bag[0] + 60.0, bag[1] - 30.0, bag[2] + 96.0],
                22.0,
            )),
        );
    }
}

// ---- the two bits of geometry the assembly needs ---------------------------

/// A capsule from `from` to `to`: a cylinder along the segment with a ball at
/// each end, so two bones meeting at a joint meet in a sphere and never in a
/// crease.
///
/// vcad's cylinder is built along +z with its base at the origin, so the
/// placement is a rotation taking +z onto the segment and then a translation.
/// `rotate(a, 0, c)` is Euler XYZ — X first, then Z — and applying it to +z
/// gives `(sin a sin c, −sin a cos c, cos a)`, which is why `a` is the polar
/// angle off +z and `c` is `atan2(dx, −dy)`.
fn bone(b: &Builder, from: [f64; 3], to: [f64; 3], r: f64) -> Shape {
    let d = sub(to, from);
    let len = norm(d).max(1e-6);
    let polar = (d[2] / len).clamp(-1.0, 1.0).acos().to_degrees();
    let about_z = d[0].atan2(-d[1]).to_degrees();
    b.cylinder(r, len)
        .rotate(polar, 0.0, about_z)
        .at(from[0], from[1], from[2])
        .union(b.sphere(r).at(from[0], from[1], from[2]))
        .union(b.sphere(r).at(to[0], to[1], to[2]))
}

/// The middle joint of a two-link chain from `root` to `tip`.
///
/// The classic two-link solve: the joint lies on a circle about the line from
/// root to tip, and `hint` picks the point on that circle — which is the only
/// thing that decides whether an elbow points backwards or forwards. When the
/// target is out of reach the chain straightens and points at it rather than
/// failing, because a pose knob that snaps is worse than one that saturates.
pub fn joint(root: [f64; 3], tip: [f64; 3], upper: f64, lower: f64, hint: [f64; 3]) -> [f64; 3] {
    let to_tip = sub(tip, root);
    let d = norm(to_tip).clamp(1e-6, upper + lower - 1e-6);
    let dir = scale(to_tip, 1.0 / norm(to_tip).max(1e-6));
    let along = (upper * upper - lower * lower + d * d) / (2.0 * d);
    let off = (upper * upper - along * along).max(0.0).sqrt();
    // the hint, with everything along the chain taken out of it
    let mut pole = sub(hint, scale(dir, dot(hint, dir)));
    if norm(pole) < 1e-9 {
        pole = sub([0.0, 0.0, -1.0], scale(dir, dot([0.0, 0.0, -1.0], dir)));
    }
    let pole = normalize(pole);
    add(add(root, scale(dir, along)), scale(pole, off))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    scale(a, 1.0 / norm(a).max(1e-12))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The figure is the height it says it is, and every part of it is a
    /// named body with a material.
    #[test]
    fn the_hero_is_a_metre_ten_and_all_of_it_is_named() -> anyhow::Result<()> {
        let r = Rig::DEFAULT;
        assert!((r.head_z + r.head_r - r.height).abs() < 10.0, "the crown is the height");
        let built = figure(&Params::default())?;
        for name in ["head", "hood", "cloak", "arm_r", "boot_l", "satchel", "eye_l"] {
            assert!(built.bodies.iter().any(|b| b.name == name), "no `{name}` body");
        }
        for body in &built.bodies {
            assert!(!body.material.is_empty(), "`{}` has no material", body.name);
        }
        Ok(())
    }

    /// A pose is the hands, and the elbows follow them. Both halves matter:
    /// the solve has to reach the target when it can — the three below are a
    /// hand at rest, a hand holding the lens up at the doorstep, and a hand
    /// straight out, all of them inside the arm's 420 mm — and straighten
    /// toward it when it cannot.
    #[test]
    fn the_elbow_follows_the_hand_and_saturates_rather_than_snapping() {
        let r = Rig::DEFAULT;
        let s = r.shoulder(1.0);
        for hand in [[250.0, 60.0, 380.0], [110.0, 216.0, 933.0], [400.0, 0.0, 600.0]] {
            let e = joint(s, hand, r.upper_arm, r.forearm, [0.8, -0.6, -0.2]);
            assert!((norm(sub(e, s)) - r.upper_arm).abs() < 1e-6, "the upper arm changed length");
            assert!((norm(sub(hand, e)) - r.forearm).abs() < 1e-6, "the forearm changed length");
        }
        // out of reach: the arm straightens and still points at the target
        let far = [0.0, 3000.0, 600.0];
        let e = joint(s, far, r.upper_arm, r.forearm, [0.8, -0.6, -0.2]);
        let straight = normalize(sub(far, s));
        let got = normalize(sub(e, s));
        assert!(dot(straight, got) > 0.999, "an unreachable target did not straighten the arm");
    }

    /// A knob read off a `Params` and a knob read off a `Builder` are the
    /// same knob — the thing that keeps the pose the caller solved and the
    /// pose the CAD built from being two different poses.
    #[test]
    fn the_rig_reads_the_same_from_either_side() -> anyhow::Result<()> {
        let mut params = Params::new();
        params.set("hand_r_z_mm", 1024.0);
        params.set("head_tilt_deg", 21.0);
        let rig = Rig::of(&params);
        assert_eq!(rig.hand_right[2], 1024.0);
        assert_eq!(rig.head_tilt, 21.0);
        let built = figure(&params)?;
        assert_eq!(built.param("hand_r_z_mm"), Some(1024.0));
        assert_eq!(built.param("head_tilt_deg"), Some(21.0));
        Ok(())
    }
}
