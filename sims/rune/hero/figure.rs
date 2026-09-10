//! The adventurer: one parametric assembly of smooth solids.
//!
//! Switch Sports proportions on a 1.11 m frame — a big head, a small body,
//! stubby rounded limbs, flat saturated colour, no texture and no hair. The
//! silhouette does all the work, which is exactly the brief a path tracer
//! given clean CAD and a soft sky answers well.
//!
//! **Not a character from another game.** The costume is a hooded cyan-teal
//! cloak with a short mantle over it, cream trousers, a cream collar, dark
//! boots and a russet satchel on the left hip with its strap across the
//! chest. The colour is chosen in [`super::stage::palette`] against the three
//! things it has to stand in front of, and the hood is a cowl rather than a
//! peak, so nothing here is a tunic and nothing is a cap.
//!
//! **Every part is a substance.** Each body is authored with
//! [`kosm::build::Body::substance`] against an entry in
//! `kosm::material` — `cloak` is wool felt under a dye, `cream` is linen,
//! `boot` is blacked leather, `skin` is skin — so the same document that is
//! traced can be asked for a density. That is the half of the rig a picture
//! does not need and `kosm::player` will.
//!
//! **The frame.** Hero-local millimetres: the origin is on the ground between
//! the boots, +y is the way it faces, +z is up, and +x is therefore its own
//! right. One [`super::stage::stand`] puts it on the sand facing anywhere.
//!
//! **How it is posed.** Every limb is a capsule between two named joints, and
//! the joints are computed, not authored: the hands are the knobs
//! (`hand_r_x_mm` and its five friends) and the elbows fall out of a two-link
//! solve ([`joint`]) with a hint that keeps them out and back. A pose is
//! therefore six numbers, a head tilt, a torso lean and a stance — and the day
//! this figure is rigged for real those same numbers become IK targets.
//! [`Rig::pivots`] hands back every hinge with its axis, in one struct, so
//! nothing downstream re-derives a joint from the CAD.
//!
//! ```text
//! ankle  (±foot_x, foot_y[side], ankle_z)   hip     (±hip_x, pelvis_y, hip_z)
//! knee   solved, bulging +y                 torso   (0, pelvis_y, hip_z), about x
//! elbow  solved, bulging −y and out         shoulder(±shoulder_x, ·, shoulder_z)
//!                                           neck    (0, ·, neck_z), about x
//! ```
//!
//! # What the last pass got wrong, and what fixed it
//!
//! **The hood was a balloon.** It was a ball a little larger than the head
//! pushed a long way back, and two spheres can only ever meet in a circle
//! wider than the face — so the head's exposed cap was a hemisphere and the
//! rest was a teal sphere with nothing on it, which is what the back of the
//! turntable showed. Cutting the face out with a `difference` is *right* and
//! is **faceted**: vcad's boolean hands back a BRep of planar faces and a
//! smooth sphere comes back a geodesic dome.
//!
//! So the cowl is three analytic primitives and no boolean, and the load has
//! moved from the ball to the *ring*: a rolled edge of cloth whose hole is
//! `0.60 · head_r` and whose outer edge stands proud of the head all round,
//! set well forward of the head's equator so the face bulges through it. The
//! crown ball behind it is now barely larger than the head and barely moved,
//! because it no longer has to reach round to the front. See [`assemble`].
//!
//! **Nothing read from behind.** The back of a hooded figure is a teal ball
//! over a teal body. A short mantle — a cone from the collar to just under
//! the shoulders, with cream piping at its hem — is what gives the back
//! three edges instead of none, and the satchel strap crosses it.
//!
//! **The strap was on the wrong side.** It ran to `y = −0.98 · chest_r`,
//! which is the *back*: +y is the way this figure faces. It now crosses the
//! front of the mantle in russet leather, which is the one thing that tells
//! the hero's left from its right at any distance.

use kosm::build::{Builder, Built, Params, Shape, build};
use kosm::material::named;

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
    /// How far the torso leans forward over the hips, degrees. Positive is
    /// toward +y — into the door, into the swing, into whatever it is looking
    /// at. See [`Rig::lean`].
    pub torso_lean: f64,
    /// How far the right boot is ahead of the left. Positive puts the right
    /// foot forward, so the *left* is the back foot and the weight goes on
    /// it. The boots split this between them.
    pub stride: f64,
    /// Where the hips sit between the feet, along the facing axis. Negative
    /// is back over the rear foot, which with a forward `torso_lean` is what
    /// "weight on the back foot" looks like from any angle.
    pub pelvis_y: f64,
    /// How far each boot is turned out, degrees. A figure whose toes point
    /// straight ahead is a figure standing to attention.
    pub toe_out: f64,
    /// Whether the satchel and its strap are drawn. One knob because a
    /// turntable wants them and a silhouette test does not.
    pub satchel: bool,
}

impl Rig {
    /// The default figure: 1.11 m, arms down, looking level, weight even.
    ///
    /// The head is 456 mm across on a 1113 mm frame — two and a half heads
    /// tall, where the last pass was two and three-quarters. That single
    /// change is most of what makes it read at twenty metres: at that size
    /// the head is the silhouette and everything else is a plinth for it.
    pub const DEFAULT: Rig = Rig {
        height: 1113.0,
        head_r: 228.0,
        head_z: 885.0,
        neck_z: 646.0,
        shoulder_x: 174.0,
        shoulder_z: 588.0,
        chest_r: 180.0,
        chest_z: 452.0,
        hip_x: 94.0,
        hip_z: 306.0,
        skirt_r: 268.0,
        skirt_z: 248.0,
        foot_x: 118.0,
        ankle_z: 96.0,
        leg_r: 56.0,
        boot_r: 100.0,
        upper_arm: 208.0,
        forearm: 196.0,
        arm_r: 64.0,
        hand_r: 80.0,
        hand_right: [258.0, 66.0, 372.0],
        hand_left: [-258.0, 66.0, 372.0],
        head_tilt: 0.0,
        torso_lean: 0.0,
        stride: 0.0,
        pelvis_y: 0.0,
        toe_out: 11.0,
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
            torso_lean: knob("torso_lean_deg", d.torso_lean),
            stride: knob("stride_mm", d.stride),
            pelvis_y: knob("pelvis_y_mm", d.pelvis_y),
            toe_out: knob("toe_out_deg", d.toe_out),
            satchel: knob("satchel", if d.satchel { 1.0 } else { 0.0 }) > 0.5,
        }
    }

    /// The rig a set of knobs describes, without building anything.
    pub fn of(params: &Params) -> Rig {
        Rig::read(&|name, default| params.get(name).unwrap_or(default))
    }

    /// The torso pivot: the point everything above the hips turns about.
    ///
    /// Between the hip joints and at their height, so a lean is a rotation of
    /// the pelvis and not a shear of the waist. This is the pivot
    /// `kosm::player` will put a revolute joint on.
    pub fn torso_pivot(&self) -> [f64; 3] {
        [0.0, self.pelvis_y, self.hip_z]
    }

    /// A point of the *upper body*, leaned. Everything from the waist up —
    /// the cloak's chest, the mantle, the collar, the shoulders, the neck and
    /// the head — passes through here, and nothing below the hips does.
    ///
    /// `rotate_x(θ)` takes `+z` to `−y·sin θ`, so a **forward** lean is a
    /// negative rotation. That sign is the one thing about this function that
    /// can be wrong without failing to compile, so it is asserted in
    /// [`tests::the_torso_leans_forward_about_the_hips`].
    pub fn lean(&self, p: [f64; 3]) -> [f64; 3] {
        let (s, c) = (-self.torso_lean).to_radians().sin_cos();
        let o = self.torso_pivot();
        let (y, z) = (p[1] - o[1], p[2] - o[2]);
        [p[0], o[1] + y * c - z * s, o[2] + y * s + z * c]
    }

    /// The same as a `Shape` transform: build it in the un-leaned frame, then
    /// take the pivot to the origin, turn, and put it back.
    fn lean_shape(&self, shape: Shape) -> Shape {
        let o = self.torso_pivot();
        shape
            .at(-o[0], -o[1], -o[2])
            .rotate_x(-self.torso_lean)
            .at(o[0], o[1], o[2])
    }

    /// The shoulder joint on a side: +1 is the hero's right. Leaned, because
    /// the shoulders are on the torso and the hands are not — which is the
    /// whole reason a lean is worth having. A hand target stays where the
    /// pose put it and the arm re-solves to reach it from the new shoulder.
    pub fn shoulder(&self, side: f64) -> [f64; 3] {
        self.lean([side * self.shoulder_x, 0.0, self.shoulder_z])
    }

    /// The hip joint on a side.
    pub fn hip(&self, side: f64) -> [f64; 3] {
        [side * self.hip_x, self.pelvis_y, self.hip_z]
    }

    /// The ankle on a side, with the stride split between the two feet.
    pub fn ankle(&self, side: f64) -> [f64; 3] {
        [side * self.foot_x, side * 0.5 * self.stride, self.ankle_z]
    }

    /// The neck pivot: where the head hinges, leaned with the torso.
    pub fn neck(&self) -> [f64; 3] {
        self.lean([0.0, 0.0, self.neck_z])
    }

    /// How far a hand can get from its shoulder: the two links, straight.
    pub fn reach(&self) -> f64 {
        self.upper_arm + self.forearm
    }

    /// The hand target on a side.
    pub fn hand(&self, side: f64) -> [f64; 3] {
        if side > 0.0 { self.hand_right } else { self.hand_left }
    }

    /// Every joint this figure has, with the axis it turns about.
    ///
    /// The point of the struct is that it is the *only* place a downstream
    /// rig has to look. `kosm::player` builds phyz's tree from it: a hinge
    /// per entry, in the order given, each one's parent the entry before it
    /// up the chain. Millimetres, hero-local, in the pose the rig describes —
    /// so a `Pivots` read off `Rig::DEFAULT` is the rest pose and a `Pivots`
    /// read off a posed rig is where those joints actually are.
    pub fn pivots(&self) -> Pivots {
        let solve_knee = |side: f64| {
            let hip = self.hip(side);
            let ankle = self.ankle(side);
            joint(hip, ankle, self.thigh(), self.shin(), KNEE_HINT)
        };
        let solve_elbow = |side: f64| {
            let s = self.shoulder(side);
            joint(s, self.hand(side), self.upper_arm, self.forearm, elbow_hint(side))
        };
        Pivots {
            torso: self.torso_pivot(),
            neck: self.neck(),
            hip: [self.hip(1.0), self.hip(-1.0)],
            knee: [solve_knee(1.0), solve_knee(-1.0)],
            ankle: [self.ankle(1.0), self.ankle(-1.0)],
            shoulder: [self.shoulder(1.0), self.shoulder(-1.0)],
            elbow: [solve_elbow(1.0), solve_elbow(-1.0)],
            hand: [self.hand_right, self.hand_left],
        }
    }

    /// How much longer the two leg bones are than the drop from hip to ankle.
    ///
    /// This number is not a detail: it is the *whole* of how far this figure
    /// can move its feet, and it fights itself. Slack is what puts a bend in
    /// the knee, and slack is also what lets a foot step forward or plant
    /// wide — but the bend goes as `sqrt(thigh² − along²)`, so a leg with
    /// enough slack to take a stride has a knee bulging eighty millimetres
    /// out through the front of a boot. At 210 mm of leg, a metre-ten Mii
    /// cannot stride, and pretending otherwise puts a bump in the picture.
    ///
    /// So: fourteen millimetres, which is a 37 mm bend at rest — a leg that
    /// is visibly not straight and is still inside its own boot — and every
    /// pose in `sims/rune/hero` keeps its feet inside the 74 mm of travel
    /// that leaves. The weight shift is done by [`Rig::pelvis_y`] and
    /// [`Rig::torso_lean`] instead, which cost nothing and read further.
    pub const LEG_SLACK: f64 = 14.0;

    /// The thigh's length. A shade over half the hip-to-ankle run, so the
    /// knee sits high and the shin is the longer bone — a child's
    /// proportions, and what keeps the boot looking heavy.
    pub fn thigh(&self) -> f64 {
        0.53 * (self.hip_z - self.ankle_z + Rig::LEG_SLACK)
    }

    /// The shin's, so that the two together are the drop plus the slack.
    pub fn shin(&self) -> f64 {
        self.hip_z - self.ankle_z + Rig::LEG_SLACK - self.thigh()
    }

    /// How far the ankle on a side is from its hip, against how far it may
    /// be. Over one and the leg has straightened and the boot is off the
    /// sand; the poses in this crate all stay under it, and
    /// [`tests::every_pose_this_crate_uses_keeps_its_boots_on_the_sand`]
    /// is what says so.
    pub fn leg_strain(&self, side: f64) -> f64 {
        let d = sub(self.ankle(side), self.hip(side));
        norm(d) / (self.thigh() + self.shin())
    }
}

/// Where every hinge is, in one struct, so nothing downstream re-derives one.
///
/// Paired fields are `[right, left]`, the hero's own right first — the same
/// `+1 / −1` side convention the whole file uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pivots {
    /// The waist. Turns about `+x`; positive lean is forward, toward `+y`.
    pub torso: [f64; 3],
    /// The base of the skull. Turns about `+x`; positive tilt looks up.
    pub neck: [f64; 3],
    /// Ball joints.
    pub hip: [[f64; 3]; 2],
    /// Hinges about `+x`, bending the shin backward.
    pub knee: [[f64; 3]; 2],
    pub ankle: [[f64; 3]; 2],
    /// Ball joints.
    pub shoulder: [[f64; 3]; 2],
    /// Hinges; the axis is `(hand − elbow) × (elbow − shoulder)`, which is
    /// what a two-link solve leaves behind and is recomputed rather than
    /// stored, because it moves with the pose.
    pub elbow: [[f64; 3]; 2],
    /// Not a joint: the IK target the arm was solved to. Here because the
    /// controller's arm PD drives exactly this.
    pub hand: [[f64; 3]; 2],
}

impl Pivots {
    /// The elbow's hinge axis on a side, normalised.
    pub fn elbow_axis(&self, side: usize) -> [f64; 3] {
        let a = sub(self.hand[side], self.elbow[side]);
        let b = sub(self.elbow[side], self.shoulder[side]);
        let n = cross(a, b);
        if norm(n) < 1e-9 { [1.0, 0.0, 0.0] } else { normalize(n) }
    }

    /// The knee's, the same way.
    pub fn knee_axis(&self, side: usize) -> [f64; 3] {
        let a = sub(self.ankle[side], self.knee[side]);
        let b = sub(self.knee[side], self.hip[side]);
        let n = cross(a, b);
        if norm(n) < 1e-9 { [1.0, 0.0, 0.0] } else { normalize(n) }
    }
}

/// Which way a knee bends: forward, because that is the way a knee bends.
const KNEE_HINT: [f64; 3] = [0.0, 1.0, 0.0];

/// Which way an elbow points, per side: out and back.
///
/// Mostly *out*. The last pass leaned this hint backward and the arms folded
/// in against the ribs; a Switch Sports figure holds its elbows clear of its
/// body, and at these proportions "clear" is the difference between a
/// silhouette with two holes in it and one without.
fn elbow_hint(side: f64) -> [f64; 3] {
    [side * 1.0, -0.34, -0.16]
}

/// The hero, at whatever pose the knobs describe.
pub fn figure(params: &Params) -> anyhow::Result<Built> {
    build(params, assemble)
}

/// A named substance from the library, or a panic naming what is missing.
///
/// Not a fallback: a costume part whose substance the library does not know
/// is a bug in one of two files, and a grey default would hide it until the
/// day somebody asked this figure how much it weighs.
fn stuff(name: &str) -> kosm::material::Material {
    named(name).unwrap_or_else(|| panic!("`{name}` is not in kosm::material"))
}

/// Emit the whole figure into a builder.
///
/// Every part is its own named body with its own substance, so the picture
/// can re-colour or drop any one of them and a later rig can hang a joint off
/// any one of them.
fn assemble(b: &Builder) {
    let r = Rig::read(&|name, default| b.param(name, default));
    let (cloak, cream, skin, ink, blush, boot, leather, brass) = (
        stuff("cloak"),
        stuff("cream"),
        stuff("skin"),
        stuff("ink"),
        stuff("blush"),
        stuff("boot"),
        stuff("leather"),
        stuff("brass"),
    );

    // ---- the legs and the boots -----------------------------------------
    // Hip → knee → ankle. The knee is solved and told to bulge forward (+y),
    // which is the way a knee bends; the boot is a ball with a toe, turned
    // out about the ankle so the stance is a stance and not an attention.
    for (side, tag) in [(1.0, "r"), (-1.0, "l")] {
        let hip = r.hip(side);
        let ankle = r.ankle(side);
        let knee = joint(hip, ankle, r.thigh(), r.shin(), KNEE_HINT);
        b.body(&format!("leg_{tag}"))
            .substance(&cream)
            .add(bone(b, hip, knee, r.leg_r).union(bone(b, knee, ankle, r.leg_r * 0.92)));
        // Built about the origin so the turn-out is a rotation about the
        // ankle, then carried to the ankle. A boot rotated in place and *then*
        // translated is a boot that has swung out from under the leg.
        // The boot also sits *outboard* of its ankle. That is what plants the
        // stance wide without asking the legs for a splay they have not got:
        // 118 mm of foot plus 26 of offset plus a 100 mm ball is a boot whose
        // outer edge is 244 mm off the centre line, against a cloak that
        // flares to 268 — so the feet come very nearly out to the hem and the
        // figure stands on a base as wide as it is.
        let out = side * 26.0;
        b.body(&format!("boot_{tag}")).substance(&boot).add(
            b.sphere(r.boot_r)
                .at(out, 0.0, r.boot_r * 0.94 - r.ankle_z)
                .union(b.sphere(r.boot_r * 0.72).at(out, 116.0, r.boot_r * 0.60 - r.ankle_z))
                .union(b.sphere(r.boot_r * 0.85).at(out, 58.0, r.boot_r * 0.84 - r.ankle_z))
                .rotate_z(-side * r.toe_out)
                .at(ankle[0], ankle[1], ankle[2]),
        );
    }

    // ---- the cloak -------------------------------------------------------
    // A flared skirt and a capsule chest, unioned in one body because they
    // are one garment. The cone's wide end is at the bottom: vcad's cone has
    // its base at z = 0, so it is placed at the hem and rises to the waist.
    //
    // The skirt hangs from the hips and the chest leans off them, so the two
    // halves are *not* under the same transform: only the chest is leaned.
    let waist_r = 0.71 * r.skirt_r;
    let skirt = b
        .cone(r.skirt_r, waist_r, r.chest_z - r.skirt_z + 30.0)
        .at(0.0, r.pelvis_y * 0.35, r.skirt_z);
    let chest = r.lean_shape(
        b.cylinder(r.chest_r, r.shoulder_z - r.chest_z)
            .at(0.0, 0.0, r.chest_z)
            .union(b.sphere(r.chest_r).at(0.0, 0.0, r.chest_z))
            .union(b.sphere(r.chest_r).at(0.0, 0.0, r.shoulder_z)),
    );
    b.body("cloak").substance(&cloak).add(skirt.union(chest));

    // The belt: a ring around the cloak at the waist, dark, so the figure has
    // a middle. The cone's radius there, read off the taper.
    let belt_z = r.skirt_z + 0.62 * (r.chest_z - r.skirt_z);
    let belt_r = r.skirt_r + (waist_r - r.skirt_r) * (belt_z - r.skirt_z) / (r.chest_z - r.skirt_z + 30.0);
    b.body("belt")
        .substance(&boot)
        .add(b.torus(belt_r, 24.0).at(0.0, r.pelvis_y * 0.35, belt_z));

    // ---- the mantle -------------------------------------------------------
    // A short cape from the collar to just under the shoulders. It is here
    // for exactly one view: the back. A hooded figure seen from behind is a
    // ball over a cylinder and reads as nothing; a mantle gives that view a
    // hem, a shoulder line and a place for the strap to cross.
    //
    // It flares 54 mm past the chest at its hem, which is enough to catch the
    // sun on one shoulder and leave the other in its own shadow.
    let mantle_lo = r.chest_z + 30.0;
    let mantle_hi = r.shoulder_z + 44.0;
    let mantle_r = 1.30 * r.chest_r;
    b.body("mantle").substance(&cloak).add(r.lean_shape(
        b.cone(mantle_r, 0.72 * r.chest_r, mantle_hi - mantle_lo).at(0.0, 0.0, mantle_lo),
    ));
    // …and its piping. Cream at the hem is the second bright ring on the
    // figure and the one that survives at twenty metres, because it is the
    // widest.
    b.body("mantle_trim").substance(&cream).add(r.lean_shape(
        b.torus(mantle_r, 15.0).at(0.0, 0.0, mantle_lo),
    ));

    // ---- the arms --------------------------------------------------------
    // Shoulder → elbow → hand, the elbow solved and told to bulge outward so
    // the arms never fold through the chest. The shoulder is leaned and the
    // hand is not, which is what makes a lean a pose rather than a rotation
    // of the whole figure.
    for (side, tag) in [(1.0, "r"), (-1.0, "l")] {
        let shoulder = r.shoulder(side);
        let hand = r.hand(side);
        let elbow = joint(shoulder, hand, r.upper_arm, r.forearm, elbow_hint(side));
        b.body(&format!("arm_{tag}"))
            .substance(&cloak)
            .add(bone(b, shoulder, elbow, r.arm_r).union(bone(b, elbow, hand, r.arm_r * 0.9)));
        // The cuff, where the sleeve ends and the hand begins. Fatter than
        // the sleeve by a tenth, so it reads as a turned-back edge of cloth
        // rather than as a paler stretch of the same sleeve.
        b.body(&format!("cuff_{tag}"))
            .substance(&cream)
            .add(b.sphere(r.arm_r * 1.10).at(
                hand[0] - 0.30 * (hand[0] - elbow[0]),
                hand[1] - 0.30 * (hand[1] - elbow[1]),
                hand[2] - 0.30 * (hand[2] - elbow[2]),
            ));
        b.body(&format!("hand_{tag}"))
            .substance(&skin)
            .add(b.sphere(r.hand_r).at(hand[0], hand[1], hand[2]));
    }

    // ---- the collar ------------------------------------------------------
    // A cream ring where the cloak meets the neck: the brightest thing on the
    // figure, and the reason the eye goes to the head.
    b.body("collar").substance(&cream).add(r.lean_shape(
        b.torus(0.80 * r.chest_r, 44.0).at(0.0, 0.0, r.shoulder_z + 40.0),
    ));

    // ---- the head, the hood and the face ----------------------------------
    // All of it is built about the *neck pivot* at the origin, tipped by
    // `head_tilt_deg`, lifted to the neck, and then leaned with the torso.
    // Two rotations about two pivots, in that order, and they are the two
    // joints a rig would put there.
    let tip = |shape: Shape| r.lean_shape(shape.rotate_x(r.head_tilt).at(0.0, 0.0, r.neck_z));
    let h = r.head_r;
    let dz = r.head_z - r.neck_z;

    b.body("neck")
        .substance(&skin)
        .add(tip(b.cylinder(0.30 * h, dz).at(0.0, 0.0, -20.0)));
    b.body("head").substance(&skin).add(tip(b.sphere(h).at(0.0, 0.0, dz)));

    // The face: two dots and a mouth, low and small, the Mii way. Three marks
    // is a face; two is a mask and four is a portrait, and this stops at
    // three plus a cheek.
    let eye_r = 0.145 * h;
    let on_head = |n: [f64; 3], sink: f64, radius: f64| {
        let n = normalize(n);
        let s = h - sink * radius;
        [n[0] * s, n[1] * s, dz + n[2] * s]
    };
    for side in [1.0, -1.0] {
        let p = on_head([side * 0.30, 0.925, 0.02], 0.60, eye_r);
        let tag = if side > 0.0 { "r" } else { "l" };
        b.body(&format!("eye_{tag}"))
            .substance(&ink)
            .add(tip(b.sphere(eye_r).at(p[0], p[1], p[2])));
        // The cheek: a rose disc, flat, outboard of the eye and below it,
        // and **inside the cream lining ring** — a blush that straddles the
        // piping is a smudge. It is sunk to four-fifths of its own radius so
        // what shows is a shallow cap and not a ball: a Mii's blush, not a
        // boil.
        let cheek_r = 0.135 * h;
        let c = on_head([side * 0.45, 0.88, -0.19], 0.82, cheek_r);
        b.body(&format!("cheek_{tag}"))
            .substance(&blush)
            .add(tip(b.sphere(cheek_r).at(c[0], c[1], c[2])));
    }
    // The mouth: a short bar rather than a dot, low, and sat *on* the surface
    // rather than under it — a dot placed a whole radius inside a 228 mm head
    // is a dot nobody ever sees. Fifty millimetres wide on a 456 mm head,
    // which is as much mouth as a Mii has.
    let mouth_r = 0.30 * eye_r;
    let m = |x: f64| on_head([x, 0.905, -0.40], 0.52, mouth_r);
    b.body("mouth").substance(&ink).add(tip(bone(b, m(-0.055), m(0.055), mouth_r)));

    // The hood, in three analytic pieces and no boolean. See the module
    // header for what was tried and why this is what is left.
    //
    // The **roll** is the piece that does the work. It is a torus set forward
    // of the head's equator and a little below its centre, with a hole of
    // `0.55 · head_r` and an outer edge at `1.15 · head_r`: the face bulges
    // through the hole, the cloth stands proud of the head all the way round,
    // and the ring's bottom arrives at the collar, which is where a cowl
    // meets a cloak. Everything else is support — a crown that reaches from
    // the nape forward to meet the roll, and a fold of cloth at the nape.
    //
    // **The three of them have to *overlap*, and that is not automatic.** The
    // first version of this had a crown ball 12 mm larger than the head and a
    // roll whose inner lip was a hundred and thirty millimetres out, and
    // between them, right over the temples and the brow, ran a three
    // millimetre band where neither reached — so the render showed a stripe
    // of bare scalp arcing over the hood. The fix is arithmetic and not
    // taste: sample the head's own sphere, ask of every point whether the
    // crown ball or the roll's tube contains it, and move the four numbers
    // until the only exposed patch is the face. What that leaves is an
    // opening 35° across at the top, 41° at the sides and 50° at the chin —
    // an egg, wider below, which is the shape a drawstring makes and is why
    // the cheeks and the mouth fit under it and the forehead does not.
    let hole = 0.55 * h; // what a face has to fit through
    let proud = 1.15 * h; // and how far the cloth stands past the head
    let ring_r = 0.5 * (hole + proud); // major radius of the roll
    let roll = 0.5 * (proud - hole); // and its tube
    let ring_y = 0.50 * h;
    let ring_z = -0.18 * h; // the opening sits low, so the cowl covers the crown
    b.body("hood").substance(&cloak).add(tip(
        b.sphere(h + 28.0)
            .at(0.0, -0.18 * h, dz + 0.06 * h)
            // the roll around the opening: it is what a face is framed by
            .union(b.torus(ring_r, roll).rotate_x(90.0).at(0.0, ring_y, dz + ring_z))
            // and a fold of cloth at the nape, where the cowl meets the mantle
            .union(b.sphere(0.46 * h).at(0.0, -0.60 * h, dz - 0.46 * h))
            // …and the seam over the crown, as an arc of capsules.
            //
            // From behind, a hood is one teal ball on one teal body with no
            // edge anywhere in it, and a raised seam gives that view a
            // highlight and a shadow without putting a stripe of a second
            // colour down the back of a head. It stands ten millimetres
            // proud, which is what makes it read at all.
            //
            // It is an **arc** and not a torus, and that is the whole reason
            // this is eight lines instead of one. A ring standing proud of
            // the crown stands proud everywhere, including where it crosses
            // the hood's own opening — where there is no hood to stand proud
            // of, only forehead, so a full torus drove a teal spike a hundred
            // and fifty millimetres down between the eyes. The arc stops at
            // `SEAM_FROM`, above the opening's 35° lip, and runs back to the
            // nape.
            .union(seam(b, [0.0, -0.18 * h, dz + 0.06 * h], h + 26.0, 13.0)),
    ));
    // The lining: cream piping laid *on the head*, just inside the roll's
    // hole. On the head and not between the two, because anything drawn in
    // the gap is inside one of them — the head is 228 mm of solid and the
    // roll is 118 mm of tube.
    let lining_r = 0.53 * h;
    let lining_y = (h * h - lining_r * lining_r).max(1.0).sqrt();
    b.body("hood_lining").substance(&cream).add(tip(
        b.torus(lining_r, 0.14 * roll).rotate_x(90.0).at(0.0, lining_y, dz),
    ));

    // ---- the satchel -------------------------------------------------------
    // On the left hip, over the cloak, with a strap across the chest. It is
    // the one warm colour on the figure and the one hard-edged box, so it
    // does two jobs: it breaks the silhouette and it says "adventurer".
    if r.satchel {
        // Outside the skirt, not inside it. The cloak flares to `skirt_r` at
        // the hem and a bag tucked within that radius is a bag nobody ever
        // sees; this one hangs clear of the cloth on the hero's left, low
        // enough to break the flare's silhouette.
        let bag = [-0.90 * r.skirt_r - 40.0, -30.0, r.hip_z + 40.0];
        b.body("satchel").substance(&leather).add(
            b.boxed(220.0, 112.0, 176.0)
                .at(bag[0], bag[1], bag[2])
                .union(b.boxed(234.0, 126.0, 64.0).at(bag[0], bag[1] - 6.0, bag[2] + 74.0)),
        );
        b.body("buckle")
            .substance(&brass)
            .add(b.cylinder(30.0, 22.0).rotate_x(90.0).at(bag[0], bag[1] - 70.0, bag[2] + 36.0));
        // The strap: over the *right* shoulder, down the front of the mantle
        // to the bag on the left hip, and back up the mantle's other side —
        // which is the whole of what a shoulder bag's strap does. Russet on
        // teal, so it is the one mark that says which way this figure is
        // facing from any angle, and the reason the satchel is worn on one
        // side at all. The last pass ran it to `y = −0.98 · chest_r`, which
        // is the *back*: `+y` is the way this figure faces.
        //
        // **Every waypoint is *on the cone*.** The mantle is a frustum and a
        // chord between two points on a frustum runs *inside* it: a version
        // with one point on each shoulder and one at the hem lost the middle
        // of the strap into the cloth and left two russet stubs. So the
        // mantle's radius is a function, the waypoints are taken off it at a
        // given height and bearing with a few millimetres of clearance, and
        // the strap is the polyline through them.
        let mantle_at = |z: f64| {
            let t = ((z - mantle_lo) / (mantle_hi - mantle_lo)).clamp(0.0, 1.0);
            mantle_r + (0.72 * r.chest_r - mantle_r) * t
        };
        // `bearing` is measured from the front, `+` toward the hero's right.
        let on_mantle = |z: f64, bearing: f64, clear: f64| {
            let a: f64 = bearing.to_radians();
            let rad = mantle_at(z) + clear;
            r.lean([rad * a.sin(), rad * a.cos(), z])
        };
        let shoulder = on_mantle(r.shoulder_z + 16.0, 62.0, 14.0);
        let bag_front = [bag[0] + 70.0, bag[1] + 40.0, bag[2] + 104.0];
        let bag_back = [bag[0] + 46.0, bag[1] - 52.0, bag[2] + 104.0];
        let strap = b.body("strap");
        strap.substance(&leather);
        let run = |points: [[f64; 3]; 4]| {
            let mut shape = bone(b, points[0], points[1], 27.0);
            for pair in points.windows(2).skip(1) {
                shape = shape.union(bone(b, pair[0], pair[1], 27.0));
            }
            strap.add(shape);
        };
        run([shoulder, on_mantle(566.0, 34.0, 16.0), on_mantle(494.0, 13.0, 18.0), bag_front]);
        run([shoulder, on_mantle(566.0, 124.0, 16.0), on_mantle(494.0, 166.0, 18.0), bag_back]);
    }
}

// ---- the two bits of geometry the assembly needs ---------------------------

/// Where the crown's seam starts and stops, degrees about the crown's centre
/// measured from straight ahead: `0` is the brow, `90` the top, `180` the
/// back of the skull, `270` the throat.
const SEAM_FROM: f64 = 48.0;
const SEAM_TO: f64 = 208.0;

/// A raised seam over a ball: an arc of capsules in the `yz` plane, centred
/// on `about`, at `radius`, each capsule `tube` thick.
///
/// Sixteen segments over a hundred and sixty degrees is a chord sagitta of
/// half a millimetre on a 254 mm arc — under a tenth of the tube — so it
/// reads as a smooth ridge and is made of nothing but capsules.
fn seam(b: &Builder, about: [f64; 3], radius: f64, tube: f64) -> Shape {
    let n = 16;
    let at = |i: usize| {
        let a = (SEAM_FROM + (SEAM_TO - SEAM_FROM) * i as f64 / n as f64).to_radians();
        [about[0], about[1] + radius * a.cos(), about[2] + radius * a.sin()]
    };
    let mut arc = bone(b, at(0), at(1), tube);
    for i in 1..n {
        arc = arc.union(bone(b, at(i), at(i + 1), tube));
    }
    arc
}

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
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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

    /// The figure is the height it says it is, every part of it is a named
    /// body, and every one of those names is a substance the library knows.
    #[test]
    fn the_hero_is_a_metre_ten_and_all_of_it_is_a_substance() -> anyhow::Result<()> {
        let r = Rig::DEFAULT;
        assert!((r.head_z + r.head_r - r.height).abs() < 10.0, "the crown is the height");
        // the head is the character: two and a half heads tall, no more
        let heads = r.height / (2.0 * r.head_r);
        assert!((2.3..2.6).contains(&heads), "the figure is {heads:.2} heads tall");
        let built = figure(&Params::default())?;
        for name in [
            "head", "hood", "hood_lining", "cloak", "mantle", "mantle_trim", "arm_r", "boot_l",
            "satchel", "strap", "eye_l", "cheek_r", "mouth",
        ] {
            assert!(built.bodies.iter().any(|b| b.name == name), "no `{name}` body");
        }
        for body in &built.bodies {
            let s = body
                .substance()
                .unwrap_or_else(|| panic!("`{}` is `{}`, which is not a substance", body.name, body.material));
            assert!(s.density > 0.0, "`{}` is made of {} and weighs nothing", body.name, s.name);
        }
        Ok(())
    }

    /// A pose is the hands, and the elbows follow them. Both halves matter:
    /// the solve has to reach the target when it can — the three below are a
    /// hand at rest, a hand holding the lens up at the doorstep, and a hand
    /// straight out, all of them inside the arm's 404 mm — and straighten
    /// toward it when it cannot.
    #[test]
    fn the_elbow_follows_the_hand_and_saturates_rather_than_snapping() {
        let r = Rig::DEFAULT;
        let s = r.shoulder(1.0);
        for hand in [[258.0, 66.0, 372.0], [110.0, 210.0, 910.0], [390.0, 0.0, 590.0]] {
            let e = joint(s, hand, r.upper_arm, r.forearm, elbow_hint(1.0));
            assert!((norm(sub(e, s)) - r.upper_arm).abs() < 1e-6, "the upper arm changed length");
            assert!((norm(sub(hand, e)) - r.forearm).abs() < 1e-6, "the forearm changed length");
            // and it points out, away from the ribs, which is the whole job
            // of the hint
            assert!(e[0] > s[0] - 1e-9, "the elbow folded across the chest");
        }
        // out of reach: the arm straightens and still points at the target
        let far = [0.0, 3000.0, 600.0];
        let e = joint(s, far, r.upper_arm, r.forearm, elbow_hint(1.0));
        let straight = normalize(sub(far, s));
        let got = normalize(sub(e, s));
        assert!(dot(straight, got) > 0.999, "an unreachable target did not straighten the arm");
    }

    /// The torso pivot: a lean turns everything above the hips about the hips
    /// and nothing below them, it goes **forward** for a positive angle, and
    /// it leaves the hands where the pose put them.
    ///
    /// That last clause is the reason the joint is worth having. The doorstep
    /// solves where the lens must be to put the sun in the keyhole; if a lean
    /// carried the hands with it the caustic would move and the shot would be
    /// solved twice. It does not, so a lean is free.
    #[test]
    fn the_torso_leans_forward_about_the_hips() -> anyhow::Result<()> {
        let mut params = Params::new();
        params.set("torso_lean_deg", 14.0);
        let leaned = Rig::of(&params);
        let level = Rig::DEFAULT;

        // the pivot itself does not move
        assert_eq!(leaned.torso_pivot(), level.torso_pivot());
        assert_eq!(leaned.lean(leaned.torso_pivot()), leaned.torso_pivot());
        // below the hips, nothing moves at all
        for side in [1.0, -1.0] {
            assert_eq!(leaned.ankle(side), level.ankle(side));
            assert_eq!(leaned.hip(side), level.hip(side));
        }
        // above them, everything leans forward — toward +y, and by the arc
        // the angle asks for
        let (a, b) = (level.shoulder(1.0), leaned.shoulder(1.0));
        assert!(b[1] > a[1] + 50.0, "the shoulder went to {b:?} from {a:?}");
        assert!(b[2] < a[2], "and it should drop as it goes");
        let arm = level.shoulder_z - level.hip_z;
        let swung = ((b[1] - level.pelvis_y).powi(2) + (b[2] - level.hip_z).powi(2)).sqrt();
        assert!((swung - arm).abs() < 1e-9, "the lean is not a rotation: {swung} vs {arm}");
        assert!(((b[1] - level.pelvis_y) / arm - 14f64.to_radians().sin()).abs() < 1e-9);
        // the head leans with it and the hands do not
        assert!(leaned.neck()[1] > level.neck()[1] + 50.0);
        assert_eq!(leaned.hand_right, level.hand_right);

        // and the whole of that reaches the CAD: a leaned figure is a
        // different document, not the same one under a camera trick
        let built = figure(&params)?;
        assert_eq!(built.param("torso_lean_deg"), Some(14.0));
        Ok(())
    }

    /// Every hinge a controller will need, in one struct, in the pose the rig
    /// describes — and each solved joint really is on its two bones.
    #[test]
    fn the_pivots_are_the_joints_the_assembly_actually_built() {
        let mut params = Params::new();
        params.set("torso_lean_deg", 9.0);
        params.set("stride_mm", 76.0);
        params.set("pelvis_y_mm", -26.0);
        let r = Rig::of(&params);
        let p = r.pivots();

        assert_eq!(p.torso, [0.0, -26.0, r.hip_z]);
        assert_eq!(p.neck, r.neck());
        // the stride splits between the feet: right forward, left back
        assert!(p.ankle[0][1] > 0.0 && p.ankle[1][1] < 0.0);
        assert!((p.ankle[0][1] - p.ankle[1][1] - 76.0).abs() < 1e-9);
        // the hips carry the pelvis offset and the ankles do not
        assert_eq!(p.hip[0][1], -26.0);
        for side in 0..2 {
            let s = if side == 0 { 1.0 } else { -1.0 };
            // the knee is on its two bones
            assert!((norm(sub(p.knee[side], p.hip[side])) - r.thigh()).abs() < 1e-6);
            assert!((norm(sub(p.ankle[side], p.knee[side])) - r.shin()).abs() < 1e-6);
            // the elbow is on its two bones, and hangs off the leaned shoulder
            assert_eq!(p.shoulder[side], r.shoulder(s));
            assert!((norm(sub(p.elbow[side], p.shoulder[side])) - r.upper_arm).abs() < 1e-6);
            assert!((norm(sub(p.hand[side], p.elbow[side])) - r.forearm).abs() < 1e-6);
            // and the two solved hinges have an axis to turn about
            assert!((norm(p.elbow_axis(side)) - 1.0).abs() < 1e-9);
            assert!((norm(p.knee_axis(side)) - 1.0).abs() < 1e-9);
        }
    }

    /// Every pose this crate uses keeps both boots inside the leg's reach.
    ///
    /// A two-link solve that cannot reach its target straightens and points
    /// at it, which for an arm is the right failure and for a *leg* means the
    /// boot has left the sand — silently, in one panel of six. So the strain
    /// is a number and the poses are checked against it.
    ///
    /// The margin is thin on purpose: see [`Rig::LEG_SLACK`]. A metre-ten
    /// figure with 210 mm legs has about 74 mm of foot travel, and this is
    /// where that budget is spent.
    #[test]
    fn every_pose_this_crate_uses_keeps_its_boots_on_the_sand() {
        for (name, stride, pelvis) in [
            ("rest", 0.0, 0.0),
            ("stand", 62.0, -22.0),
            ("doorstep", 76.0, -26.0),
            ("portrait", 58.0, -20.0),
        ] {
            let mut params = Params::new();
            params.set("stride_mm", stride);
            params.set("pelvis_y_mm", pelvis);
            let r = Rig::of(&params);
            for side in [1.0, -1.0] {
                let strain = r.leg_strain(side);
                assert!(strain < 1.0, "`{name}` strains its leg to {strain:.3} of straight");
                // and the knee really is on both bones, which is the same
                // claim seen from the CAD's side
                let k = joint(r.hip(side), r.ankle(side), r.thigh(), r.shin(), KNEE_HINT);
                assert!((norm(sub(k, r.hip(side))) - r.thigh()).abs() < 1e-6);
                assert!((norm(sub(r.ankle(side), k)) - r.shin()).abs() < 1e-6);
            }
        }
        // and the budget is what the doc says it is: past 80 mm of foot
        // travel the leg is straight, which is why nothing above asks for it
        let mut params = Params::new();
        params.set("stride_mm", 200.0);
        assert!(Rig::of(&params).leg_strain(1.0) > 1.0, "the leg reached further than it can");
    }

    /// A knob read off a `Params` and a knob read off a `Builder` are the
    /// same knob — the thing that keeps the pose the caller solved and the
    /// pose the CAD built from being two different poses.
    #[test]
    fn the_rig_reads_the_same_from_either_side() -> anyhow::Result<()> {
        let mut params = Params::new();
        params.set("hand_r_z_mm", 1024.0);
        params.set("head_tilt_deg", 21.0);
        params.set("torso_lean_deg", 7.0);
        let rig = Rig::of(&params);
        assert_eq!(rig.hand_right[2], 1024.0);
        assert_eq!(rig.head_tilt, 21.0);
        assert_eq!(rig.torso_lean, 7.0);
        let built = figure(&params)?;
        assert_eq!(built.param("hand_r_z_mm"), Some(1024.0));
        assert_eq!(built.param("head_tilt_deg"), Some(21.0));
        assert_eq!(built.param("torso_lean_deg"), Some(7.0));
        Ok(())
    }
}
