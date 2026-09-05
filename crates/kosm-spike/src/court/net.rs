//! The net: cord hung from the rim, integrated beside phyz rather than in it.
//!
//! A basketball net is a few hundred short lengths of nylon in a diamond
//! mesh. phyz is a rigid-body simulator, and a net is not rigid, so the net
//! lives here: nodes with mass, cords as springs, symplectic Euler at the
//! court's own `dt`, and a Jakobsen-style length pass afterwards so the mesh
//! cannot stretch absurdly when a ball drives through it.
//!
//! The rings alternate by half a strand and each node is tied to the two
//! nodes nearest it on the ring below, which is the mesh a real net is
//! knotted into: the diamonds scissor open when a ball goes through and the
//! cords pull them shut again.
//!
//! The coupling with the balls is **one way**. Each step, any node that has
//! ended up inside a ball is pushed out to the ball's surface and loses the
//! part of its velocity that was heading into the ball (measured against the
//! ball's own velocity, so a rising ball carries the net up with it). The
//! ball feels nothing back — a shot is not slowed by the net it passes
//! through. That is wrong by about a per cent of the ball's momentum and
//! right enough for the picture; giving the net a reaction force on the ball
//! means feeding it into phyz's contact solve, which is a later job.
//!
//! Metres and seconds here, like the rest of the court; `placed_solids` is
//! the one crossing into vcad's millimetres.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use phyz_math::{GRAVITY, Vec3};
use vcad_kernel::Solid;
use vcad_kernel_math::{Dir3, Transform};

use crate::scene::MM;

use super::parts::PlacedSolid;
use super::CourtScene;

/// One cord: two nodes and the length it wants to be.
#[derive(Clone, Copy, Debug)]
struct Cord {
    a: usize,
    b: usize,
    rest: f64,
}

/// A net hung from a rim.
pub struct Net {
    /// Node positions, ring by ring from the rim down; metres.
    pub nodes: Vec<Vec3>,
    /// Node velocities, metres per second.
    pub vel: Vec<Vec3>,
    /// Cord diameter in millimetres, for the picture.
    pub cord_mm: f64,
    cords: Vec<Cord>,
    /// The top ring hangs on the rim's rod and does not move.
    pinned: usize,
    strands: usize,
    rows: usize,
    dt: f64,
    /// Sub-steps per court step: the cords are stiffer than phyz's dt.
    substeps: usize,
    stiffness: f64,
    damping: f64,
    node_mass: f64,
    /// Cord radius in metres: how far a node stays off the floor.
    cord_r: f64,
    /// One cylinder per length, reused: the segments repeat every frame.
    solids: Mutex<HashMap<i64, Arc<Solid>>>,
}

/// How far a cord is allowed past its rest length before the constraint pass
/// pulls it back, and how many passes.
const MAX_STRETCH: f64 = 0.02;
const RELAX_PASSES: usize = 2;
/// Cylinders are cached per this much length. A cord is held within 2% of
/// its rest length, about a millimetre, so at 2 mm each cord lands in one or
/// two buckets for the whole run and the renderer's per-solid BVH cache
/// actually hits; at 0.2 mm every sway minted a new solid, a new BVH, and a
/// 20 s frame. Two millimetres on a 5 mm cord is not visible.
const LENGTH_BUCKET_MM: f64 = 2.0;

impl Net {
    /// Hang a net on the scene's rim, or `None` if the level asked for no
    /// strands.
    pub fn from_scene(scene: &CourtScene) -> anyhow::Result<Option<Self>> {
        let a = &scene.authored;
        let strands = a.parameter_or("net_strands", 0.0).round().max(0.0) as usize;
        let rows = a.parameter_or("net_rows", 0.0).round().max(0.0) as usize;
        if strands < 3 || rows == 0 {
            return Ok(None);
        }
        let length = a.parameter_or("net_length_mm", 400.0) * MM;
        let bottom_r = a.parameter_or("net_bottom_r_mm", 150.0) * MM;
        let cord_mm = a.parameter_or("net_cord_mm", 5.0);
        let stiffness = a.parameter_or("net_stiffness", 900.0);
        let damping = a.parameter_or("net_damping", 0.9);
        let mass = a.parameter_or("net_mass_g", 110.0) * 1e-3;
        let rod = a.millimetres("rim_rod_mm").unwrap_or(0.016);

        // The top ring sits on the rod's centreline, which is where the
        // level's rim ring is: radius rim_r + rod/2, top of the rod at rim_z.
        let top_r = scene.hoop.rim_r + 0.5 * rod;
        let centre = scene.hoop.rim_centre;
        let top_z = centre.z - 0.5 * rod;

        let step = std::f64::consts::TAU / strands as f64;
        let mut nodes = Vec::with_capacity(strands * (rows + 1));
        for ring in 0..=rows {
            let t = ring as f64 / rows as f64;
            let r = top_r + (bottom_r - top_r) * t;
            let z = top_z - length * t;
            // each ring turned half a strand from the one above it
            let phase = 0.5 * step * ring as f64;
            for s in 0..strands {
                let angle = phase + step * s as f64;
                nodes.push(Vec3::new(centre.x + r * angle.cos(), centre.y + r * angle.sin(), z));
            }
        }

        // the diamonds: each node to the two nodes half a strand either way
        // on the ring below
        let mut cords = Vec::with_capacity(2 * strands * rows);
        for ring in 0..rows {
            for s in 0..strands {
                let a = ring * strands + s;
                for b in [(ring + 1) * strands + s, (ring + 1) * strands + (s + strands - 1) % strands] {
                    let rest = (nodes[b] - nodes[a]).norm();
                    cords.push(Cord { a, b, rest });
                }
            }
        }

        let n = nodes.len();
        let node_mass = (mass / n as f64).max(1e-6);

        // Explicit springs are only stable while the step is short against
        // the node's own period, and a node hangs on up to four cords. Take
        // the busiest node, ask for a quarter of its period and half its
        // damping time, and substep the court's dt down to that.
        let mut degree = vec![0.0f64; n];
        for cord in &cords {
            degree[cord.a] += 1.0;
            degree[cord.b] += 1.0;
        }
        let busiest = degree.iter().copied().fold(1.0, f64::max);
        let h_spring = 0.25 / (busiest * stiffness / node_mass).sqrt();
        let h_damping = 0.5 * node_mass / (busiest * damping).max(1e-12);
        let substeps = (scene.dt / h_spring.min(h_damping)).ceil().clamp(1.0, 64.0) as usize;

        Ok(Some(Self {
            vel: vec![Vec3::zeros(); n],
            nodes,
            cord_mm,
            cords,
            pinned: strands,
            strands,
            rows,
            dt: scene.dt,
            substeps,
            stiffness,
            damping,
            node_mass,
            cord_r: 0.5 * cord_mm * MM,
            solids: Mutex::new(HashMap::new()),
        }))
    }

    /// Sub-steps the net takes for one court step.
    pub fn substeps(&self) -> usize {
        self.substeps
    }

    pub fn strands(&self) -> usize {
        self.strands
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn segments(&self) -> usize {
        self.cords.len()
    }

    /// The lowest node's height above the floor.
    pub fn lowest(&self) -> f64 {
        self.nodes.iter().map(|p| p.z).fold(f64::INFINITY, f64::min)
    }

    /// The worst cord's stretch as a fraction of its rest length.
    pub fn worst_stretch(&self) -> f64 {
        self.cords
            .iter()
            .map(|c| (self.nodes[c.b] - self.nodes[c.a]).norm() / c.rest - 1.0)
            .fold(0.0, f64::max)
    }

    /// One step: gravity and the cords, then the balls and the floor, then a
    /// length pass. `balls` is each ball's centre, its velocity and its
    /// radius.
    pub fn step(&mut self, balls: &[(Vec3, Vec3, f64)]) {
        let h = self.dt / self.substeps as f64;
        let mut force = vec![Vec3::zeros(); self.nodes.len()];
        for _ in 0..self.substeps {
            force.fill(Vec3::new(0.0, 0.0, -GRAVITY * self.node_mass));
            for cord in &self.cords {
                let d = self.nodes[cord.b] - self.nodes[cord.a];
                let len = d.norm();
                if len < 1e-9 {
                    continue;
                }
                let dir = d / len;
                let along = (self.vel[cord.b] - self.vel[cord.a]).dot(dir);
                let f = dir * (self.stiffness * (len - cord.rest) + self.damping * along);
                force[cord.a] += f;
                force[cord.b] -= f;
            }

            let inv_m = 1.0 / self.node_mass;
            for i in self.pinned..self.nodes.len() {
                self.vel[i] += force[i] * (inv_m * h);
                let v = self.vel[i];
                self.nodes[i] += v * h;
            }

            for _ in 0..RELAX_PASSES {
                self.relax();
            }
        }
        self.collide(balls);
    }

    /// Pull over-long cords back towards their rest length. Position only —
    /// the springs already carry the velocity — and only the free ends move.
    fn relax(&mut self) {
        let limit = 1.0 + MAX_STRETCH;
        for k in 0..self.cords.len() {
            let cord = self.cords[k];
            let d = self.nodes[cord.b] - self.nodes[cord.a];
            let len = d.norm();
            let want = cord.rest * limit;
            if len <= want || len < 1e-9 {
                continue;
            }
            let fix = d * ((len - want) / len);
            let (a_free, b_free) = (cord.a >= self.pinned, cord.b >= self.pinned);
            match (a_free, b_free) {
                (true, true) => {
                    self.nodes[cord.a] += fix * 0.5;
                    self.nodes[cord.b] -= fix * 0.5;
                }
                (true, false) => self.nodes[cord.a] += fix,
                (false, true) => self.nodes[cord.b] -= fix,
                (false, false) => {}
            }
        }
    }

    /// The balls push the net; the net does not push back (see the module
    /// note). The floor is a hard stop a cord radius up.
    fn collide(&mut self, balls: &[(Vec3, Vec3, f64)]) {
        for i in self.pinned..self.nodes.len() {
            for &(c, bv, r) in balls {
                let d = self.nodes[i] - c;
                let len = d.norm();
                if len >= r || len < 1e-9 {
                    continue;
                }
                let n = d / len;
                self.nodes[i] = c + n * r;
                // the ball's frame: only motion into the ball is removed
                let rel = self.vel[i] - bv;
                let into = rel.dot(n);
                if into < 0.0 {
                    self.vel[i] = bv + (rel - n * into);
                }
            }
            if self.nodes[i].z < self.cord_r {
                self.nodes[i].z = self.cord_r;
                if self.vel[i].z < 0.0 {
                    self.vel[i].z = 0.0;
                }
            }
        }
    }

    /// The net as vcad solids, in millimetres: one thin cylinder per cord,
    /// placed base-to-tip between its two nodes.
    pub fn placed_solids(&self, cord_mm: f64) -> Vec<PlacedSolid> {
        let r = 0.5 * cord_mm;
        let mut out = Vec::with_capacity(self.cords.len());
        for cord in &self.cords {
            let a = self.nodes[cord.a] / MM;
            let b = self.nodes[cord.b] / MM;
            let d = b - a;
            let len = d.norm();
            if !len.is_finite() || len < 1e-6 {
                continue;
            }
            out.push(PlacedSolid {
                solid: self.cylinder(r, len),
                to_world: Transform::translation(a.x, a.y, a.z).then(&up_to(d / len)),
                material: "net".into(),
            });
        }
        out
    }

    /// A cylinder of this length, from the cache if the length has been seen
    /// (bucketed: the segments repeat, and a fresh BRep per segment per frame
    /// is the expensive part).
    fn cylinder(&self, r: f64, len_mm: f64) -> Arc<Solid> {
        let key = (len_mm / LENGTH_BUCKET_MM).round() as i64;
        let mut cache = self.solids.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| Arc::new(Solid::cylinder(r, (key as f64 * LENGTH_BUCKET_MM).max(1e-3), 8)))
            .clone()
    }
}

/// The rotation that takes +z, which is where a vcad cylinder points, onto
/// the unit vector `d`.
fn up_to(d: Vec3) -> Transform {
    let cos = d.z.clamp(-1.0, 1.0);
    if cos > 1.0 - 1e-12 {
        return Transform::identity();
    }
    if cos < -1.0 + 1e-12 {
        return Transform::rotation_x(std::f64::consts::PI);
    }
    // axis = z × d
    let axis = Dir3::new_normalize(vcad_kernel_math::Vec3::new(-d.y, d.x, 0.0));
    Transform::rotation_about_axis(&axis, cos.acos())
}
