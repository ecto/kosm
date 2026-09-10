//! The sea, as the shader has it.
//!
//! `sims/pool/scene.wgsl` is the construction and this is the cove's version
//! of it: a surface at a fixed height whose normal comes from an analytic
//! swell, a Fresnel split between the sky reflected off it and the seabed
//! refracted through it, Beer–Lambert absorption over the path inside, and a
//! horizon fade so the lattice's far edge is not an edge.
//!
//! Two things differ from the pool's. The cove's seabed is a **plane at a
//! grade** rather than a flat floor — the sand runs on under the water at
//! `beach_slope` — so the refracted ray is intersected with that plane and
//! shaded with the *sand's* own material, which is the same entry the beach
//! above the waterline is drawn with. And the swell is the level's:
//! `sims/rune/render.rs::sea_field` sums two sine waves crossing at shallow
//! angles, and the same two are evaluated here in closed form rather than
//! sampled off a height lattice, so the raster's water and the tracer's are
//! the same surface written twice.
//!
//! ```
//! use kosm_view::raster::Sea;
//! // the cove's authored swell: 30 mm at 7 m, 20 mm at 3 m
//! let sea = Sea::cove(0.0, 0.06, -20.0, 40.0).with_swell(0.03, 7.0, 0.02, 3.0);
//! // a surface, not a plane: the height wanders by the two amplitudes
//! let h = sea.height(3.0, -4.0, 0.0);
//! assert!(h.abs() <= 0.05 + 1e-9);
//! ```

/// The sea's knobs, in world metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sea {
    /// The flat waterline height.
    pub z: f64,
    /// The beach's grade, `dz/dy`, which the seabed carries on at.
    pub slope: f64,
    /// The `y` at which the sand meets the water.
    pub waterline_y: f64,
    /// How far out to sea the surface is drawn before the horizon fade has
    /// finished, metres.
    pub reach: f64,
    /// The first swell: amplitude and wavelength, metres, and the angle its
    /// crests run at either side of the shore normal, radians.
    pub swell: [f64; 2],
    pub swell_angle: f64,
    /// The second.
    pub swell_b: [f64; 2],
    pub swell_b_angle: f64,
    /// How fast the crests walk in, radians a second per wave.
    pub speed: f64,
    /// The index of the water.
    pub ior: f64,
    /// What a metre of it takes out, per linear RGB channel.
    pub absorption: [f64; 3],
    /// The material the seabed is shaded with, and the one the surface's own
    /// body colour comes from.
    pub seabed_material: u32,
    pub water_material: u32,
}

impl Sea {
    /// The cove's sea: a waterline, the beach's grade, where they meet, and
    /// how far out the surface reaches.
    pub fn cove(z: f64, slope: f64, waterline_y: f64, reach: f64) -> Self {
        Self {
            z,
            slope,
            waterline_y,
            reach,
            // the level's own defaults, in metres — `swell_a_mm` and friends
            swell: [0.03, 7.0],
            swell_angle: 0.20,
            swell_b: [0.02, 3.0],
            swell_b_angle: -0.55,
            speed: 0.6,
            ior: 1.333,
            // teal: red goes first, blue travels furthest
            absorption: [0.45, 0.10, 0.06],
            seabed_material: 0,
            water_material: 0,
        }
    }

    /// The two authored swell components, metres.
    pub fn with_swell(mut self, a1: f64, l1: f64, a2: f64, l2: f64) -> Self {
        self.swell = [a1, l1.max(1e-3)];
        self.swell_b = [a2, l2.max(1e-3)];
        self
    }

    pub fn with_materials(mut self, seabed: u32, water: u32) -> Self {
        self.seabed_material = seabed;
        self.water_material = water;
        self
    }

    /// The surface's height above [`Self::z`] at a point and a time.
    ///
    /// The same sum `sea_field` fills its lattice with, evaluated rather than
    /// sampled. The shader has this function too; this one is what the tests
    /// hold it to, and what the CPU side uses to place the sea's own mesh.
    pub fn height(&self, x: f64, y: f64, t: f64) -> f64 {
        let (c1, s1) = self.swell_angle.sin_cos();
        let (c2, s2) = self.swell_b_angle.sin_cos();
        let k1 = std::f64::consts::TAU / self.swell[1];
        let k2 = std::f64::consts::TAU / self.swell_b[1];
        self.swell[0] * (k1 * (x * c1 + y * s1) - self.speed * t).sin()
            + self.swell_b[0] * (k2 * (x * c2 + y * s2) + 1.7 - self.speed * t).sin()
    }

    /// The surface's unit normal, from the analytic gradient of [`Self::height`].
    pub fn normal(&self, x: f64, y: f64, t: f64) -> [f64; 3] {
        let (c1, s1) = self.swell_angle.sin_cos();
        let (c2, s2) = self.swell_b_angle.sin_cos();
        let k1 = std::f64::consts::TAU / self.swell[1];
        let k2 = std::f64::consts::TAU / self.swell_b[1];
        let p1 = k1 * (x * c1 + y * s1) - self.speed * t;
        let p2 = k2 * (x * c2 + y * s2) + 1.7 - self.speed * t;
        let (a1, a2) = (self.swell[0] * k1 * p1.cos(), self.swell_b[0] * k2 * p2.cos());
        let dx = a1 * c1 + a2 * c2;
        let dy = a1 * s1 + a2 * s2;
        let n = [-dx, -dy, 1.0];
        let l = (n[0] * n[0] + n[1] * n[1] + 1.0).sqrt();
        [n[0] / l, n[1] / l, n[2] / l]
    }

    /// The seabed's height at a point: the beach's own plane, carried on
    /// under the water.
    pub fn seabed_z(&self, y: f64) -> f64 {
        self.z + self.slope * (y - self.waterline_y)
    }

    /// A flat lattice for the surface, `n × n` quads over the sea's own
    /// footprint. The swell is applied in the *vertex* shader from
    /// [`Self::height`], so this carries no heights of its own — which is why
    /// a moving swell costs one uniform and not a re-upload.
    pub fn lattice(&self, half_x: f64, n: u32) -> (Vec<[f64; 3]>, Vec<u32>) {
        let n = n.max(1);
        let (y0, y1) = (self.waterline_y - self.reach, self.waterline_y);
        let mut p = Vec::with_capacity(((n + 1) * (n + 1)) as usize);
        for j in 0..=n {
            let y = y0 + (y1 - y0) * j as f64 / n as f64;
            for i in 0..=n {
                let x = -half_x + 2.0 * half_x * i as f64 / n as f64;
                p.push([x, y, self.z]);
            }
        }
        let mut idx = Vec::with_capacity((n * n * 6) as usize);
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                idx.extend_from_slice(&[a, a + 1, a + n + 2]);
                idx.extend_from_slice(&[a, a + n + 2, a + n + 1]);
            }
        }
        (p, idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The normal is the analytic gradient of the height — checked against a
    /// central difference, because a swell whose normal does not match its
    /// surface is a sea with the light coming from the wrong place.
    #[test]
    fn the_normal_is_the_gradient_of_the_height() {
        let sea = Sea::cove(0.0, 0.06, -20.0, 40.0);
        let h = 1e-5;
        for (x, y, t) in [(0.0, -5.0, 0.0), (3.7, -12.3, 1.4), (-8.1, -21.0, 3.0)] {
            let dx = (sea.height(x + h, y, t) - sea.height(x - h, y, t)) / (2.0 * h);
            let dy = (sea.height(x, y + h, t) - sea.height(x, y - h, t)) / (2.0 * h);
            let want = {
                let n = [-dx, -dy, 1.0];
                let l = (n[0] * n[0] + n[1] * n[1] + 1.0).sqrt();
                [n[0] / l, n[1] / l, n[2] / l]
            };
            let got = sea.normal(x, y, t);
            for a in 0..3 {
                assert!((got[a] - want[a]).abs() < 1e-5, "at ({x}, {y}): {got:?} vs {want:?}");
            }
        }
    }

    /// The seabed is the beach's own plane: at the waterline they are the
    /// same height, and it drops going out to sea.
    #[test]
    fn the_seabed_is_the_beach_carried_on() {
        let sea = Sea::cove(1.5, 0.06, -20.0, 40.0);
        assert!((sea.seabed_z(-20.0) - 1.5).abs() < 1e-12);
        assert!(sea.seabed_z(-30.0) < sea.seabed_z(-20.0));
    }

    /// The lattice is closed and covers the footprint exactly once.
    #[test]
    fn the_lattice_closes_over_its_footprint() {
        let sea = Sea::cove(0.0, 0.06, -20.0, 40.0);
        let (p, idx) = sea.lattice(20.0, 4);
        assert_eq!(p.len(), 25);
        assert_eq!(idx.len(), 4 * 4 * 6);
        assert!(idx.iter().all(|&i| (i as usize) < p.len()));
        assert_eq!(p[0], [-20.0, -60.0, 0.0]);
        assert_eq!(p[24], [20.0, -20.0, 0.0]);
    }
}
