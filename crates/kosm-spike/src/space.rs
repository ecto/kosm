//! Earth-centred inertial SI propagation, independent of pool computations.
//! RK4 translation with central gravity + J2; torque-free rigid attitude via
//! Euler's equations and a unit quaternion. Earth rotation is render-only.
use crate::scene::AuthoredScene;
use std::f64::consts::TAU;
use tang::Vec3 as V;
#[derive(Clone)]
pub struct SpaceScene {
    pub radius: f64,
    pub mu: f64,
    pub j2: f64,
    pub altitude: f64,
    pub inclination: f64,
    pub phase: f64,
    pub mass: f64,
    pub bus: [f64; 3],
    pub panel_span: f64,
    pub panel_chord: f64,
    pub panel_thickness: f64,
    pub spin: V<f64>,
}
impl SpaceScene {
    pub fn load() -> anyhow::Result<Self> {
        let a = AuthoredScene::load_bundled("space.loon")?;
        let s = Self {
            radius: a.parameter("earth_radius_m")?,
            mu: a.parameter("earth_mu")?,
            j2: a.parameter("earth_j2")?,
            altitude: a.parameter("altitude_m")?,
            inclination: a.parameter("inclination_deg")?.to_radians(),
            phase: a.parameter("phase_deg")?.to_radians(),
            mass: a.parameter("mass_kg")?,
            bus: [
                a.millimetres("bus_x_mm")?,
                a.millimetres("bus_y_mm")?,
                a.millimetres("bus_z_mm")?,
            ],
            panel_span: a.millimetres("panel_span_mm")?,
            panel_chord: a.millimetres("panel_chord_mm")?,
            panel_thickness: a.millimetres("panel_thickness_mm")?,
            spin: V::new(
                a.parameter("spin_x_deg_s")?.to_radians(),
                a.parameter("spin_y_deg_s")?.to_radians(),
                a.parameter("spin_z_deg_s")?.to_radians(),
            ),
        };
        anyhow::ensure!(
            [
                s.radius,
                s.mu,
                s.altitude,
                s.mass,
                s.panel_span,
                s.panel_chord,
                s.panel_thickness
            ]
            .iter()
            .chain(s.bus.iter())
            .all(|x| x.is_finite() && *x > 0.0),
            "space dimensions and physical parameters must be finite and positive"
        );
        anyhow::ensure!(
            [s.j2, s.inclination, s.phase, s.spin.x, s.spin.y, s.spin.z]
                .iter()
                .all(|x| x.is_finite()),
            "space angles and J2 must be finite"
        );
        Ok(s)
    }
    pub fn acceleration(&self, r: V<f64>) -> V<f64> {
        let r2 = r.dot(&r);
        let rn = r2.sqrt();
        let k = 1.5 * self.j2 * self.mu * self.radius * self.radius / (r2 * r2 * rn);
        let z = 5.0 * r.z * r.z / r2;
        r * (-self.mu / (r2 * rn)) + V::new(r.x * (z - 1.0), r.y * (z - 1.0), r.z * (z - 3.0)) * k
    }
    pub fn potential(&self, r: V<f64>) -> f64 {
        let n = r.norm();
        -self.mu / n
            * (1.0 - self.j2 * (self.radius / n).powi(2) * 0.5 * (3.0 * (r.z / n).powi(2) - 1.0))
    }
    pub fn inertia(&self) -> V<f64> {
        let [x, y, z] = self.bus;
        V::new(
            self.mass * (y * y + z * z) / 12.0,
            self.mass * (x * x + z * z) / 12.0,
            self.mass * (x * x + y * y) / 12.0,
        )
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    pub t: f64,
    pub r: V<f64>,
    pub v: V<f64>,
    pub q: [f64; 4],
    pub omega: V<f64>,
    pub delta_v: f64,
}
impl Snapshot {
    pub fn initial(s: &SpaceScene) -> Self {
        let (p, c) = s.phase.sin_cos();
        let (si, ci) = s.inclination.sin_cos();
        let r = s.radius + s.altitude;
        let speed = (s.mu / r).sqrt();
        Self {
            t: 0.0,
            r: V::new(c, p * ci, p * si) * r,
            v: V::new(-p, c * ci, c * si) * speed,
            q: [1.0, 0.0, 0.0, 0.0],
            omega: s.spin,
            delta_v: 0.0,
        }
    }
    pub fn axes(&self) -> [V<f64>; 3] {
        let [w, x, y, z] = self.q;
        [
            V::new(
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            ),
            V::new(
                2.0 * (x * y - w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + w * x),
            ),
            V::new(
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ]
    }
    pub fn energy(&self, s: &SpaceScene) -> f64 {
        0.5 * self.v.dot(&self.v) + s.potential(self.r)
    }
    pub fn elements(&self, s: &SpaceScene) -> (f64, f64, f64) {
        let h = self.r.cross(&self.v);
        let e = (self.v.cross(&h) / s.mu - self.r / self.r.norm()).norm();
        let a = 1.0 / (2.0 / self.r.norm() - self.v.dot(&self.v) / s.mu);
        (
            a * (1.0 - e) - s.radius,
            a * (1.0 + e) - s.radius,
            if a > 0.0 {
                TAU * (a.powi(3) / s.mu).sqrt()
            } else {
                f64::NAN
            },
        )
    }
    pub fn impulse(&mut self, direction: V<f64>, dv: f64) {
        self.v = self.v + direction.normalize() * dv;
        self.delta_v += dv.abs();
    }
    pub fn step(&mut self, s: &SpaceScene, dt: f64) {
        let (r, v) = (self.r, self.v);
        let a1 = s.acceleration(r);
        let v2 = v + a1 * (dt * 0.5);
        let a2 = s.acceleration(r + v * (dt * 0.5));
        let v3 = v + a2 * (dt * 0.5);
        let a3 = s.acceleration(r + v2 * (dt * 0.5));
        let v4 = v + a3 * dt;
        let a4 = s.acceleration(r + v3 * dt);
        self.r = r + (v + v2 * 2.0 + v3 * 2.0 + v4) * (dt / 6.0);
        self.v = v + (a1 + a2 * 2.0 + a3 * 2.0 + a4) * (dt / 6.0);
        // RK4 of coupled Euler/quaternion dynamics, not a cosmetic spin animation.
        let deriv = |q: [f64; 4], w: V<f64>| {
            let i = s.inertia();
            let dw = V::new(
                (i.y - i.z) * w.y * w.z / i.x,
                (i.z - i.x) * w.z * w.x / i.y,
                (i.x - i.y) * w.x * w.y / i.z,
            );
            let [a, b, c, d] = q;
            (
                [
                    -0.5 * (b * w.x + c * w.y + d * w.z),
                    0.5 * (a * w.x + c * w.z - d * w.y),
                    0.5 * (a * w.y + d * w.x - b * w.z),
                    0.5 * (a * w.z + b * w.y - c * w.x),
                ],
                dw,
            )
        };
        let add = |q: [f64; 4], d: [f64; 4], k: f64| std::array::from_fn(|i| q[i] + d[i] * k);
        let (q, w) = (self.q, self.omega);
        let (k1, d1) = deriv(q, w);
        let (k2, d2) = deriv(add(q, k1, dt * 0.5), w + d1 * (dt * 0.5));
        let (k3, d3) = deriv(add(q, k2, dt * 0.5), w + d2 * (dt * 0.5));
        let (k4, d4) = deriv(add(q, k3, dt), w + d3 * dt);
        self.q =
            std::array::from_fn(|i| q[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
        let n = self.q.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.q.iter_mut().for_each(|x| *x /= n);
        self.omega = w + (d1 + d2 * 2.0 + d3 * 2.0 + d4) * (dt / 6.0);
        self.t += dt;
    }
    pub fn advance(&mut self, s: &SpaceScene, seconds: f64) {
        let mut left = seconds.max(0.0);
        while left > 1e-10 {
            let h = left.min(1.0);
            self.step(s, h);
            left -= h;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn circular_orbit_closes() {
        let mut s = SpaceScene::load().unwrap();
        s.j2 = 0.0;
        let mut p = Snapshot::initial(&s);
        let r = p.r;
        let e = p.energy(&s);
        let period = p.elements(&s).2;
        p.advance(&s, period);
        assert!((p.r - r).norm() < 0.01, "closure {} m", (p.r - r).norm());
        assert!(((p.energy(&s) - e) / e).abs() < 1e-11);
    }
    #[test]
    fn j2_force_matches_potential_gradient() {
        let s = SpaceScene::load().unwrap();
        let p = Snapshot::initial(&s);
        let a = s.acceleration(p.r);
        for axis in [
            V::new(1.0, 0.0, 0.0),
            V::new(0.0, 1.0, 0.0),
            V::new(0.0, 0.0, 1.0),
        ] {
            let fd = -(s.potential(p.r + axis) - s.potential(p.r - axis)) / 2.0;
            assert!((fd - a.dot(&axis)).abs() < 1e-7);
        }
    }
    #[test]
    fn j2_conserves_energy_and_axial_momentum() {
        let s = SpaceScene::load().unwrap();
        let mut p = Snapshot::initial(&s);
        let (e, h) = (p.energy(&s), p.r.cross(&p.v).z);
        p.advance(&s, 18000.0);
        assert!(((p.energy(&s) - e) / e).abs() < 1e-10);
        assert!(((p.r.cross(&p.v).z - h) / h).abs() < 1e-10);
    }
    #[test]
    fn torque_free_attitude_conserves_invariants() {
        let s = SpaceScene::load().unwrap();
        let mut p = Snapshot::initial(&s);
        let i = s.inertia();
        let energy = |w: V<f64>| 0.5 * (i.x * w.x * w.x + i.y * w.y * w.y + i.z * w.z * w.z);
        let momentum = |p: Snapshot| {
            let a = p.axes();
            a[0] * (i.x * p.omega.x) + a[1] * (i.y * p.omega.y) + a[2] * (i.z * p.omega.z)
        };
        let (e, h) = (energy(p.omega), momentum(p));
        p.advance(&s, 3600.0);
        assert!((p.q.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(((energy(p.omega) - e) / e).abs() < 1e-9);
        assert!((momentum(p) - h).norm() / h.norm() < 1e-8);
    }
    #[test]
    fn prograde_burn_raises_apogee() {
        let s = SpaceScene::load().unwrap();
        let mut p = Snapshot::initial(&s);
        let old = p.elements(&s);
        p.impulse(p.v, 10.0);
        assert!(p.elements(&s).1 > old.1 + 30000.0);
        assert_eq!(p.delta_v, 10.0);
    }
}
