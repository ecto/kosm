//! Immutable observations of a pool simulation.
//!
//! A snapshot deliberately contains no solver, GPU resources, or phyz state.
//! Renderers and viewers can retain it without retaining the mutable world.

use super::{Foam, Melon, PoolSimulation, Surface};
use crate::splash::Droplet;
use tang::Vec3 as V;

#[derive(Clone)]
pub struct PoolSnapshot {
    pub time: f64,
    pub melon: Melon,
    pub melon_velocity: V<f64>,
    pub surface: Surface,
    pub droplets: Vec<Droplet>,
    pub foam: Vec<Foam>,
    pub fluid_force: V<f64>,
    pub water_particles: usize,
}

impl PoolSnapshot {
    pub fn capture(simulation: &PoolSimulation) -> Self {
        Self {
            time: simulation.state.time,
            melon: simulation.melon(),
            melon_velocity: V::new(
                simulation.state.v[3],
                simulation.state.v[4],
                simulation.state.v[5],
            ),
            surface: simulation.surface.clone(),
            droplets: simulation.droplets.clone(),
            foam: simulation.foam.clone(),
            fluid_force: simulation.fluid_force,
            water_particles: simulation.water_particle_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_is_independent_of_the_mutable_simulation() {
        let mut simulation = PoolSimulation::new(1.3);
        let snapshot = PoolSnapshot::capture(&simulation);

        simulation.state.q[5] = -0.4;
        simulation.state.v[5] = 2.0;
        simulation.surface.t = 7.0;

        assert_eq!(snapshot.melon.centre.z, 1.3);
        assert_eq!(snapshot.melon_velocity.z, 0.0);
        assert_eq!(snapshot.surface.t, 0.0);
    }

    #[test]
    fn taking_a_snapshot_resets_only_frame_local_diagnostics() {
        let mut simulation = PoolSimulation::new(1.3);
        simulation.fluid_force = V::new(1.0, 2.0, 3.0);

        let snapshot = simulation.take_snapshot();

        assert_eq!(snapshot.fluid_force, V::new(1.0, 2.0, 3.0));
        assert_eq!(simulation.fluid_force, V::zero());
        assert_eq!(simulation.centre(), snapshot.melon.centre);
    }
}
