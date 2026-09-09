//! The pool's level, in Rust.
//!
//! This was `pool.loon`. Authorship lives here; `mod.rs` supplies this
//! scene's fluid, rigid-body and light computations. vcad lengths are
//! millimetres and simulation lengths cross to metres exactly once, in
//! `Built::millimetres`.

use kosm::build::{Built, Params, build};

use super::WaterConfig;

pub use kosm::fluid::PoolGeometry;

/// The authored pool: a basin and the watermelon's bounding volume. vcad's
/// cube has a corner at the origin, so both are placed by their corner. These
/// roots make the authored scene inspectable; the coarse simulation and the
/// renderers consume the authored dimensions, and the fine-water solver still
/// requires the reference basin.
pub fn scene(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        // environment
        let length = b.param("pool_length_mm", 50000.0);
        let width = b.param("pool_width_mm", 25000.0);
        let depth = b.param("pool_depth_mm", 2000.0);

        // watermelon
        let drop = b.param("drop_height_mm", 1300.0);
        let a = b.param("melon_a_mm", 150.0);
        let melon_b = b.param("melon_b_mm", 105.0);
        let c = b.param("melon_c_mm", 105.0);
        b.param("melon_density", 950.0);

        // water computation
        b.param("water_cell_mm", 25.0);
        b.param("water_bulk_modulus", 2.0e6);
        b.param("water_air_above_mm", 1000.0);
        b.param("water_settle_seconds", 2.0);
        b.param("water_use_gpu", 1.0);
        b.param("fps", 60.0);

        b.body("pool-volume")
            .material("water")
            .add(b.cube(length, width, depth).at(-0.5 * length, -0.5 * width, -depth));
        b.body("melon-bounds")
            .material("watermelon")
            .add(b.cube(2.0 * a, 2.0 * melon_b, 2.0 * c).at(-a, -melon_b, drop - c));
    })
}

pub struct PoolScene {
    pub authored: Built,
    pub geometry: PoolGeometry,
    pub drop_height: f64,
    pub melon_axes: [f64; 3],
    pub melon_density: f64,
    pub water: WaterConfig,
    pub fps: f64,
}

impl PoolScene {
    /// The level with these knobs turned.
    pub fn build(params: &Params) -> anyhow::Result<Self> {
        Self::interpret(scene(params)?)
    }

    fn interpret(authored: Built) -> anyhow::Result<Self> {
        let half_x = authored.millimetres("pool_length_mm")? * 0.5;
        let half_y = authored.millimetres("pool_width_mm")? * 0.5;
        let depth = authored.millimetres("pool_depth_mm")?;

        let scene = Self {
            geometry: PoolGeometry {
                half_extents: [half_x, half_y],
                depth,
            },
            drop_height: authored.millimetres("drop_height_mm")?,
            melon_axes: [
                authored.millimetres("melon_a_mm")?,
                authored.millimetres("melon_b_mm")?,
                authored.millimetres("melon_c_mm")?,
            ],
            melon_density: authored.parameter("melon_density")?,
            water: WaterConfig {
                cell_size: authored.millimetres("water_cell_mm")?,
                bulk_modulus: authored.parameter("water_bulk_modulus")?,
                air_above: authored.millimetres("water_air_above_mm")?,
                settle_seconds: authored.parameter("water_settle_seconds")?,
                use_gpu: authored.parameter_or("water_use_gpu", 1.0) != 0.0,
            },
            fps: authored.parameter("fps")?,
            authored,
        };
        anyhow::ensure!(
            scene.melon_axes.iter().all(|axis| *axis > 0.0),
            "melon axes must be positive"
        );
        anyhow::ensure!(
            scene.geometry.half_extents.iter().all(|extent| *extent > 0.0)
                && scene.geometry.depth > 0.0,
            "pool dimensions must be positive"
        );
        anyhow::ensure!(scene.melon_density > 0.0, "melon density must be positive");
        anyhow::ensure!(scene.fps > 0.0, "fps must be positive");
        Ok(scene)
    }

    /// Preserve the spike's environment controls at the executable boundary;
    /// the authored scene itself remains deterministic and testable.
    pub fn with_env_overrides(mut self) -> Self {
        self.water = self.water.with_env_overrides();
        self.fps = std::env::var("KOSM_FPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.fps);
        self
    }

    /// The level at its authored defaults.
    pub fn reference() -> anyhow::Result<Self> {
        Self::build(&Params::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::{MELON_AXES, PoolSimulation};

    #[test]
    fn reference_pool_is_an_authored_scene() {
        let scene = PoolScene::reference().unwrap();
        assert_eq!(scene.geometry, PoolGeometry::reference());
        assert_eq!(scene.melon_axes, MELON_AXES);
        assert_eq!(scene.drop_height, 1.3);
        assert_eq!(scene.water, WaterConfig::default());
        assert!(!scene.authored.document.roots.is_empty());
    }

    #[test]
    fn authored_actor_values_reach_the_scene_computation() {
        let mut scene = PoolScene::reference().unwrap();
        scene.drop_height = 0.9;
        scene.melon_axes = [0.2, 0.1, 0.08];
        scene.geometry = PoolGeometry {
            half_extents: [4.0, 2.0],
            depth: 1.25,
        };

        let simulation = PoolSimulation::from_scene(&scene);
        let melon = simulation.melon();
        let fluid_body = simulation.fluid_body(tang::Vec3::zero());

        assert_eq!(melon.centre.z, 0.9);
        assert_eq!(melon.axes, [0.2, 0.1, 0.08]);
        assert_eq!(fluid_body.semi, melon.axes);
        let long_axis_surface = fluid_body.centre + fluid_body.axis * fluid_body.semi[0];
        assert!(fluid_body.sdf(long_axis_surface).0.abs() < 1e-12);
        assert_eq!(simulation.recording_fps(), scene.fps);
        assert_eq!(simulation.geometry(), scene.geometry);
        assert_eq!(simulation.snapshot().geometry, scene.geometry);
    }

    #[test]
    fn fine_water_rejects_geometry_its_grids_do_not_yet_support() {
        let mut scene = PoolScene::reference().unwrap();
        scene.geometry.half_extents = [4.0, 2.0];

        let error = PoolSimulation::from_scene(&scene)
            .with_water_config(scene.water)
            .err()
            .expect("unsupported geometry");

        assert!(error.to_string().contains("reference pool geometry"));
    }

    #[test]
    fn authored_dimensions_are_not_reference_only() {
        let mut params = Params::new();
        params.set("pool_length_mm", 8000.0);
        params.set("pool_width_mm", 4000.0);
        params.set("pool_depth_mm", 1250.0);
        let loaded = PoolScene::build(&params);

        assert_eq!(
            loaded.unwrap().geometry,
            PoolGeometry {
                half_extents: [4.0, 2.0],
                depth: 1.25,
            }
        );
    }

    #[test]
    fn reference_renderer_reads_snapshot_geometry() {
        let simulation = PoolSimulation::new(10.0);
        let mut snapshot = simulation.snapshot();
        snapshot.geometry = PoolGeometry {
            half_extents: [0.5, 0.5],
            depth: 1.0,
        };
        let view = crate::pool::View {
            eye: tang::Vec3::new(1.0, -1.0, 1.0),
            target: tang::Vec3::new(1.0, 0.0, 0.0),
            width: 1,
            height: 1,
            vfov: 0.1,
        };
        let caustic = crate::pool::Caustic {
            origin: [-10.0, -10.0],
            cell: 20.0,
            nx: 2,
            ny: 2,
            e: vec![1.0; 4],
        };

        let deck = crate::pool::render_snapshot(&view, &snapshot, &caustic);
        snapshot.geometry = PoolGeometry::reference();
        let water = crate::pool::render_snapshot(&view, &snapshot, &caustic);

        assert_ne!(deck.as_raw(), water.as_raw());
    }
}
