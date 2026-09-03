//! Pool-specific interpretation of an authored scene.

use std::path::Path;

use crate::scene::AuthoredScene;

use super::{DEPTH, POOL_X, POOL_Y, WaterConfig};

pub const DEFAULT_POOL_SCENE: &str = "levels/pool.loon";
const DEFAULT_POOL_FILE: &str = "pool.loon";

/// Authored pool knobs consumed by the pool's scene-specific computations.
pub struct PoolScene {
    pub authored: AuthoredScene,
    pub drop_height: f64,
    pub melon_axes: [f64; 3],
    pub melon_density: f64,
    pub water: WaterConfig,
    pub fps: f64,
}

impl PoolScene {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let authored = AuthoredScene::load(path)?;
        Self::interpret(authored)
    }

    fn interpret(authored: AuthoredScene) -> anyhow::Result<Self> {
        // The hand-written renderer still bakes these dimensions into its ray
        // tests. Refuse divergence until it consumes evaluated geometry.
        let half_x = authored.millimetres("pool_length_mm")? * 0.5;
        let half_y = authored.millimetres("pool_width_mm")? * 0.5;
        let depth = authored.millimetres("pool_depth_mm")?;
        anyhow::ensure!(
            (half_x - POOL_X).abs() < 1e-9
                && (half_y - POOL_Y).abs() < 1e-9
                && (depth - DEPTH).abs() < 1e-9,
            "authored pool dimensions are not yet supported by the legacy renderer"
        );

        let scene = Self {
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
        anyhow::ensure!(scene.melon_density > 0.0, "melon density must be positive");
        anyhow::ensure!(scene.fps > 0.0, "fps must be positive");
        Ok(scene)
    }

    /// Preserve the spike's environment controls at the executable boundary;
    /// the authored scene itself remains deterministic and testable.
    pub fn with_env_overrides(mut self) -> Self {
        self.water = self.water.with_env_overrides();
        self.fps = std::env::var("NEWT_FPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.fps);
        self
    }

    pub fn reference() -> anyhow::Result<Self> {
        let authored = AuthoredScene::load_bundled(DEFAULT_POOL_FILE)?;
        Self::interpret(authored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::{MELON_AXES, PoolSimulation};

    #[test]
    fn reference_pool_is_an_authored_scene() {
        let scene = PoolScene::reference().unwrap();
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

        let simulation = PoolSimulation::from_scene(&scene);
        let melon = simulation.melon();

        assert_eq!(melon.centre.z, 0.9);
        assert_eq!(melon.axes, [0.2, 0.1, 0.08]);
        assert_eq!(simulation.recording_fps(), scene.fps);
    }
}
