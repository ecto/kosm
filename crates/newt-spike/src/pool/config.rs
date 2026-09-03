//! Typed configuration for constructing the fine-water simulation.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterConfig {
    pub cell_size: f64,
    pub bulk_modulus: f64,
    pub air_above: f64,
    pub settle_seconds: f64,
    pub use_gpu: bool,
}

impl WaterConfig {
    pub fn from_env(default_cell_size: f64) -> Self {
        Self {
            cell_size: default_cell_size,
            ..Self::default()
        }
        .with_env_overrides()
    }

    pub fn with_env_overrides(mut self) -> Self {
        self.cell_size = std::env::var("NEWT_H")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.cell_size);
        self.use_gpu = std::env::var("NEWT_GPU")
            .map(|value| value != "0")
            .unwrap_or(self.use_gpu);
        self
    }
}

impl Default for WaterConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.025,
            bulk_modulus: 2.0e6,
            air_above: 1.0,
            settle_seconds: 2.0,
            use_gpu: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_capture_the_reference_water_model() {
        let config = WaterConfig::default();
        assert_eq!(config.cell_size, 0.025);
        assert_eq!(config.bulk_modulus, 2.0e6);
        assert_eq!(config.air_above, 1.0);
        assert_eq!(config.settle_seconds, 2.0);
        assert!(config.use_gpu);
    }
}
