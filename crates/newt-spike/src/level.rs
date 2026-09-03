//! Marble-specific interpretation of an authored scene.

use newt_spike::scene::AuthoredScene;
use phyz_math::{Mat3, SpatialTransform, Vec3};
use vcad_ir::{CsgOp, Document, Node};

pub(crate) const DT: f64 = 1e-3;
pub(crate) use newt_spike::scene::MM;
pub(crate) type Level = AuthoredScene;

pub(crate) trait MarbleLevel {
    fn p(&self, name: &str) -> anyhow::Result<f64>;
    fn tilt(&self) -> anyhow::Result<Tilt>;
    fn start(&self) -> anyhow::Result<[f64; 2]>;
    fn marble_r(&self) -> anyhow::Result<f64>;
    fn steps(&self) -> anyhow::Result<usize>;
    fn with_params(&self, updates: &[(&str, f64)]) -> String;
}

impl MarbleLevel for AuthoredScene {
    fn p(&self, name: &str) -> anyhow::Result<f64> {
        self.parameter(name)
    }

    fn tilt(&self) -> anyhow::Result<Tilt> {
        Ok(Tilt {
            pitch: self.parameter("pitch_deg")?.to_radians(),
            roll: self.parameter("roll_deg")?.to_radians(),
        })
    }

    fn start(&self) -> anyhow::Result<[f64; 2]> {
        Ok([self.millimetres("start_x")?, self.millimetres("start_y")?])
    }

    fn marble_r(&self) -> anyhow::Result<f64> {
        self.millimetres("marble_r")
    }

    fn steps(&self) -> anyhow::Result<usize> {
        Ok((self.parameter("t_end")? / DT).round() as usize)
    }

    fn with_params(&self, updates: &[(&str, f64)]) -> String {
        self.with_parameters(updates)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Tilt {
    /// About the plate's y axis (radians). Positive lowers the +x end.
    pub(crate) pitch: f64,
    /// About the plate's x axis (radians). Positive lowers the −y end.
    pub(crate) roll: f64,
}

impl Tilt {
    /// Plate-frame → world rotation.
    pub(crate) fn rotation(self) -> Mat3 {
        Mat3::rotation_y(self.pitch) * Mat3::rotation_x(self.roll)
    }

    /// The track body's pose. `SpatialTransform::rot` is world→body.
    pub(crate) fn pose(self) -> SpatialTransform {
        SpatialTransform::new(self.rotation().transpose(), Vec3::new(0.0, 0.0, 0.25))
    }
}

/// The document with every root wrapped in a `Rotate`, for the tilted view.
pub(crate) fn tilted(doc: &Document, tilt: Tilt) -> Document {
    let mut result = doc.clone();
    let mut next = result.nodes.keys().copied().max().unwrap_or(0) + 1;
    for root in &mut result.roots {
        result.nodes.insert(
            next,
            Node {
                id: next,
                name: Some("tilt".into()),
                op: CsgOp::Rotate {
                    child: root.root,
                    angles: vcad_ir::Vec3::new(
                        tilt.roll.to_degrees(),
                        tilt.pitch.to_degrees(),
                        0.0,
                    ),
                },
            },
        );
        root.root = next;
        next += 1;
    }
    result
}
