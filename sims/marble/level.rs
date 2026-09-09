//! Marble-specific interpretation of the built scene.

use kosm::build::Built;
use phyz_math::{Mat3, SpatialTransform, Vec3};
use vcad_ir::{CsgOp, Document, Node};

pub(crate) const DT: f64 = 1e-3;
pub(crate) use kosm::scene::MM;

/// The level is what `scene.rs` built. It carries the document the printer,
/// the colliders and the picture read, and the knobs that used to be
/// `defparam`s.
pub(crate) type Level = Built;

pub(crate) trait MarbleLevel {
    fn p(&self, name: &str) -> anyhow::Result<f64>;
    fn opt(&self, name: &str) -> Option<f64>;
    fn tilt(&self) -> anyhow::Result<Tilt>;
    fn start(&self) -> anyhow::Result<[f64; 2]>;
    fn marble_r(&self) -> anyhow::Result<f64>;
    fn steps(&self) -> anyhow::Result<usize>;
}

impl MarbleLevel for Built {
    fn p(&self, name: &str) -> anyhow::Result<f64> {
        self.param(name).ok_or_else(|| anyhow::anyhow!("the level has no `{name}`"))
    }

    fn opt(&self, name: &str) -> Option<f64> {
        self.param(name)
    }

    fn tilt(&self) -> anyhow::Result<Tilt> {
        Ok(Tilt {
            pitch: self.p("pitch_deg")?.to_radians(),
            roll: self.p("roll_deg")?.to_radians(),
        })
    }

    fn start(&self) -> anyhow::Result<[f64; 2]> {
        Ok([self.p("start_x")? * MM, self.p("start_y")? * MM])
    }

    fn marble_r(&self) -> anyhow::Result<f64> {
        Ok(self.p("marble_r")? * MM)
    }

    fn steps(&self) -> anyhow::Result<usize> {
        Ok((self.p("t_end")? / DT).round() as usize)
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
