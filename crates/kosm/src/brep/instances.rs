//! A root as placed primitives, not as a boolean.
//!
//! The court's markings are sixty bars, the rim twenty-four, the ball's seams
//! four tori. Every one of those roots is authored as a union of transformed
//! primitives, and a union of solids that do not overlap needs no boolean at
//! all: it is a list of placements, which is exactly what the tracer's
//! per-object transform — and a GPU TLAS — already wants. Evaluating it
//! instead pays a BRep union per bar (3.7 s for the markings) and hands back a
//! mesh with no BRep, which the GPU tier then has to skip.
//!
//! So the same walk the colliders take over the IR (see [`crate::colliders`])
//! is taken here: `Union` recurses, `Translate` and `Rotate` compose a frame,
//! the patterns are unions of transformed copies, and a primitive is a leaf —
//! one `Solid` per distinct (kind, dimensions), shared by every instance of it
//! and placed by the frame it was found in. Only a subtree that is genuinely
//! boolean — `Difference`, `Intersection`, an op with no instancing reading —
//! goes to `vcad_eval`, and only that subtree does.
//!
//! Millimetres throughout, like every vcad solid; the placement is object →
//! world.

use std::collections::HashMap;
use std::sync::Arc;

use vcad_ir::{CsgOp, Document, NodeId};
use vcad_kernel::Solid;
use vcad_kernel_math::{Dir3, Transform, Vec3};

/// One primitive of a root, and where it sits.
pub struct Instance {
    pub solid: Arc<Solid>,
    /// Object → world, in millimetres.
    pub to_world: Transform,
}

/// The primitives built so far, so a pattern of sixty bars holds one cube.
/// Keyed by kind and dimensions bit-for-bit — two bars that agree to the last
/// ulp are the same solid, and anything else is a different one.
#[derive(Default)]
pub struct Prims {
    prims: HashMap<(u8, [u64; 4]), Arc<Solid>>,
    /// Solids from evaluated (boolean) subtrees, by node.
    evaluated: HashMap<NodeId, Option<Arc<Solid>>>,
    /// vcad-eval's own cache, shared across every subtree that needs it.
    cache: HashMap<NodeId, Option<Solid>>,
}

impl Prims {
    fn primitive(&mut self, kind: u8, dims: [f64; 4], build: impl FnOnce() -> Solid) -> Arc<Solid> {
        let key = (kind, [dims[0].to_bits(), dims[1].to_bits(), dims[2].to_bits(), dims[3].to_bits()]);
        self.prims.entry(key).or_insert_with(|| Arc::new(build())).clone()
    }

    /// How many distinct primitive solids have been built.
    pub fn distinct(&self) -> usize {
        self.prims.len()
    }
}

/// The instances one root is made of.
pub fn instances(doc: &Document, root: NodeId, prims: &mut Prims) -> anyhow::Result<Vec<Instance>> {
    let mut out = Vec::new();
    walk(doc, root, &Transform::identity(), prims, &mut out)?;
    Ok(out)
}

/// A child's placement carried by the frame it was found in. `Transform`'s
/// matrix acts on column vectors, so the outer frame is on the left.
fn compose(outer: &Transform, inner: &Transform) -> Transform {
    Transform { matrix: outer.matrix * inner.matrix }
}

/// vcad `Rotate` is Euler XYZ in degrees: X first, then Y, then Z.
fn euler_xyz_deg(a: &vcad_ir::Vec3) -> Transform {
    let rz = Transform::rotation_z(a.z.to_radians());
    let ry = Transform::rotation_y(a.y.to_radians());
    let rx = Transform::rotation_x(a.x.to_radians());
    compose(&compose(&rz, &ry), &rx)
}

fn walk(
    doc: &Document,
    id: NodeId,
    frame: &Transform,
    prims: &mut Prims,
    out: &mut Vec<Instance>,
) -> anyhow::Result<()> {
    let node = doc.nodes.get(&id).ok_or_else(|| anyhow::anyhow!("node {id} missing"))?;
    match &node.op {
        CsgOp::Empty => Ok(()),
        CsgOp::Union { left, right } => {
            walk(doc, *left, frame, prims, out)?;
            walk(doc, *right, frame, prims, out)
        }
        CsgOp::Translate { child, offset } => {
            let f = compose(frame, &Transform::translation(offset.x, offset.y, offset.z));
            walk(doc, *child, &f, prims, out)
        }
        CsgOp::Rotate { child, angles } => walk(doc, *child, &compose(frame, &euler_xyz_deg(angles)), prims, out),
        // Patterns are unions of transformed copies, laid out exactly as
        // `vcad_kernel::Solid::{linear,circular}_pattern` lays them out: copy 0
        // is the child itself; copy i is offset by i·spacing, or turned by
        // i·(angle/count) about the axis through `axis_origin`.
        CsgOp::LinearPattern { child, direction, count, spacing } => {
            let d = Vec3::new(direction.x, direction.y, direction.z);
            let n = d.norm();
            let copies = if *count < 2 || n < 1e-12 { 1 } else { *count };
            for i in 0..copies {
                let s = if n < 1e-12 { Vec3::zeros() } else { d * (spacing * i as f64 / n) };
                let f = compose(frame, &Transform::translation(s.x, s.y, s.z));
                walk(doc, *child, &f, prims, out)?;
            }
            Ok(())
        }
        CsgOp::CircularPattern { child, axis_origin, axis_dir, count, angle_deg } => {
            let axis = Vec3::new(axis_dir.x, axis_dir.y, axis_dir.z);
            let n = axis.norm();
            let copies = if *count < 2 || n < 1e-12 { 1 } else { *count };
            let step = angle_deg.to_radians() / *count as f64;
            let (ox, oy, oz) = (axis_origin.x, axis_origin.y, axis_origin.z);
            let dir = Dir3::new_normalize(axis);
            for i in 0..copies {
                // p ↦ o + R (p − o): a rotation about the off-origin axis.
                let r = Transform::rotation_about_axis(&dir, step * i as f64);
                let to = Transform::translation(ox, oy, oz);
                let back = Transform::translation(-ox, -oy, -oz);
                let about = compose(&compose(&to, &r), &back);
                walk(doc, *child, &compose(frame, &about), prims, out)?;
            }
            Ok(())
        }
        // Primitives, at the kernel's own origins: a cube has a corner at the
        // origin, a cylinder's base is at z = 0, a sphere and a torus are centred.
        CsgOp::Cube { size } => {
            let (x, y, z) = (size.x, size.y, size.z);
            let solid = prims.primitive(0, [x, y, z, 0.0], || Solid::cube(x, y, z));
            out.push(Instance { solid, to_world: frame.clone() });
            Ok(())
        }
        CsgOp::Cylinder { radius, height, segments } => {
            let (r, h, s) = (*radius, *height, *segments);
            let solid = prims.primitive(1, [r, h, s as f64, 0.0], || Solid::cylinder(r, h, s));
            out.push(Instance { solid, to_world: frame.clone() });
            Ok(())
        }
        CsgOp::Sphere { radius, segments } => {
            let (r, s) = (*radius, *segments);
            let solid = prims.primitive(2, [r, s as f64, 0.0, 0.0], || Solid::sphere(r, s));
            out.push(Instance { solid, to_world: frame.clone() });
            Ok(())
        }
        CsgOp::Torus { major_radius, minor_radius, segments } => {
            let (a, b, s) = (*major_radius, *minor_radius, *segments);
            let solid = prims.primitive(3, [a, b, s as f64, 0.0], || Solid::torus(a, b, s));
            out.push(Instance { solid, to_world: frame.clone() });
            Ok(())
        }
        CsgOp::Cone { radius_bottom, radius_top, height, segments } => {
            let (a, b, h, s) = (*radius_bottom, *radius_top, *height, *segments);
            let solid = prims.primitive(4, [a, b, h, s as f64], || Solid::cone(a, b, h, s));
            out.push(Instance { solid, to_world: frame.clone() });
            Ok(())
        }
        CsgOp::Wedge { size } => {
            let (x, y, z) = (size.x, size.y, size.z);
            let solid = prims.primitive(5, [x, y, z, 0.0], || Solid::wedge(x, y, z));
            out.push(Instance { solid, to_world: frame.clone() });
            Ok(())
        }
        CsgOp::Prism { sides, radius, height } => {
            let (n, r, h) = (*sides, *radius, *height);
            let solid = prims.primitive(6, [n as f64, r, h, 0.0], || Solid::prism(n, r, h));
            out.push(Instance { solid, to_world: frame.clone() });
            Ok(())
        }
        // A genuine boolean, or an op with no instancing reading: evaluate that
        // subtree and place the result. This is what a whole root used to be.
        _ => {
            let solid = match prims.evaluated.get(&id) {
                Some(s) => s.clone(),
                None => {
                    let s = vcad_eval::evaluate_node(id, &doc.nodes, &mut prims.cache)
                        .map_err(|e| anyhow::anyhow!("{e:?}"))?
                        .map(Arc::new);
                    prims.evaluated.insert(id, s.clone());
                    s
                }
            };
            if let Some(solid) = solid {
                out.push(Instance { solid, to_world: frame.clone() });
            }
            Ok(())
        }
    }
}
