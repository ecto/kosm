//! Colliders from a vcad document.
//!
//! The document is the level. Physics does not get its own copy of the
//! geometry; it gets a derivation of the same tree the printer gets. The
//! convex-primitive algebra (`Union` of `Translate`d / `Rotate`d `Cube`,
//! `Cylinder`, `Sphere`) maps one-to-one onto phyz colliders. Anything else
//! (`Difference`, `Intersection`, `Scale`, sketches...) falls back to the
//! convex hull of the subtree's tessellation, which phyz's `Geometry::Mesh`
//! is, and says so in a warning: a hull is honest for a bracket and wrong for
//! a cup.
//!
//! Units: vcad millimetres in, phyz metres out.

use std::collections::HashMap;

use phyz_math::{Mat3, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry};
use vcad_ir::{CsgOp, Document, NodeId};

const MM: f64 = 1e-3;

pub struct Derived {
    pub colliders: Vec<GeomInstance>,
    pub warnings: Vec<String>,
}

/// A rigid placement in document coordinates (mm): `p_doc = rot * p_local + pos`.
#[derive(Clone, Copy)]
struct Frame {
    rot: Mat3,
    pos: Vec3,
}

impl Frame {
    fn identity() -> Self {
        Self { rot: Mat3::identity(), pos: Vec3::zeros() }
    }
    fn then_translate(self, o: Vec3) -> Self {
        Self { rot: self.rot, pos: self.pos + self.rot * o }
    }
    fn then_rotate(self, r: Mat3) -> Self {
        Self { rot: self.rot * r, pos: self.pos }
    }
    fn apply(&self, p: Vec3) -> Vec3 {
        self.rot * p + self.pos
    }
    /// A shape centred at `local_center` (mm) in this frame, as a phyz placement (m).
    fn instance(&self, geometry: Geometry, local_center: Vec3) -> GeomInstance {
        let c = self.apply(local_center) * MM;
        // `GeomInstance::origin` is a Plücker transform: `rot` maps body → shape,
        // the transpose of the shape's orientation in the body.
        GeomInstance::new(geometry, SpatialTransform::new(self.rot.transpose(), c))
    }
}

/// vcad `Rotate` is Euler XYZ in degrees: X first, then Y, then Z.
fn euler_xyz_deg(a: &vcad_ir::Vec3) -> Mat3 {
    Mat3::rotation_z(a.z.to_radians()) * Mat3::rotation_y(a.y.to_radians()) * Mat3::rotation_x(a.x.to_radians())
}

pub fn colliders_from_document(doc: &Document) -> anyhow::Result<Derived> {
    let mut out = Derived { colliders: Vec::new(), warnings: Vec::new() };
    let mut cache = HashMap::new();
    for root in &doc.roots {
        walk(doc, root.root, Frame::identity(), &mut out, &mut cache)?;
    }
    Ok(out)
}

fn walk(
    doc: &Document,
    id: NodeId,
    frame: Frame,
    out: &mut Derived,
    cache: &mut HashMap<NodeId, Option<vcad_kernel::Solid>>,
) -> anyhow::Result<()> {
    let node = doc.nodes.get(&id).ok_or_else(|| anyhow::anyhow!("node {id} missing"))?;
    match &node.op {
        CsgOp::Union { left, right } => {
            walk(doc, *left, frame, out, cache)?;
            walk(doc, *right, frame, out, cache)
        }
        CsgOp::Translate { child, offset } => {
            walk(doc, *child, frame.then_translate(Vec3::new(offset.x, offset.y, offset.z)), out, cache)
        }
        CsgOp::Rotate { child, angles } => walk(doc, *child, frame.then_rotate(euler_xyz_deg(angles)), out, cache),
        // Patterns are unions of transformed copies, so they are exact here.
        // Both mirror `vcad_kernel::Solid::{linear,circular}_pattern`: copy 0 is
        // the child itself; copy i is offset by i·spacing, or turned by
        // i·(angle/count) about the axis through `axis_origin`.
        CsgOp::LinearPattern { child, direction, count, spacing } => {
            let d = Vec3::new(direction.x, direction.y, direction.z);
            let n = d.norm();
            let copies = if *count < 2 || n < 1e-12 { 1 } else { *count };
            for i in 0..copies {
                let step = if n < 1e-12 { Vec3::zeros() } else { d * (spacing * i as f64 / n) };
                walk(doc, *child, frame.then_translate(step), out, cache)?;
            }
            Ok(())
        }
        CsgOp::CircularPattern { child, axis_origin, axis_dir, count, angle_deg } => {
            let axis = Vec3::new(axis_dir.x, axis_dir.y, axis_dir.z);
            let o = Vec3::new(axis_origin.x, axis_origin.y, axis_origin.z);
            let n = axis.norm();
            let copies = if *count < 2 || n < 1e-12 { 1 } else { *count };
            let step = angle_deg.to_radians() / *count as f64;
            for i in 0..copies {
                let r = Mat3::rotation_axis(axis / n, step * i as f64);
                // p ↦ o + R (p − o): a rotation about the off-origin axis.
                let f = Frame { rot: frame.rot * r, pos: frame.apply(o - r * o) };
                walk(doc, *child, f, out, cache)?;
            }
            Ok(())
        }
        // Primitives. vcad's origins: cube has a corner at the origin, the
        // cylinder's base is at z = 0, the sphere is centred.
        CsgOp::Cube { size } => {
            let half = Vec3::new(size.x, size.y, size.z) * (0.5 * MM);
            out.colliders.push(frame.instance(Geometry::Box { half_extents: half }, Vec3::new(size.x, size.y, size.z) * 0.5));
            Ok(())
        }
        CsgOp::Cylinder { radius, height, .. } => {
            out.colliders.push(frame.instance(
                Geometry::Cylinder { radius: radius * MM, height: height * MM },
                Vec3::new(0.0, 0.0, height * 0.5),
            ));
            Ok(())
        }
        CsgOp::Sphere { radius, .. } => {
            out.colliders.push(frame.instance(Geometry::Sphere { radius: radius * MM }, Vec3::zeros()));
            Ok(())
        }
        CsgOp::Empty => Ok(()),
        other => {
            // Convex hull of the tessellated subtree, in the subtree's own frame.
            let name = node.name.clone().unwrap_or_else(|| format!("node {id}"));
            let kind = format!("{other:?}");
            let kind = kind.split(' ').next().unwrap_or("?").trim_end_matches('{').to_string();
            let Some(solid) = vcad_eval::evaluate_node(id, &doc.nodes, cache).map_err(|e| anyhow::anyhow!("{e:?}"))? else {
                return Ok(());
            };
            let mesh = solid.to_mesh(24);
            let vertices: Vec<Vec3> = mesh
                .vertices
                .chunks(3)
                .map(|v| frame.apply(Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)) * MM)
                .collect();
            let faces: Vec<[usize; 3]> =
                mesh.indices.chunks(3).map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]).collect();
            out.warnings.push(format!(
                "'{name}' is a {kind}: collider is the convex hull of its {} tessellated vertices",
                vertices.len()
            ));
            out.colliders.push(GeomInstance::new(Geometry::Mesh { vertices, faces }, SpatialTransform::identity()));
            Ok(())
        }
    }
}

/// Check every primitive collider against the tessellation of the whole
/// document: along a set of directions, the collider's support point must not
/// exceed the mesh's, and the union of colliders must reach the mesh's extent.
/// Returns the worst discrepancy in metres.
pub fn verify_against_mesh(doc: &Document, derived: &Derived) -> anyhow::Result<f64> {
    let scene = vcad_eval::evaluate_document(doc, &vcad_eval::EvalOptions::default())
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let verts: Vec<Vec3> = scene
        .parts
        .iter()
        .flat_map(|p| p.mesh.positions.chunks(3).map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64) * MM).collect::<Vec<_>>())
        .collect();
    anyhow::ensure!(!verts.is_empty(), "document tessellated to nothing");

    let mut dirs = vec![Vec3::x(), Vec3::y(), Vec3::z(), -Vec3::x(), -Vec3::y(), -Vec3::z()];
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                dirs.push(Vec3::new(sx, sy, sz).normalize());
            }
        }
    }
    let mut worst = 0.0f64;
    for d in &dirs {
        let mesh_max = verts.iter().map(|v| v.dot(d)).fold(f64::MIN, f64::max);
        let mut coll_max = f64::MIN;
        for inst in &derived.colliders {
            let geom = to_collision_geometry(&inst.geometry);
            // origin.rot is body→shape; the collision crate wants shape→body.
            let s = geom.support(d, &inst.origin.pos, &inst.origin.rot.transpose());
            coll_max = coll_max.max(s.dot(d));
        }
        // Colliders may not stick out past the mesh (a tessellated cylinder is
        // inscribed, so allow the chord sagitta), and the union must reach it.
        worst = worst.max((coll_max - mesh_max).abs());
    }
    Ok(worst)
}

fn to_collision_geometry(g: &Geometry) -> phyz_collision::Geometry {
    use phyz_collision::Geometry as C;
    match g {
        Geometry::Sphere { radius } => C::Sphere { radius: *radius },
        Geometry::Capsule { radius, length } => C::Capsule { radius: *radius, length: *length },
        Geometry::Box { half_extents } => C::Box { half_extents: *half_extents },
        Geometry::Cylinder { radius, height } => C::Cylinder { radius: *radius, height: *height },
        Geometry::Mesh { vertices, faces } => C::Mesh { vertices: vertices.clone(), faces: faces.clone() },
        Geometry::Plane { normal } => C::Plane { normal: *normal },
    }
}
