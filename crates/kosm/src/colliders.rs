//! Colliders from a vcad document.
//!
//! The document is the level. Physics does not get its own copy of the
//! geometry; it gets a derivation of the same tree the printer gets. The
//! convex-primitive algebra (`Union` of `Translate`d / `Rotate`d `Cube`,
//! `Cylinder`, `Sphere`) maps one-to-one onto phyz colliders.
//!
//! `Difference` is the interesting case. phyz colliders are convex, so a
//! difference cannot be *one* collider: the convex hull of a cup is a puck and
//! the marble bounces off the lid. What it can be is a handful of convex pieces
//! whose union is the cup and none of which reaches into the hole — an
//! approximate convex decomposition, computed geometrically from the
//! tessellated subtree and *scored* before it is accepted (see [`decompose`]).
//! Anything still undecomposable — `Scale`, sketches, a decomposition that
//! failed its own check — falls back to the convex hull of the subtree's
//! tessellation, which phyz's `Geometry::Mesh` is, and says so in a warning: a
//! hull is honest for a bracket and wrong for a cup.
//!
//! Units: vcad millimetres in, phyz metres out.

use std::collections::{HashMap, HashSet};

use phyz_math::{Mat3, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry};
use vcad_ir::{CsgOp, Document, NodeId};

const MM: f64 = 1e-3;

/// A piece may not reach this far into a removed volume (metres).
const INTRUSION_TOL: f64 = 0.5e-3;
/// Sectors are grown by this much before their contents are hulled (metres).
/// Adjacent sectors already share their boundary polygon exactly — the clip
/// puts points *on* each plane — so this only has to absorb round-off, and it
/// is kept small because every micron of it widens a wedge and so deepens the
/// chord it cuts across a bore.
const SECTOR_OVERLAP: f64 = 5.0e-5;
/// Tessellation density for a decomposition. Denser than the 24 used for the
/// hull fallback: the sector planes cut between vertices, so vertices are the
/// resolution.
const SEGMENTS: u32 = 48;
/// Sector counts tried, cheapest (fewest pieces) first.
const SECTOR_COUNTS: [usize; 7] = [6, 8, 12, 16, 24, 32, 48];
/// How many points a piece's hull is built from. The hull is brute-forced in
/// O(n⁴), and this whole derivation runs once per candidate tilt, so the
/// constant matters; a sector of one solid has nothing like this many corners.
const HULL_DIRS: usize = 26;

pub type Tri = [Vec3; 3];

pub struct Derived {
    pub colliders: Vec<GeomInstance>,
    pub warnings: Vec<String>,
    /// Informational lines about decompositions that succeeded.
    pub notes: Vec<String>,
    /// Volumes cut out of the level, with the pieces they constrain.
    pub removed: Vec<Removed>,
}

/// A removed volume and the colliders derived alongside it.
pub struct Removed {
    /// The subtrahend's triangles, in body coordinates (metres).
    pub tris: Vec<Tri>,
    /// Indices into [`Derived::colliders`] of the pieces this cut applies to.
    pub pieces: std::ops::Range<usize>,
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
    let mut out = Derived { colliders: Vec::new(), warnings: Vec::new(), notes: Vec::new(), removed: Vec::new() };
    let mut cache = HashMap::new();
    for root in &doc.roots {
        walk(doc, root.root, Frame::identity(), &mut out, &mut cache)?;
    }
    Ok(out)
}

type Cache = HashMap<NodeId, Option<vcad_kernel::Solid>>;

fn walk(doc: &Document, id: NodeId, frame: Frame, out: &mut Derived, cache: &mut Cache) -> anyhow::Result<()> {
    let node = doc.nodes.get(&id).ok_or_else(|| anyhow::anyhow!("node {id} missing"))?;
    let name = || node.name.clone().unwrap_or_else(|| format!("node {id}"));
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
        // The two booleans that are not a union of convex parts. Both get a
        // geometric decomposition; `Difference` additionally records what was
        // removed, so the pieces can be checked against the hole afterwards.
        CsgOp::Difference { .. } => carve(doc, id, cut_chain(doc, id), frame, &name(), out, cache),
        CsgOp::Intersection { .. } => carve(doc, id, Vec::new(), frame, &name(), out, cache),
        other => {
            let kind = format!("{other:?}");
            let kind = kind.split(' ').next().unwrap_or("?").trim_end_matches('{').to_string();
            hull_fallback(doc, id, frame, &name(), &kind, out, cache)
        }
    }
}

/// Every volume removed along a chain of differences. Cuts compose down the
/// `left` (subject) edge — `[difference gap [difference bore blank]]` is one
/// node whose subject is another — so following only the outermost `right`
/// would place sectors around the gap and never learn the bore exists.
fn cut_chain(doc: &Document, id: NodeId) -> Vec<NodeId> {
    let mut cuts = Vec::new();
    let mut cur = id;
    while let Some(CsgOp::Difference { left, right }) = doc.nodes.get(&cur).map(|n| &n.op) {
        cuts.push(*right);
        cur = *left;
    }
    cuts
}

/// A boolean subtree: try a decomposition, fall back to a single hull.
///
/// `subtrahends` are the nodes whose solids were removed — they are both what
/// the sector planes are placed around and what the result is checked against.
fn carve(
    doc: &Document,
    id: NodeId,
    subtrahends: Vec<NodeId>,
    frame: Frame,
    name: &str,
    out: &mut Derived,
    cache: &mut Cache,
) -> anyhow::Result<()> {
    let Some(solid) = vcad_eval::evaluate_node(id, &doc.nodes, cache).map_err(|e| anyhow::anyhow!("{e:?}"))? else {
        return Ok(());
    };
    let mesh = solid.to_mesh(SEGMENTS);
    let (pts, tris) = to_body(&frame, &mesh.vertices, &mesh.indices);
    if pts.len() < 4 {
        return Ok(());
    }

    // Fast path: a boolean whose result is already convex — a chamfered block,
    // an intersection of two convex solids — is one exact collider, with no
    // sectors and nothing to warn about.
    let concave = concavity(&pts, &tris);
    if concave <= INTRUSION_TOL {
        let faces = hull_faces(&pts);
        if !faces.is_empty() {
            out.notes.push(format!("'{name}' is convex to {:.3} mm: one exact hull collider", concave / MM));
            out.colliders.push(mesh_instance(pts, faces));
            return Ok(());
        }
    }

    // The removed volumes, in the same body coordinates. They are kept apart
    // rather than merged: a gap box and a bore overlap, and ray parity through
    // two overlapping shells cancels in the overlap — exactly where a wrong
    // piece would hide.
    let mut cuts: Vec<Cut> = Vec::new();
    for sub in &subtrahends {
        let Some(solid) = vcad_eval::evaluate_node(*sub, &doc.nodes, cache).map_err(|e| anyhow::anyhow!("{e:?}"))?
        else {
            continue;
        };
        let m = solid.to_mesh(SEGMENTS);
        let (p, t) = to_body(&frame, &m.vertices, &m.indices);
        if p.len() >= 4 && !t.is_empty() {
            cuts.push((p, t));
        }
    }
    if cuts.is_empty() {
        // A non-convex intersection, or a difference by nothing: no hole to
        // place sectors around and nothing to check against. A hull is all we
        // can justify.
        let kind = if subtrahends.is_empty() { "Intersection" } else { "Difference" };
        return hull_fallback(doc, id, frame, name, kind, out, cache);
    }

    match decompose(&pts, &tris, &cuts) {
        Some((pieces, label, intrusion)) => {
            let first = out.colliders.len();
            for (p, f) in pieces {
                out.colliders.push(mesh_instance(p, f));
            }
            let n = out.colliders.len() - first;
            let pieces = first..out.colliders.len();
            out.notes.push(format!(
                "'{name}' is a Difference of {} cut(s): {n} convex pieces by {label}, reaching {:.3} mm into the cuts",
                cuts.len(),
                intrusion / MM
            ));
            for (_, tris) in cuts {
                out.removed.push(Removed { tris, pieces: pieces.clone() });
            }
            Ok(())
        }
        None => hull_fallback(doc, id, frame, name, "Difference", out, cache),
    }
}

fn mesh_instance(vertices: Vec<Vec3>, faces: Vec<[usize; 3]>) -> GeomInstance {
    GeomInstance::new(Geometry::Mesh { vertices, faces }, SpatialTransform::identity())
}

/// The old behaviour: one collider that is the convex hull of the subtree's
/// tessellation, and a warning saying so.
fn hull_fallback(
    doc: &Document,
    id: NodeId,
    frame: Frame,
    name: &str,
    kind: &str,
    out: &mut Derived,
    cache: &mut Cache,
) -> anyhow::Result<()> {
    let Some(solid) = vcad_eval::evaluate_node(id, &doc.nodes, cache).map_err(|e| anyhow::anyhow!("{e:?}"))? else {
        return Ok(());
    };
    let mesh = solid.to_mesh(24);
    let vertices: Vec<Vec3> =
        mesh.vertices.chunks(3).map(|v| frame.apply(Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)) * MM).collect();
    let faces: Vec<[usize; 3]> =
        mesh.indices.chunks(3).map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]).collect();
    out.warnings.push(format!(
        "'{name}' is a {kind}: collider is the convex hull of its {} tessellated vertices",
        vertices.len()
    ));
    out.colliders.push(mesh_instance(vertices, faces));
    Ok(())
}

// =============================================================================
// Approximate convex decomposition
// =============================================================================
//
// The decomposition is geometric, not syntactic: evaluate the subtree to a
// mesh, clip that mesh against a family of convex sectors, and hull each
// sector's share of it. The convex hull of points inside a convex sector stays
// inside that sector, so the only way a piece can be wrong is by spanning a
// concavity — which is exactly what the sector planes are placed to prevent.
//
// A sector is an intersection of half-spaces, and clipping means more than
// binning vertices: a sector's share of the mesh is its vertices *plus* the
// points where the mesh's edges cross the sector's planes. Without those
// crossings a box would contribute only its eight corners, so a wedge that
// contained no corner would hull down to a splinter and the union of wedges
// would be a sieve with a plate-shaped outline.
//
// Two families of sector, both keyed off a removed volume's bounding box:
//
//   * **grid** — split at the box's own faces. 27 cells minus the one inside
//     the box: exact for a rectangular pocket, and only eight-ish pieces.
//   * **wedges** — angular sectors about an axis through the box's centre,
//     crossed with three slabs along that axis. A hollow cylinder becomes a
//     ring of wedge hulls; the residual intrusion is the sagitta of the chord
//     across the bore, which shrinks as the sector count rises.
//
// Every candidate is scored before it is accepted:
//
//   * **coverage** — points sampled all over the result's *surface*, not just
//     its vertices, must each lie inside some piece, so the union is the solid
//     and not a sieve. This is what rejects a grid for a *round* hole: the
//     material between the arc and the bounding box's corner belongs to the
//     excluded centre cell and no other cell claims it.
//   * **intrusion** — no sampled point on any piece's hull may lie more than
//     `INTRUSION_TOL` inside any subtrahend's mesh.
//
// Candidates are tried cheapest-first and the first that passes both wins.

type Piece = (Vec<Vec3>, Vec<[usize; 3]>);
type Cut = (Vec<Vec3>, Vec<Tri>);
/// A half-space: inside where `n · p ≤ d`, with `n` a unit normal.
type Plane = (Vec3, f64);
/// A convex region, as the intersection of its half-spaces.
type Sector = Vec<Plane>;

/// Returns the pieces, a human label for the strategy, and the worst intrusion.
fn decompose(pts: &[Vec3], tris: &[Tri], cuts: &[Cut]) -> Option<(Vec<Piece>, String, f64)> {
    let edges = edges_of(tris);
    let samples = surface_samples(tris);

    // Sector planes are placed around each cut in turn: with a bore and a slot
    // in the same solid only one of the two bounding boxes leads to a
    // decomposition that clears both, and trying them is cheaper than guessing.
    let mut candidates: Vec<(String, Vec<Sector>)> = Vec::new();
    for (c, (cut_pts, _)) in cuts.iter().enumerate() {
        let (lo, hi) = bounds(cut_pts);
        candidates.push((format!("a grid on cut {c}'s bounding box"), grid_sectors(lo, hi)));
    }
    for n in SECTOR_COUNTS {
        for (c, (cut_pts, _)) in cuts.iter().enumerate() {
            let (lo, hi) = bounds(cut_pts);
            for axis in [2usize, 0, 1] {
                let label = format!("{n} wedges about {} through cut {c}", ["x", "y", "z"][axis]);
                candidates.push((label, wedge_sectors(lo, hi, axis, n)));
            }
        }
    }

    for (label, sectors) in candidates {
        let pieces: Vec<Piece> = sectors
            .iter()
            .filter_map(|s| {
                let p = sector_points(s, pts, &edges);
                let f = hull_faces(&p);
                (!f.is_empty()).then_some((p, f))
            })
            .collect();
        if pieces.len() < 2 {
            continue;
        }
        // Coverage first: it is the cheaper of the two to fail, because one
        // uncovered sample ends the candidate.
        if !covers(&pieces, &samples) {
            continue;
        }
        let Some(intrusion) = intrusion_of(&pieces, cuts) else {
            continue;
        };
        return Some((pieces, label, intrusion));
    }
    None
}

/// Worst reach of any piece into any cut, or `None` once that exceeds the
/// tolerance — the early exit is what keeps trying dozens of candidates cheap.
fn intrusion_of(pieces: &[Piece], cuts: &[Cut]) -> Option<f64> {
    let mut worst = 0.0f64;
    for (p, f) in pieces {
        for s in hull_samples(p, f) {
            for (_, tris) in cuts {
                worst = worst.max(depth_inside(tris, s));
            }
            if worst > INTRUSION_TOL {
                return None;
            }
        }
    }
    Some(worst)
}

/// Every sampled point of the solid's surface must lie inside (or on) a piece.
fn covers(pieces: &[Piece], samples: &[Vec3]) -> bool {
    samples.iter().all(|p| pieces.iter().any(|(hp, hf)| hull_signed_distance(hp, hf, *p) <= INTRUSION_TOL))
}

fn unit(axis: usize) -> Vec3 {
    match axis {
        0 => Vec3::x(),
        1 => Vec3::y(),
        _ => Vec3::z(),
    }
}

/// The `k`-th of the three slabs `(below lo, between, above hi)` along `axis`.
fn slab(axis: usize, lo: f64, hi: f64, k: usize) -> Vec<Plane> {
    let u = unit(axis);
    match k {
        0 => vec![(u, lo)],
        1 => vec![(-u, -lo), (u, hi)],
        _ => vec![(-u, -hi)],
    }
}

/// 27 cells cut at the removed box's own faces, minus the cell inside it.
fn grid_sectors(lo: Vec3, hi: Vec3) -> Vec<Sector> {
    let mut out = Vec::new();
    for ix in 0..3 {
        for iy in 0..3 {
            for iz in 0..3 {
                if (ix, iy, iz) == (1, 1, 1) {
                    continue; // inside the cut: no material here
                }
                let mut s = slab(0, lo.x, hi.x, ix);
                s.extend(slab(1, lo.y, hi.y, iy));
                s.extend(slab(2, lo.z, hi.z, iz));
                out.push(s);
            }
        }
    }
    out
}

/// `n` angular wedges about `axis` through the cut's centre, crossed with the
/// three slabs along that axis.
fn wedge_sectors(lo: Vec3, hi: Vec3, axis: usize, n: usize) -> Vec<Sector> {
    let (eu, ev) = (unit((axis + 1) % 3), unit((axis + 2) % 3));
    let mid = (lo + hi) * 0.5;
    let w = std::f64::consts::TAU / n as f64;
    let mut out = Vec::new();
    for i in 0..n {
        // The wedge [i·w, (i+1)·w] is the inner side of two half-planes hinged
        // on the axis. Both are written as `n · p ≤ d` about `mid`.
        let (a0, a1) = (i as f64 * w, (i + 1) as f64 * w);
        let n0 = eu * a0.sin() - ev * a0.cos();
        let n1 = -eu * a1.sin() + ev * a1.cos();
        for k in 0..3 {
            let mut s = vec![(n0, n0.dot(mid)), (n1, n1.dot(mid))];
            s.extend(slab(axis, comp(lo, axis), comp(hi, axis), k));
            out.push(s);
        }
    }
    out
}

/// A sector's share of the mesh: the vertices it contains, plus the points
/// where the mesh's edges cross its planes. The crossings are what make the
/// pieces tile the solid rather than merely sample it.
///
/// The result is reduced to its extreme points before it is returned. Clipping
/// a 48-segment tessellation produces hundreds of points per sector, nearly all
/// of them interior to their own hull, and the brute-force hull below is
/// O(n⁴). Keeping only what is extreme along a spread of directions leaves the
/// hull essentially unchanged and can only ever *shrink* it, which is the safe
/// direction: a piece cannot intrude into a hole by losing a vertex.
fn sector_points(sector: &Sector, pts: &[Vec3], edges: &[(Vec3, Vec3)]) -> Vec<Vec3> {
    let held = |p: Vec3| sector.iter().all(|(n, d)| n.dot(p) <= d + SECTOR_OVERLAP);
    let mut out: Vec<Vec3> = pts.iter().copied().filter(|p| held(*p)).collect();
    for (a, b) in edges {
        for (n, d) in sector {
            let (da, db) = (n.dot(a) - d, n.dot(b) - d);
            if (da > 0.0) == (db > 0.0) {
                continue;
            }
            let q = a + (b - a) * (da / (da - db));
            if held(q) {
                out.push(q);
            }
        }
    }
    extreme_points(&dedup(&out))
}

/// The points of `pts` that are furthest along each of `HULL_DIRS` evenly
/// spread directions — at most that many, and a good stand-in for the hull's
/// vertices when the hull is as simple as one sector of one solid.
fn extreme_points(pts: &[Vec3]) -> Vec<Vec3> {
    if pts.len() <= HULL_DIRS {
        return pts.to_vec();
    }
    // A Fibonacci sphere: evenly spread without a preferred axis, which an
    // octahedral or cubic direction set is not.
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    let mut keep = Vec::new();
    for i in 0..HULL_DIRS {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / HULL_DIRS as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let a = golden * i as f64;
        let d = Vec3::new(r * a.cos(), r * a.sin(), z);
        let mut best = (0usize, f64::MIN);
        for (j, p) in pts.iter().enumerate() {
            let s = p.dot(d);
            if s > best.1 {
                best = (j, s);
            }
        }
        keep.push(best.0);
    }
    // Sorted and deduplicated, not collected through a hash set: the *order* of
    // a piece's points decides which anchor `hull_faces` measures its planes
    // from, and a run-to-run reshuffle there is a run-to-run reshuffle of the
    // colliders the physics gets.
    keep.sort_unstable();
    keep.dedup();
    keep.into_iter().map(|i| pts[i]).collect()
}

/// The mesh's distinct edges.
fn edges_of(tris: &[Tri]) -> Vec<(Vec3, Vec3)> {
    let key = |p: &Vec3| {
        let q = |x: f64| (x * 1e9).round() as i64;
        (q(p.x), q(p.y), q(p.z))
    };
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (ka, kb) = (key(&a), key(&b));
            if seen.insert(if ka <= kb { (ka, kb) } else { (kb, ka) }) {
                out.push((a, b));
            }
        }
    }
    out
}

/// Points spread over the mesh's surface, for the coverage test. Vertices alone
/// would not do: a plate's vertices are its eight corners, and a decomposition
/// that covered only those could still leave the middle of the plate hollow.
fn surface_samples(tris: &[Tri]) -> Vec<Vec3> {
    const N: usize = 3;
    const CAP: usize = 8000;
    let per = (N + 1) * (N + 2) / 2;
    let stride = (tris.len() * per / CAP).max(1);
    let mut out = Vec::new();
    for t in tris.iter().step_by(stride) {
        for i in 0..=N {
            for j in 0..=(N - i) {
                out.push(t[0] + (t[1] - t[0]) * (i as f64 / N as f64) + (t[2] - t[0]) * (j as f64 / N as f64));
            }
        }
    }
    out
}

// =============================================================================
// Mesh and hull utilities
// =============================================================================

/// Flat f32 vcad mesh (mm, local) → deduplicated points and triangles in body
/// coordinates (metres).
fn to_body(frame: &Frame, vertices: &[f32], indices: &[u32]) -> (Vec<Vec3>, Vec<Tri>) {
    let pts: Vec<Vec3> =
        vertices.chunks(3).map(|v| frame.apply(Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)) * MM).collect();
    let tris = indices
        .chunks(3)
        .filter_map(|t| {
            let (a, b, c) = (*t.first()? as usize, *t.get(1)? as usize, *t.get(2)? as usize);
            Some([*pts.get(a)?, *pts.get(b)?, *pts.get(c)?])
        })
        .collect();
    (dedup(&pts), tris)
}

fn dedup(pts: &[Vec3]) -> Vec<Vec3> {
    let key = |p: &Vec3| {
        let q = |x: f64| (x * 1e9).round() as i64;
        (q(p.x), q(p.y), q(p.z))
    };
    let mut seen = HashSet::new();
    pts.iter().filter(|p| seen.insert(key(p))).copied().collect()
}

/// One component of a vector, by axis index.
fn comp(v: Vec3, i: usize) -> f64 {
    match i {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn bounds(pts: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Vec3::new(f64::MIN, f64::MIN, f64::MIN);
    for p in pts {
        lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    (lo, hi)
}

/// How far the mesh departs from its own convex hull: the deepest any vertex
/// sits *outside* a face's plane, with the winding oriented outward. Zero for a
/// convex solid, roughly the bore radius for a tube.
fn concavity(pts: &[Vec3], tris: &[Tri]) -> f64 {
    if tris.is_empty() {
        return f64::MAX;
    }
    // Orient by signed volume, so the test does not depend on the tessellator's
    // winding convention.
    let vol: f64 = tris.iter().map(|t| t[0].dot(t[1].cross(t[2]))).sum();
    let sign = if vol < 0.0 { -1.0 } else { 1.0 };
    let mut worst = 0.0f64;
    for t in tris {
        let nrm = (t[1] - t[0]).cross(t[2] - t[0]) * sign;
        let len = nrm.norm();
        if len < 1e-16 {
            continue;
        }
        let nrm = nrm / len;
        for p in pts {
            worst = worst.max((p - t[0]).dot(nrm));
        }
    }
    worst
}

/// Squared distance from `p` to a triangle (Ericson's closest-point routine).
fn dist2_tri(p: Vec3, t: &Tri) -> f64 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (p - a).norm_squared();
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (p - b).norm_squared();
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (p - c).norm_squared();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (p - (a + ab * (d1 / (d1 - d3)))).norm_squared();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (p - (a + ac * (d2 / (d2 - d6)))).norm_squared();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let s = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * s)).norm_squared();
    }
    let denom = 1.0 / (va + vb + vc);
    (p - (a + ab * (vb * denom) + ac * (vc * denom))).norm_squared()
}

/// Ray-parity containment. The direction is deliberately irrational so it does
/// not graze the axis-aligned faces a tessellated box is made of.
fn inside(tris: &[Tri], p: Vec3) -> bool {
    let d = Vec3::new(0.5773502691, 0.3251344395, 0.7495029123).normalize();
    let mut crossings = 0usize;
    for t in tris {
        let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
        let h = d.cross(e2);
        let det = e1.dot(h);
        if det.abs() < 1e-16 {
            continue;
        }
        let f = 1.0 / det;
        let s = p - t[0];
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = f * d.dot(q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        if f * e2.dot(q) > 1e-12 {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

/// How far `p` reaches inside the volume bounded by `tris`; 0 when outside.
fn depth_inside(tris: &[Tri], p: Vec3) -> f64 {
    if !inside(tris, p) {
        return 0.0;
    }
    tris.iter().map(|t| dist2_tri(p, t)).fold(f64::MAX, f64::min).max(0.0).sqrt()
}

/// Outward-wound hull faces of a small point set, by brute force: a triple is a
/// face when every other point lies on one side of its plane. Point sets here
/// are one sector of one tessellated subtree — tens of points, not thousands —
/// so O(n⁴) on tens is free. A coplanar patch yields several coincident
/// triangles, which is harmless for rendering and deduplicated for sampling.
fn hull_faces(pts: &[Vec3]) -> Vec<[usize; 3]> {
    const EPS: f64 = 1e-9;
    let n = pts.len();
    let mut faces = Vec::new();
    if !(4..=HULL_DIRS).contains(&n) {
        return faces;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let nrm = (pts[j] - pts[i]).cross(pts[k] - pts[i]);
                let len = nrm.norm();
                if len < 1e-16 {
                    continue;
                }
                let nrm = nrm / len;
                let (mut lo, mut hi) = (0.0f64, 0.0f64);
                for p in pts {
                    let d = (p - pts[i]).dot(nrm);
                    lo = lo.min(d);
                    hi = hi.max(d);
                }
                if hi <= EPS {
                    faces.push([i, j, k]);
                } else if lo >= -EPS {
                    faces.push([i, k, j]);
                }
            }
        }
    }
    faces
}

/// Signed distance to the hull, positive outside. Exact when the nearest
/// feature is a face and a lower bound otherwise, which is the safe direction
/// for a coverage test.
fn hull_signed_distance(pts: &[Vec3], faces: &[[usize; 3]], p: Vec3) -> f64 {
    let mut worst = f64::MIN;
    for f in faces {
        let (a, b, c) = (pts[f[0]], pts[f[1]], pts[f[2]]);
        let nrm = (b - a).cross(c - a);
        let len = nrm.norm();
        if len < 1e-16 {
            continue;
        }
        worst = worst.max((p - a).dot(nrm / len));
    }
    if worst == f64::MIN { f64::MAX } else { worst }
}

/// Points spread through a piece, for the intrusion test.
///
/// The deepest reach into a removed volume is never at a vertex — a wedge
/// across a bore touches the solid at both ends of its chord and cuts the hole
/// in between — so vertices alone would report zero for exactly the pieces that
/// are wrong. Two independent families are sampled:
///
///   * every distinct hull face, subdivided barycentrically; and
///   * the midpoints and triangle centroids of the point set itself.
///
/// The second family is the belt to the first's braces. It needs no faces at
/// all, so a piece whose near-coplanar point cloud confuses `hull_faces` into
/// reporting a partial surface still gets probed across its own chords — and
/// the midpoint of the two vertices that straddle a bore is precisely the
/// deepest point of the intrusion it would cause.
fn hull_samples(pts: &[Vec3], faces: &[[usize; 3]]) -> Vec<Vec3> {
    const N: usize = 4;
    let mut out: Vec<Vec3> = pts.to_vec();
    let mut seen = HashSet::new();
    for f in faces {
        let mut key = *f;
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        let (a, b, c) = (pts[f[0]], pts[f[1]], pts[f[2]]);
        for i in 0..=N {
            for j in 0..=(N - i) {
                out.push(a + (b - a) * (i as f64 / N as f64) + (c - a) * (j as f64 / N as f64));
            }
        }
    }
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            out.push((pts[i] + pts[j]) * 0.5);
            for k in (j + 1)..pts.len() {
                out.push((pts[i] + pts[j] + pts[k]) * (1.0 / 3.0));
            }
        }
    }
    out
}

// =============================================================================
// Verification
// =============================================================================

/// Check every collider against the tessellation of the whole document: along a
/// set of directions, the collider's support point must not exceed the mesh's,
/// and the union of colliders must reach the mesh's extent. Returns the worst
/// discrepancy in metres.
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

/// The other half of the check, for colliders derived from a `Difference`.
///
/// The support test above only ever sees the outside of the level, and a puck
/// and a cup have the same outside — it is exactly blind to the bug this
/// module exists to fix. This one looks in the hole: for every piece derived
/// alongside a cut, sample its hull's surface and measure how far the deepest
/// sample lies inside the volume that was removed. Returns metres; zero means
/// nothing reaches into the cut at all.
pub fn verify_no_intrusion(derived: &Derived) -> f64 {
    let mut worst = 0.0f64;
    for cut in &derived.removed {
        for inst in &derived.colliders[cut.pieces.clone()] {
            let Geometry::Mesh { vertices, faces } = &inst.geometry else {
                continue;
            };
            for s in hull_samples(vertices, faces) {
                worst = worst.max(depth_inside(&cut.tris, s));
            }
        }
    }
    worst
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{Builder, Params, Shape, build};

    /// The colliders of a one-body level, built in Rust.
    fn derive(shape: impl Fn(&Builder) -> Shape + Send + Sync + 'static) -> Derived {
        let built = build(&Params::default(), move |b| {
            b.body("part").add(shape(b));
        })
        .expect("builds");
        let mut bodies = built.bodies;
        bodies.remove(0).colliders
    }

    fn hulls(d: &Derived) -> Vec<(&Vec<Vec3>, &Vec<[usize; 3]>)> {
        d.colliders
            .iter()
            .filter_map(|i| match &i.geometry {
                Geometry::Mesh { vertices, faces } => Some((vertices, faces)),
                _ => None,
            })
            .collect()
    }

    /// A tube: outer radius 25, bore 20, 30 mm tall, bored right through.
    fn hollow_cylinder(b: &Builder) -> Shape {
        b.cylinder(25.0, 30.0).difference(b.cylinder(20.0, 40.0))
    }

    /// A 100×80×10 plate with a 12 mm bore straight through it at (50, 40).
    fn holed_plate(b: &Builder) -> Shape {
        b.cube(100.0, 80.0, 10.0).difference(b.cylinder(12.0, 20.0).at(50.0, 40.0, -5.0))
    }

    /// The cup `sims/marble` models as a bore and a slot, at its own numbers:
    /// centre (90, 0), inside radius 22, wall 3, 14 mm of it.
    fn marble_cup(b: &Builder) -> Shape {
        let (cup_x, cup_r, cup_wall, cup_h) = (90.0, 22.0, 3.0, 14.0);
        let cup_big = cup_r + cup_wall;
        // Mouth first, then the bore — see `sims/marble/scene.rs::cup_hollow`
        // for why the order is load-bearing under vcad 0.10.
        let mouth = b
            .boxed(1.4444 * cup_big, 1.6630 * cup_big, 1.2 * cup_h)
            .at(cup_x - 1.2778 * cup_big, 0.0, 0.5 * cup_h);
        b.cylinder(cup_big, cup_h)
            .at(cup_x, 0.0, 0.0)
            .difference(mouth)
            .difference(b.cylinder(cup_r, 1.1 * cup_h).at(cup_x, 0.0, 0.0))
    }

    #[test]
    fn hollow_cylinder_decomposes_and_leaves_the_bore_empty() {
        let d = derive(hollow_cylinder);
        assert!(d.warnings.is_empty(), "should not have fallen back: {:?}", d.warnings);
        assert!(d.colliders.len() >= 6, "expected a ring of pieces, got {}", d.colliders.len());
        assert_eq!(hulls(&d).len(), d.colliders.len(), "every piece should be a hull");

        // The tube's axis runs from z = 0 to z = 30 mm. No piece may contain a
        // point on it: that is the bore.
        for z in [1.0, 7.5, 15.0, 22.5, 29.0] {
            let axis = Vec3::new(0.0, 0.0, z * MM);
            for (i, (p, f)) in hulls(&d).iter().enumerate() {
                assert!(
                    hull_signed_distance(p, f, axis) > INTRUSION_TOL,
                    "piece {i} contains the axis point at z = {z} mm"
                );
            }
        }
        let worst = verify_no_intrusion(&d);
        assert!(worst < INTRUSION_TOL, "pieces reach {:.4} mm into the bore", worst / MM);
    }

    #[test]
    fn plate_with_a_through_hole_keeps_the_hole() {
        let d = derive(holed_plate);
        assert!(d.warnings.is_empty(), "should not have fallen back: {:?}", d.warnings);
        assert!(d.colliders.len() >= 6, "expected several pieces, got {}", d.colliders.len());

        for z in [1.0, 5.0, 9.0] {
            let axis = Vec3::new(50.0 * MM, 40.0 * MM, z * MM);
            for (i, (p, f)) in hulls(&d).iter().enumerate() {
                assert!(hull_signed_distance(p, f, axis) > INTRUSION_TOL, "piece {i} plugs the bore at z = {z} mm");
            }
        }
        let worst = verify_no_intrusion(&d);
        assert!(worst < INTRUSION_TOL, "pieces reach {:.4} mm into the bore", worst / MM);
    }

    #[test]
    fn the_plate_beside_the_hole_is_still_solid() {
        // Sanity in the other direction: a decomposition that leaves the hole
        // empty by leaving *everything* empty would pass the test above.
        let d = derive(holed_plate);
        for (dx, dy) in [(-40.0, -30.0), (40.0, 30.0), (-40.0, 30.0), (20.0, -10.0)] {
            let p = Vec3::new((50.0 + dx) * MM, (40.0 + dy) * MM, 5.0 * MM);
            assert!(
                hulls(&d).iter().any(|(hp, hf)| hull_signed_distance(hp, hf, p) < 0.0),
                "plate is hollow at ({dx}, {dy}) mm from the bore"
            );
        }
    }

    #[test]
    fn flat_primitives_still_match_the_tessellation() {
        // The unchanged path. Boxes tessellate exactly, so their support
        // functions must agree with the mesh to round-off — no tolerance to
        // hide a regression behind.
        let built = build(&Params::default(), |b| {
            b.body("part")
                .add(b.cube(20.0, 20.0, 20.0).at(30.0, 0.0, 0.0))
                .add(b.cube(40.0, 20.0, 10.0).rotate_z(15.0))
                .add(b.cube(10.0, 10.0, 40.0).rotate_x(30.0).at(-30.0, 5.0, 0.0));
        })
        .expect("builds");
        let d = &built.bodies[0].colliders;
        assert!(d.warnings.is_empty(), "primitives should not warn: {:?}", d.warnings);
        assert_eq!(d.colliders.len(), 3);
        assert!(d.removed.is_empty());
        let worst = verify_against_mesh(&built.document, d).expect("verify");
        assert!(worst < 1e-6, "support functions disagree by {worst:.3e} m");
        assert_eq!(verify_no_intrusion(d), 0.0);
    }

    #[test]
    fn curved_primitives_are_inscribed_by_their_tessellation() {
        // A sphere and a cylinder are exact colliders but chordal meshes, so
        // the collider legitimately stands proud of the tessellation by the
        // sagitta. It is a few microns, and it is one-sided.
        let built = build(&Params::default(), |b| {
            b.body("part")
                .add(b.sphere(8.0).at(30.0, 0.0, 0.0))
                .add(b.cylinder(6.0, 20.0).at(-30.0, 0.0, 0.0));
        })
        .expect("builds");
        let d = &built.bodies[0].colliders;
        assert!(d.warnings.is_empty(), "primitives should not warn: {:?}", d.warnings);
        assert_eq!(d.colliders.len(), 2);
        let worst = verify_against_mesh(&built.document, d).expect("verify");
        assert!(worst < 1e-5, "support functions disagree by {worst:.3e} m, more than a chord sagitta");
    }

    /// The level this module exists for: `sims/marble` models its cup as a
    /// bore and a slot, and the marble has to be able to get inside it.
    #[test]
    fn the_cup_level_leaves_its_cup_hollow() {
        let d = derive(marble_cup);
        assert!(d.warnings.is_empty(), "the cup should not have fallen back: {:?}", d.warnings);
        // Cup centre is (cup_x, 0) = (90, 0) mm; the bore is 22 mm in radius
        // and runs the full 14 mm of the wall.
        for z in [1.0, 7.0, 13.0] {
            let inside_cup = Vec3::new(90.0 * MM, 0.0, z * MM);
            for (i, (p, f)) in hulls(&d).iter().enumerate() {
                assert!(
                    hull_signed_distance(p, f, inside_cup) > INTRUSION_TOL,
                    "piece {i} fills the cup at z = {z} mm"
                );
            }
        }
        let worst = verify_no_intrusion(&d);
        assert!(worst < INTRUSION_TOL, "pieces reach {:.4} mm into the cup", worst / MM);
    }

    #[test]
    fn an_intersection_of_convex_solids_is_one_exact_collider() {
        // Two overlapping boxes: the result is a box, so the convex fast path
        // takes it and there is nothing to decompose or warn about.
        let d = derive(|b| b.cube(20.0, 20.0, 20.0).intersection(b.cube(20.0, 20.0, 20.0).at(10.0, 10.0, 10.0)));
        assert!(d.warnings.is_empty(), "a convex result should not warn: {:?}", d.warnings);
        assert_eq!(d.colliders.len(), 1);
        assert_eq!(d.notes.len(), 1);
    }
}
