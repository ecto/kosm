//! Convex pieces and what they weigh: the physics layer of a *thing*.
//!
//! A place gets a signed-distance field, because a place does not move and a
//! field is exact everywhere ([`crate::sdf`]). A thing moves, and a field
//! baked in the world frame stops being true the moment somebody kicks it.
//! So an object's collision layer is a small set of **convex pieces** carried
//! in the object's own frame, which the contact pipeline transforms with the
//! body every step for free.
//!
//! Three pieces of arithmetic live here, in the order a bake needs them:
//!
//! 1. [`convex_hull`] — the hull of a point cloud, as a closed triangle mesh.
//! 2. [`mass_properties`] — volume, centre of mass and the inertia tensor of
//!    a closed mesh, by exact tetrahedron integration. This is what makes a
//!    captured object *fall right*: a bucket and a bowling ball of the same
//!    size differ almost entirely in these six numbers.
//! 3. [`decompose`] — a concavity-driven split into several hulls, because
//!    one hull over a mug is a lump with no handle and one hull over a bucket
//!    is a solid cylinder you cannot put a foot inside.
//!
//! # Why hulls and not the scan mesh
//!
//! phyz's `Geometry::Mesh` is documented "vertices only; convex hull
//! assumed" — handing it a concave scan does not produce concave contact, it
//! produces the hull silently. Making the decomposition explicit means the
//! error is a *number in the manifest* (`hull_error`) instead of a surprise
//! at the first contact, and it means the pieces can be looked at.
//!
//! # Honest edges
//!
//! * The decomposition is greedy axis-aligned splitting, not V-HACD. It is a
//!   few hundred lines instead of a few thousand, it is deterministic, and it
//!   is measurably good enough for the shapes a robot lab contains (an L, a
//!   bucket, a mug). It will do poorly on something like a spiral.
//! * Pieces may overlap at a seam. That is deliberate and it is the safe
//!   direction: overlapping convex colliders are fine, gaps between them are
//!   not, and every triangle of the source surface is inside some piece.
//! * Mass properties assume **uniform density**. Real objects are not
//!   uniform — a drill is nearly all battery — so the bake records the
//!   assumption and lets a measured mass override the density-derived one.

use phyz_math::{Mat3, Vec3};

/// One convex piece: vertices and the closed triangle surface over them.
///
/// Faces are carried as well as vertices even though phyz's mesh collider
/// only reads vertices, because the viewer draws the pieces and a hull whose
/// faces are re-derived on the other side of the wire is a second
/// implementation of this file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Hull {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<[u32; 3]>,
}

impl Hull {
    /// Axis-aligned bounds.
    pub fn aabb(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f64::INFINITY);
        let mut hi = Vec3::splat(f64::NEG_INFINITY);
        for v in &self.vertices {
            lo = lo.component_min(*v);
            hi = hi.component_max(*v);
        }
        (lo, hi)
    }

    /// Signed distance to the hull's boundary, negative inside.
    ///
    /// A convex hull is the intersection of its face half-spaces, so this is
    /// the max over faces — exact inside, and an under-estimate outside near
    /// an edge, which is all the containment tests here need.
    pub fn plane_distance(&self, p: Vec3) -> f64 {
        let mut worst = f64::NEG_INFINITY;
        for f in &self.faces {
            let (a, b, c) = (
                self.vertices[f[0] as usize],
                self.vertices[f[1] as usize],
                self.vertices[f[2] as usize],
            );
            let n = (b - a).cross(c - a);
            let len = n.norm();
            if len < 1e-18 {
                continue;
            }
            worst = worst.max((p - a).dot(n / len));
        }
        worst
    }
}

/// The convex hull of a point cloud, as a closed outward-wound mesh.
///
/// `None` when the points are degenerate — fewer than four, or all within
/// `1e-9 × extent` of a single plane. A caller with a flat point set wants to
/// know that rather than receive a zero-volume "solid" the contact pipeline
/// will divide by.
///
/// Incremental (add a point, remove the faces it can see, stitch the horizon)
/// with the outside-set pruning that makes it quickhull-shaped: after every
/// insertion the candidate list drops to the points still outside, which on
/// scan data collapses from a hundred thousand to a few hundred within a
/// dozen insertions.
pub fn convex_hull(points: &[Vec3]) -> Option<Hull> {
    if points.len() < 4 {
        return None;
    }
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for p in points {
        lo = lo.component_min(*p);
        hi = hi.component_max(*p);
    }
    let scale = (hi - lo).max_element();
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    // Relative, because a 3 cm object and a 30 m room cannot share an
    // absolute tolerance and the same code bakes both.
    let eps = scale * 1e-9;

    // ── the seed tetrahedron ──
    let far = |dir: Vec3| -> usize {
        let mut best = 0;
        let mut score = f64::NEG_INFINITY;
        for (i, p) in points.iter().enumerate() {
            let s = p.dot(dir);
            if s > score {
                score = s;
                best = i;
            }
        }
        best
    };
    let a = far(Vec3::new(1.0, 0.0, 0.0));
    let b = far(Vec3::new(-1.0, 0.0, 0.0));
    if (points[a] - points[b]).norm() < eps {
        return None;
    }
    let ab = points[b] - points[a];
    // Farthest from the line ab.
    let mut c = usize::MAX;
    let mut best = eps;
    for (i, p) in points.iter().enumerate() {
        let d = (*p - points[a]).cross(ab).norm() / ab.norm();
        if d > best {
            best = d;
            c = i;
        }
    }
    if c == usize::MAX {
        return None;
    }
    // Farthest from the plane abc.
    let n = ab.cross(points[c] - points[a]);
    let nl = n.norm();
    if nl < eps {
        return None;
    }
    let n = n / nl;
    let mut d = usize::MAX;
    let mut best = eps;
    for (i, p) in points.iter().enumerate() {
        let h = (*p - points[a]).dot(n).abs();
        if h > best {
            best = h;
            d = i;
        }
    }
    if d == usize::MAX {
        return None;
    }

    // Wind the seed so every face points away from the interior.
    let mut faces: Vec<[usize; 3]> = if (points[d] - points[a]).dot(n) < 0.0 {
        vec![[a, b, c], [a, c, d], [a, d, b], [b, d, c]]
    } else {
        vec![[a, c, b], [a, b, d], [a, d, c], [b, c, d]]
    };

    let plane = |f: &[usize; 3]| -> (Vec3, f64) {
        let (p0, p1, p2) = (points[f[0]], points[f[1]], points[f[2]]);
        let n = (p1 - p0).cross(p2 - p0);
        let len = n.norm();
        if len < 1e-18 {
            (Vec3::zeros(), 0.0)
        } else {
            let n = n / len;
            (n, n.dot(p0))
        }
    };

    // Points still outside the hull. Everything already inside can never
    // become relevant again, so the list only shrinks.
    let mut outside: Vec<usize> = (0..points.len()).collect();
    let mut guard = points.len() * 4 + 64;

    loop {
        guard = guard.saturating_sub(1);
        if guard == 0 {
            break;
        }
        let planes: Vec<(Vec3, f64)> = faces.iter().map(plane).collect();
        // Keep only points outside some face, and take the farthest as next.
        let mut next = usize::MAX;
        let mut next_depth = eps;
        outside.retain(|&i| {
            let p = points[i];
            let mut worst = f64::NEG_INFINITY;
            for (n, o) in &planes {
                worst = worst.max(p.dot(*n) - *o);
            }
            if worst > next_depth {
                next_depth = worst;
                next = i;
            }
            worst > eps
        });
        if next == usize::MAX {
            break;
        }

        // Faces this point can see, and the horizon: edges owned by exactly
        // one visible face.
        let p = points[next];
        let mut visible = vec![false; faces.len()];
        for (k, (n, o)) in planes.iter().enumerate() {
            visible[k] = p.dot(*n) - *o > eps;
        }
        let mut edge_count: std::collections::HashMap<(usize, usize), i32> =
            std::collections::HashMap::new();
        for (k, f) in faces.iter().enumerate() {
            if !visible[k] {
                continue;
            }
            for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                let key = if e.0 < e.1 { e } else { (e.1, e.0) };
                let dir = if e.0 < e.1 { 1 } else { -1 };
                *edge_count.entry(key).or_insert(0) += dir;
            }
        }
        let horizon: Vec<(usize, usize)> = edge_count
            .into_iter()
            .filter(|(_, c)| *c != 0)
            .map(|((u, v), c)| if c > 0 { (u, v) } else { (v, u) })
            .collect();
        if horizon.is_empty() {
            // Numerically the point sees nothing coherent; dropping it can
            // only shrink the hull by less than eps, and keeping the mesh
            // closed matters more.
            continue;
        }
        let mut k = 0;
        faces.retain(|_| {
            let keep = !visible[k];
            k += 1;
            keep
        });
        for (u, v) in horizon {
            if u != next && v != next {
                faces.push([u, v, next]);
            }
        }
    }

    // Compact to only the vertices the surface uses.
    let mut remap = vec![u32::MAX; points.len()];
    let mut vertices = Vec::new();
    let mut out_faces = Vec::with_capacity(faces.len());
    for f in &faces {
        let mut ids = [0u32; 3];
        for (k, &i) in f.iter().enumerate() {
            if remap[i] == u32::MAX {
                remap[i] = vertices.len() as u32;
                vertices.push(points[i]);
            }
            ids[k] = remap[i];
        }
        if ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2] {
            out_faces.push(ids);
        }
    }
    if vertices.len() < 4 || out_faces.len() < 4 {
        return None;
    }
    Some(Hull { vertices, faces: out_faces })
}

/// Volume, centre of mass, and inertia of a closed triangle mesh.
///
/// Inertia is **about the centre of mass**, for the given uniform density —
/// which is the convention `SpatialInertia { mass, com, inertia }` wants, and
/// getting it wrong (handing over the tensor about the origin) makes a
/// dropped object tumble about a point it does not have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassProperties {
    /// Enclosed volume, m³. Negative means the surface is wound inward.
    pub volume: f64,
    /// Centre of mass in the mesh's own frame.
    pub com: Vec3,
    /// Inertia tensor about `com`, kg·m², for the density passed in.
    pub inertia: Mat3,
    /// `density × volume`.
    pub mass: f64,
}

/// Integrate a closed surface. Exact for any closed mesh — each triangle
/// makes a tetrahedron with the origin and the signed volumes cancel over
/// the parts of space the surface encloses twice.
pub fn mass_properties(vertices: &[Vec3], faces: &[[u32; 3]], density: f64) -> MassProperties {
    let mut volume = 0.0;
    let mut com_num = Vec3::zeros();
    // ∫x², ∫y², ∫z², ∫xy, ∫yz, ∫zx over the solid, about the origin.
    let mut m = [0.0f64; 6];

    for f in faces {
        let (a, b, c) = (
            vertices[f[0] as usize],
            vertices[f[1] as usize],
            vertices[f[2] as usize],
        );
        let det = a.dot(b.cross(c));
        volume += det / 6.0;
        com_num = com_num + (a + b + c) * (det / 24.0);

        // Second moments of the tetrahedron (0, a, b, c), the standard
        // closed forms: ∫x² = det/60 · Σx² + Σ_{i<j} x_i x_j, and
        // ∫xy = det/120 · (2Σx_iy_i + Σ_{i≠j} x_i y_j).
        let sq = |x: f64, y: f64, z: f64| x * x + y * y + z * z + x * y + x * z + y * z;
        m[0] += det / 60.0 * sq(a.x, b.x, c.x);
        m[1] += det / 60.0 * sq(a.y, b.y, c.y);
        m[2] += det / 60.0 * sq(a.z, b.z, c.z);
        let cross = |x: (f64, f64, f64), y: (f64, f64, f64)| {
            2.0 * (x.0 * y.0 + x.1 * y.1 + x.2 * y.2)
                + x.0 * y.1
                + x.1 * y.0
                + x.0 * y.2
                + x.2 * y.0
                + x.1 * y.2
                + x.2 * y.1
        };
        m[3] += det / 120.0 * cross((a.x, b.x, c.x), (a.y, b.y, c.y));
        m[4] += det / 120.0 * cross((a.y, b.y, c.y), (a.z, b.z, c.z));
        m[5] += det / 120.0 * cross((a.z, b.z, c.z), (a.x, b.x, c.x));
    }

    if volume.abs() < 1e-15 {
        return MassProperties {
            volume: 0.0,
            com: Vec3::zeros(),
            inertia: Mat3::zero(),
            mass: 0.0,
        };
    }
    let com = com_num / volume;
    let mass = density * volume;

    // Inertia about the origin, then the parallel-axis shift to the com.
    let ix = density * (m[1] + m[2]);
    let iy = density * (m[2] + m[0]);
    let iz = density * (m[0] + m[1]);
    let ixy = -density * m[3];
    let iyz = -density * m[4];
    let izx = -density * m[5];
    let (cx, cy, cz) = (com.x, com.y, com.z);
    let inertia = Mat3::new(
        ix - mass * (cy * cy + cz * cz),
        ixy + mass * cx * cy,
        izx + mass * cz * cx,
        ixy + mass * cx * cy,
        iy - mass * (cz * cz + cx * cx),
        iyz + mass * cy * cz,
        izx + mass * cz * cx,
        iyz + mass * cy * cz,
        iz - mass * (cx * cx + cy * cy),
    );
    MassProperties { volume, com, inertia, mass }
}

/// The mass properties of a compound: several pieces, one rigid body.
///
/// Sums volumes and shifts each piece's tensor to the shared centre of mass.
/// Overlap between pieces is counted twice — see [`decompose`] on why that is
/// the direction to err in, and why the bake reports `hull_error` rather than
/// hiding it.
pub fn compound_mass_properties(pieces: &[Hull], density: f64) -> MassProperties {
    let parts: Vec<MassProperties> = pieces
        .iter()
        .map(|h| mass_properties(&h.vertices, &h.faces, density))
        .collect();
    let mass: f64 = parts.iter().map(|p| p.mass).sum();
    let volume: f64 = parts.iter().map(|p| p.volume).sum();
    if mass.abs() < 1e-15 {
        return MassProperties {
            volume,
            com: Vec3::zeros(),
            inertia: Mat3::zero(),
            mass,
        };
    }
    let com = parts
        .iter()
        .fold(Vec3::zeros(), |acc, p| acc + p.com * p.mass)
        / mass;
    let mut inertia = Mat3::zero();
    for p in &parts {
        let d = p.com - com;
        let (dx, dy, dz) = (d.x, d.y, d.z);
        let shift = Mat3::new(
            p.mass * (dy * dy + dz * dz),
            -p.mass * dx * dy,
            -p.mass * dz * dx,
            -p.mass * dx * dy,
            p.mass * (dz * dz + dx * dx),
            -p.mass * dy * dz,
            -p.mass * dz * dx,
            -p.mass * dy * dz,
            p.mass * (dx * dx + dy * dy),
        );
        inertia = inertia + p.inertia + shift;
    }
    MassProperties { volume, com, inertia, mass }
}

/// How wrong one hull is about a shape: the fraction of its volume that is
/// not in the shape at all.
///
/// `0` for anything already convex, `0.5` for an L, `0.66` for a bucket. The
/// number the bake prints and the manifest records, so "the physics of this
/// object is a lump" is a fact somebody can read instead of discover.
pub fn hull_error(hull_volume: f64, true_volume: f64) -> f64 {
    if hull_volume.abs() < 1e-15 {
        return 0.0;
    }
    ((hull_volume - true_volume) / hull_volume).clamp(0.0, 1.0)
}

/// Split a closed mesh into at most `max_pieces` convex hulls, stopping early
/// once the worst piece is within `tolerance` of convex.
///
/// Greedy: hull the whole thing, and while the error is over budget, take the
/// piece contributing the most excess volume and cut it with the
/// axis-aligned plane that removes the most. Triangles are assigned to a side
/// by their centroid and carry all three vertices with them, so the two
/// children's hulls together cover every triangle of the parent — pieces can
/// overlap at the seam, and cannot leave a gap.
///
/// Deterministic: the candidate planes are a fixed lattice, ties break by
/// axis order. Two bakes of the same mesh produce the same pieces.
pub fn decompose(
    vertices: &[Vec3],
    faces: &[[u32; 3]],
    max_pieces: usize,
    tolerance: f64,
) -> Vec<Hull> {
    let whole = mass_properties(vertices, faces, 1.0).volume.abs();
    let all: Vec<usize> = (0..faces.len()).collect();
    let mut parts: Vec<(Vec<usize>, Hull, f64)> = match hull_of(vertices, faces, &all) {
        Some((h, v)) => vec![(all, h, v)],
        None => return Vec::new(),
    };

    while parts.len() < max_pieces.max(1) {
        let total: f64 = parts.iter().map(|p| p.2).sum();
        if hull_error(total, whole) <= tolerance {
            break;
        }
        // The piece wasting the most volume — measured against its own
        // triangles' share of the solid, not against the whole, so a big
        // convex piece is never split just for being big.
        let worst = parts
            .iter()
            .enumerate()
            .map(|(i, (tris, _, hv))| {
                let own = subset_volume(vertices, faces, tris).abs();
                (i, hv - own)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i);
        let Some(worst) = worst else { break };
        let Some((left, right)) = best_split(vertices, faces, &parts[worst].0) else {
            break;
        };
        parts.swap_remove(worst);
        parts.push(left);
        parts.push(right);
    }

    parts.into_iter().map(|(_, h, _)| h).collect()
}

/// The hull over a subset of triangles, and its volume.
fn hull_of(vertices: &[Vec3], faces: &[[u32; 3]], tris: &[usize]) -> Option<(Hull, f64)> {
    let mut pts = Vec::with_capacity(tris.len() * 3);
    for &t in tris {
        for i in faces[t] {
            pts.push(vertices[i as usize]);
        }
    }
    let hull = convex_hull(&pts)?;
    let v = mass_properties(&hull.vertices, &hull.faces, 1.0).volume.abs();
    Some((hull, v))
}

/// Signed volume of the tetrahedra over a triangle subset — the subset's own
/// share of the solid, used to score how much a hull over-claims.
fn subset_volume(vertices: &[Vec3], faces: &[[u32; 3]], tris: &[usize]) -> f64 {
    let mut v = 0.0;
    for &t in tris {
        let f = faces[t];
        let (a, b, c) = (
            vertices[f[0] as usize],
            vertices[f[1] as usize],
            vertices[f[2] as usize],
        );
        v += a.dot(b.cross(c)) / 6.0;
    }
    v
}

/// Try a lattice of axis-aligned planes and keep the split whose two hulls
/// have the least total volume. Returns `None` when nothing separates.
fn best_split(
    vertices: &[Vec3],
    faces: &[[u32; 3]],
    tris: &[usize],
) -> Option<((Vec<usize>, Hull, f64), (Vec<usize>, Hull, f64))> {
    if tris.len() < 8 {
        return None;
    }
    let centroid = |t: usize| -> Vec3 {
        let f = faces[t];
        (vertices[f[0] as usize] + vertices[f[1] as usize] + vertices[f[2] as usize]) / 3.0
    };
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for &t in tris {
        let c = centroid(t);
        lo = lo.component_min(c);
        hi = hi.component_max(c);
    }

    // Seven interior planes per axis. Finer sampling costs hulls (each
    // candidate builds two) and buys very little: the useful cut on a real
    // object is a limb boundary, which is wide.
    const CUTS: usize = 7;
    let mut best: Option<(f64, (Vec<usize>, Hull, f64), (Vec<usize>, Hull, f64))> = None;
    for axis in 0..3 {
        let (a, b) = (lo.as_array()[axis], hi.as_array()[axis]);
        if b - a < 1e-9 {
            continue;
        }
        for k in 1..=CUTS {
            let at = a + (b - a) * k as f64 / (CUTS + 1) as f64;
            let (mut l, mut r) = (Vec::new(), Vec::new());
            for &t in tris {
                if centroid(t).as_array()[axis] <= at {
                    l.push(t);
                } else {
                    r.push(t);
                }
            }
            if l.len() < 4 || r.len() < 4 {
                continue;
            }
            let (Some((lh, lv)), Some((rh, rv))) =
                (hull_of(vertices, faces, &l), hull_of(vertices, faces, &r))
            else {
                continue;
            };
            let score = lv + rv;
            if best.as_ref().is_none_or(|(s, _, _)| score < *s) {
                best = Some((score, (l, lh, lv), (r, rh, rv)));
            }
        }
    }
    best.map(|(_, l, r)| (l, r))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed, outward-wound box.
    fn box_mesh(half: Vec3) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let (x, y, z) = (half.x, half.y, half.z);
        let v: Vec<Vec3> = vec![
            Vec3::new(-x, -y, -z),
            Vec3::new(x, -y, -z),
            Vec3::new(x, y, -z),
            Vec3::new(-x, y, -z),
            Vec3::new(-x, -y, z),
            Vec3::new(x, -y, z),
            Vec3::new(x, y, z),
            Vec3::new(-x, y, z),
        ];
        let f: Vec<[u32; 3]> = vec![
            [0, 2, 1],
            [0, 3, 2], // -z
            [4, 5, 6],
            [4, 6, 7], // +z
            [0, 1, 5],
            [0, 5, 4], // -y
            [2, 3, 7],
            [2, 7, 6], // +y
            [1, 2, 6],
            [1, 6, 5], // +x
            [0, 4, 7],
            [0, 7, 3], // -x
        ];
        (v, f)
    }

    /// Translate a mesh, so the origin is not accidentally the answer.
    fn shifted(v: &[Vec3], by: Vec3) -> Vec<Vec3> {
        v.iter().map(|p| *p + by).collect()
    }

    #[test]
    fn a_box_weighs_what_a_box_weighs() {
        let half = Vec3::new(0.10, 0.20, 0.05);
        let (v, f) = box_mesh(half);
        let v = shifted(&v, Vec3::new(0.3, -0.7, 1.1));
        let m = mass_properties(&v, &f, 800.0);

        let volume = 8.0 * half.x * half.y * half.z;
        assert!((m.volume - volume).abs() < 1e-12, "volume {}", m.volume);
        assert!((m.mass - 800.0 * volume).abs() < 1e-9);
        assert!((m.com - Vec3::new(0.3, -0.7, 1.1)).norm() < 1e-12, "com {:?}", m.com);

        // m/12 · (b² + c²) about each axis, with the full side lengths.
        let (lx, ly, lz) = (2.0 * half.x, 2.0 * half.y, 2.0 * half.z);
        let want = |a: f64, b: f64| m.mass / 12.0 * (a * a + b * b);
        assert!((m.inertia.get(0, 0) - want(ly, lz)).abs() < 1e-12);
        assert!((m.inertia.get(1, 1) - want(lz, lx)).abs() < 1e-12);
        assert!((m.inertia.get(2, 2) - want(lx, ly)).abs() < 1e-12);
        // A box about its own axes has no products of inertia — and this is
        // the term that would be wrong if the parallel-axis shift were.
        for (r, c) in [(0, 1), (0, 2), (1, 2)] {
            assert!(m.inertia.get(r, c).abs() < 1e-12, "product {r}{c}");
        }
    }

    #[test]
    fn the_hull_of_a_cloud_in_a_box_is_the_box() {
        // Corners plus interior noise: the hull must be the eight corners
        // and nothing else, at the right volume.
        let half = Vec3::new(0.1, 0.15, 0.2);
        let (corners, _) = box_mesh(half);
        let mut pts = corners.clone();
        let mut seed = 12345u64;
        for _ in 0..500 {
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            };
            pts.push(Vec3::new(
                next() * half.x * 0.9,
                next() * half.y * 0.9,
                next() * half.z * 0.9,
            ));
        }
        let hull = convex_hull(&pts).expect("a box is not degenerate");
        assert_eq!(hull.vertices.len(), 8, "interior points reached the hull");
        let m = mass_properties(&hull.vertices, &hull.faces, 1.0);
        let want = 8.0 * half.x * half.y * half.z;
        assert!((m.volume - want).abs() < 1e-9, "hull volume {}", m.volume);
        // Outward winding: the centre is inside.
        assert!(hull.plane_distance(Vec3::zeros()) < 0.0);
        assert!(hull.plane_distance(Vec3::new(1.0, 0.0, 0.0)) > 0.0);
    }

    #[test]
    fn a_plane_of_points_is_refused() {
        let flat: Vec<Vec3> = (0..50)
            .map(|i| Vec3::new(i as f64 * 0.01, (i % 7) as f64 * 0.01, 0.0))
            .collect();
        assert!(convex_hull(&flat).is_none());
    }

    /// Two boxes in an L. One hull bridges the notch; two do not.
    fn l_shape() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let (mut v, mut f) = box_mesh(Vec3::new(0.20, 0.05, 0.05));
        let (v2, f2) = box_mesh(Vec3::new(0.05, 0.05, 0.20));
        let off = v.len() as u32;
        v.extend(v2.iter().map(|p| *p + Vec3::new(0.15, 0.0, 0.25)));
        f.extend(f2.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
        (v, f)
    }

    #[test]
    fn one_hull_over_an_l_is_a_lump_and_two_are_not() {
        let (v, f) = l_shape();
        let true_volume = mass_properties(&v, &f, 1.0).volume.abs();

        let one = decompose(&v, &f, 1, 0.0);
        assert_eq!(one.len(), 1);
        let one_v = mass_properties(&one[0].vertices, &one[0].faces, 1.0).volume.abs();
        let lump = hull_error(one_v, true_volume);
        assert!(lump > 0.3, "a single hull over an L should be badly wrong: {lump}");

        let many = decompose(&v, &f, 4, 0.05);
        assert!(many.len() > 1, "the decomposition refused to split");
        let sum: f64 = many
            .iter()
            .map(|h| mass_properties(&h.vertices, &h.faces, 1.0).volume.abs())
            .sum();
        let err = hull_error(sum, true_volume);
        assert!(err < lump * 0.6, "splitting did not help: {err} vs {lump}");

        // The whole surface must still be covered: every source vertex is
        // inside some piece. A decomposition with a gap is a hole a foot
        // falls through.
        for p in &v {
            let inside = many.iter().any(|h| h.plane_distance(*p) < 1e-6);
            assert!(inside, "vertex {p:?} fell out of every piece");
        }
    }

    #[test]
    fn compound_and_whole_agree_on_a_split_box() {
        // Two halves of one box, as a compound, must weigh and spin like the
        // box — the check that the parallel-axis assembly is right.
        let (v, f) = box_mesh(Vec3::new(0.1, 0.1, 0.2));
        let whole = mass_properties(&v, &f, 500.0);
        let (lv, lf) = box_mesh(Vec3::new(0.1, 0.1, 0.1));
        let lower = Hull {
            vertices: shifted(&lv, Vec3::new(0.0, 0.0, -0.1)),
            faces: lf.clone(),
        };
        let upper = Hull {
            vertices: shifted(&lv, Vec3::new(0.0, 0.0, 0.1)),
            faces: lf,
        };
        let compound = compound_mass_properties(&[lower, upper], 500.0);
        assert!((compound.mass - whole.mass).abs() < 1e-9);
        assert!((compound.com - whole.com).norm() < 1e-12);
        for i in 0..3 {
            let (a, b) = (compound.inertia.get(i, i), whole.inertia.get(i, i));
            assert!((a - b).abs() < 1e-9, "axis {i}: {a} vs {b}");
        }
    }
}
