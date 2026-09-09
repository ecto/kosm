//! The welded collision mesh and its signed-distance query.
//!
//! Sign comes from the angle-weighted pseudonormal of the closest feature
//! (Bærentzen & Aanæs 2005), not from ray parity: scan meshes are never
//! watertight, and parity flips on every crack, while the pseudonormal only
//! asks that triangle winding be consistent — which TSDF/marching-cubes
//! output is. The one thing that breaks it is a soup whose duplicate vertices
//! are not bitwise identical across triangles; STL from any single exporter
//! is, and [`TriMesh::from_soup`] welds on exact f32 bit patterns for that
//! reason (a tolerance weld would quietly merge genuinely distinct geometry
//! on a coarse scan).

use phyz_math::Vec3;

/// A welded triangle mesh with the precomputed adjacency the signed-distance
/// query needs: face normals, edge pseudonormals, angle-weighted vertex
/// pseudonormals, and an AABB tree over the triangles.
pub struct TriMesh {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    face_normals: Vec<Vec3>,
    /// Angle-weighted pseudonormal per vertex.
    vertex_normals: Vec<Vec3>,
    /// Pseudonormal per directed edge key (min, max): sum of the two adjacent
    /// face normals (or one, on an open boundary).
    edge_normals: std::collections::HashMap<(u32, u32), Vec3>,
    bvh: Bvh,
}

impl TriMesh {
    /// Weld a triangle soup (exact f32 bit-pattern match) and build adjacency.
    ///
    /// Degenerate triangles (zero-area after welding) are dropped: they carry
    /// no surface and a zero face normal would poison the pseudonormal sums.
    pub fn from_soup(soup: &[crate::stl::SoupTri]) -> Self {
        use std::collections::HashMap;
        let mut index: HashMap<[u32; 3], u32> = HashMap::new();
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut triangles: Vec<[u32; 3]> = Vec::new();

        for tri in soup {
            let mut ids = [0u32; 3];
            for (k, v) in tri.iter().enumerate() {
                let key = [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
                ids[k] = *index.entry(key).or_insert_with(|| {
                    vertices.push(Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64));
                    (vertices.len() - 1) as u32
                });
            }
            if ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2] {
                triangles.push(ids);
            }
        }
        Self::new(vertices, triangles)
    }

    /// Build from already-welded vertices and triangles.
    pub fn new(vertices: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Self {
        let mut face_normals = Vec::with_capacity(triangles.len());
        let mut vertex_normals = vec![Vec3::zeros(); vertices.len()];
        let mut edge_normals: std::collections::HashMap<(u32, u32), Vec3> =
            std::collections::HashMap::new();
        let mut kept: Vec<[u32; 3]> = Vec::with_capacity(triangles.len());

        for t in &triangles {
            let [a, b, c] = [
                vertices[t[0] as usize],
                vertices[t[1] as usize],
                vertices[t[2] as usize],
            ];
            let cross = (b - a).cross(c - a);
            let Some(n) = cross.try_normalize() else {
                continue; // zero-area sliver
            };
            kept.push(*t);
            face_normals.push(n);

            // Angle-weighted vertex contributions.
            let corners = [(a, b, c, t[0]), (b, c, a, t[1]), (c, a, b, t[2])];
            for (p, q, r, vid) in corners {
                let u = (q - p).try_normalize().unwrap_or(Vec3::zeros());
                let w = (r - p).try_normalize().unwrap_or(Vec3::zeros());
                let angle = u.dot(w).clamp(-1.0, 1.0).acos();
                vertex_normals[vid as usize] = vertex_normals[vid as usize] + n * angle;
            }
            // Edge sums.
            for (i, j) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = (i.min(j), i.max(j));
                let e = edge_normals.entry(key).or_insert(Vec3::zeros());
                *e = *e + n;
            }
        }

        let bvh = Bvh::build(&vertices, &kept);
        TriMesh {
            vertices,
            triangles: kept,
            face_normals,
            vertex_normals,
            edge_normals,
            bvh,
        }
    }

    /// Axis-aligned bounds of the mesh.
    pub fn aabb(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f64::INFINITY);
        let mut hi = Vec3::splat(f64::NEG_INFINITY);
        for v in &self.vertices {
            lo = lo.component_min(*v);
            hi = hi.component_max(*v);
        }
        (lo, hi)
    }

    /// Signed distance from `p` to the surface: positive outside (the side
    /// the winding's normals face), negative inside.
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        let hit = self.closest(p);
        let delta = p - hit.point;
        let dist = delta.norm();
        if dist == 0.0 {
            return 0.0;
        }
        let sign = if delta.dot(hit.pseudonormal) >= 0.0 { 1.0 } else { -1.0 };
        sign * dist
    }

    /// Closest surface point to `p`, with the pseudonormal of the feature
    /// (face interior, edge, or vertex) it landed on.
    fn closest(&self, p: Vec3) -> ClosestHit {
        let mut best = ClosestHit {
            dist_sq: f64::INFINITY,
            point: Vec3::zeros(),
            pseudonormal: Vec3::z(),
        };
        self.bvh.nearest(p, &mut |tri_idx| {
            let t = self.triangles[tri_idx];
            let [a, b, c] = [
                self.vertices[t[0] as usize],
                self.vertices[t[1] as usize],
                self.vertices[t[2] as usize],
            ];
            let (cp, feature) = closest_point_triangle(p, a, b, c);
            let d2 = (p - cp).norm_squared();
            if d2 < best.dist_sq {
                best.dist_sq = d2;
                best.point = cp;
                best.pseudonormal = self.feature_normal(tri_idx, feature);
            }
            best.dist_sq
        });
        best
    }

    fn feature_normal(&self, tri_idx: usize, feature: Feature) -> Vec3 {
        let t = self.triangles[tri_idx];
        match feature {
            Feature::Face => self.face_normals[tri_idx],
            Feature::Vertex(k) => {
                let n = self.vertex_normals[t[k] as usize];
                n.try_normalize().unwrap_or(self.face_normals[tri_idx])
            }
            Feature::Edge(k) => {
                let (i, j) = (t[k], t[(k + 1) % 3]);
                let key = (i.min(j), i.max(j));
                self.edge_normals
                    .get(&key)
                    .and_then(|n| n.try_normalize())
                    .unwrap_or(self.face_normals[tri_idx])
            }
        }
    }
}

struct ClosestHit {
    dist_sq: f64,
    point: Vec3,
    pseudonormal: Vec3,
}

/// Which feature of the triangle the closest point landed on. Edge `k` is the
/// edge from corner `k` to corner `(k+1) % 3`.
#[derive(Clone, Copy)]
enum Feature {
    Face,
    Edge(usize),
    Vertex(usize),
}

/// Closest point on triangle `abc` to `p`, and the feature it lies on.
/// Ericson, *Real-Time Collision Detection*, §5.1.5, with the Voronoi-region
/// outcome kept because the sign query needs to know face vs edge vs vertex.
fn closest_point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Feature) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Feature::Edge(0));
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Feature::Edge(2));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, Feature::Edge(1));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, Feature::Face)
}

// ---------------------------------------------------------------------------
// AABB tree
// ---------------------------------------------------------------------------

/// Median-split AABB tree over triangle indices. Flat array, leaves hold up
/// to 8 triangles. Built once at mesh construction; queried millions of times
/// per bake, so `nearest` prunes on squared box distance against the caller's
/// running best.
struct Bvh {
    nodes: Vec<BvhNode>,
    /// Triangle indices, leaf ranges point into this.
    tris: Vec<u32>,
}

struct BvhNode {
    lo: Vec3,
    hi: Vec3,
    /// Left child index, or `usize::MAX` for a leaf.
    left: usize,
    /// Right child index, or start of the leaf's triangle range.
    right: usize,
    /// Leaf range length; 0 for internal nodes.
    count: usize,
}

const LEAF_SIZE: usize = 8;

impl Bvh {
    fn build(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Bvh {
        let centroids: Vec<Vec3> = triangles
            .iter()
            .map(|t| {
                (vertices[t[0] as usize] + vertices[t[1] as usize] + vertices[t[2] as usize])
                    * (1.0 / 3.0)
            })
            .collect();
        let mut tris: Vec<u32> = (0..triangles.len() as u32).collect();
        let mut nodes = Vec::new();
        if !triangles.is_empty() {
            let len = tris.len();
            Self::build_node(vertices, triangles, &centroids, &mut tris, 0, len, &mut nodes);
        }
        Bvh { nodes, tris }
    }

    fn build_node(
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
        centroids: &[Vec3],
        tris: &mut [u32],
        start: usize,
        end: usize,
        nodes: &mut Vec<BvhNode>,
    ) -> usize {
        let mut lo = Vec3::splat(f64::INFINITY);
        let mut hi = Vec3::splat(f64::NEG_INFINITY);
        for &ti in &tris[start..end] {
            for &vi in &triangles[ti as usize] {
                lo = lo.component_min(vertices[vi as usize]);
                hi = hi.component_max(vertices[vi as usize]);
            }
        }
        let idx = nodes.len();
        nodes.push(BvhNode { lo, hi, left: usize::MAX, right: start, count: end - start });

        if end - start > LEAF_SIZE {
            let extent = hi - lo;
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            let mid = start + (end - start) / 2;
            tris[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
                let ca = centroids[a as usize].as_array()[axis];
                let cb = centroids[b as usize].as_array()[axis];
                ca.total_cmp(&cb)
            });
            // A degenerate split (all centroids identical) still divides the
            // range in half, so recursion always terminates.
            let left = Self::build_node(vertices, triangles, centroids, tris, start, mid, nodes);
            let right = Self::build_node(vertices, triangles, centroids, tris, mid, end, nodes);
            nodes[idx].left = left;
            nodes[idx].right = right;
            nodes[idx].count = 0;
        }
        idx
    }

    /// Visit triangles in best-first order. `visit` is called with a triangle
    /// index and returns the current best squared distance; subtrees farther
    /// than that are pruned.
    fn nearest(&self, p: Vec3, visit: &mut dyn FnMut(usize) -> f64) {
        if self.nodes.is_empty() {
            return;
        }
        let mut best = f64::INFINITY;
        let mut stack = vec![0usize];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            if aabb_dist_sq(p, node.lo, node.hi) > best {
                continue;
            }
            if node.count > 0 {
                for &ti in &self.tris[node.right..node.right + node.count] {
                    best = best.min(visit(ti as usize));
                }
            } else {
                // Push the farther child first so the nearer one pops first.
                let (l, r) = (node.left, node.right);
                let dl = aabb_dist_sq(p, self.nodes[l].lo, self.nodes[l].hi);
                let dr = aabb_dist_sq(p, self.nodes[r].lo, self.nodes[r].hi);
                if dl <= dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
    }
}

fn aabb_dist_sq(p: Vec3, lo: Vec3, hi: Vec3) -> f64 {
    let dx = (lo.x - p.x).max(0.0).max(p.x - hi.x);
    let dy = (lo.y - p.y).max(0.0).max(p.y - hi.y);
    let dz = (lo.z - p.z).max(0.0).max(p.z - hi.z);
    dx * dx + dy * dy + dz * dz
}

/// Test-only helpers, `pub` so sibling modules' tests can build fixtures.
/// An axis-aligned box as a triangle soup, exactly — 12 triangles, outward
/// wound.
///
/// Here rather than in a bake tool because two of them now want it: a
/// mechanism's parts are primitives (`ipse_sim::objects::rigs`) and something
/// has to turn them into the STL a viewer draws. Exact, not sampled: a 27 mm
/// wheel through a 1 cm surface-nets pass is a lumpy potato.
pub fn box_soup(centre: Vec3, half: Vec3) -> Vec<[Vec3; 3]> {
    let v = |i: usize| {
        Vec3::new(
            centre.x + if i & 1 == 0 { -half.x } else { half.x },
            centre.y + if i & 2 == 0 { -half.y } else { half.y },
            centre.z + if i & 4 == 0 { -half.z } else { half.z },
        )
    };
    let quads = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    let mut out = Vec::with_capacity(12);
    for q in quads {
        out.push([v(q[0]), v(q[1]), v(q[2])]);
        out.push([v(q[0]), v(q[2]), v(q[3])]);
    }
    out
}

/// A UV sphere as a triangle soup, outward wound.
///
/// `rings` bands of latitude by `segments` of longitude. 12 x 16 is plenty
/// for a skateboard wheel on screen and is a few hundred triangles, which is
/// nothing next to the map it rolls on.
pub fn sphere_soup(centre: Vec3, radius: f64, rings: usize, segments: usize) -> Vec<[Vec3; 3]> {
    let rings = rings.max(2);
    let segments = segments.max(3);
    let at = |ring: usize, seg: usize| -> Vec3 {
        let phi = std::f64::consts::PI * ring as f64 / rings as f64;
        let theta = 2.0 * std::f64::consts::PI * (seg % segments) as f64 / segments as f64;
        centre
            + Vec3::new(
                radius * phi.sin() * theta.cos(),
                radius * phi.sin() * theta.sin(),
                radius * phi.cos(),
            )
    };
    let mut out = Vec::with_capacity(rings * segments * 2);
    for r in 0..rings {
        for s in 0..segments {
            let (a, b, c, d) = (at(r, s), at(r, s + 1), at(r + 1, s + 1), at(r + 1, s));
            // The caps degenerate to one triangle each; dropping the zero-area
            // half keeps the soup free of triangles with no normal.
            if r > 0 {
                out.push([a, b, c]);
            }
            if r + 1 < rings {
                out.push([a, c, d]);
            }
        }
    }
    out
}

#[cfg(test)]#[cfg(test)]
pub mod tests {
    use super::*;

    /// An axis-aligned box as a welded 12-triangle mesh, outward winding.
    pub fn box_mesh(lo: Vec3, hi: Vec3) -> TriMesh {
        let v = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
        let corners = [
            v(lo.x, lo.y, lo.z), // 0
            v(hi.x, lo.y, lo.z), // 1
            v(hi.x, hi.y, lo.z), // 2
            v(lo.x, hi.y, lo.z), // 3
            v(lo.x, lo.y, hi.z), // 4
            v(hi.x, lo.y, hi.z), // 5
            v(hi.x, hi.y, hi.z), // 6
            v(lo.x, hi.y, hi.z), // 7
        ];
        let quads = [
            [4, 5, 6, 7], // +z
            [1, 0, 3, 2], // -z
            [5, 1, 2, 6], // +x
            [0, 4, 7, 3], // -x
            [6, 2, 3, 7], // +y
            [0, 1, 5, 4], // -y
        ];
        let mut triangles = Vec::new();
        for q in quads {
            triangles.push([q[0], q[1], q[2]]);
            triangles.push([q[0], q[2], q[3]]);
        }
        TriMesh::new(corners.to_vec(), triangles)
    }

    #[test]
    fn signed_distance_box() {
        let m = box_mesh(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        // Above the top face.
        assert!((m.signed_distance(Vec3::new(0.0, 0.0, 1.5)) - 0.5).abs() < 1e-12);
        // Inside.
        assert!((m.signed_distance(Vec3::new(0.0, 0.0, 0.5)) - (-0.5)).abs() < 1e-12);
        // Outside a vertex: distance to the corner, positive.
        let d = m.signed_distance(Vec3::new(2.0, 2.0, 2.0));
        assert!((d - (3.0f64).sqrt()).abs() < 1e-12);
        // Outside an edge.
        let d = m.signed_distance(Vec3::new(2.0, 0.0, 2.0));
        assert!((d - (2.0f64).sqrt()).abs() < 1e-12);
        // On the surface.
        assert!(m.signed_distance(Vec3::new(0.3, -0.2, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn weld_dedupes() {
        // Two triangles sharing an edge, as soup.
        let soup = vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ];
        let m = TriMesh::from_soup(&soup);
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.triangles.len(), 2);
    }
}
