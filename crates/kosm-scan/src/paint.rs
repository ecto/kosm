//! Per-vertex colour for a baked map mesh.
//!
//! The mesh is geometry that survived a vote; colour is the appearance the
//! same capture measured. They travel in separate files because binary STL
//! has nowhere to put a vertex colour — its only spare field is a per-*face*
//! u16 that VisCAM overloads, and `mesh.stl` is triangle soup with no vertex
//! identity at all. So colours live beside it in `mesh_rgb.bin`, one RGB
//! triple per soup vertex, in exactly the order the STL's triangles are
//! written. Two files, one order, no index to get out of step.
//!
//! # Where the colour comes from
//!
//! `room_fuse.py` already writes `cloud.ply` next to the geometry: a 500 k
//! point subsample of the fused cloud, **metric and in the floor frame** —
//! the same frame `roombake` meshes into. So painting is a spatial lookup,
//! not a re-projection: no camera model, no visibility test, no second
//! inference. Each mesh vertex takes the mean colour of the cloud points
//! within a radius.
//!
//! One honest caveat: `roombake` applies a depth-scale calibration (~1.6 % on
//! the garage) and re-zeroes the floor before meshing, and the cloud predates
//! both. The two therefore disagree by a couple of centimetres at room scale,
//! which is why the search radius is centimetres rather than millimetres and
//! why the mean over neighbours is taken rather than the nearest point.
//!
//! Vertices with no cloud point in range stay **white**, which is the tint
//! that changes nothing: a renderer multiplying vertex colour by an instance
//! albedo draws them exactly as it drew every unpainted mesh before.

use std::collections::HashMap;
use std::path::Path;

use phyz_math::Vec3;
use rayon::prelude::*;

use crate::MapError;

/// The tint that leaves a renderer's instance albedo untouched.
pub const UNPAINTED: [u8; 3] = [255, 255, 255];

const MAGIC: &[u8; 4] = b"IRGB";
const VERSION: u32 = 1;

/// A coloured point cloud: positions and their colours, same length.
#[derive(Debug, Clone, Default)]
pub struct ColourCloud {
    pub points: Vec<Vec3>,
    pub colours: Vec<[u8; 3]>,
}

/// Read an ASCII PLY of `x y z red green blue` vertices.
///
/// Deliberately narrow: this reads the file `room_fuse.py` writes, and says
/// so when handed anything else, rather than growing into a PLY library. The
/// binary splat PLYs are handled in [`crate::stl`] and share nothing with
/// this beyond the four letters at the top.
pub fn read_colour_cloud(path: &Path) -> Result<ColourCloud, MapError> {
    let text = std::fs::read_to_string(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    let bad = |msg: String| MapError::Format(path.to_path_buf(), msg);

    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err(bad("not a ply".into()));
    }
    let mut count: Option<usize> = None;
    let mut props: Vec<String> = Vec::new();
    let mut ascii = false;
    for line in lines.by_ref() {
        let line = line.trim();
        if line.starts_with("format ascii") {
            ascii = true;
        } else if let Some(rest) = line.strip_prefix("element vertex ") {
            count = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("property ") {
            props.push(rest.split_whitespace().last().unwrap_or("").to_string());
        } else if line == "end_header" {
            break;
        }
    }
    if !ascii {
        return Err(bad("only the ascii cloud.ply is read here".into()));
    }
    let count = count.ok_or_else(|| bad("no element vertex".into()))?;
    let want = ["x", "y", "z", "red", "green", "blue"];
    for w in want {
        if !props.iter().any(|p| p == w) {
            return Err(bad(format!("missing property {w}; got {props:?}")));
        }
    }
    let col = |name: &str| props.iter().position(|p| p == name).unwrap();
    let (ix, iy, iz) = (col("x"), col("y"), col("z"));
    let (ir, ig, ib) = (col("red"), col("green"), col("blue"));

    let mut cloud = ColourCloud {
        points: Vec::with_capacity(count),
        colours: Vec::with_capacity(count),
    };
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < props.len() {
            continue;
        }
        let num = |i: usize| f[i].parse::<f64>().ok();
        let (Some(x), Some(y), Some(z)) = (num(ix), num(iy), num(iz)) else {
            continue;
        };
        let byte = |i: usize| f[i].parse::<f64>().ok().map(|v| v.clamp(0.0, 255.0) as u8);
        let (Some(r), Some(g), Some(b)) = (byte(ir), byte(ig), byte(ib)) else {
            continue;
        };
        cloud.points.push(Vec3::new(x, y, z));
        cloud.colours.push([r, g, b]);
    }
    if cloud.points.is_empty() {
        return Err(bad("no vertices parsed".into()));
    }
    Ok(cloud)
}

/// Colour every soup vertex from the cloud, returning one RGB per vertex.
///
/// `tris` is the triangle soup exactly as it is written to `mesh.stl`, so the
/// result is index-aligned with the file: vertex `3t + i` of triangle `t`.
/// Positions repeat across shared corners, so the average is computed once
/// per *distinct* position (bit-identical f32, the same rule the STL weld
/// uses) and fanned back out — surface nets emits about three triangles per
/// vertex, so that is a 3x saving for free.
pub fn paint_soup(tris: &[[Vec3; 3]], cloud: &ColourCloud, radius: f64) -> Vec<[u8; 3]> {
    let mut out = vec![UNPAINTED; tris.len() * 3];
    if cloud.points.is_empty() || radius <= 0.0 {
        return out;
    }

    // Hash the cloud at the search radius, so a lookup visits 27 buckets.
    let key = |p: Vec3| {
        [
            (p.x / radius).floor() as i64,
            (p.y / radius).floor() as i64,
            (p.z / radius).floor() as i64,
        ]
    };
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (i, &p) in cloud.points.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i as u32);
    }

    // Distinct positions, keyed by the f32 bits that land in the STL — the
    // same identity `TriMesh::from_soup` welds on, so painting agrees with
    // any consumer that welds.
    let bits = |v: Vec3| {
        [
            (v.x as f32).to_bits(),
            (v.y as f32).to_bits(),
            (v.z as f32).to_bits(),
        ]
    };
    let mut unique: HashMap<[u32; 3], usize> = HashMap::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut of_vertex: Vec<usize> = Vec::with_capacity(tris.len() * 3);
    for t in tris {
        for v in t {
            let k = bits(*v);
            let next = positions.len();
            let id = *unique.entry(k).or_insert_with(|| {
                positions.push(*v);
                next
            });
            of_vertex.push(id);
        }
    }

    let r2 = radius * radius;
    let painted: Vec<[u8; 3]> = positions
        .par_iter()
        .map(|&p| {
            let c = key(p);
            let mut sum = [0.0f64; 3];
            let mut n = 0.0f64;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(bucket) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                            continue;
                        };
                        for &i in bucket {
                            let q = cloud.points[i as usize];
                            if (q - p).norm_squared() > r2 {
                                continue;
                            }
                            let rgb = cloud.colours[i as usize];
                            for (s, v) in sum.iter_mut().zip(rgb) {
                                *s += v as f64;
                            }
                            n += 1.0;
                        }
                    }
                }
            }
            if n == 0.0 {
                return UNPAINTED;
            }
            [
                (sum[0] / n).round() as u8,
                (sum[1] / n).round() as u8,
                (sum[2] / n).round() as u8,
            ]
        })
        .collect();

    for (slot, &id) in out.iter_mut().zip(&of_vertex) {
        *slot = painted[id];
    }
    out
}

/// Fraction of vertices that found colour, for the bake to report.
pub fn painted_fraction(colours: &[[u8; 3]]) -> f64 {
    if colours.is_empty() {
        return 0.0;
    }
    let n = colours.iter().filter(|c| **c != UNPAINTED).count();
    n as f64 / colours.len() as f64
}

/// Write `mesh_rgb.bin`: magic, version, count, then RGB triples.
pub fn write_vertex_colours(path: &Path, colours: &[[u8; 3]]) -> Result<(), MapError> {
    let mut bytes = Vec::with_capacity(12 + colours.len() * 3);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&(colours.len() as u32).to_le_bytes());
    for c in colours {
        bytes.extend_from_slice(c);
    }
    std::fs::write(path, bytes).map_err(|e| MapError::Io(path.to_path_buf(), e))
}

/// Read `mesh_rgb.bin`, checking it against the mesh it belongs to.
///
/// `expect` is the soup vertex count of `mesh.stl`. A colour file that does
/// not match is refused rather than truncated: silently painting two thirds
/// of a room and leaving the rest white is the kind of failure that gets
/// blamed on the capture.
pub fn read_vertex_colours(path: &Path, expect: usize) -> Result<Vec<[u8; 3]>, MapError> {
    let bytes = std::fs::read(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    let bad = |msg: String| MapError::Format(path.to_path_buf(), msg);
    if bytes.len() < 12 || &bytes[..4] != MAGIC {
        return Err(bad("not an IRGB vertex-colour file".into()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(bad(format!("version {version}, want {VERSION}")));
    }
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if bytes.len() < 12 + n * 3 {
        return Err(bad(format!(
            "{n} colours need {} B, file has {}",
            12 + n * 3,
            bytes.len()
        )));
    }
    if n != expect {
        return Err(bad(format!(
            "{n} colours for a mesh with {expect} vertices"
        )));
    }
    Ok(bytes[12..12 + n * 3]
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(z: f64) -> [Vec3; 3] {
        [
            Vec3::new(0.0, 0.0, z),
            Vec3::new(1.0, 0.0, z),
            Vec3::new(0.0, 1.0, z),
        ]
    }

    #[test]
    fn a_vertex_takes_the_mean_colour_of_the_points_around_it() {
        let cloud = ColourCloud {
            points: vec![Vec3::new(0.01, 0.0, 0.0), Vec3::new(0.0, 0.01, 0.0)],
            colours: vec![[200, 0, 0], [100, 0, 0]],
        };
        let c = paint_soup(&[tri(0.0)], &cloud, 0.05);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], [150, 0, 0], "mean of the two neighbours");
    }

    #[test]
    fn a_vertex_with_nothing_near_it_stays_the_neutral_tint() {
        let cloud = ColourCloud {
            points: vec![Vec3::new(5.0, 5.0, 5.0)],
            colours: vec![[10, 20, 30]],
        };
        let c = paint_soup(&[tri(0.0)], &cloud, 0.05);
        assert!(
            c.iter().all(|v| *v == UNPAINTED),
            "far cloud must not bleed onto the mesh: {c:?}"
        );
        assert_eq!(painted_fraction(&c), 0.0);
    }

    #[test]
    fn shared_corners_get_one_answer_and_the_soup_order_is_kept() {
        // Two triangles sharing a corner at the origin, one red cloud point
        // there and a blue one out at the far corner of the second.
        let a = tri(0.0);
        let b = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
        ];
        let cloud = ColourCloud {
            points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)],
            colours: vec![[255, 0, 0], [0, 0, 255]],
        };
        let c = paint_soup(&[a, b], &cloud, 0.05);
        assert_eq!(c.len(), 6);
        assert_eq!(c[0], [255, 0, 0]);
        assert_eq!(c[3], c[0], "the same corner in both triangles");
        assert_eq!(c[5], [0, 0, 255], "third vertex of the second triangle");
    }

    #[test]
    fn the_colour_file_round_trips_and_refuses_a_mismatched_mesh() {
        let dir = std::env::temp_dir().join(format!("ipse-map-paint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mesh_rgb.bin");
        let colours = vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        write_vertex_colours(&path, &colours).unwrap();
        assert_eq!(read_vertex_colours(&path, 3).unwrap(), colours);
        let err = read_vertex_colours(&path, 4).unwrap_err();
        assert!(
            format!("{err}").contains("3 colours for a mesh with 4"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_ascii_cloud_is_read_and_anything_else_is_refused() {
        let dir = std::env::temp_dir().join(format!("ipse-map-cloud-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("cloud.ply");
        std::fs::write(
            &good,
            "ply\nformat ascii 1.0\nelement vertex 2\n\
             property float x\nproperty float y\nproperty float z\n\
             property uchar red\nproperty uchar green\nproperty uchar blue\n\
             end_header\n0 0 0 10 20 30\n1 2 3 40 50 60\n",
        )
        .unwrap();
        let c = read_colour_cloud(&good).unwrap();
        assert_eq!(c.points.len(), 2);
        assert_eq!(c.colours[1], [40, 50, 60]);
        assert!((c.points[1].y - 2.0).abs() < 1e-9);

        let bad = dir.join("binary.ply");
        std::fs::write(
            &bad,
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n\
             property float x\nend_header\n",
        )
        .unwrap();
        assert!(read_colour_cloud(&bad).is_err(), "binary must be refused");
        std::fs::remove_dir_all(&dir).ok();
    }
}
