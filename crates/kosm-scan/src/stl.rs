//! Binary STL, read and write.
//!
//! STL because it is the one mesh format the rest of this workspace already
//! speaks (`phyz-camera` refuses everything else) and every phone-scan
//! exporter can produce it. STL is a triangle soup — no shared vertices — so
//! [`crate::mesh::TriMesh::from_soup`] welds it before any adjacency is built.
//!
//! ASCII STL is rejected, loudly: the ASCII header sniff exists because a
//! Polycam "STL" export once turned out to be ASCII and the binary reader
//! happily parsed its length field as a garbage triangle count.

use std::io::{Read, Write};
use std::path::Path;

use phyz_math::Vec3;

use crate::MapError;

/// One soup triangle: three vertices, in file order.
pub type SoupTri = [[f32; 3]; 3];

/// Read a binary STL into a triangle soup.
pub fn read_binary_stl(path: &Path) -> Result<Vec<SoupTri>, MapError> {
    let bytes = std::fs::read(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    parse_binary_stl(&bytes).map_err(|msg| MapError::Format(path.to_path_buf(), msg))
}

/// Parse binary STL bytes. Split from the file read so tests can feed bytes.
pub fn parse_binary_stl(bytes: &[u8]) -> Result<Vec<SoupTri>, String> {
    if bytes.len() < 84 {
        return Err(format!("{} bytes is too short for binary STL", bytes.len()));
    }
    // "solid" opening + mostly-printable body means an ASCII STL.
    if bytes.starts_with(b"solid")
        && bytes[..bytes.len().min(512)]
            .iter()
            .all(|&b| b == b'\n' || b == b'\r' || b == b'\t' || (0x20..0x7f).contains(&b))
    {
        return Err("ASCII STL — export binary STL instead".into());
    }
    let count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    let expected = 84 + count * 50;
    if bytes.len() < expected {
        return Err(format!(
            "header claims {count} triangles ({expected} bytes) but file has {}",
            bytes.len()
        ));
    }

    let mut tris = Vec::with_capacity(count);
    for t in 0..count {
        let base = 84 + t * 50 + 12; // skip the stored normal — recomputed from winding
        let mut tri: SoupTri = [[0.0; 3]; 3];
        for (v, vert) in tri.iter_mut().enumerate() {
            for (c, coord) in vert.iter_mut().enumerate() {
                let off = base + (v * 3 + c) * 4;
                *coord = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            }
        }
        if tri.iter().flatten().all(|c| c.is_finite()) {
            tris.push(tri);
        }
    }
    Ok(tris)
}

/// Write a binary STL from world-space triangles.
///
/// `mapbake` uses this to emit the *aligned* mesh into the map directory, so
/// the mesh on disk and the SDF baked from it are in the same frame — the map
/// frame — rather than whatever frame the phone exported.
pub fn write_binary_stl(path: &Path, tris: &[[Vec3; 3]]) -> Result<(), MapError> {
    let io = |e| MapError::Io(path.to_path_buf(), e);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
    let mut header = [0u8; 80];
    header[..12].copy_from_slice(b"ipse-map stl");
    f.write_all(&header).map_err(io)?;
    f.write_all(&(tris.len() as u32).to_le_bytes()).map_err(io)?;
    for t in tris {
        let n = (t[1] - t[0]).cross(t[2] - t[0]).try_normalize().unwrap_or(Vec3::zeros());
        for v in [n, t[0], t[1], t[2]] {
            for c in [v.x, v.y, v.z] {
                f.write_all(&(c as f32).to_le_bytes()).map_err(io)?;
            }
        }
        f.write_all(&0u16.to_le_bytes()).map_err(io)?;
    }
    f.flush().map_err(io)
}

/// A quick validity sniff of a splat `.ply` — is this actually a Gaussian
/// splat, not just any point cloud?
///
/// Checks the header for the `f_dc_0` property every 3DGS exporter writes.
/// Nothing here parses the splat; the point is that a wrong export fails at
/// map-load time with a message, not at render time with silence.
pub fn sniff_splat_ply(path: &Path) -> Result<(), MapError> {
    let mut head = vec![0u8; 4096];
    let mut f = std::fs::File::open(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    let n = f.read(&mut head).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    head.truncate(n);
    let text = String::from_utf8_lossy(&head);
    if !text.starts_with("ply") {
        return Err(MapError::Format(path.to_path_buf(), "not a .ply file".into()));
    }
    if !text.contains("f_dc_0") {
        return Err(MapError::Format(
            path.to_path_buf(),
            "a .ply but not a Gaussian splat (no f_dc_0 property in header)".into(),
        ));
    }
    Ok(())
}

/// Which way is up in a splat's own frame.
///
/// Not a preference — a property of where the file came from, and there is no
/// way to read it out of the file itself. Phone scanners (Polycam, Scaniverse,
/// Luma) export **y-up**. Anything trained from photographs through COLMAP —
/// which is every splat from the 3DGS papers and most from the community — is
/// in COLMAP's camera convention, **y-down**. Guessing wrong tips the room on
/// its side, so this is stated at import and recorded in the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpAxis {
    /// `(x, y, z) -> (x, -z, y)`.
    YUp,
    /// `(x, y, z) -> (x, z, -y)`. COLMAP, and therefore most trained splats.
    YDown,
    /// Already z-up: no rotation.
    ZUp,
}

impl UpAxis {
    pub fn manifest_name(self) -> &'static str {
        match self {
            UpAxis::YUp => "y-up-to-z-up",
            UpAxis::YDown => "y-down-to-z-up",
            UpAxis::ZUp => "none",
        }
    }

    pub fn parse(s: &str) -> Option<UpAxis> {
        match s {
            "y" | "y-up" => Some(UpAxis::YUp),
            "-y" | "y-down" => Some(UpAxis::YDown),
            "z" | "none" => Some(UpAxis::ZUp),
            _ => None,
        }
    }

    /// Apply the rotation to a point in the capture frame.
    pub fn apply(self, p: [f32; 3]) -> Vec3 {
        match self {
            UpAxis::YUp => Vec3::new(p[0] as f64, -(p[2] as f64), p[1] as f64),
            UpAxis::YDown => Vec3::new(p[0] as f64, p[2] as f64, -(p[1] as f64)),
            UpAxis::ZUp => Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64),
        }
    }
}

/// What a splat `.ply`'s header says about where its rows are.
///
/// One parse, three consumers ([`read_splat_positions`], [`crop_splat_ply`],
/// and anything that comes later), because a second reader of this header
/// would be a second opinion about a byte offset.
struct SplatHeader {
    /// Bytes up to and including `end_header\n`.
    data_start: usize,
    /// Vertex rows.
    count: usize,
    /// Bytes per row.
    stride: usize,
    /// Byte offsets of x, y, z within a row.
    xyz: (usize, usize, usize),
    /// How many `element` lines the header declares.
    elements: usize,
}

fn splat_header(bytes: &[u8], path: &Path) -> Result<SplatHeader, MapError> {
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(65536)]).into_owned();
    let data_start = head
        .find("end_header\n")
        .ok_or_else(|| MapError::Format(path.to_path_buf(), "no end_header".into()))?
        + "end_header\n".len();

    let (mut count, mut stride) = (0usize, 0usize);
    let (mut ox, mut oy, mut oz) = (None, None, None);
    let mut in_vertex = false;
    let mut elements = 0usize;
    for line in head[..data_start].lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["format", kind, ..] if !kind.contains("little") => {
                return Err(MapError::Format(path.to_path_buf(), format!("{kind} splat")));
            }
            ["element", name, n] => {
                elements += 1;
                in_vertex = *name == "vertex";
                if in_vertex {
                    count = n.parse().unwrap_or(0);
                }
            }
            ["property", ty, name] if in_vertex => {
                let w = match *ty {
                    "float" | "float32" | "int" | "int32" | "uint" | "uint32" => 4,
                    "double" | "float64" => 8,
                    "short" | "ushort" | "int16" | "uint16" => 2,
                    "char" | "uchar" | "int8" | "uint8" => 1,
                    other => {
                        return Err(MapError::Format(
                            path.to_path_buf(),
                            format!("property type {other}"),
                        ));
                    }
                };
                match *name {
                    "x" => ox = Some(stride),
                    "y" => oy = Some(stride),
                    "z" => oz = Some(stride),
                    _ => {}
                }
                stride += w;
            }
            _ => {}
        }
    }
    let xyz = match (ox, oy, oz) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return Err(MapError::Format(path.to_path_buf(), "no x/y/z".into())),
    };
    if count == 0 || stride == 0 || data_start + count * stride > bytes.len() {
        return Err(MapError::Format(path.to_path_buf(), "truncated".into()));
    }
    Ok(SplatHeader { data_start, count, stride, xyz, elements })
}

/// Copy a splat, keeping only the gaussians a predicate accepts.
///
/// Returns `(kept, total)`. The predicate sees each centre in the file's own
/// frame; a caller that thinks in the object frame passes a closure that
/// applies the alignment itself, which keeps this function ignorant of what
/// a frame is.
///
/// **Rows are copied byte for byte.** Nothing here understands a covariance
/// or a spherical harmonic, and that is the whole point: cropping is the one
/// edit to a splat that cannot be silently wrong. Rotating one — which is
/// what levelling a capture would need — is a renderer's job, and the
/// manifest's `[align]` is how that request travels instead.
///
/// Refuses a header with more than one element: the extra one would have to
/// be copied through with offsets nothing here tracks, and a splat with a
/// second element is not a file this house has ever produced.
pub fn crop_splat_ply(
    src: &Path,
    dst: &Path,
    keep: impl Fn([f32; 3]) -> bool,
) -> Result<(usize, usize), MapError> {
    let bytes = std::fs::read(src).map_err(|e| MapError::Io(src.to_path_buf(), e))?;
    let h = splat_header(&bytes, src)?;
    if h.elements > 1 {
        return Err(MapError::Format(
            src.to_path_buf(),
            format!("{} elements — this crops single-element splats only", h.elements),
        ));
    }
    let at = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let (ox, oy, oz) = h.xyz;
    let mut rows: Vec<u8> = Vec::with_capacity(bytes.len() / 2);
    let mut kept = 0usize;
    for i in 0..h.count {
        let b = h.data_start + i * h.stride;
        if keep([at(b + ox), at(b + oy), at(b + oz)]) {
            rows.extend_from_slice(&bytes[b..b + h.stride]);
            kept += 1;
        }
    }
    // The header, with the vertex count rewritten. Everything else — the
    // property list, the comments, whatever the exporter felt like saying —
    // is passed through untouched, because a re-emitted header would be this
    // file's opinion of a format it only reads.
    let head = String::from_utf8_lossy(&bytes[..h.data_start]).into_owned();
    let mut out = String::with_capacity(h.data_start);
    for line in head.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if let ["element", "vertex", _] = f.as_slice() {
            out.push_str(&format!("element vertex {kept}\n"));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    let mut bytes_out = out.into_bytes();
    bytes_out.extend_from_slice(&rows);
    std::fs::write(dst, bytes_out).map_err(|e| MapError::Io(dst.to_path_buf(), e))?;
    Ok((kept, h.count))
}

/// Every gaussian centre in a splat `.ply`, in the file's own frame.
///
/// Positions only. The renderer reads the whole file itself; this exists so
/// the *importer* can answer two questions a manifest has to record — which
/// way is up, and where the floor is — without a second full parser.
pub fn read_splat_positions(path: &Path) -> Result<Vec<[f32; 3]>, MapError> {
    let bytes = std::fs::read(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    let h = splat_header(&bytes, path)?;
    let at = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let (ox, oy, oz) = h.xyz;
    Ok((0..h.count)
        .map(|i| {
            let b = h.data_start + i * h.stride;
            [at(b + ox), at(b + oy), at(b + oz)]
        })
        .collect())
}

/// How much of the cloud lies in a thin slab just above the floor.
///
/// The statistic that tells one candidate up axis from another. A room's
/// floor is a *dense, flat* thing: point it the right way and a few percent
/// of every gaussian lands in the first 25 cm. Point it wrongly and the floor
/// projects onto a diagonal, smearing across the whole range — measured at
/// 4.0% against 1.5% on `drjohnson`, and 7.2% against 0.0% on `train`.
pub fn floor_slab_fraction(points: &[[f32; 3]], up: UpAxis) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let floor = floor_height(points, up);
    let n = points
        .iter()
        .filter(|p| {
            let z = up.apply(**p).z - floor;
            (-0.05..0.25).contains(&z)
        })
        .count();
    n as f64 / points.len() as f64
}

/// Work out which way is up by measuring, rather than assuming.
///
/// There is no convention to rely on. Trained splats inherit whatever frame
/// COLMAP happened to solve, and that is a property of how the cameras moved
/// — not of the tool. Measured across three scenes from the 3DGS papers,
/// `drjohnson` comes out y-up and `train` y-down, so *any* fixed default is
/// wrong about half the time. [`floor_slab_fraction`] separates them
/// cleanly; this picks the winner and the caller records it.
///
/// Returns the axis and its score, so a caller can say how sure it is. A
/// scene with no flat floor anywhere — an object on a turntable, a drone
/// orbit — scores low on all three, and then the answer is a guess that
/// should be overridden by hand.
pub fn detect_up(points: &[[f32; 3]]) -> (UpAxis, f64) {
    [UpAxis::YUp, UpAxis::YDown, UpAxis::ZUp]
        .into_iter()
        .map(|u| (u, floor_slab_fraction(points, u)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((UpAxis::YDown, 0.0))
}

/// Where the floor is, along the up axis, after rotation.
///
/// The low tail of the height distribution, not the minimum: a splat has
/// stray gaussians well below the floor (COLMAP noise, reflections under
/// furniture), and taking the minimum would hang the room in the air. The
/// first percentile is robust to that and still lands on the floor rather
/// than on the lowest table leg.
pub fn floor_height(points: &[[f32; 3]], up: UpAxis) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let mut zs: Vec<f64> = points.iter().map(|p| up.apply(*p).z).collect();
    let k = (zs.len() as f64 * 0.01) as usize;
    let (_, nth, _) = zs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *nth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let tris = vec![[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]];
        let dir = std::env::temp_dir().join("ipse-map-stl-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tri.stl");
        write_binary_stl(&path, &tris).unwrap();
        let soup = read_binary_stl(&path).unwrap();
        assert_eq!(soup.len(), 1);
        assert_eq!(soup[0][1][0], 1.0);
        assert_eq!(soup[0][2][1], 1.0);
    }

    /// A minimal but real splat: header, then rows of `x y z f_dc_0 opacity`.
    fn write_fake_splat(path: &Path, points: &[[f32; 3]]) {
        let mut bytes = format!(
            "ply\nformat binary_little_endian 1.0\ncomment made by a test\n\
             element vertex {}\nproperty float x\nproperty float y\nproperty float z\n\
             property float f_dc_0\nproperty float opacity\nend_header\n",
            points.len()
        )
        .into_bytes();
        for (i, p) in points.iter().enumerate() {
            for c in [p[0], p[1], p[2], 0.5, i as f32] {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn cropping_keeps_the_thing_and_the_rows_it_keeps_are_untouched() {
        let dir = std::env::temp_dir().join("ipse-map-splat-crop");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("scan.ply");
        // Three gaussians on the object, two out on the table it stood on.
        let pts = [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.05],
            [-0.01, 0.01, 0.09],
            [1.4, 0.0, 0.0],
            [-2.0, 3.0, 0.0],
        ];
        write_fake_splat(&src, &pts);
        let dst = dir.join("splat.ply");
        let (kept, total) =
            crop_splat_ply(&src, &dst, |p| p[0].abs() < 0.1 && p[1].abs() < 0.1).unwrap();
        assert_eq!((kept, total), (3, 5));

        // The survivors are the ones asked for, byte-identically: the
        // opacity column carries each row's index, so a shuffled or
        // re-encoded row shows up here.
        let back = read_splat_positions(&dst).unwrap();
        assert_eq!(back.len(), 3);
        for (a, b) in back.iter().zip(&pts[..3]) {
            assert_eq!(a, b);
        }
        let bytes = std::fs::read(&dst).unwrap();
        let tail = &bytes[bytes.len() - 20..];
        let opacity = f32::from_le_bytes([tail[16], tail[17], tail[18], tail[19]]);
        assert_eq!(opacity, 2.0, "the last surviving row is not row 2");
        // And it is still a splat as far as every other consumer is concerned.
        sniff_splat_ply(&dst).unwrap();
    }

    #[test]
    fn ascii_rejected() {
        let ascii = b"solid thing\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid thing\n";
        let mut padded = ascii.to_vec();
        padded.resize(200, b' ');
        assert!(parse_binary_stl(&padded).is_err());
    }
}
