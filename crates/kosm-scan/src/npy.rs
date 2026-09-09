//! A minimal `.npy` reader — just enough to consume `room_fuse.py`'s output.
//!
//! The room pipeline's Python half saves per-view geometry as NumPy arrays
//! (`extri.npy`, `intri.npy`, `depth.npy`, `floor.npy`). This crate reads
//! them; it never writes them, and it never needs anything beyond
//! little-endian floats in C order. So rather than pull in a dependency for
//! a format whose whole spec fits on one page, this parses the header dict
//! by hand and converts every supported dtype to `f32`.
//!
//! Supported: format versions 1.0, 2.0, 3.0; dtypes `<f2`, `<f4`, `<f8`
//! (and their `=`/`|` byte-order spellings on little-endian hosts);
//! `fortran_order: False`. Anything else is a [`MapError::Format`] naming
//! the file — a wrong dtype should fail loudly at load, not fuse garbage.

use std::path::Path;

use crate::MapError;

/// A dense array with its shape, values widened to `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    /// C order (last axis fastest).
    pub data: Vec<f32>,
}

impl NpyArray {
    /// Number of elements the shape describes.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Read `path` as a `.npy` array.
pub fn read_npy(path: &Path) -> Result<NpyArray, MapError> {
    let bytes = std::fs::read(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    parse_npy(&bytes).map_err(|msg| MapError::Format(path.to_path_buf(), msg))
}

/// Parse `.npy` bytes. Split from [`read_npy`] so tests can feed buffers.
pub fn parse_npy(bytes: &[u8]) -> Result<NpyArray, String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not a .npy file (bad magic)".into());
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated header".into());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let data_start = header_start + header_len;
    if bytes.len() < data_start {
        return Err("truncated header".into());
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .map_err(|_| "header is not UTF-8".to_string())?;

    let descr = dict_str(header, "descr").ok_or("header has no 'descr'")?;
    let fortran = dict_raw(header, "fortran_order").ok_or("header has no 'fortran_order'")?;
    if fortran.starts_with("True") {
        return Err("fortran_order arrays are not supported".into());
    }
    let shape_raw = dict_raw(header, "shape").ok_or("header has no 'shape'")?;
    let shape = parse_shape(shape_raw)?;

    // Byte order: '<' or '=' or '|' are all little-endian for our purposes.
    let (order, kind) = descr.split_at(1);
    if order == ">" {
        return Err(format!("big-endian dtype {descr} is not supported"));
    }
    let width: usize = match kind {
        "f2" => 2,
        "f4" => 4,
        "f8" => 8,
        _ => return Err(format!("unsupported dtype {descr} (want <f2, <f4, or <f8)")),
    };
    let n: usize = shape.iter().product();
    let body = &bytes[data_start..];
    if body.len() < n * width {
        return Err(format!(
            "data too short: shape {shape:?} × {width} B needs {} B, have {}",
            n * width,
            body.len()
        ));
    }
    let data: Vec<f32> = match width {
        2 => body[..n * 2]
            .chunks_exact(2)
            .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        4 => body[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => body[..n * 8]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
    };
    Ok(NpyArray { shape, data })
}

/// The quoted string value of `key` in the header dict.
fn dict_str<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let raw = dict_raw(header, key)?;
    let raw = raw.trim_start();
    let quote = raw.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &raw[1..];
    let end = rest.find(quote)?;
    Some(&rest[..end])
}

/// The raw text following `'key':` in the header dict, up to end of header.
fn dict_raw<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("'{key}':");
    let at = header.find(&needle)?;
    Some(header[at + needle.len()..].trim_start())
}

fn parse_shape(raw: &str) -> Result<Vec<usize>, String> {
    let raw = raw.trim_start();
    if !raw.starts_with('(') {
        return Err("shape is not a tuple".into());
    }
    let end = raw.find(')').ok_or("shape tuple not closed")?;
    raw[1..end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| format!("shape entry {s:?}: {e}")))
        .collect()
}

/// IEEE 754 binary16 → f32.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// Serialize an array as `.npy` v1.0, `<f4`, C order. Used by tests (and by
/// anyone who wants to hand a synthetic scene to `roombake`).
pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), MapError> {
    let bytes = encode_npy_f32(shape, data);
    std::fs::write(path, bytes).map_err(|e| MapError::Io(path.to_path_buf(), e))
}

/// The bytes [`write_npy_f32`] writes.
pub fn encode_npy_f32(shape: &[usize], data: &[f32]) -> Vec<u8> {
    assert_eq!(shape.iter().product::<usize>(), data.len(), "shape/data mismatch");
    let shape_txt = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    };
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_txt}, }}");
    // Pad so the total preamble (10 B + header) is a multiple of 64, ending
    // in a newline, as the format requires.
    let total = 10 + header.len() + 1;
    let pad = (64 - total % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_f32() {
        let data = vec![1.0, -2.5, 3.25, 0.0, 1e-3, 7.0];
        let bytes = encode_npy_f32(&[2, 3], &data);
        let arr = parse_npy(&bytes).unwrap();
        assert_eq!(arr.shape, vec![2, 3]);
        assert_eq!(arr.data, data);
    }

    #[test]
    fn one_d_shape_and_f8() {
        // Hand-built v1.0 header with a 1-tuple shape and float64 payload.
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (4,), }";
        let mut h = header.to_string();
        let total = 10 + h.len() + 1;
        h.push_str(&" ".repeat((64 - total % 64) % 64));
        h.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(h.len() as u16).to_le_bytes());
        bytes.extend_from_slice(h.as_bytes());
        for v in [-0.0074f64, -0.9999, 0.0073, 0.8103] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let arr = parse_npy(&bytes).unwrap();
        assert_eq!(arr.shape, vec![4]);
        assert!((arr.data[3] - 0.8103).abs() < 1e-6);
    }

    #[test]
    fn half_precision_decodes() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert!((half_to_f32(0x3555) - 0.333252).abs() < 1e-5);
        // Smallest subnormal.
        assert!((half_to_f32(0x0001) - 5.960464e-8).abs() < 1e-12);
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());

        // A v2 header with an f2 payload.
        let header = "{'descr': '<f2', 'fortran_order': False, 'shape': (1, 2), }";
        let mut h = header.to_string();
        let total = 12 + h.len() + 1;
        h.push_str(&" ".repeat((64 - total % 64) % 64));
        h.push('\n');
        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend_from_slice(&(h.len() as u32).to_le_bytes());
        bytes.extend_from_slice(h.as_bytes());
        bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
        bytes.extend_from_slice(&0xc000u16.to_le_bytes());
        let arr = parse_npy(&bytes).unwrap();
        assert_eq!(arr.shape, vec![1, 2]);
        assert_eq!(arr.data, vec![1.0, -2.0]);
    }

    #[test]
    fn rejects_wrong_things() {
        assert!(parse_npy(b"nope").is_err());
        let bytes = encode_npy_f32(&[2], &[1.0, 2.0]);
        let mut fortran = bytes.clone();
        let at = fortran.windows(5).position(|w| w == b"False").unwrap();
        fortran[at..at + 5].copy_from_slice(b"True ");
        assert!(parse_npy(&fortran).is_err());
        assert!(parse_npy(&bytes[..bytes.len() - 1]).is_err());
    }
}
