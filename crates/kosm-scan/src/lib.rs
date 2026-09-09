//! Captured environment maps: one directory, one frame, two layers.
//!
//! A *map* is a captured place — the lab, first — packaged so every consumer
//! reads the representation it actually needs, in one shared frame:
//!
//! * **Appearance layer** (`splat.ply`): a 3D Gaussian splat trained from the
//!   capture (Scaniverse/Polycam/nerfstudio all export the standard splat
//!   `.ply`). Consumed by the sim head camera (world-model training frames)
//!   and by Dojo. This crate only validates and points at it — rendering
//!   lives with the renderers.
//! * **Physics layer** (`mesh.stl` + `sdf.bin`): the scan's TSDF/LiDAR mesh,
//!   and a dense signed-distance grid baked from it. The SDF — not the splat,
//!   whose geometry is mush exactly where contact cares — is what the contact
//!   solver stands on, via [`contact::find_terrain_contacts_model`].
//!
//! The frame contract: **z-up, metres, gravity −z, floor at z ≈ 0**. Phone
//! scans arrive y-up with the floor wherever the phone started; `mapbake`
//! rotates and shifts them into the contract and records the transform in the
//! manifest, so the splat can be aligned with the *same* transform later
//! without re-deriving it.
//!
//! # Directory format
//!
//! ```text
//! maps/lab/
//!   map.toml    manifest: layers, alignment applied at bake, provenance
//!   splat.ply   appearance (optional until the phone capture lands)
//!   mesh.stl    collision mesh, binary STL, already in the map frame
//!   sdf.bin     baked SDF grid, ISDF v1 (see `sdf` module docs)
//! ```
//!
//! Two bakers write this directory: `mapbake` from a phone-scan mesh (or the
//! synthetic garage), and `roombake` from a room journal's VGGT-derived
//! per-view depth ([`room`]), which never touches a mesh on the way in —
//! it TSDF-fuses the views with a persistence vote and meshes the field.
//!
//! # Why an SDF and not a heightfield or trimesh collider
//!
//! The phyz ground path already works by generating per-shape support
//! candidates and measuring `depth = ground_height − p.z` against an implicit
//! `+z` normal. An SDF is the same idea with the plane generalized away:
//! `depth = −sdf(p)`, `normal = ∇sdf(p)`. One code path then covers the flat
//! floor, stairs, and furniture, and it degenerates to bit-near-identical
//! contacts on a flat scan — which is the regression test
//! (`ipse-sim/tests/terrain_parity.rs`).

pub mod contact;
pub mod fuse;
pub mod hull;
pub mod journal;
// Liveness note for capture consumers: the K1's camera topics keep
// publishing stale buffers at nominal rates after the vendor's vision
// daemon dies (docs/vendor-camera-ticket.md). Rate is not liveness;
// recorders must verify content variance. `scripts/capture_scan.py`
// carries the tripwire.
pub mod manifest;
pub mod mesh;
pub mod npy;
pub mod object;
pub mod paint;
pub mod props;
pub mod room;
pub mod sdf;
pub mod stl;
pub mod stream;

pub use contact::find_terrain_contacts_model;
pub use fuse::{DepthFrame, FusedGrid};
pub use manifest::{Map, MapManifest};
pub use mesh::TriMesh;
pub use object::{Object, ObjectManifest};
pub use paint::{read_vertex_colours, ColourCloud};
pub use room::{FloorFrame, RoomBakeParams, RoomDerived};
pub use sdf::SdfGrid;
pub use stream::{BrickKey, BrickMesh, BrickWalker, Bricked, MapInfo};

/// Errors from loading, baking, or validating a map.
#[derive(Debug)]
pub enum MapError {
    /// I/O, with the path that failed.
    Io(std::path::PathBuf, std::io::Error),
    /// A file that was not what its extension promised.
    Format(std::path::PathBuf, String),
    /// A manifest that parsed but describes an unusable map.
    Manifest(String),
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            MapError::Format(p, msg) => write!(f, "{}: {msg}", p.display()),
            MapError::Manifest(msg) => write!(f, "map.toml: {msg}"),
        }
    }
}

impl std::error::Error for MapError {}
