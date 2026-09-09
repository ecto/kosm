//! `map.toml` — the manifest that names a map's layers and remembers how the
//! capture was bent into the map frame.
//!
//! ```toml
//! name = "lab"
//!
//! [splat]                      # appearance layer; absent until captured
//! path = "splat.ply"
//!
//! [collision]                  # physics layer, already in the map frame
//! mesh = "mesh.stl"
//! sdf = "sdf.bin"
//! cell = 0.02                  # bake resolution, recorded for provenance
//!
//! [align]                      # source capture frame -> map frame
//! rotate = "y-up-to-z-up"      # or "none"
//! translate = [0.0, 0.0, -1.31]
//!
//! [provenance]
//! captured = "2026-08-12"
//! device = "iphone-lidar"
//! ```
//!
//! `[align]` is the transform `mapbake` *already applied* to the mesh before
//! baking — recorded so the splat, which trains in the raw capture frame, can
//! be brought into the same map frame later by applying the identical
//! rotation-then-translation, instead of someone re-deriving the floor shift
//! by eye and getting a robot that stands 4 cm inside the ground.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{MapError, SdfGrid};

/// The parsed `map.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapManifest {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splat: Option<SplatLayer>,
    /// The physics layer, absent on an **appearance-only** map.
    ///
    /// Optional since splats became something you can add on their own. A
    /// phone scan produces a splat in minutes and a baked SDF in however long
    /// the bake takes, and in between there is a real place you can look
    /// around and cannot stand in. Refusing to load that map would mean the
    /// format could not represent a state the workflow actually passes
    /// through; representing it means every *consumer* has to say what it
    /// does without a floor, which is the honest version of the same
    /// question. `StandConfig::terrain` cannot be filled from such a map and
    /// the caller is told so by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision: Option<CollisionLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<Align>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    /// The place's bounds in the map frame, when nothing else can say.
    ///
    /// A map with a physics layer announces its extent through the bricks it
    /// streams. An appearance-only one streams none, so without this the
    /// viewer knows a room exists and not whether it is a cupboard or a
    /// warehouse — and opens the camera two metres from an origin that may be
    /// inside a wall. Measured at import from the gaussian centres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extent: Option<Extent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extent {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplatLayer {
    /// Gaussian splat `.ply`, relative to the map directory. In the *raw
    /// capture frame* until a renderer applies `[align]`.
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionLayer {
    /// Binary STL in the map frame, relative to the map directory.
    pub mesh: String,
    /// Baked ISDF grid, relative to the map directory.
    pub sdf: String,
    /// Bake resolution, metres. Provenance, not a runtime knob — the grid
    /// carries its own spacing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell: Option<f64>,
    /// Per-vertex colour for `mesh`, relative to the map directory
    /// (`mesh_rgb.bin`, see [`crate::paint`]). Absent on maps baked without
    /// a colour cloud, and on every map baked before it existed — a renderer
    /// that finds nothing here draws the mesh exactly as it always did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colour: Option<String>,
}

/// The transform from the capture frame to the map frame, applied as
/// rotation first, then translation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Align {
    /// `"none"` or `"y-up-to-z-up"` (the rotation `(x, y, z) → (x, −z, y)`).
    pub rotate: String,
    /// Metres, applied after the rotation.
    pub translate: [f64; 3],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl MapManifest {
    /// Write `dir/map.toml`.
    pub fn save(&self, dir: &Path) -> Result<(), MapError> {
        let path = dir.join("map.toml");
        let text = toml::to_string_pretty(self)
            .map_err(|e| MapError::Manifest(format!("serialize: {e}")))?;
        std::fs::write(&path, text).map_err(|e| MapError::Io(path, e))
    }
}

/// A loaded map: manifest plus the SDF resident in memory. The mesh and
/// splat stay on disk as paths — their consumers (renderers, the viewer)
/// load them with their own machinery.
#[derive(Debug)]
pub struct Map {
    pub dir: PathBuf,
    pub manifest: MapManifest,
    /// The physics layer, or `None` on an appearance-only map. See
    /// [`MapManifest::collision`], and [`Map::standable`] for the check every
    /// consumer that needs a floor should make by name.
    pub sdf: Option<SdfGrid>,
}

impl Map {
    /// Load `dir/map.toml` and the SDF it points at, if it has one.
    /// Validates that the files the manifest names actually exist — and that
    /// the splat, when present, really is one — so a broken map fails at load
    /// with a path in the message, not mid-rollout.
    pub fn load(dir: &Path) -> Result<Map, MapError> {
        let manifest_path = dir.join("map.toml");
        let text = std::fs::read_to_string(&manifest_path)
            .map_err(|e| MapError::Io(manifest_path.clone(), e))?;
        let manifest: MapManifest =
            toml::from_str(&text).map_err(|e| MapError::Manifest(e.to_string()))?;

        if manifest.collision.is_none() && manifest.splat.is_none() {
            return Err(MapError::Manifest(
                "no layers: a map with neither collision nor splat is an empty directory".into(),
            ));
        }
        if let Some(splat) = &manifest.splat {
            crate::stl::sniff_splat_ply(&dir.join(&splat.path))?;
        }
        let sdf = match &manifest.collision {
            Some(c) => {
                let mesh_path = dir.join(&c.mesh);
                if !mesh_path.is_file() {
                    return Err(MapError::Manifest(format!(
                        "collision.mesh {} does not exist",
                        mesh_path.display()
                    )));
                }
                Some(SdfGrid::load(&dir.join(&c.sdf))?)
            }
            None => None,
        };
        Ok(Map { dir: dir.to_path_buf(), manifest, sdf })
    }

    /// Absolute path of the collision mesh, when there is a physics layer.
    pub fn mesh_path(&self) -> Option<PathBuf> {
        self.manifest.collision.as_ref().map(|c| self.dir.join(&c.mesh))
    }

    /// Absolute path of the mesh's per-vertex colour, when it was baked.
    ///
    /// Lazy like [`Self::mesh_path`]: the mesh itself is never loaded here,
    /// so its colours are not either — a caller that wants painted geometry
    /// reads both and pairs them by index.
    pub fn colour_path(&self) -> Option<PathBuf> {
        self.manifest
            .collision
            .as_ref()
            .and_then(|c| c.colour.as_ref())
            .map(|c| self.dir.join(c))
    }

    /// Absolute path of the splat, if the appearance layer exists yet.
    pub fn splat_path(&self) -> Option<PathBuf> {
        self.manifest.splat.as_ref().map(|s| self.dir.join(&s.path))
    }

    /// The SDF, or an error naming what is missing.
    ///
    /// A named accessor rather than an `unwrap` at every call site: "you
    /// cannot run a robot in a map that has no floor" is a sentence an
    /// operator should read, and it is the same sentence everywhere.
    pub fn standable(&self) -> Result<&SdfGrid, MapError> {
        self.sdf.as_ref().ok_or_else(|| {
            MapError::Manifest(format!(
                "{} is appearance-only — there is no collision layer to stand on. \
                 Bake one with `mapbake <scan.stl> --out {}`.",
                self.manifest.name,
                self.dir.display()
            ))
        })
    }
}

/// Copy a Gaussian splat into a map directory and record it in the manifest,
/// creating the map when the directory has none.
///
/// The one way a splat gets added, so the app, the CLI and any future
/// importer are the same three lines. Creating is the interesting case: a
/// splat with no bake behind it is an [appearance-only](MapManifest::collision)
/// map — somewhere to walk a camera around, and explicitly nowhere to put a
/// robot down.
///
/// Validates before it copies. A `.ply` that is not a splat would otherwise
/// land in the directory, be recorded, and fail at load time — with the
/// mistake already committed to disk.
pub fn attach_splat(
    dir: &Path,
    ply: &Path,
    name: Option<&str>,
    up: Option<crate::stl::UpAxis>,
) -> Result<MapManifest, MapError> {
    crate::stl::sniff_splat_ply(ply)?;
    // Read the centres once, to record the two things a manifest must say
    // about a capture frame and the file cannot: which way is up, and where
    // the floor is. `up` is the caller's (there is no way to read it out of a
    // ply); the floor shift is measured.
    let (up, floor, extent) = match crate::stl::read_splat_positions(ply) {
        Ok(points) => {
            // Measured unless the caller insisted. See `stl::detect_up` —
            // there is no convention to default to.
            let up = up.unwrap_or_else(|| {
                let (best, score) = crate::stl::detect_up(&points);
                eprintln!(
                    "attach_splat: up looks like {} ({:.1}% of gaussians in the floor slab)",
                    best.manifest_name(),
                    score * 100.0
                );
                best
            });
            let floor = crate::stl::floor_height(&points, up);
            // Bounds in the finished map frame: rotated, then levelled, so
            // what the viewer reads is where things actually are. Trimmed to
            // the 1st and 99th percentile per axis — a splat has stray
            // gaussians hundreds of metres out (COLMAP puts background sky
            // somewhere), and framing a camera on those shows a room the size
            // of a pixel.
            let mut axes: [Vec<f64>; 3] = [vec![], vec![], vec![]];
            for p in &points {
                let q = up.apply(*p);
                axes[0].push(q.x);
                axes[1].push(q.y);
                axes[2].push(q.z - floor);
            }
            let mut lo = [0.0; 3];
            let mut hi = [0.0; 3];
            for (i, v) in axes.iter_mut().enumerate() {
                v.sort_by(|a, b| a.total_cmp(b));
                lo[i] = v[v.len() / 100];
                hi[i] = v[v.len() - 1 - v.len() / 100];
            }
            (up, floor, Some(Extent { lo, hi }))
        }
        Err(e) => {
            // A splat whose positions cannot be read is still a splat worth
            // importing — it just arrives unlevelled, and says so.
            eprintln!("attach_splat: floor not measured ({e}); leaving it at z = 0");
            (up.unwrap_or(crate::stl::UpAxis::YDown), 0.0, None)
        }
    };
    std::fs::create_dir_all(dir).map_err(|e| MapError::Io(dir.to_path_buf(), e))?;
    let dst = dir.join("splat.ply");
    // A splat already in place is the file we would be copying onto itself;
    // `fs::copy` on the same path truncates it to nothing.
    if dst.canonicalize().ok() != ply.canonicalize().ok() {
        std::fs::copy(ply, &dst).map_err(|e| MapError::Io(dst.clone(), e))?;
    }

    let manifest_path = dir.join("map.toml");
    let mut manifest = match std::fs::read_to_string(&manifest_path) {
        Ok(text) => toml::from_str::<MapManifest>(&text)
            .map_err(|e| MapError::Manifest(e.to_string()))?,
        Err(_) => MapManifest {
            name: name
                .map(str::to_string)
                .or_else(|| dir.file_name().map(|s| s.to_string_lossy().into_owned()))
                .unwrap_or_else(|| "map".into()),
            splat: None,
            collision: None,
            // The rotation is the caller's claim; the shift is measured.
            //
            // Nothing in a `.ply` says which way is up, so `up` has to come
            // from whoever knows where the file came from — phone scanners
            // export y-up, anything trained through COLMAP is y-down. The
            // floor, though, is in the data: `floor_height` finds it and this
            // records the translation that puts it at z = 0, which is the
            // frame contract every other consumer assumes.
            //
            // An *existing* map keeps whatever `align` it already has: that
            // one was measured at bake time and outranks anything here.
            align: Some(Align {
                rotate: up.manifest_name().into(),
                translate: [0.0, 0.0, -floor],
            }),
            provenance: Some(Provenance {
                captured: None,
                device: None,
                notes: Some(format!("splat imported from {}", ply.display())),
            }),
            extent: None,
        },
    };
    manifest.splat = Some(SplatLayer { path: "splat.ply".into() });
    if manifest.extent.is_none() {
        manifest.extent = extent;
    }
    manifest.save(dir)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrip() {
        let m = MapManifest {
            name: "lab".into(),
            splat: None,
            collision: Some(CollisionLayer {
                mesh: "mesh.stl".into(),
                sdf: "sdf.bin".into(),
                cell: Some(0.02),
            colour: None,
        }),
            align: Some(Align { rotate: "y-up-to-z-up".into(), translate: [0.0, 0.0, -1.31] }),
            provenance: Some(Provenance {
                captured: Some("2026-08-12".into()),
                device: Some("iphone-lidar".into()),
                notes: None,
            }),
            extent: None,
        };
        let text = toml::to_string_pretty(&m).unwrap();
        let back: MapManifest = toml::from_str(&text).unwrap();
        assert_eq!(back.name, "lab");
        assert_eq!(back.collision.unwrap().cell, Some(0.02));
        assert_eq!(back.align.unwrap().translate[2], -1.31);
    }

    /// The smallest thing `sniff_splat_ply` will call a splat.
    fn fake_splat(path: &Path) {
        std::fs::write(
            path,
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n\
             property float x\nproperty float f_dc_0\nend_header\n",
        )
        .unwrap();
    }

    #[test]
    fn a_splat_alone_is_a_loadable_map() {
        let dir = std::env::temp_dir().join("ipse-map-splat-only");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ply = dir.join("scan.ply");
        fake_splat(&ply);

        let m = attach_splat(&dir, &ply, Some("hallway"), Some(crate::stl::UpAxis::YDown)).unwrap();
        assert_eq!(m.name, "hallway");
        assert!(m.collision.is_none());

        let map = Map::load(&dir).expect("an appearance-only map must load");
        assert!(map.sdf.is_none());
        assert_eq!(map.mesh_path(), None);
        assert!(map.splat_path().is_some());
        // And it must refuse to be stood on, in words an operator can act on.
        let err = map.standable().unwrap_err().to_string();
        assert!(err.contains("appearance-only"), "{err}");
        assert!(err.contains("mapbake"), "the refusal must say what to do: {err}");
    }

    #[test]
    fn attaching_a_splat_keeps_the_physics_layer() {
        let dir = std::env::temp_dir().join("ipse-map-splat-attach");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("map.toml"),
            "name = \"lab\"\n[collision]\nmesh = \"mesh.stl\"\nsdf = \"sdf.bin\"\ncell = 0.02\n",
        )
        .unwrap();
        let ply = dir.join("scan.ply");
        fake_splat(&ply);
        let m = attach_splat(&dir, &ply, None, Some(crate::stl::UpAxis::YDown)).unwrap();
        assert_eq!(m.name, "lab", "attaching a splat renamed the map");
        assert!(m.collision.is_some(), "attaching a splat dropped the physics layer");
        assert_eq!(m.splat.unwrap().path, "splat.ply");
    }

    #[test]
    fn a_ply_that_is_not_a_splat_is_refused_before_it_is_copied() {
        let dir = std::env::temp_dir().join("ipse-map-splat-bad");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ply = dir.join("mesh.ply");
        std::fs::write(&ply, "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n").unwrap();
        assert!(attach_splat(&dir, &ply, None, Some(crate::stl::UpAxis::YDown)).is_err());
        // Nothing written: the mistake must not reach the directory.
        assert!(!dir.join("splat.ply").exists());
        assert!(!dir.join("map.toml").exists());
    }

    #[test]
    fn load_reports_missing_mesh() {
        let dir = std::env::temp_dir().join("ipse-map-manifest-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("map.toml"),
            "name = \"x\"\n[collision]\nmesh = \"nope.stl\"\nsdf = \"nope.bin\"\n",
        )
        .unwrap();
        let err = Map::load(&dir).unwrap_err();
        assert!(err.to_string().contains("nope.stl"), "{err}");
    }
}
