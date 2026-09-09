//! Captured *things*: one directory, one frame, three layers.
//!
//! A [map](crate::manifest) is a place — it does not move, so its physics is
//! a field baked in the world frame. An **object** is a thing you can pick
//! up, kick, or drop a robot onto, so everything about it has to travel with
//! it: its appearance, its convex pieces, and the six numbers that decide how
//! it falls.
//!
//! ```text
//! objects/bucket/
//!   object.toml   manifest: layers, mass properties, alignment, provenance
//!   splat.ply     appearance, cropped to the thing (optional)
//!   mesh.stl      the surface, in the object frame — drawn, and decomposed
//!   hulls.bin     the convex pieces the contact solver actually sees
//! ```
//!
//! # The object frame
//!
//! **z-up, metres, the resting base at `z = 0`, the origin at the centre of
//! the footprint.** This is the map contract's little sibling and it exists
//! for the same reason: so that placing a thing is `(x, y, yaw)` and nothing
//! else. A format that recorded the object wherever the scanner happened to
//! see it would make every placement a three-translation puzzle, and the
//! first person to get it wrong would spawn a bucket halfway through the
//! floor and blame the physics.
//!
//! `objbake` establishes the frame and records what it did in `[align]`, the
//! same way `mapbake` does — so the splat, which is never rewritten, can be
//! bent into the object frame by a renderer applying the identical
//! rotation-then-translation.
//!
//! # Why the splat is cropped and not transformed
//!
//! Rotating a Gaussian splat means rotating every covariance and every
//! spherical-harmonic band, which is a rendering-grade operation this crate
//! has no business doing badly. Cropping is a *filter over rows* — no
//! per-gaussian arithmetic, no chance of a silently wrong appearance — and
//! it is the operation an object actually needs, because a scan of a mug is
//! mostly the table it was standing on.
//!
//! # What an object does not have
//!
//! No SDF. The field would be baked in the object frame and would have to be
//! resampled through the body's pose every query — which is what the convex
//! pieces already do, exactly, for free. See [`crate::hull`].

use std::path::{Path, PathBuf};

use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use serde::{Deserialize, Serialize};

use crate::hull::Hull;
use crate::manifest::{Align, Extent, Provenance, SplatLayer};
use crate::MapError;

/// `"IHUL"` little-endian — the convex-piece file's magic.
const HULL_MAGIC: u32 = 0x4C55_4849;
const HULL_VERSION: u32 = 1;

/// The parsed `object.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectManifest {
    pub name: String,
    /// Appearance, cropped to the thing, in the *capture* frame — apply
    /// `[align]` to bring it into the object frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splat: Option<SplatLayer>,
    /// The surface, in the object frame. Drawn by viewers; decomposed into
    /// [`Physics::hulls`] at bake time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visual: Option<VisualLayer>,
    /// What the contact solver sees, and what makes it fall right.
    ///
    /// Optional for the same reason a map's collision layer is: a splat of a
    /// thing, cropped and levelled, is a real state the workflow passes
    /// through — something you can put in a scene and look at, and cannot
    /// touch. Consumers that need to simulate it say so by name
    /// ([`Object::simulatable`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physics: Option<Physics>,
    /// A parameterized mechanism, for things whose behaviour cannot be
    /// captured — see [`RigLayer`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig: Option<RigLayer>,
    /// Capture frame → object frame, applied rotation first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<Align>,
    /// Bounds in the object frame. `lo.z` is 0 by construction; `hi.z` is how
    /// tall the thing is, which is the number a person placing it wants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extent: Option<Extent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualLayer {
    /// Binary STL in the object frame, relative to the object directory.
    pub mesh: String,
}

/// A **mechanism**: several bodies on joints, built by named code from
/// numbered parameters.
///
/// The layer that exists because some things cannot be captured. A
/// skateboard is seven bodies and six joints, and what makes it a skateboard
/// rather than a plank is that its truck pivots about an axis *canted* from
/// vertical — the mechanism that turns deck roll into wheel steer. Scan one
/// and you get a convex lump that cannot carve; the physics of a board is
/// modelling knowledge (`ipse_sim::skate`), not geometry.
///
/// So this layer names the builder and carries its **parameters as data**,
/// which is the half a training session actually varies. The topology stays
/// in code, where its sign conventions have tests protecting them; the
/// numbers come out here, where a scenario can randomize them and an
/// operator can read them.
///
/// ```toml
/// [rig]
/// builder = "skateboard"
/// mass = 3.15
///
/// [rig.params]      # every physical number the builder takes
/// cant = 0.7853981633974483
/// bushing_k = 40.0
/// wheel_radius = 0.027
///
/// [rig.randomize]   # fractional spread per parameter, for domain randomization
/// bushing_k = 0.40  # least-known until the mini-sysid session
/// wheel_radius = 0.05  # tape-measured
///
/// [[rig.parts]]     # what a viewer draws, one per body
/// name = "skate_deck"
/// mesh = "deck.stl"
/// ```
///
/// The builder is resolved by `ipse_sim::objects::rigs`, not here: this crate
/// owns the *format* and deliberately cannot construct a mechanism. A
/// manifest naming a builder nobody has is a load-time error with the name in
/// it, which is the same bargain the rest of the format makes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigLayer {
    /// Which builder makes this, e.g. `"skateboard"`.
    pub builder: String,
    /// Total mass, kg — on the manifest so a row can be drawn and a scenario
    /// checked without constructing the mechanism.
    pub mass: f64,
    /// Every physical number the builder takes. Names are the builder's, and
    /// an unknown one is refused rather than ignored — a typo'd `bushing_c`
    /// that silently kept the default would be a training run measuring the
    /// wrong board.
    #[serde(default)]
    pub params: std::collections::BTreeMap<String, f64>,
    /// Fractional spread per parameter for domain randomization: `0.40` means
    /// ±40% about the value in `params`.
    ///
    /// Here rather than in a scenario because the spread is a property of how
    /// well the number is *known* — the bushings are guesses until the
    /// mini-sysid session, the geometry is tape-measured — and that does not
    /// change per training run. A scenario scales all of them at once.
    #[serde(default)]
    pub randomize: std::collections::BTreeMap<String, f64>,
    /// One entry per body, in the builder's own order, so a viewer can draw
    /// the mechanism and the pose stream can say which part moved.
    #[serde(default)]
    pub parts: Vec<RigPart>,
}

/// One body of a mechanism, as far as a viewer is concerned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigPart {
    /// The body's name in the built model — how a streamer finds it again.
    pub name: String,
    /// Binary STL in the *part's own* frame, relative to the object
    /// directory. Several parts share one file when they share a shape,
    /// which for four wheels is the usual case.
    pub mesh: String,
}

/// The six numbers that decide how a thing falls, plus the pieces it falls on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Physics {
    /// Kilograms.
    pub mass: f64,
    /// `"measured"` or `"density"` — how [`Physics::mass`] was arrived at.
    ///
    /// Recorded rather than inferred because these are not the same claim. A
    /// kitchen scale is ground truth. `density × volume` is a guess whose
    /// error is the guess about density *and* the scan's over-estimate of
    /// volume, and a policy that learns to kick a 4 kg bucket which really
    /// weighs 1 kg has learned about a world that does not exist.
    pub mass_source: String,
    /// kg/m³, when the mass came from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density: Option<f64>,
    /// Enclosed volume of the source surface, m³.
    pub volume: f64,
    /// Centre of mass in the object frame, metres. Uniform density.
    pub com: [f64; 3],
    /// Inertia tensor **about the centre of mass**, kg·m²:
    /// `[ixx, iyy, izz, ixy, iyz, izx]`.
    pub inertia: [f64; 6],
    /// The convex pieces, relative to the object directory.
    pub hulls: String,
    /// How many pieces are in that file — in the manifest so a row can be
    /// drawn without opening it.
    pub pieces: usize,
    /// Fraction of the pieces' volume that is not in the object at all.
    ///
    /// The decomposition's honesty number. 0 for anything convex; 0.4 means
    /// the physics of this thing is two-fifths imagination, which is
    /// something an operator should be able to read off a row before
    /// wondering why the robot tripped on nothing.
    pub hull_error: f64,
    /// Coulomb friction against everything else.
    pub friction: f64,
    /// Restitution, 0 = dead.
    pub restitution: f64,
}

impl ObjectManifest {
    /// Write `dir/object.toml`.
    pub fn save(&self, dir: &Path) -> Result<(), MapError> {
        let path = dir.join("object.toml");
        let text = toml::to_string_pretty(self)
            .map_err(|e| MapError::Manifest(format!("serialize: {e}")))?;
        std::fs::write(&path, text).map_err(|e| MapError::Io(path, e))
    }
}

/// A loaded object: manifest plus its convex pieces in memory.
///
/// The mesh and the splat stay on disk as paths — their consumers are
/// renderers, and a physics crate that parsed a splat would be the second
/// implementation of something that already exists in Metal.
#[derive(Debug, Clone)]
pub struct Object {
    pub dir: PathBuf,
    pub manifest: ObjectManifest,
    /// Convex pieces in the object frame. Empty on an appearance-only object.
    pub hulls: Vec<Hull>,
}

impl Object {
    /// Load `dir/object.toml` and the pieces it names.
    pub fn load(dir: &Path) -> Result<Object, MapError> {
        let manifest_path = dir.join("object.toml");
        let text = std::fs::read_to_string(&manifest_path)
            .map_err(|e| MapError::Io(manifest_path.clone(), e))?;
        let manifest: ObjectManifest =
            toml::from_str(&text).map_err(|e| MapError::Manifest(e.to_string()))?;

        if manifest.physics.is_none() && manifest.splat.is_none() && manifest.rig.is_none() {
            return Err(MapError::Manifest(
                "no layers: an object with no physics, rig or splat is an empty directory".into(),
            ));
        }
        if manifest.physics.is_some() && manifest.rig.is_some() {
            // Two answers to "what does the solver see". Whichever were
            // chosen, the other would be silently ignored, and a mechanism
            // quietly simulated as a lump is exactly the failure the rig
            // layer exists to prevent.
            return Err(MapError::Manifest(
                "both [physics] and [rig]: a thing is either captured or built, not both".into(),
            ));
        }
        if let Some(r) = &manifest.rig {
            if r.mass <= 0.0 {
                return Err(MapError::Manifest(format!(
                    "rig mass = {} kg — a mechanism with no mass cannot be simulated",
                    r.mass
                )));
            }
            for p in &r.parts {
                let path = dir.join(&p.mesh);
                if !path.is_file() {
                    return Err(MapError::Manifest(format!(
                        "rig part {} names {} and it is not there",
                        p.name,
                        path.display()
                    )));
                }
            }
        }
        if let Some(splat) = &manifest.splat {
            crate::stl::sniff_splat_ply(&dir.join(&splat.path))?;
        }
        let hulls = match &manifest.physics {
            Some(p) => {
                let hulls = read_hulls(&dir.join(&p.hulls))?;
                if hulls.len() != p.pieces {
                    return Err(MapError::Manifest(format!(
                        "{} holds {} pieces, the manifest says {}",
                        p.hulls,
                        hulls.len(),
                        p.pieces
                    )));
                }
                if p.mass <= 0.0 {
                    return Err(MapError::Manifest(format!(
                        "mass = {} kg — a thing with no mass cannot be simulated",
                        p.mass
                    )));
                }
                hulls
            }
            None => Vec::new(),
        };
        Ok(Object { dir: dir.to_path_buf(), manifest, hulls })
    }

    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Absolute path of the surface mesh, if there is one.
    pub fn mesh_path(&self) -> Option<PathBuf> {
        self.manifest.visual.as_ref().map(|v| self.dir.join(&v.mesh))
    }

    /// Absolute path of the splat, if the appearance layer exists.
    pub fn splat_path(&self) -> Option<PathBuf> {
        self.manifest.splat.as_ref().map(|s| self.dir.join(&s.path))
    }

    /// How tall the thing is, metres — `extent.hi.z`, or the pieces' own
    /// bound when the manifest predates the field.
    pub fn height(&self) -> f64 {
        if let Some(e) = &self.manifest.extent {
            return e.hi[2];
        }
        self.hulls
            .iter()
            .map(|h| h.aabb().1.z)
            .fold(0.0f64, f64::max)
    }

    /// The physics layer, or an error naming what is missing.
    ///
    /// The sibling of [`Map::standable`](crate::Map::standable), and for the
    /// same reason: "you cannot drop a thing that has no mass into a
    /// simulation" is a sentence an operator should read once, in the same
    /// words, wherever they hit it.
    pub fn simulatable(&self) -> Result<&Physics, MapError> {
        self.manifest.physics.as_ref().ok_or_else(|| {
            if self.manifest.rig.is_some() {
                // A mechanism has no convex pieces of its own: its bodies and
                // their shapes come from the builder. Callers that wanted
                // pieces should have asked `kind` first, so this says what to
                // ask instead rather than what to go and bake.
                return MapError::Manifest(format!(
                    "{} is a rig — its bodies come from its builder, not from convex pieces. \
                     Use `Object::rig` and `ipse_sim::objects::rigs`.",
                    self.manifest.name
                ));
            }
            MapError::Manifest(format!(
                "{} is appearance-only — there are no convex pieces to collide with. \
                 Bake them with `objbake <mesh.stl> --out {}`.",
                self.manifest.name,
                self.dir.display()
            ))
        })
    }

    /// The mechanism, when this thing is one.
    pub fn rig(&self) -> Option<&RigLayer> {
        self.manifest.rig.as_ref()
    }

    /// Whether anything at all can be simulated: convex pieces or a builder.
    pub fn can_simulate(&self) -> bool {
        self.manifest.physics.is_some() || self.manifest.rig.is_some()
    }

    /// Kilograms, from whichever layer knows — 0 for appearance-only.
    pub fn mass(&self) -> f64 {
        self.manifest
            .physics
            .as_ref()
            .map(|p| p.mass)
            .or_else(|| self.manifest.rig.as_ref().map(|r| r.mass))
            .unwrap_or(0.0)
    }

    /// What a viewer draws, as `(part name, absolute mesh path)`.
    ///
    /// One entry for a captured thing, one per body for a mechanism — and the
    /// order is the order poses arrive in, which is what lets the wire refer
    /// to a wheel by an index instead of by a name it would have to match.
    pub fn parts(&self) -> Vec<(String, PathBuf)> {
        if let Some(r) = &self.manifest.rig {
            return r
                .parts
                .iter()
                .map(|p| (p.name.clone(), self.dir.join(&p.mesh)))
                .collect();
        }
        match self.mesh_path() {
            Some(p) => vec![(self.manifest.name.clone(), p)],
            None => Vec::new(),
        }
    }

    /// The rigid body's inertia, ready for `Body::new`.
    ///
    /// `SpatialInertia`'s tensor is about the centre of mass and its `com` is
    /// the offset in the body frame — which is exactly how the manifest
    /// stores them, on purpose, so nothing here has to shift an axis.
    pub fn spatial_inertia(&self) -> Result<SpatialInertia, MapError> {
        let p = self.simulatable()?;
        let [ixx, iyy, izz, ixy, iyz, izx] = p.inertia;
        Ok(SpatialInertia::new(
            p.mass,
            Vec3::new(p.com[0], p.com[1], p.com[2]),
            Mat3::new(ixx, ixy, izx, ixy, iyy, iyz, izx, iyz, izz),
        ))
    }

    /// The convex pieces as phyz collision geometry, centred on the body
    /// frame — which is the object frame, so no offsets are involved.
    pub fn collisions(&self) -> Vec<phyz_model::GeomInstance> {
        self.hulls
            .iter()
            .map(|h| {
                phyz_model::GeomInstance::new(
                    phyz_model::Geometry::Mesh {
                        vertices: h.vertices.clone(),
                        faces: h
                            .faces
                            .iter()
                            .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
                            .collect(),
                    },
                    SpatialTransform::identity(),
                )
            })
            .collect()
    }
}

/// Every object directory under `root` — one level down, `object.toml` inside.
///
/// Bad directories are skipped with a line on stderr rather than failing the
/// scan: one object with a truncated hull file must not make a library of
/// forty disappear.
pub fn scan(root: &Path) -> Vec<Object> {
    let mut out = Vec::new();
    let Ok(kids) = std::fs::read_dir(root) else {
        return out;
    };
    let mut dirs: Vec<PathBuf> = kids.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    dirs.sort();
    for d in dirs {
        if !d.join("object.toml").is_file() {
            continue;
        }
        match Object::load(&d) {
            Ok(o) => out.push(o),
            Err(e) => eprintln!("objects: skipping {}: {e}", d.display()),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// hulls.bin
// ---------------------------------------------------------------------------

/// Write the convex pieces.
///
/// ```text
/// magic  u32  "IHUL"
/// version u32 1
/// pieces u32
/// pad    u32  0
/// per piece:
///   nverts u32, nfaces u32
///   nverts × 3 × f32   vertices, object frame, metres
///   nfaces × 3 × u32   triangles, outward wound
/// ```
///
/// `f32` because an object is at most a couple of metres across and f32 holds
/// that to a tenth of a micron — the file is a tenth the size of an f64 one
/// for precision nothing downstream can use. The same reasoning as the
/// brick stream's `u16`s, one order of magnitude up.
pub fn write_hulls(path: &Path, hulls: &[Hull]) -> Result<(), MapError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&HULL_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&HULL_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(hulls.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    for h in hulls {
        bytes.extend_from_slice(&(h.vertices.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(h.faces.len() as u32).to_le_bytes());
        for v in &h.vertices {
            for c in [v.x as f32, v.y as f32, v.z as f32] {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        for f in &h.faces {
            for i in f {
                bytes.extend_from_slice(&i.to_le_bytes());
            }
        }
    }
    std::fs::write(path, bytes).map_err(|e| MapError::Io(path.to_path_buf(), e))
}

/// Read the convex pieces, refusing anything that does not add up.
pub fn read_hulls(path: &Path) -> Result<Vec<Hull>, MapError> {
    let bytes = std::fs::read(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    let bad = |msg: &str| MapError::Format(path.to_path_buf(), msg.to_string());
    if bytes.len() < 16 {
        return Err(bad("too short for a header"));
    }
    let u32_at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    if u32_at(0) != HULL_MAGIC {
        return Err(bad("not an IHUL file"));
    }
    if u32_at(4) != HULL_VERSION {
        return Err(MapError::Format(
            path.to_path_buf(),
            format!("IHUL version {} — this build reads {HULL_VERSION}", u32_at(4)),
        ));
    }
    let pieces = u32_at(8) as usize;
    let mut at = 16;
    let mut out = Vec::with_capacity(pieces);
    for _ in 0..pieces {
        if at + 8 > bytes.len() {
            return Err(bad("truncated piece header"));
        }
        let nv = u32_at(at) as usize;
        let nf = u32_at(at + 4) as usize;
        at += 8;
        if at + nv * 12 + nf * 12 > bytes.len() {
            return Err(bad("truncated piece"));
        }
        let mut vertices = Vec::with_capacity(nv);
        for i in 0..nv {
            let b = at + i * 12;
            let f = |o: usize| {
                f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as f64
            };
            vertices.push(Vec3::new(f(b), f(b + 4), f(b + 8)));
        }
        at += nv * 12;
        let mut faces = Vec::with_capacity(nf);
        for i in 0..nf {
            let b = at + i * 12;
            let f = [u32_at(b), u32_at(b + 4), u32_at(b + 8)];
            if f.iter().any(|&i| i as usize >= nv) {
                return Err(bad("triangle indexes a vertex that is not there"));
            }
            faces.push(f);
        }
        at += nf * 12;
        out.push(Hull { vertices, faces });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn unit_cube() -> Hull {
        let pts: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -0.05 } else { 0.05 },
                    if i & 2 == 0 { -0.05 } else { 0.05 },
                    if i & 4 == 0 { 0.0 } else { 0.10 },
                )
            })
            .collect();
        crate::hull::convex_hull(&pts).unwrap()
    }

    fn cube_object(dir: &Path) -> ObjectManifest {
        let hulls = vec![unit_cube()];
        write_hulls(&dir.join("hulls.bin"), &hulls).unwrap();
        let m = crate::hull::compound_mass_properties(&hulls, 700.0);
        let manifest = ObjectManifest {
            name: "cube".into(),
            splat: None,
            visual: None,
            rig: None,
            physics: Some(Physics {
                mass: m.mass,
                mass_source: "density".into(),
                density: Some(700.0),
                volume: m.volume,
                com: [m.com.x, m.com.y, m.com.z],
                inertia: [
                    m.inertia.get(0, 0),
                    m.inertia.get(1, 1),
                    m.inertia.get(2, 2),
                    m.inertia.get(0, 1),
                    m.inertia.get(1, 2),
                    m.inertia.get(2, 0),
                ],
                hulls: "hulls.bin".into(),
                pieces: 1,
                hull_error: 0.0,
                friction: 0.8,
                restitution: 0.0,
            }),
            align: None,
            extent: Some(Extent { lo: [-0.05, -0.05, 0.0], hi: [0.05, 0.05, 0.10] }),
            provenance: None,
        };
        manifest.save(dir).unwrap();
        manifest
    }

    #[test]
    fn hulls_round_trip() {
        let dir = tmp("ipse-object-hulls");
        let hulls = vec![unit_cube(), unit_cube()];
        let path = dir.join("hulls.bin");
        write_hulls(&path, &hulls).unwrap();
        let back = read_hulls(&path).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].faces, hulls[0].faces);
        for (a, b) in back[0].vertices.iter().zip(&hulls[0].vertices) {
            assert!((*a - *b).norm() < 1e-6);
        }
    }

    #[test]
    fn a_truncated_hull_file_is_refused_not_guessed() {
        let dir = tmp("ipse-object-truncated");
        let path = dir.join("hulls.bin");
        write_hulls(&path, &[unit_cube()]).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 30);
        std::fs::write(&path, bytes).unwrap();
        let err = read_hulls(&path).unwrap_err().to_string();
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn an_object_loads_with_the_inertia_it_was_baked_with() {
        let dir = tmp("ipse-object-load");
        let baked = cube_object(&dir);
        let obj = Object::load(&dir).unwrap();
        assert_eq!(obj.name(), "cube");
        assert_eq!(obj.hulls.len(), 1);
        assert!((obj.height() - 0.10).abs() < 1e-9);

        let si = obj.spatial_inertia().unwrap();
        let want = baked.physics.unwrap();
        assert!((si.mass - want.mass).abs() < 1e-9);
        // A 10×10×10 cm box of 700 kg/m³ is 0.7 kg, and its com is half way
        // up — the placement contract's whole point is that this is *not* the
        // origin.
        assert!((si.mass - 0.7).abs() < 1e-6, "{}", si.mass);
        assert!((si.com.z - 0.05).abs() < 1e-6, "{}", si.com.z);
        let i = 0.7 / 12.0 * (0.1 * 0.1 + 0.1 * 0.1);
        assert!((si.inertia.get(0, 0) - i).abs() < 1e-6);

        // And the collision geometry comes out as one convex mesh.
        let c = obj.collisions();
        assert_eq!(c.len(), 1);
        assert!(matches!(c[0].geometry, phyz_model::Geometry::Mesh { .. }));
    }

    #[test]
    fn a_manifest_that_lies_about_its_pieces_is_refused() {
        let dir = tmp("ipse-object-miscount");
        let mut m = cube_object(&dir);
        m.physics.as_mut().unwrap().pieces = 3;
        m.save(&dir).unwrap();
        let err = Object::load(&dir).unwrap_err().to_string();
        assert!(err.contains("pieces"), "{err}");
    }

    #[test]
    fn an_appearance_only_object_loads_and_refuses_to_be_simulated() {
        let dir = tmp("ipse-object-look-only");
        std::fs::write(
            dir.join("splat.ply"),
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n\
             property float x\nproperty float f_dc_0\nend_header\n",
        )
        .unwrap();
        ObjectManifest {
            name: "vase".into(),
            splat: Some(SplatLayer { path: "splat.ply".into() }),
            visual: None,
            physics: None,
            rig: None,
            align: None,
            extent: None,
            provenance: None,
        }
        .save(&dir)
        .unwrap();

        let obj = Object::load(&dir).unwrap();
        assert!(obj.hulls.is_empty());
        let err = obj.simulatable().unwrap_err().to_string();
        assert!(err.contains("appearance-only"), "{err}");
        assert!(err.contains("objbake"), "the refusal must say what to do: {err}");
    }

    #[test]
    fn scan_finds_objects_and_survives_a_broken_one() {
        let root = tmp("ipse-object-scan");
        cube_object(&std::fs::create_dir_all(root.join("cube"))
            .map(|_| root.join("cube"))
            .unwrap());
        std::fs::create_dir_all(root.join("broken")).unwrap();
        std::fs::write(root.join("broken/object.toml"), "name = \"broken\"\n").unwrap();
        std::fs::create_dir_all(root.join("not-an-object")).unwrap();

        let found = scan(&root);
        assert_eq!(found.len(), 1, "scan should skip the broken one, not fail");
        assert_eq!(found[0].name(), "cube");
    }
}
