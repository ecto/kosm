//! Turn a scan of a *thing* into an object directory.
//!
//! ```text
//! objbake <mesh.stl> --out objects/bucket [--name bucket] [--up y]
//!         [--crop x0,y0,z0,x1,y1,z1] [--splat scan.ply] [--margin 0.05]
//!         [--mass 1.2 | --density 950] [--pieces 8] [--tolerance 0.05]
//!         [--friction 0.9] [--restitution 0.0]
//! objbake --splat scan.ply --out objects/vase --crop ... --up y
//! objbake <mesh.stl> --inspect [--splat scan.ply] [--up y]
//! ```
//!
//! The whole job is four measurements and one refusal:
//!
//! 1. **Which way is up** — stated, not guessed. A room can be measured by
//!    finding its floor ([`stl::detect_up`]); a mug on a table has no floor
//!    of its own, so there is nothing to measure and the caller has to say.
//! 2. **Which of these gaussians are the thing** — a scan of an object is
//!    mostly the room it was scanned in. `--crop` is the box that is the
//!    thing; without one, the mesh's own bounds plus `--margin` are it.
//! 3. **Where the thing's bottom is** — so it can be placed with `(x, y,
//!    yaw)` and land on the floor rather than in it. See the object-frame
//!    contract in `kosm_scan::object`.
//! 4. **What it weighs and how it spins** — from the convex pieces, which
//!    are closed by construction, scaled to whichever of `--mass` and
//!    `--density` was given.
//!
//! And the refusal: **neither `--mass` nor `--density` is a thing this tool
//! will invent.** A scan carries no mass, every default would be wrong, and
//! a policy that learns to shove a 4 kg bucket which really weighs 1 kg has
//! learned about a world that does not exist. A kitchen scale takes ten
//! seconds.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use kosm_scan::hull;
use kosm_scan::manifest::{Align, Extent, Provenance, SplatLayer};
use kosm_scan::object::{Object, ObjectManifest, Physics, VisualLayer};
use kosm_scan::{MapError, TriMesh, stl};
use phyz_math::Vec3;

struct Args {
    input: Option<PathBuf>,
    /// A measured prop to synthesize instead of reading a scan.
    prop: Option<String>,
    out: PathBuf,
    name: Option<String>,
    up: stl::UpAxis,
    crop: Option<(Vec3, Vec3)>,
    margin: f64,
    splat: Option<PathBuf>,
    mass: Option<f64>,
    density: Option<f64>,
    pieces: usize,
    tolerance: f64,
    friction: f64,
    restitution: f64,
    inspect: bool,
}

fn usage() -> &'static str {
    "usage: objbake <mesh.stl> --out <dir> (--mass <kg> | --density <kg/m3>)\n\
     \x20      [--name <name>] [--up y|-y|z] [--crop x0,y0,z0,x1,y1,z1]\n\
     \x20      [--splat <scan.ply>] [--margin 0.05] [--pieces 8]\n\
     \x20      [--tolerance 0.05] [--friction 0.9] [--restitution 0.0]\n\
            objbake --prop bucket --out <dir> (--mass <kg> | --density <kg/m3>)\n\
            objbake --splat <scan.ply> --out <dir> --crop ... --up y\n\
            objbake <mesh.stl> --inspect [--splat <scan.ply>] [--up y]\n\
     \n\
     --prop         build one of the measured garage props instead of\n\
     \x20              reading a scan: bucket, plyo, bench, mat, ladder.\n\
     \x20              Something to put in a room before anything has been\n\
     \x20              scanned. The geometry is measured (kosm_scan::props);\n\
     \x20              the mass is still yours to give.\n\
     --inspect      report the frame, the bounds and the volume instead of\n\
     \x20              baking, so a crop box can be chosen from numbers\n\
     --up           which way is up in the SOURCE files: y, y-down (-y), or z\n\
     \x20              (default z). An object has no floor to measure, so\n\
     \x20              unlike a map this is never inferred.\n\
     --crop         keep only what is inside this box, in the up-rotated\n\
     \x20              capture frame — the numbers --inspect prints. Required\n\
     \x20              for a splat-only object; otherwise defaults to the\n\
     \x20              mesh's bounds grown by --margin.\n\
     --mass         measured mass, kg. The honest source.\n\
     --density      kg/m3, when nothing has been weighed. Mass becomes\n\
     \x20              density x the convex pieces' volume, and the manifest\n\
     \x20              records that it was derived.\n\
     --pieces       maximum convex pieces (default 8)\n\
     --tolerance    stop splitting once the pieces waste less than this\n\
     \x20              fraction of their volume (default 0.05)\n\
     --friction     Coulomb friction (default 0.9)\n\
     --restitution  bounce, 0 = dead (default 0.0)"
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut input = None;
    let mut prop = None;
    let mut out = None;
    let mut name = None;
    let mut up = stl::UpAxis::ZUp;
    let mut crop = None;
    let mut margin = 0.05;
    let mut splat = None;
    let mut mass = None;
    let mut density = None;
    let mut pieces = 8usize;
    let mut tolerance = 0.05;
    let mut friction = 0.9;
    let mut restitution = 0.0;
    let mut inspect = false;

    let value = |args: &mut dyn Iterator<Item = String>, flag: &str| {
        args.next().ok_or(format!("{flag} needs a value"))
    };
    let number = |v: String, flag: &str| -> Result<f64, String> {
        v.parse().map_err(|e| format!("{flag}: {e}"))
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(value(&mut args, "--out")?)),
            "--prop" => prop = Some(value(&mut args, "--prop")?),
            "--name" => name = Some(value(&mut args, "--name")?),
            "--up" => {
                let v = value(&mut args, "--up")?;
                up = stl::UpAxis::parse(&v)
                    .ok_or(format!("--up {v}: expected y, y-down (or -y), or z"))?;
            }
            "--crop" => {
                let v = value(&mut args, "--crop")?;
                let n: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                if n.len() != 6 {
                    return Err("--crop wants six numbers: x0,y0,z0,x1,y1,z1".into());
                }
                let lo = Vec3::new(n[0].min(n[3]), n[1].min(n[4]), n[2].min(n[5]));
                let hi = Vec3::new(n[0].max(n[3]), n[1].max(n[4]), n[2].max(n[5]));
                crop = Some((lo, hi));
            }
            "--margin" => margin = number(value(&mut args, "--margin")?, "--margin")?,
            "--splat" => splat = Some(PathBuf::from(value(&mut args, "--splat")?)),
            "--mass" => mass = Some(number(value(&mut args, "--mass")?, "--mass")?),
            "--density" => density = Some(number(value(&mut args, "--density")?, "--density")?),
            "--pieces" => {
                pieces = value(&mut args, "--pieces")?
                    .parse()
                    .map_err(|e| format!("--pieces: {e}"))?
            }
            "--tolerance" => tolerance = number(value(&mut args, "--tolerance")?, "--tolerance")?,
            "--friction" => friction = number(value(&mut args, "--friction")?, "--friction")?,
            "--restitution" => {
                restitution = number(value(&mut args, "--restitution")?, "--restitution")?
            }
            "--inspect" => inspect = true,
            "--help" | "-h" => return Err(String::new()),
            other if input.is_none() && !other.starts_with('-') => {
                input = Some(PathBuf::from(other))
            }
            other => return Err(format!("unrecognized argument {other}")),
        }
    }

    let args = Args {
        input,
        prop,
        out: out.clone().unwrap_or_default(),
        name,
        up,
        crop,
        margin,
        splat,
        mass,
        density,
        pieces,
        tolerance,
        friction,
        restitution,
        inspect,
    };
    if args.inspect {
        if args.input.is_none() && args.splat.is_none() {
            return Err("--inspect needs a mesh or a --splat to look at".into());
        }
        return Ok(args);
    }
    if out.is_none() {
        return Err("--out is required".into());
    }
    if args.prop.is_some() && args.input.is_some() {
        return Err("--prop builds its own geometry; do not also pass a scan".into());
    }
    if args.input.is_none() && args.splat.is_none() && args.prop.is_none() {
        return Err("no input: give a mesh, a --prop, a --splat, or a combination".into());
    }
    if args.input.is_none() && args.prop.is_none() && args.crop.is_none() {
        // Nothing else says where the thing ends. A splat-only object with
        // no crop is a whole room being called a mug.
        return Err(
            "a splat-only object needs --crop: without a mesh there is nothing \
             else that says where the thing stops and the room starts. \
             Run --inspect to see the bounds and pick a box."
                .into(),
        );
    }
    if (args.input.is_some() || args.prop.is_some())
        && args.mass.is_none()
        && args.density.is_none()
    {
        return Err(
            "give --mass (weigh it) or --density (state the assumption). \
             A scan does not carry mass and this tool will not invent one."
                .into(),
        );
    }
    if args.pieces == 0 {
        return Err("--pieces must be at least 1".into());
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("objbake: {msg}");
            }
            eprintln!("{}", usage());
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("objbake: {e}");
            ExitCode::FAILURE
        }
    }
}

/// One of the measured garage props, as a surface in the object frame.
///
/// `kosm_scan::props` models these from the real objects' measured dimensions
/// — that is the whole point of the module — and bakes them as analytic
/// solids. Meshing the field with the same surface-nets pass `mapbake
/// --garage` uses gives the object pipeline something to decompose, and gives
/// a room something to have in it before anything has been scanned.
///
/// The **mass is still the caller's**. Geometry can be modelled from a tape
/// measure; weight cannot, and a plyo box that is hollow plywood and one that
/// is solid oak have the same outline.
fn prop_mesh(name: &str) -> Result<TriMesh, MapError> {
    use kosm_scan::props::{self, Solid};
    let zero = Vec3::zeros();
    // Sized to the prop, at the garage cell — the resolution the mat's
    // 1-inch lip forced, and small enough that a 40 cm bucket's wall
    // survives meshing.
    let (solid, hi): (Solid, Vec3) = match name {
        "bucket" => (props::bucket(zero), Vec3::new(0.22, 0.22, 0.42)),
        "plyo" => (props::plyo_box(zero, 0.508, 0.0), Vec3::new(0.45, 0.45, 0.56)),
        "bench" => (props::bench(zero, 0.0), Vec3::new(0.75, 0.35, 0.50)),
        "mat" => (props::mat(zero, 0.6, 0.0), Vec3::new(0.35, 0.35, 0.06)),
        "ladder" => (props::ladder_flat(zero, 1.8, 0.0), Vec3::new(1.0, 0.35, 0.08)),
        other => {
            return Err(MapError::Manifest(format!(
                "unknown prop {other} — try bucket, plyo, bench, mat, or ladder"
            )));
        }
    };
    let lo = Vec3::new(-hi.x, -hi.y, -0.03);
    eprintln!("objbake: meshing the measured {name} at {} m cells...", props::GARAGE_CELL);
    let sdf = solid.bake(lo, hi, props::GARAGE_CELL);
    let soup = kosm_scan::FusedGrid::from_sdf(&sdf).extract_mesh();
    let soup: Vec<stl::SoupTri> = soup
        .iter()
        .map(|t| {
            [
                [t[0].x as f32, t[0].y as f32, t[0].z as f32],
                [t[1].x as f32, t[1].y as f32, t[1].z as f32],
                [t[2].x as f32, t[2].y as f32, t[2].z as f32],
            ]
        })
        .collect();
    Ok(TriMesh::from_soup(&soup))
}

/// Read a mesh into the up-rotated capture frame.
fn load_rotated(path: &Path, up: stl::UpAxis) -> Result<TriMesh, MapError> {
    let soup = stl::read_binary_stl(path)?;
    let rotated: Vec<[Vec3; 3]> = soup
        .iter()
        .map(|t| {
            let mut out = [Vec3::zeros(); 3];
            for (k, v) in t.iter().enumerate() {
                out[k] = up.apply(*v);
            }
            out
        })
        .collect();
    // Back through the soup welder so the mesh carries the adjacency the
    // volume integral and the decomposition both read.
    let soup: Vec<stl::SoupTri> = rotated
        .iter()
        .map(|t| {
            [
                [t[0].x as f32, t[0].y as f32, t[0].z as f32],
                [t[1].x as f32, t[1].y as f32, t[1].z as f32],
                [t[2].x as f32, t[2].y as f32, t[2].z as f32],
            ]
        })
        .collect();
    Ok(TriMesh::from_soup(&soup))
}

/// Triangles whose centroid is inside the box, welded again.
fn crop_mesh(mesh: &TriMesh, lo: Vec3, hi: Vec3) -> TriMesh {
    let inside = |p: Vec3| {
        p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z
    };
    let soup: Vec<stl::SoupTri> = mesh
        .triangles
        .iter()
        .filter_map(|t| {
            let v = [
                mesh.vertices[t[0] as usize],
                mesh.vertices[t[1] as usize],
                mesh.vertices[t[2] as usize],
            ];
            let c = (v[0] + v[1] + v[2]) / 3.0;
            inside(c).then(|| {
                [
                    [v[0].x as f32, v[0].y as f32, v[0].z as f32],
                    [v[1].x as f32, v[1].y as f32, v[1].z as f32],
                    [v[2].x as f32, v[2].y as f32, v[2].z as f32],
                ]
            })
        })
        .collect();
    TriMesh::from_soup(&soup)
}

/// How many edges the surface leaves open. Zero means closed, and a closed
/// surface is the only kind whose enclosed volume means anything.
///
/// Reported rather than enforced: scans are never watertight, and refusing
/// them would refuse the whole point. What it changes is *which* volume the
/// bake trusts — see [`run`].
fn open_edges(mesh: &TriMesh) -> usize {
    let mut count: std::collections::HashMap<(u32, u32), i32> = std::collections::HashMap::new();
    for t in &mesh.triangles {
        for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (key, dir) = if e.0 < e.1 { (e, 1) } else { ((e.1, e.0), -1) };
            *count.entry(key).or_insert(0) += dir;
        }
    }
    count.values().filter(|c| **c != 0).count()
}

/// Bounds, and the low percentile of height that counts as the base.
///
/// The 1st percentile, not the minimum, for the reason the map importer uses
/// it on a floor: a scan has stray vertices below the object, and a base set
/// by the lowest of them hangs the thing in the air by however far the worst
/// artefact reaches.
fn bounds_and_base(vertices: &[Vec3]) -> (Vec3, Vec3, f64) {
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for v in vertices {
        lo = lo.component_min(*v);
        hi = hi.component_max(*v);
    }
    let mut zs: Vec<f64> = vertices.iter().map(|v| v.z).collect();
    zs.sort_by(f64::total_cmp);
    let base = zs.get(zs.len() / 100).copied().unwrap_or(lo.z);
    (lo, hi, base)
}

fn run(args: Args) -> Result<(), MapError> {
    let mesh = match (&args.input, &args.prop) {
        (Some(p), _) => Some(load_rotated(p, args.up)?),
        (None, Some(name)) => Some(prop_mesh(name)?),
        (None, None) => None,
    };

    if args.inspect {
        return inspect(&args, mesh.as_ref());
    }

    let name = args
        .name
        .clone()
        .or_else(|| args.prop.clone())
        .or_else(|| {
            args.out
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "object".into());

    // ── the crop box, in the up-rotated capture frame ──
    let (crop_lo, crop_hi) = match (args.crop, &mesh) {
        (Some(b), _) => b,
        (None, Some(m)) => {
            let (lo, hi, _) = bounds_and_base(&m.vertices);
            (lo - Vec3::splat(args.margin), hi + Vec3::splat(args.margin))
        }
        (None, None) => unreachable!("parse_args refuses a splat-only object with no crop"),
    };

    // ── the object frame ──
    let mesh = mesh.map(|m| crop_mesh(&m, crop_lo, crop_hi));
    let shift = match &mesh {
        Some(m) => {
            if m.triangles.is_empty() {
                return Err(MapError::Manifest(
                    "the crop box kept no triangles — check --crop against --inspect".into(),
                ));
            }
            let (lo, hi, base) = bounds_and_base(&m.vertices);
            Vec3::new(-(lo.x + hi.x) * 0.5, -(lo.y + hi.y) * 0.5, -base)
        }
        // Splat-only: the crop box is all there is, so its floor and its
        // middle are the frame. Stated here rather than measured from the
        // gaussians because a splat's stray outliers are exactly what the
        // box was drawn to exclude.
        None => Vec3::new(
            -(crop_lo.x + crop_hi.x) * 0.5,
            -(crop_lo.y + crop_hi.y) * 0.5,
            -crop_lo.z,
        ),
    };
    let to_object = |p: Vec3| p + shift;

    std::fs::create_dir_all(&args.out).map_err(|e| MapError::Io(args.out.clone(), e))?;

    // ── physics ──
    let mut physics = None;
    let mut extent = None;
    if let Some(m) = &mesh {
        let vertices: Vec<Vec3> = m.vertices.iter().map(|p| to_object(*p)).collect();
        let faces = m.triangles.clone();

        let holes = open_edges(m);
        let surface_volume = hull::mass_properties(&vertices, &faces, 1.0).volume.abs();
        if holes > 0 {
            eprintln!(
                "objbake: the surface is open ({holes} boundary edges) — normal for a scan. \
                 Volume and mass come from the convex pieces, which are closed."
            );
        }

        eprintln!("objbake: decomposing {} triangles...", faces.len());
        let pieces = hull::decompose(&vertices, &faces, args.pieces, args.tolerance);
        if pieces.is_empty() {
            return Err(MapError::Manifest(
                "no convex pieces came out — the mesh is degenerate or empty".into(),
            ));
        }
        let unit = hull::compound_mass_properties(&pieces, 1.0);
        if unit.volume <= 0.0 {
            return Err(MapError::Manifest(format!(
                "the pieces enclose {} m3 — nothing can be simulated with that",
                unit.volume
            )));
        }
        // One density, whichever end it came from, then everything scales.
        let (mass, density, source) = match (args.mass, args.density) {
            (Some(m), _) => (m, m / unit.volume, "measured"),
            (None, Some(d)) => (d * unit.volume, d, "density"),
            (None, None) => unreachable!("parse_args requires one of them"),
        };
        let scale = mass / unit.volume;
        let i = unit.inertia * scale;
        let err = if holes == 0 {
            hull::hull_error(unit.volume, surface_volume)
        } else {
            // An open surface has no volume to be wrong about. Comparing the
            // pieces against a garbage number and printing the result would
            // be worse than saying nothing.
            0.0
        };

        write_object_mesh(&args.out.join("mesh.stl"), &vertices, &faces)?;
        kosm_scan::object::write_hulls(&args.out.join("hulls.bin"), &pieces)?;

        let (lo, hi, _) = bounds_and_base(&vertices);
        extent = Some(Extent {
            lo: [lo.x, lo.y, lo.z],
            hi: [hi.x, hi.y, hi.z],
        });
        eprintln!(
            "objbake: {} pieces, {:.2} L, {:.3} kg at {:.0} kg/m3 ({source}), \
             com {:.3} {:.3} {:.3}, {:.0}% of the pieces is imagination",
            pieces.len(),
            unit.volume * 1000.0,
            mass,
            density,
            unit.com.x,
            unit.com.y,
            unit.com.z,
            err * 100.0
        );
        physics = Some(Physics {
            mass,
            mass_source: source.into(),
            density: Some(density),
            volume: unit.volume,
            com: [unit.com.x, unit.com.y, unit.com.z],
            inertia: [
                i.get(0, 0),
                i.get(1, 1),
                i.get(2, 2),
                i.get(0, 1),
                i.get(1, 2),
                i.get(2, 0),
            ],
            hulls: "hulls.bin".into(),
            pieces: pieces.len(),
            hull_error: err,
            friction: args.friction,
            restitution: args.restitution,
        });
    }

    // ── appearance ──
    let mut splat_layer = None;
    if let Some(src) = &args.splat {
        stl::sniff_splat_ply(src)?;
        let dst = args.out.join("splat.ply");
        let up = args.up;
        let (lo, hi) = (crop_lo, crop_hi);
        let (kept, total) = stl::crop_splat_ply(src, &dst, |p| {
            let q = up.apply(p);
            q.x >= lo.x && q.x <= hi.x && q.y >= lo.y && q.y <= hi.y && q.z >= lo.z && q.z <= hi.z
        })?;
        if kept == 0 {
            let _ = std::fs::remove_file(&dst);
            return Err(MapError::Manifest(
                "the crop box kept no gaussians — is --up right? Try --inspect.".into(),
            ));
        }
        eprintln!(
            "objbake: splat cropped to {kept} of {total} gaussians ({:.1}%)",
            kept as f64 / total as f64 * 100.0
        );
        splat_layer = Some(SplatLayer { path: "splat.ply".into() });
    }

    let manifest = ObjectManifest {
        name: name.clone(),
        splat: splat_layer,
        visual: mesh
            .as_ref()
            .map(|_| VisualLayer { mesh: "mesh.stl".into() }),
        physics,
        rig: None,
        // The transform a renderer must apply to bring the *uncropped-frame*
        // splat into the object frame: the same rotation, then the same
        // shift, as the mesh got. Recorded rather than applied for the reason
        // in `kosm_scan::object` — rotating gaussians is a renderer's job.
        align: Some(Align {
            rotate: args.up.manifest_name().into(),
            translate: [shift.x, shift.y, shift.z],
        }),
        extent,
        provenance: Some(Provenance {
            captured: None,
            device: None,
            notes: Some(match (&args.input, &args.prop) {
                (_, Some(p)) => format!("objbake --prop {p} (measured dimensions, not a scan)"),
                (Some(p), _) => format!("objbake from {}", p.display()),
                _ => args
                    .splat
                    .as_ref()
                    .map(|p| format!("objbake from {}", p.display()))
                    .unwrap_or_default(),
            }),
        }),
    };
    manifest.save(&args.out)?;

    // Load it back before claiming success: everything this tool wrote has
    // to satisfy the reader every consumer uses, and finding out otherwise
    // at the start of a rollout is worse by an hour.
    let obj = Object::load(&args.out)?;
    eprintln!(
        "objbake: wrote {} — {} tall, {} pieces{}",
        args.out.display(),
        format_args!("{:.3} m", obj.height()),
        obj.hulls.len(),
        obj.splat_path().map(|_| ", splat present").unwrap_or_default()
    );
    Ok(())
}

/// Write the object-frame surface as binary STL.
fn write_object_mesh(path: &Path, vertices: &[Vec3], faces: &[[u32; 3]]) -> Result<(), MapError> {
    let tris: Vec<[Vec3; 3]> = faces
        .iter()
        .map(|f| {
            [
                vertices[f[0] as usize],
                vertices[f[1] as usize],
                vertices[f[2] as usize],
            ]
        })
        .collect();
    stl::write_binary_stl(path, &tris)
}

/// Report the frame instead of baking it.
///
/// Choosing a crop box means knowing the numbers, and the alternative to
/// printing them is a person opening the scan in another application and
/// typing what they see there into this one.
fn inspect(args: &Args, mesh: Option<&TriMesh>) -> Result<(), MapError> {
    if let Some(m) = mesh {
        let (lo, hi, base) = bounds_and_base(&m.vertices);
        let holes = open_edges(m);
        println!("mesh — {} triangles, up = {}", m.triangles.len(), args.up.manifest_name());
        println!("  bounds  {lo:?}");
        println!("       -> {hi:?}");
        println!("  base (1st pct of height)  {base:.4}");
        println!(
            "  surface is {}",
            if holes == 0 {
                "closed — its enclosed volume is meaningful".to_string()
            } else {
                format!("open, {holes} boundary edges — volume comes from the pieces")
            }
        );
        println!(
            "  --crop {:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            lo.x - args.margin,
            lo.y - args.margin,
            lo.z - args.margin,
            hi.x + args.margin,
            hi.y + args.margin,
            hi.z + args.margin
        );
    }
    if let Some(p) = &args.splat {
        stl::sniff_splat_ply(p)?;
        let points = stl::read_splat_positions(p)?;
        println!("splat — {} gaussians", points.len());
        for up in [stl::UpAxis::YUp, stl::UpAxis::YDown, stl::UpAxis::ZUp] {
            let mut lo = Vec3::splat(f64::INFINITY);
            let mut hi = Vec3::splat(f64::NEG_INFINITY);
            for q in points.iter().map(|p| up.apply(*p)) {
                lo = lo.component_min(q);
                hi = hi.component_max(q);
            }
            println!(
                "  {:>14}: {:.2}..{:.2} x  {:.2}..{:.2} y  {:.2}..{:.2} z",
                up.manifest_name(),
                lo.x,
                hi.x,
                lo.y,
                hi.y,
                lo.z,
                hi.z
            );
        }
        if let Some((lo, hi)) = args.crop {
            let up = args.up;
            let kept = points
                .iter()
                .filter(|p| {
                    let q = up.apply(**p);
                    q.x >= lo.x
                        && q.x <= hi.x
                        && q.y >= lo.y
                        && q.y <= hi.y
                        && q.z >= lo.z
                        && q.z <= hi.z
                })
                .count();
            println!(
                "  --crop keeps {kept} of {} ({:.1}%) with --up {}",
                points.len(),
                kept as f64 / points.len() as f64 * 100.0,
                args.up.manifest_name()
            );
        }
    }
    Ok(())
}
