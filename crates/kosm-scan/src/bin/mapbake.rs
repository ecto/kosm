//! Turn a phone scan into a map directory.
//!
//! ```text
//! mapbake scan.stl --out maps/lab [--cell 0.02] [--pad 0.5] [--up y]
//!         [--set-floor] [--splat splat.ply] [--name lab]
//! ```
//!
//! Reads the scan mesh, optionally rotates y-up → z-up and shifts the
//! detected floor to `z = 0` (the frame contract the sim's spawn logic
//! assumes), writes the *aligned* mesh, the baked SDF, and a `map.toml` that
//! records exactly what was applied — because the splat trains in the raw
//! capture frame and must be bent by the same transform later.
//!
//! Floor detection: area-weighted histogram (1 cm bins) of the heights of
//! upward-facing triangles (`n_z > 0.8` after rotation); the floor is the
//! *lowest* bin whose cumulative area reaches 10% of the upward total, not
//! the modal bin — a lab with more desk than visible carpet should still put
//! the carpet, not the desks, at `z = 0`.

use std::path::PathBuf;
use std::process::ExitCode;

use kosm_scan::manifest::{Align, CollisionLayer, MapManifest, Provenance, SplatLayer};
use kosm_scan::{MapError, SdfGrid, TriMesh, stl};
use phyz_math::Vec3;

struct Args {
    /// The scan to bake, or `None` when `--garage` synthesizes one.
    input: Option<PathBuf>,
    /// Build the measured garage scene instead of reading a scan.
    garage: bool,
    out: PathBuf,
    cell: f64,
    pad: f64,
    y_up: bool,
    set_floor: bool,
    splat: Option<PathBuf>,
    name: Option<String>,
    /// Report the frame instead of importing.
    inspect: bool,
    /// Which way is up in the *splat's* frame, or `None` to measure it.
    up: Option<kosm_scan::stl::UpAxis>,
}

fn usage() -> &'static str {
    "usage: mapbake <scan.stl> --out <dir> [--cell 0.02] [--pad 0.5] [--up y] \
     [--set-floor] [--splat <splat.ply>] [--name <name>]\n\
            mapbake --garage --out <dir> [--cell 0.0064] [--name <name>]\n\
     \n\
     --inspect    with --splat: report the frame along each candidate up axis\n\
     \x20            instead of importing, so the choice can be checked\n\
     --garage     synthesize the measured garage scene instead of reading a\n\
     \x20            scan: floor, mats, bench, bucket, plyo box. Something to\n\
     \x20            stand in before the robot has scanned anything.\n\
     --cell       SDF sample spacing, metres (default 0.02)\n\
     --pad        volume padding around the mesh AABB, metres (default 0.5)\n\
     --up         which way is up in the source: y, y-down (or -y), or z.\n\
     \x20            For --splat this is MEASURED when omitted; there is no\n\
     \x20            convention to assume. Use --inspect to see the evidence.\n\
     --set-floor  detect the floor plane and shift it to z = 0\n\
     --splat      Gaussian splat .ply to reference (copied into the map dir)\n\
     --name       map name for the manifest (default: the out dir's name)"
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut input = None;
    let mut garage = false;
    let mut out = None;
    let mut cell = 0.02;
    let mut pad = 0.5;
    let mut y_up = false;
    // Splat imports need three cases, not two: a mesh scan is y-up or z-up,
    // but a trained splat is usually COLMAP's y-DOWN.
    let mut up: Option<kosm_scan::stl::UpAxis> = None;
    let mut inspect = false;
    let mut set_floor = false;
    let mut splat = None;
    let mut name = None;

    let value = |args: &mut dyn Iterator<Item = String>, flag: &str| {
        args.next().ok_or(format!("{flag} needs a value"))
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(value(&mut args, "--out")?)),
            "--cell" => {
                cell = value(&mut args, "--cell")?
                    .parse()
                    .map_err(|e| format!("--cell: {e}"))?
            }
            "--pad" => {
                pad = value(&mut args, "--pad")?.parse().map_err(|e| format!("--pad: {e}"))?
            }
            "--up" => {
                let v = value(&mut args, "--up")?;
                let parsed = kosm_scan::stl::UpAxis::parse(&v)
                    .ok_or(format!("--up {v}: expected y, y-down (or -y), or z"))?;
                y_up = parsed == kosm_scan::stl::UpAxis::YUp;
                up = Some(parsed);
            }
            "--inspect" => inspect = true,
            "--garage" => garage = true,
            "--set-floor" => set_floor = true,
            "--splat" => splat = Some(PathBuf::from(value(&mut args, "--splat")?)),
            "--name" => name = Some(value(&mut args, "--name")?),
            "--help" | "-h" => return Err(String::new()),
            other if input.is_none() && !other.starts_with('-') => {
                input = Some(PathBuf::from(other))
            }
            other => return Err(format!("unrecognized argument {other}")),
        }
    }
    if inspect {
        return Ok(Args { input, garage, out: out.unwrap_or_default(), cell, pad, y_up,
            set_floor, splat, name, up, inspect });
    }
    if !(cell > 0.0) {
        return Err("--cell must be positive".into());
    }
    if input.is_none() && !garage && splat.is_some() {
        // Splat with nothing to bake: an appearance-only map, the CLI
        // counterpart of Dojo's "Add Splat…". Same one function underneath
        // (`manifest::attach_splat`), so the two cannot drift.
        return Ok(Args { input, garage, out: out.ok_or("--out is required")?, cell, pad,
            y_up, set_floor, splat, name, up, inspect });
    }
    if garage && input.is_some() {
        return Err("--garage builds its own scene; do not also pass a scan".into());
    }
    if !garage && input.is_none() {
        return Err("no input mesh given (or pass --garage)".into());
    }
    Ok(Args {
        input,
        garage,
        out: out.ok_or("--out is required")?,
        cell,
        pad,
        y_up,
        set_floor,
        splat,
        name,
        up,
        inspect,
    })
}

/// The y-up → z-up rotation: `(x, y, z) → (x, −z, y)`.
fn y_up_to_z_up(v: Vec3) -> Vec3 {
    Vec3::new(v.x, -v.z, v.y)
}

/// Height of the floor: the lowest 1 cm height bin at which the cumulative
/// area of upward-facing triangles reaches 10% of their total.
fn detect_floor_z(mesh: &TriMesh) -> Option<f64> {
    let mut up: Vec<(f64, f64)> = Vec::new(); // (z of centroid, area)
    for t in &mesh.triangles {
        let [a, b, c] = [
            mesh.vertices[t[0] as usize],
            mesh.vertices[t[1] as usize],
            mesh.vertices[t[2] as usize],
        ];
        let cross = (b - a).cross(c - a);
        let area = 0.5 * cross.norm();
        let Some(n) = cross.try_normalize() else { continue };
        if n.z > 0.8 {
            up.push(((a.z + b.z + c.z) / 3.0, area));
        }
    }
    if up.is_empty() {
        return None;
    }
    let total: f64 = up.iter().map(|(_, a)| a).sum();
    let lo = up.iter().map(|(z, _)| *z).fold(f64::INFINITY, f64::min);
    let hi = up.iter().map(|(z, _)| *z).fold(f64::NEG_INFINITY, f64::max);
    let bins = (((hi - lo) / 0.01).ceil() as usize).max(1);
    let mut hist = vec![0.0f64; bins];
    for (z, a) in &up {
        let b = (((z - lo) / 0.01) as usize).min(bins - 1);
        hist[b] += a;
    }
    let mut acc = 0.0;
    for (b, a) in hist.iter().enumerate() {
        acc += a;
        if acc >= 0.10 * total {
            // Area-weighted mean height within the winning bin, not the bin
            // centre — the centre is off by up to half a bin (5 mm) on a
            // clean scan, and 5 mm of floor error is a real interpenetration.
            let (mut wz, mut w) = (0.0, 0.0);
            for (z, a) in &up {
                if ((((z - lo) / 0.01) as usize).min(bins - 1)) == b {
                    wz += z * a;
                    w += a;
                }
            }
            return (w > 0.0).then(|| wz / w);
        }
    }
    None
}

/// The measured garage, as a triangle soup already in the map frame.
///
/// `kosm_scan::props` builds this scene as analytic solids and bakes it
/// straight to an SDF — which is all a *rollout* needs, and is how
/// `k1_learn_garage` trains. A map directory owes the viewer a surface as
/// well, so the field is meshed by the same surface-nets pass fusion uses
/// (see [`kosm_scan::FusedGrid::from_sdf`]) rather than by a second extractor.
///
/// The props are the ones `k1_learn_garage` documents as measured, placed
/// where a person walking into their own garage would meet them: a mat under
/// the feet, a bench to the side, and two things at ankle height that a
/// flat-ground policy cannot see.
fn garage_soup(cell: f64) -> Vec<[Vec3; 3]> {
    use kosm_scan::props;
    let scene = props::garage(vec![
        props::mat(Vec3::new(0.0, 0.0, 0.0), 0.6, 0.0),
        props::mat(Vec3::new(0.6, 0.0, 0.0), 0.6, 0.0),
        props::bench(Vec3::new(0.3, -1.0, 0.0), 0.3),
        props::bucket(Vec3::new(-0.85, 0.6, 0.0)),
        props::plyo_box(Vec3::new(1.0, 0.9, 0.0), 0.5, 0.0),
    ]);
    // The same volume `k1_learn_garage` trains in, deliberately. A map you
    // can walk around in the viewer and a map the policy was trained against
    // should be the same map; making the explorable one bigger would mean the
    // operator can stand the robot somewhere no rollout has ever been.
    // Off the map there is no floor, by design (`docs/maps.md`).
    let lo = Vec3::new(-1.6, -1.6, -0.15);
    let hi = Vec3::new(1.6, 1.6, 1.0);
    eprintln!("baking the garage scene at {cell} m cells...");
    let sdf = scene.bake(lo, hi, cell);
    kosm_scan::FusedGrid::from_sdf(&sdf).extract_mesh()
}

/// Report what a splat's frame looks like along each candidate up axis.
///
/// There is no field in a `.ply` that says which way is up, so importing one
/// means asserting it — and an assertion nobody checks is how every scene in
/// a library ends up subtly on its side. A room has a signature that settles
/// it: a dense floor slab, a ceiling a plausible height above it, and most of
/// the mass in between. Under the wrong axis that structure smears out.
fn inspect(ply: &std::path::Path) -> Result<(), String> {
    use kosm_scan::stl::UpAxis;
    let points = kosm_scan::stl::read_splat_positions(ply).map_err(|e| e.to_string())?;
    eprintln!("{}: {} gaussians", ply.display(), points.len());
    for up in [UpAxis::YUp, UpAxis::YDown, UpAxis::ZUp] {
        let floor = kosm_scan::stl::floor_height(&points, up);
        let mut z: Vec<f64> = points.iter().map(|p| up.apply(*p).z - floor).collect();
        z.sort_by(|a, b| a.total_cmp(b));
        let at = |f: f64| z[((z.len() - 1) as f64 * f) as usize];
        // How much of the cloud sits in the first 25 cm above the floor. A
        // real floor is a thin dense slab; a smeared axis has almost nothing
        // concentrated there.
        let slab = z.iter().filter(|v| **v > -0.05 && **v < 0.25).count();
        eprintln!(
            "  {:16} floor {:8.2}  p50 {:7.2}  p95 {:7.2}  floor-slab {:5.1}%",
            up.manifest_name(),
            floor,
            at(0.50),
            at(0.95),
            100.0 * slab as f64 / z.len() as f64
        );
    }
    Ok(())
}

fn run(args: Args) -> Result<(), String> {
    if args.inspect {
        let ply = args.splat.as_ref().ok_or("--inspect needs --splat <file>")?;
        return inspect(ply);
    }
    // Appearance only: no scan, no scene, just a splat becoming a place.
    if args.input.is_none() && !args.garage {
        let splat = args.splat.as_ref().ok_or("nothing to bake and no --splat")?;
        let m = kosm_scan::manifest::attach_splat(&args.out, splat, args.name.as_deref(), args.up)
            .map_err(|e| e.to_string())?;
        kosm_scan::Map::load(&args.out).map_err(|e: MapError| format!("verify: {e}"))?;
        eprintln!(
            "appearance-only map {} written to {} — somewhere to look around, \
             nowhere to stand a robot",
            m.name,
            args.out.display()
        );
        return Ok(());
    }
    let soup: Vec<stl::SoupTri> = match &args.input {
        Some(path) => {
            let s = stl::read_binary_stl(path).map_err(|e| e.to_string())?;
            if s.is_empty() {
                return Err(format!("{}: no triangles", path.display()));
            }
            eprintln!("read {} triangles from {}", s.len(), path.display());
            s
        }
        None => {
            let tris = garage_soup(args.cell);
            if tris.is_empty() {
                return Err("the garage scene meshed to nothing — is --cell sane?".into());
            }
            eprintln!("garage: {} triangles", tris.len());
            tris.iter()
                .map(|t| t.map(|v| [v.x as f32, v.y as f32, v.z as f32]))
                .collect()
        }
    };

    // Rotation first.
    let rotated: Vec<[Vec3; 3]> = soup
        .iter()
        .map(|t| {
            t.map(|v| {
                let p = Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
                if args.y_up { y_up_to_z_up(p) } else { p }
            })
        })
        .collect();

    // Weld once (pre-shift) so floor detection sees real adjacency.
    let soup_f32: Vec<stl::SoupTri> = rotated
        .iter()
        .map(|t| t.map(|v| [v.x as f32, v.y as f32, v.z as f32]))
        .collect();
    let mesh = TriMesh::from_soup(&soup_f32);

    // Then the floor shift.
    let mut translate = Vec3::zeros();
    if args.set_floor {
        let floor = detect_floor_z(&mesh)
            .ok_or("--set-floor: no upward-facing triangles to detect a floor from")?;
        translate = Vec3::new(0.0, 0.0, -floor);
        eprintln!("floor detected at z = {floor:.3} m; shifting to z = 0");
    }

    let aligned: Vec<[Vec3; 3]> = rotated.iter().map(|t| t.map(|v| v + translate)).collect();

    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let mesh_path = args.out.join("mesh.stl");
    stl::write_binary_stl(&mesh_path, &aligned).map_err(|e| e.to_string())?;

    let aligned_f32: Vec<stl::SoupTri> = aligned
        .iter()
        .map(|t| t.map(|v| [v.x as f32, v.y as f32, v.z as f32]))
        .collect();
    let welded = TriMesh::from_soup(&aligned_f32);
    let (lo, hi) = welded.aabb();
    eprintln!(
        "baking SDF: bounds [{:.2} {:.2} {:.2}] .. [{:.2} {:.2} {:.2}] m, cell {} m",
        lo.x, lo.y, lo.z, hi.x, hi.y, hi.z, args.cell
    );
    let t0 = std::time::Instant::now();
    let sdf = SdfGrid::bake(&welded, args.cell, args.pad);
    eprintln!(
        "baked {}x{}x{} = {:.1}M samples in {:.1} s ({:.1} MB)",
        sdf.nx,
        sdf.ny,
        sdf.nz,
        sdf.data.len() as f64 / 1e6,
        t0.elapsed().as_secs_f64(),
        (sdf.data.len() * 4) as f64 / 1e6,
    );
    sdf.save(&args.out.join("sdf.bin")).map_err(|e| e.to_string())?;

    // Splat: copy in and validate, or leave the layer absent.
    let splat_layer = match &args.splat {
        Some(src) => {
            stl::sniff_splat_ply(src).map_err(|e| e.to_string())?;
            let dst = args.out.join("splat.ply");
            std::fs::copy(src, &dst).map_err(|e| format!("{}: {e}", dst.display()))?;
            Some(SplatLayer { path: "splat.ply".into() })
        }
        None => None,
    };

    let name = args.name.clone().unwrap_or_else(|| {
        args.out
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "map".into())
    });
    let manifest = MapManifest {
        name,
        splat: splat_layer,
        collision: Some(CollisionLayer {
            mesh: "mesh.stl".into(),
            sdf: "sdf.bin".into(),
            cell: Some(args.cell),
            colour: None,
        }),
        align: Some(Align {
            rotate: if args.y_up { "y-up-to-z-up".into() } else { "none".into() },
            translate: [translate.x, translate.y, translate.z],
        }),
        provenance: Some(Provenance {
            captured: None,
            device: None,
            notes: Some(match &args.input {
                Some(p) => format!("mapbake from {}", p.display()),
                None => "mapbake --garage: the measured garage scene".into(),
            }),
        }),
        // A baked map announces its extent through the bricks it streams.
        extent: None,
    };
    let text = toml::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    std::fs::write(args.out.join("map.toml"), text)
        .map_err(|e| format!("{}: {e}", args.out.display()))?;

    // Load back what was just written — the map a consumer gets is the map
    // that was verified, or the bake fails here and now.
    kosm_scan::Map::load(&args.out).map_err(|e: MapError| format!("verify: {e}"))?;
    eprintln!("map written to {}", args.out.display());
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(args) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(msg) => {
                eprintln!("mapbake: {msg}");
                ExitCode::FAILURE
            }
        },
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("mapbake: {msg}\n");
            }
            eprintln!("{}", usage());
            ExitCode::FAILURE
        }
    }
}
