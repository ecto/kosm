//! Turn a room journal's `.derived` directory into a map directory.
//!
//! ```text
//! roombake <journal>.derived --out maps/garage [--cell 0.02] [--min-views 3]
//!          [--trunc 3] [--max-depth 8] [--pad 0.3] [--scale <m/unit>]
//!          [--journal <dir>] [--depth-scale <k>] [--no-calibrate] [--name garage]
//! ```
//!
//! Reads the per-view VGGT geometry `scripts/room_fuse.py` saved
//! (`extri/intri/depth/floor.npy`, `report.json`), TSDF-fuses every view
//! through its solved pose into the floor frame (z-up, metres, floor at
//! `z = 0`), keeps only voxels seen from `--min-views` views (the
//! persistence vote that deletes movers and depth noise), meshes the field
//! by the crate's surface-nets pass, freezes the field into `sdf.bin`, and
//! writes `map.toml`. Then loads the result back and prints where the floor
//! and ceiling came out — the acceptance numbers, so a bad frame is visible
//! at bake time rather than at the first fall.
//!
//! Two calibrations happen on the way, both against the geometry actually
//! fused (the depth head), because `floor.npy` and the scale were fit
//! upstream on other heads: the depth is anchored to the robot's measured
//! camera height when the journal is at hand (`<journal>` next to
//! `<journal>.derived`, or `--journal`), and the floor is re-zeroed on the
//! depth cloud's lowest slab. Both are printed and recorded in `map.toml`.
//!
//! No inference here: VGGT ran once, upstream, and this consumes what it
//! left. Re-running with different `--cell`/`--min-views` costs seconds.

use std::path::PathBuf;
use std::process::ExitCode;

use kosm_scan::manifest::{Align, CollisionLayer, MapManifest, Provenance};
use kosm_scan::{paint, room, MapError, RoomBakeParams, RoomDerived, TriMesh, stl};
use phyz_math::Vec3;

struct Args {
    derived: PathBuf,
    out: PathBuf,
    params: RoomBakeParams,
    scale: Option<f64>,
    /// The journal directory, for measured camera heights. `None` = look
    /// for `<derived minus .derived>`.
    journal: Option<PathBuf>,
    /// Manual depth multiplier instead of the kinematic one.
    depth_scale: Option<f64>,
    /// Skip both the depth anchor and the floor re-zero.
    calibrate: bool,
    name: Option<String>,
    /// Coloured cloud to paint the mesh from. `None` = `<derived>/cloud.ply`
    /// when it exists; painting is skipped when it does not.
    colour: Option<PathBuf>,
    /// Skip painting even when a cloud is sitting right there.
    paint: bool,
    /// Search radius for a vertex's colour, metres. See `--colour-radius`.
    paint_radius: f64,
}

fn usage() -> &'static str {
    "usage: roombake <journal>.derived --out <dir> [--cell 0.02] [--min-views 3]\n\
     \x20        [--trunc 3] [--max-depth 8] [--pad 0.3] [--scale <m/unit>] [--name <name>]\n\
     \n\
     --cell        voxel / SDF spacing, metres (default 0.02)\n\
     --min-views   a voxel must be seen from this many views to survive (default 3)\n\
     --trunc       TSDF truncation band, in cells (default 3)\n\
     --max-depth   ignore depth beyond this many metres (default 8)\n\
     --min-depth   ignore depth nearer than this many metres (default 0.5;\n\
     \x20            the robot's own torso and near hallucinations)\n\
     --pad         volume padding around the cloud, metres (default 0.3)\n\
     --scale       metres per VGGT unit (default: report.json scale_m_per_unit)\n\
     --journal     the journal dir, for measured camera heights (default: the\n\
     \x20            .derived path minus its suffix, if that exists)\n\
     --colour      coloured cloud to paint the mesh from (default:\n\
     \x20            <derived>/cloud.ply when it exists)\n\
     --no-colour   bake geometry only, leaving the mesh unpainted\n\
     --colour-radius  how far a vertex looks for colour, metres (default 0.10;\n\
     \x20            measured on the garage: 6 cm paints 77% of vertices, 10 cm\n\
     \x20            90%, 15 cm 96% — and a wider radius averages across object\n\
     \x20            edges, so this is matched to what a 64 px sensor resolves\n\
     \x20            at room distance, roughly 6 cm per pixel at 2 m)\n\
     --depth-scale depth multiplier, instead of anchoring to camera heights\n\
     --no-calibrate leave depth and floor exactly as upstream fit them\n\
     --name        map name for the manifest (default: the out dir's name)"
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut derived = None;
    let mut out = None;
    let mut params = RoomBakeParams::default();
    let mut scale = None;
    let mut journal = None;
    let mut depth_scale = None;
    let mut calibrate = true;
    let mut name = None;
    let mut colour = None;
    let mut do_paint = true;
    let mut paint_radius = 0.10;
    let value = |args: &mut dyn Iterator<Item = String>, flag: &str| {
        args.next().ok_or(format!("{flag} needs a value"))
    };
    let num = |args: &mut dyn Iterator<Item = String>, flag: &str| -> Result<f64, String> {
        value(args, flag)?.parse().map_err(|e| format!("{flag}: {e}"))
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(value(&mut args, "--out")?)),
            "--cell" => params.cell = num(&mut args, "--cell")?,
            "--min-views" => params.min_views = num(&mut args, "--min-views")? as u32,
            "--trunc" => params.trunc_cells = num(&mut args, "--trunc")?,
            "--max-depth" => params.max_depth_m = num(&mut args, "--max-depth")?,
            "--min-depth" => params.min_depth_m = num(&mut args, "--min-depth")?,
            "--pad" => params.pad = num(&mut args, "--pad")?,
            "--scale" => scale = Some(num(&mut args, "--scale")?),
            "--journal" => journal = Some(PathBuf::from(value(&mut args, "--journal")?)),
            "--depth-scale" => depth_scale = Some(num(&mut args, "--depth-scale")?),
            "--no-calibrate" => calibrate = false,
            "--name" => name = Some(value(&mut args, "--name")?),
            "--colour" | "--color" => colour = Some(PathBuf::from(value(&mut args, "--colour")?)),
            "--no-colour" | "--no-color" => do_paint = false,
            "--colour-radius" | "--color-radius" => {
                paint_radius = num(&mut args, "--colour-radius")?
            }
            "--help" | "-h" => return Err(String::new()),
            other if derived.is_none() && !other.starts_with('-') => {
                derived = Some(PathBuf::from(other))
            }
            other => return Err(format!("unrecognized argument {other}")),
        }
    }
    if !(params.cell > 0.0) {
        return Err("--cell must be positive".into());
    }
    if params.min_views == 0 {
        return Err("--min-views must be at least 1".into());
    }
    Ok(Args {
        derived: derived.ok_or("no .derived directory given")?,
        out: out.ok_or("--out is required")?,
        params,
        scale,
        journal,
        depth_scale,
        calibrate,
        name,
        colour,
        paint: do_paint,
        paint_radius,
    })
}

/// Paint the mesh from the capture's colour cloud, returning the manifest
/// entry for the file it wrote.
///
/// Everything here is optional by design. A map with no cloud, or with one
/// this cannot read, is still a complete map — it just renders in whatever
/// flat colour its consumer chooses, exactly as every map did before. So the
/// failures are reported to stderr and swallowed, not propagated.
fn paint_mesh(
    args: &Args,
    derived: &RoomDerived,
    tris: &[[Vec3; 3]],
) -> Result<Option<String>, String> {
    if !args.paint {
        return Ok(None);
    }
    let cloud_path = args
        .colour
        .clone()
        .unwrap_or_else(|| args.derived.join("cloud.ply"));
    if !cloud_path.exists() {
        if args.colour.is_some() {
            return Err(format!("{}: no such file", cloud_path.display()));
        }
        eprintln!("colour: no cloud.ply beside the geometry — mesh left unpainted");
        return Ok(None);
    }
    let mut cloud = match paint::read_colour_cloud(&cloud_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("colour: {e} — mesh left unpainted");
            return Ok(None);
        }
    };
    // Into the map frame. `room_fuse.py` calls this cloud "metric, floor
    // frame" in its header; it is metric but **not** floor-rotated — its
    // vertical axis is still VGGT's, so the room's 2.6 m height shows up in
    // y. Painting without this hit 7% of vertices, which is what a frame
    // mismatch looks like when nothing checks. Scale is 1.0 because the
    // points already carry metres; only the rotation and floor offset are
    // owed. The depth-scale calibration this bake applies (~1.6%) predates
    // the cloud, which is part of why the search radius is centimetres.
    for p in &mut cloud.points {
        *p = derived.floor.to_map(*p, 1.0);
    }
    let colours = paint::paint_soup(tris, &cloud, args.paint_radius);
    let frac = paint::painted_fraction(&colours);
    // A cloud that covers the room paints nearly all of it. A low number here
    // is not "a few gaps": it is the two inputs sitting in different frames,
    // which is exactly the failure that shipped a 7%-painted mesh once.
    if frac < 0.5 {
        eprintln!(
            "colour: WARNING only {:.1}% painted — check that the cloud and the \
             mesh share a frame, or raise --colour-radius",
            100.0 * frac
        );
    }
    paint::write_vertex_colours(&args.out.join("mesh_rgb.bin"), &colours)
        .map_err(|e| e.to_string())?;
    eprintln!(
        "colour: {} points from {}, {:.1}% of {} vertices painted within {:.0} cm",
        cloud.points.len(),
        cloud_path.display(),
        100.0 * frac,
        colours.len(),
        args.paint_radius * 100.0
    );
    Ok(Some("mesh_rgb.bin".into()))
}

fn run(args: Args) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let mut derived = RoomDerived::load(&args.derived, args.scale).map_err(|e| e.to_string())?;
    let v0 = &derived.views[0];
    eprintln!(
        "{}: {} views at {}x{}, scale {:.4} m/unit, floor d {:.3} m ({:.1} s)",
        args.derived.display(),
        derived.views.len(),
        v0.width,
        v0.height,
        derived.scale,
        derived.floor.d,
        t0.elapsed().as_secs_f64()
    );

    let p = &args.params;

    // Calibrate against the depth head, which is what gets fused.
    let mut floor_shift = 0.0;
    if args.calibrate {
        let d_before = derived.floor.d;
        match args.depth_scale {
            Some(k) => {
                derived.depth_scale = k;
                eprintln!("depth scale {k:.4} (manual)");
            }
            None => {
                let journal = args.journal.clone().or_else(|| {
                    let s = args.derived.to_string_lossy();
                    s.strip_suffix(".derived").map(PathBuf::from).filter(|p| p.is_dir())
                });
                match journal {
                    Some(j) => {
                        let n = derived
                            .attach_journal(&j, &args.derived)
                            .map_err(|e| format!("{}: {e}", j.display()))?;
                        match derived.calibrate_depth(p) {
                            Some(k) => eprintln!(
                                "depth scale {k:.4}: anchored to measured head height over {n} views \
                                 (from {})",
                                j.display()
                            ),
                            None => eprintln!(
                                "depth scale left at 1: {n} views with a measured head height, \
                                 no usable floor"
                            ),
                        }
                    }
                    None => eprintln!(
                        "depth scale left at 1: no journal found next to {} (pass --journal)",
                        args.derived.display()
                    ),
                }
            }
        }
        match derived.refit_floor(p) {
            Some(_) => {
                floor_shift = derived.floor.d - d_before;
                eprintln!("floor re-zeroed on the depth cloud: shifted {floor_shift:+.3} m");
            }
            None => eprintln!("floor not re-zeroed: no dense slab near z = 0"),
        }
    }

    let (lo, hi) = derived.bounds(&args.params);
    let n = |a: f64, b: f64| ((b - a) / p.cell).ceil() as usize + 1;
    eprintln!(
        "volume [{:.2} {:.2} {:.2}] .. [{:.2} {:.2} {:.2}] m = {}x{}x{} at {} m ({:.1}M voxels)",
        lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,
        n(lo.x, hi.x), n(lo.y, hi.y), n(lo.z, hi.z), p.cell,
        (n(lo.x, hi.x) * n(lo.y, hi.y) * n(lo.z, hi.z)) as f64 / 1e6
    );

    let t1 = std::time::Instant::now();
    let grid = derived.bake_in(lo, hi, p);
    let seen = grid.weight.iter().filter(|&&w| w > 0.0).count();
    eprintln!(
        "fused {} views in {:.1} s; {:.1}% of voxels persist at >= {} views",
        derived.views.len(),
        t1.elapsed().as_secs_f64(),
        100.0 * seen as f64 / grid.weight.len() as f64,
        p.min_views
    );

    let tris = grid.extract_mesh();
    if tris.is_empty() {
        return Err("fusion meshed to nothing — check --scale / --min-views".into());
    }
    let soup: Vec<stl::SoupTri> = tris
        .iter()
        .map(|t| t.map(|v| [v.x as f32, v.y as f32, v.z as f32]))
        .collect();
    let welded = TriMesh::from_soup(&soup);
    let levels = room::surface_levels(&tris);
    eprintln!(
        "mesh: {} triangles, {} vertices; floor z = {}, ceiling z = {}, floor area {:.1} m²",
        tris.len(),
        welded.vertices.len(),
        levels.floor_z.map(|z| format!("{z:+.3} m")).unwrap_or("none".into()),
        levels.ceiling_z.map(|z| format!("{z:.2} m")).unwrap_or("none".into()),
        levels.floor_area
    );

    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    stl::write_binary_stl(&args.out.join("mesh.stl"), &tris).map_err(|e| e.to_string())?;

    // Colour, when the capture left a cloud to take it from. Geometry is the
    // vote's answer and appearance is a lookup against it, so a failure here
    // must not cost the map: anything unreadable is reported and skipped.
    let colour_file = paint_mesh(&args, &derived, &tris)?;

    let sdf = grid.into_sdf();
    sdf.save(&args.out.join("sdf.bin")).map_err(|e| e.to_string())?;
    eprintln!(
        "sdf: {}x{}x{} ({:.1} MB), truncation {:.3} m",
        sdf.nx,
        sdf.ny,
        sdf.nz,
        (sdf.data.len() * 4) as f64 / 1e6,
        grid.truncation
    );

    let name = args.name.clone().unwrap_or_else(|| {
        args.out
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "room".into())
    });
    let manifest = MapManifest {
        name,
        splat: None,
        collision: Some(CollisionLayer {
            mesh: "mesh.stl".into(),
            sdf: "sdf.bin".into(),
            cell: Some(p.cell),
            colour: colour_file,
        }),
        align: Some(Align {
            rotate: "floor-frame".into(),
            translate: [0.0, 0.0, derived.floor.d],
        }),
        provenance: Some(Provenance {
            captured: None,
            device: Some("K1 head stereo via VGGT (room_fuse.py)".into()),
            notes: Some(format!(
                "roombake from {}: {} views, cell {} m, min-views {}, trunc {} cells, \
                 depth {}..{} m, pose scale {:.4} m/unit, depth scale {:.4}, floor shift \
                 {:+.3} m; floor normal (unit frame) [{:.4}, {:.4}, {:.4}]",
                args.derived.display(),
                derived.views.len(),
                p.cell,
                p.min_views,
                p.trunc_cells,
                p.min_depth_m,
                p.max_depth_m,
                derived.scale,
                derived.depth_scale,
                floor_shift,
                derived.floor.normal.x,
                derived.floor.normal.y,
                derived.floor.normal.z,
            )),
        }),
        extent: None,
    };
    manifest.save(&args.out).map_err(|e| e.to_string())?;

    // Load back what was just written and stand on it.
    let map = kosm_scan::Map::load(&args.out).map_err(|e: MapError| format!("verify: {e}"))?;
    let sdf = map.standable().map_err(|e| format!("verify: {e}"))?;
    if let Some(fz) = levels.floor_z {
        // A point 5 cm above the fused floor, at the cloud's centre, must
        // read ~5 cm.
        let c = (lo + hi) * 0.5;
        let probe = Vec3::new(c.x, c.y, fz + 0.05);
        match sdf.sample(probe) {
            Some(d) => eprintln!("probe {:.2} {:.2} {:.2}: sdf {d:+.3} m", probe.x, probe.y, probe.z),
            None => eprintln!("probe {:?} is outside the grid", probe),
        }
    }
    eprintln!("map written to {} in {:.1} s", args.out.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(args) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(msg) => {
                eprintln!("roombake: {msg}");
                ExitCode::FAILURE
            }
        },
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("roombake: {msg}\n");
            }
            eprintln!("{}", usage());
            ExitCode::FAILURE
        }
    }
}
