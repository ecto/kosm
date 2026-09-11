//! The raster tier on a real device, headless.
//!
//! Everything here goes through [`kosm_view::raster::Raster::draw`] and comes
//! back through [`kosm_view::frame::read_back`], which is the same pair the
//! window uses — so a number measured here is a number about the picture the
//! window shows, not about a test harness that resembles it.
//!
//! They run under a plain `cargo test --release -p kosm-view`. A machine with
//! no adapter is handled by [`ctx_or_skip`], which prints the skip and the
//! reason — the convention `motion.rs` next door and `kosm-render`'s own
//! `gpu_*` tests use. Marking them `#[ignore]` as well would mean they did
//! not run on the machines that *can* run them, which is the case that
//! matters.

use kosm_render::gpu::GpuContext;
use kosm_render::math::{Point3, Vec3};
use kosm_view::frame::read_back;
use kosm_view::raster::{
    CausticQuad, Frame, Instance, Kind, Mesh, Raster, Scene, Sun,
    material::from_rgb as rgb_material, probes::uniform as uniform_probes,
};

/// The sun's `x` and `z`, both `1/√2`: 45° up, leaning along +x, so a caster
/// two metres up throws its shadow two metres along -x.
const SUN_X: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn ctx_or_skip(name: &str) -> Option<&'static GpuContext> {
    // wgpu reports a bad pipeline through `log`, and a test with no logger
    // gets a black frame and no reason for it.
    kosm_view::init();
    match GpuContext::init_blocking() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping {name}: no GPU ({e})");
            None
        }
    }
}

/// A unit box centred on the origin, in metres.
fn box_mesh(hx: f64, hy: f64, hz: f64) -> Mesh {
    let mut p = Vec::new();
    for k in 0..8 {
        p.push([
            if k & 1 == 0 { -hx } else { hx },
            if k & 2 == 0 { -hy } else { hy },
            if k & 4 == 0 { -hz } else { hz },
        ]);
    }
    // six faces, two triangles each, wound outward
    let f: [[u32; 4]; 6] = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    let mut idx = Vec::new();
    for q in f {
        idx.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
    }
    // facet normals, not welded: a box's corners are creases
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut flat = Vec::new();
    for t in idx.chunks_exact(3) {
        let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        for q in [a, b, c] {
            flat.push(pos.len() as u32);
            pos.push(q);
            nrm.push([n[0] / l, n[1] / l, n[2] / l]);
        }
    }
    Mesh::from_m(&pos, &nrm, &flat)
}

/// A 20 × 20 m ground quad at `z = 0`, facing up.
fn floor_mesh() -> Mesh {
    let p = [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0]];
    let n = [[0.0, 0.0, 1.0]; 4];
    Mesh::from_m(&p, &n, &[0, 1, 2, 0, 2, 3])
}

/// The scene both tests draw: a white floor under a sky, a sun straight
/// overhead, and (optionally) a box a metre above the floor.
fn scene(with_box: bool) -> Scene {
    let probes = uniform_probes([-12.0, -12.0, -1.0], 4.0, [7, 7, 4], 0.25);
    // A sun at 45°, so the box's shadow falls *beside* it: a sun straight
    // overhead would put the shadow exactly under the caster, where an
    // overhead camera cannot see it.
    let sun = Sun::from_rgb([SUN_X, 0.0, SUN_X], [3.0, 3.0, 3.0], 0.01);
    let (mut s, _by_name) = Scene::new(sun, probes);
    s.bounds = ([-11.0, -11.0, -0.5], [11.0, 11.0, 4.0]);
    s.exposure = 1.0;
    let white = s.push_material(rgb_material([0.8, 0.8, 0.8], 1.0, 0.0));

    let mut floor = floor_mesh();
    floor.instances.push(Instance { material: white, ..Default::default() });
    s.push_mesh(floor);

    if with_box {
        let mut b = box_mesh(1.0, 1.0, 0.25);
        b.instances.push(Instance::rigid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 2.0],
            white,
        ));
        s.push_mesh(b);
    }
    s
}

/// A camera looking straight down at the floor from high up, so the picture
/// is the floor and nothing else.
fn overhead(size: (u32, u32)) -> Frame {
    let cam = kosm_render::Camera::look_at(
        Point3::new(0.0, -0.001, 14.0),
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        40.0,
    );
    Frame::new(cam, size)
}

fn draw(ctx: &'static GpuContext, s: &Scene, f: &Frame) -> image::RgbaImage {
    // A pipeline wgpu refuses draws nothing and says so only through the
    // uncaptured-error handler, which a test does not have. The scope makes
    // it a panic with the validation message in it.
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut r = Raster::new(&ctx.device, &ctx.queue, s, wgpu::TextureFormat::Rgba8Unorm)
        .expect("the raster tier builds");
    let tex = r.draw(&ctx.device, &ctx.queue, s, f);
    let img = read_back(&ctx.device, &ctx.queue, &tex, f.size);
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("wgpu refused the raster tier: {e}");
    }
    img
}

/// Mean luminance of a square of the frame, 0..1.
fn patch(img: &image::RgbaImage, cx: u32, cy: u32, half: u32) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0f64;
    for y in cy.saturating_sub(half)..(cy + half).min(img.height()) {
        for x in cx.saturating_sub(half)..(cx + half).min(img.width()) {
            let p = img.get_pixel(x, y);
            sum += 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
            n += 1.0;
        }
    }
    sum / n.max(1.0) / 255.0
}

/// **The shadow map puts a shadow under a box and none in the open.**
///
/// A sun straight overhead, a box two metres up, a camera looking straight
/// down: the floor under the box must be markedly darker than the floor
/// beside it, and the two must agree once the box is taken away.
#[test]
fn the_shadow_map_darkens_the_floor_under_a_box() {
    let Some(ctx) = ctx_or_skip("the_shadow_map_darkens_the_floor_under_a_box") else { return };
    let size = (256u32, 256u32);
    let lit = draw(ctx, &scene(false), &overhead(size));
    let shadowed = draw(ctx, &scene(true), &overhead(size));

    // The frame: an eye 14 m up under a 40° field is 5.1 m of floor either
    // side of the centre, so a pixel is 40 mm and a metre is 25 px. The box
    // is 2 m across at z = 2, which the camera magnifies to about 29 px of
    // half-width; its shadow is the same square translated 2 m along -x,
    // which is 50 px — clear of the box itself, which is the whole reason the
    // sun is not overhead.
    let (cx, cy) = (size.0 / 2, size.1 / 2);
    let open = (patch(&lit, cx + 100, cy, 8), patch(&shadowed, cx + 100, cy, 8));
    let under = (patch(&lit, cx - 50, cy, 6), patch(&shadowed, cx - 50, cy, 6));

    eprintln!(
        "shadow: open {:.3} → {:.3}, under the box {:.3} → {:.3}",
        open.0, open.1, under.0, under.1
    );
    assert!(
        (open.1 - open.0).abs() < 0.03,
        "the open floor changed when a box was added elsewhere: {:.3} → {:.3}",
        open.0,
        open.1
    );
    assert!(
        under.1 < 0.75 * under.0,
        "the floor under the box is not shadowed: {:.3} against {:.3} in the open",
        under.1,
        under.0
    );
    assert!(under.1 > 0.0, "a shadow is not black — the sky still reaches it");
}

/// **The furnace.** A uniform sky of radiance `L` and a Lambert surface of
/// albedo `a` reflect `a · L` back, whatever the geometry — the closed form
/// every renderer is checked against.
///
/// The raster tier's version: the probes are a white furnace, the sun is off,
/// and the floor's radiance must be `albedo × L` before the tonemap. It is
/// read back *through* the tonemap, so the check is against the same ACES and
/// sRGB curves the shader applies, which is also a check that those two match
/// `Film::to_srgb8`.
#[test]
fn the_furnace_reflects_the_albedo_it_is_given() {
    let Some(ctx) = ctx_or_skip("the_furnace_reflects_the_albedo_it_is_given") else { return };
    let l = 0.6f32;
    let albedo = 0.5f32;
    let probes = uniform_probes([-12.0, -12.0, -1.0], 4.0, [7, 7, 4], l);
    // no sun at all
    let (mut s, _) = Scene::new(Sun::from_rgb([0.0, 0.0, 1.0], [0.0; 3], 0.01), probes);
    s.materials[0] = rgb_material([albedo; 3], 1.0, 0.0);
    s.bounds = ([-11.0, -11.0, -0.5], [11.0, 11.0, 1.0]);
    let m = s.push_material(rgb_material([albedo; 3], 1.0, 0.0));
    let mut floor = floor_mesh();
    floor.instances.push(Instance { material: m, ..Default::default() });
    s.push_mesh(floor);

    let mut f = overhead((128, 128));
    f.exposure = 1.0;
    let img = draw(ctx, &s, &f);

    // what the shader should have written: albedo · L, through ACES and sRGB
    let want = {
        let x = albedo * l;
        let t = ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
        if t <= 0.0031308 { 12.92 * t } else { 1.055 * t.powf(1.0 / 2.4) - 0.055 }
    } as f64;
    let got = patch(&img, 64, 64, 20);
    eprintln!("furnace: {got:.4} against the closed form {want:.4}");
    assert!(
        (got - want).abs() < 0.02,
        "the furnace reflected {got:.4} where a·L is {want:.4}"
    );
}

/// A caustic quad's irradiance lands on the rectangle it was given and
/// nowhere else — the two-dot-product test the fragment shader does.
#[test]
fn a_caustic_quad_brightens_only_its_own_rectangle() {
    let Some(ctx) = ctx_or_skip("a_caustic_quad_brightens_only_its_own_rectangle") else { return };
    let mut s = scene(false);
    s.caustics.push({
        let mut q = CausticQuad::empty((32, 32));
        // a two-metre square in the middle of the floor, carrying 4 units of
        // irradiance everywhere inside it
        q.origin = [-1.0, -1.0, 0.0];
        q.u = [2.0, 0.0, 0.0];
        q.v = [0.0, 2.0, 0.0];
        q.data.fill(4.0);
        q
    });
    let size = (256u32, 256u32);
    let plain = draw(ctx, &scene(false), &overhead(size));
    let lit = draw(ctx, &s, &overhead(size));
    let inside = (patch(&plain, 128, 128, 6), patch(&lit, 128, 128, 6));
    let outside = (patch(&plain, 228, 128, 6), patch(&lit, 228, 128, 6));
    eprintln!(
        "caustic: inside {:.3} → {:.3}, outside {:.3} → {:.3}",
        inside.0, inside.1, outside.0, outside.1
    );
    assert!(inside.1 > inside.0 + 0.05, "the caustic did not land on its rectangle");
    assert!(
        (outside.1 - outside.0).abs() < 0.01,
        "the caustic leaked off its rectangle"
    );
}

/// **The sea is water and not wet sand.** The same lattice, at the same
/// place, drawn once as an ordinary sand surface and once as the sea: the
/// water must come back *bluer* — a larger blue-minus-red — because that is
/// what Beer–Lambert over a metre of sea water does and it is the one thing
/// about this surface that is not a matter of taste.
///
/// Stated as a difference rather than as an absolute colour on purpose. The
/// absolute is a function of the exposure, the sky's radiance and the sun's,
/// and a test that pinned it would fail the first time somebody moved the
/// level's sky; the *ordering* is a fact about the transport.
#[test]
fn the_sea_is_bluer_than_the_sand_under_it() {
    let Some(ctx) = ctx_or_skip("the_sea_is_bluer_than_the_sand_under_it") else { return };
    let build = |as_water: bool| {
        let probes = uniform_probes([-30.0, -60.0, -8.0], 10.0, [7, 7, 3], 0.4);
        let (mut s, by_name) = Scene::new(Sun::from_rgb([0.0, -0.4, 0.9], [3.0; 3], 0.01), probes);
        s.bounds = ([-25.0, -55.0, -8.0], [25.0, 5.0, 3.0]);
        let sand = by_name["dry sand"];
        let water = s.push_material(rgb_material([0.03, 0.30, 0.30], 0.1, 0.5));
        // a steep beach, so twelve metres out is three metres of water and
        // the absorption has a path to work over
        let sea = kosm_view::raster::Sea::cove(0.0, 0.25, 0.0, 40.0).with_materials(sand, water);
        let (p, idx) = sea.lattice(25.0, 32);
        let mut mesh = Mesh::from_m(&p, &[], &idx);
        let kind = if as_water { Kind::Sea } else { Kind::Solid };
        let mat = if as_water { water } else { sand };
        mesh.instances.push(
            Instance { material: mat, ..Default::default() }.with_kind(kind).casting(false),
        );
        s.push_mesh(mesh);
        if as_water {
            s.sea = Some(sea);
        }
        s
    };
    let cam = kosm_render::Camera::look_at(
        Point3::new(0.0, 6.0, 4.0),
        Point3::new(0.0, -12.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        50.0,
    );
    let mut f = Frame::new(cam, (256, 144));
    f.exposure = 0.5;
    let dry = draw(ctx, &build(false), &f);
    let wet = draw(ctx, &build(true), &f);
    let blueness = |img: &image::RgbaImage| {
        let p = img.get_pixel(128, 72);
        (p[2] as f64 - p[0] as f64, p[1] as f64 - p[0] as f64, *p)
    };
    let (db, dg, dp) = blueness(&dry);
    let (wb, wg, wp) = blueness(&wet);
    eprintln!("sea: sand {dp:?} (b−r {db:+.0}, g−r {dg:+.0}); water {wp:?} (b−r {wb:+.0}, g−r {wg:+.0})");
    assert!(
        wb > db + 8.0 && wg > dg + 8.0,
        "the sea is no cooler than the sand under it: {wp:?} against {dp:?}"
    );
}

// ── the datasheet's parity ────────────────────────────────────────────────

/// A UV sphere of radius `r` centred at `c`, metres, with exact normals.
fn sphere_mesh(c: [f64; 3], r: f64, segments: usize, rings: usize) -> Mesh {
    let (mut pos, mut nrm, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..=rings {
        let theta = std::f64::consts::PI * j as f64 / rings as f64;
        for i in 0..=segments {
            let phi = std::f64::consts::TAU * i as f64 / segments as f64;
            let n = [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()];
            nrm.push(n);
            pos.push([c[0] + n[0] * r, c[1] + n[1] * r, c[2] + n[2] * r]);
        }
    }
    let w = segments + 1;
    for j in 0..rings {
        for i in 0..segments {
            let a = (j * w + i) as u32;
            let b = a + w as u32;
            idx.extend_from_slice(&[a, b, b + 1, a, b + 1, a + 1]);
        }
    }
    Mesh::from_m(&pos, &nrm, &idx)
}

/// An axis-aligned box from a corner to a corner, metres.
fn slab(lo: [f64; 3], hi: [f64; 3]) -> Mesh {
    let mut m = box_mesh(
        0.5 * (hi[0] - lo[0]),
        0.5 * (hi[1] - lo[1]),
        0.5 * (hi[2] - lo[2]),
    );
    let c = [
        0.5 * (lo[0] + hi[0]) as f32,
        0.5 * (lo[1] + hi[1]) as f32,
        0.5 * (lo[2] + hi[2]) as f32,
    ];
    for v in &mut m.vertices {
        for k in 0..3 {
            v.pos[k] += c[k];
        }
    }
    m
}

/// **The datasheet ball, on both tiers, under one light.**
///
/// `kosm::material::Ball` is a [`Lens`](kosm::lens::Lens) over the canonical
/// `ball_world` — a 30 mm sphere on a 300 mm plate under `studio_rig` — and
/// it is what every material's datasheet page shows. This draws the same
/// pose, from the same camera, at the same exposure, on the raster tier,
/// lit by `probes::studio_probes()`: the *same rig*, solved once into SH
/// instead of sampled per pixel. What comes out is the number that says
/// whether the two tiers agree about a substance.
///
/// **The tolerances are per material and they are not equal, on purpose.**
///
/// - **Dry sand, porcelain, granite** — opaque dielectrics, mostly Lambert.
///   The raster tier's whole model (albedo × irradiance ÷ π, plus one GGX
///   lobe) *is* what the tracer solves for these, so the disagreement is the
///   L2 truncation of the rig's four lights and nothing else.
/// - **Brass** is a conductor: its picture is almost entirely what it
///   *reflects*, and the raster tier reflects a nine-coefficient SH read of
///   the room where the tracer reflects the room. A softbox comes back as a
///   broad glow rather than as a rectangle with an edge.
/// - **N-BK7 is glass, and this tier does not refract.** The reference's ball
///   is a lens showing an inverted plate through it; the raster's is a
///   Fresnel rim over the sky behind. They are not the same picture and no
///   tolerance makes them one — which is the entire reason
///   [`Settle`](kosm_view::raster::Settle) exists. The number is here to be
///   *reported* and to catch a regression that makes it worse, not to claim
///   the two agree.
#[test]
fn the_datasheet_ball_agrees_on_both_tiers() {
    let Some(ctx) = ctx_or_skip("the_datasheet_ball_agrees_on_both_tiers") else { return };
    use kosm::lens::Lens as _;
    use kosm::material::{BALL_RADIUS, Ball, ball_world};

    let out = std::path::Path::new("../../out");
    let Ok(probes) = kosm::light::probes::studio_probes(out) else {
        eprintln!("skipping the_datasheet_ball_agrees_on_both_tiers: no studio probe volume");
        return;
    };
    let world = ball_world().expect("the ball world builds");
    let size = (160u32, 120u32);

    // The camera is `Ball`'s own, restated: the datasheet's eye, its target
    // and its 0.6 rad vertical field. A parity number measured from a second
    // camera would be measuring the camera.
    let cam = kosm_render::Camera::look_at(
        Point3::new(0.075, -0.105, 0.055),
        Point3::new(0.0, 0.0, BALL_RADIUS),
        Vec3::new(0.0, 0.0, 1.0),
        (0.6f64).to_degrees(),
    );

    // The tracer's ball is lit by the rig directly; the raster's by the rig
    // baked into `studio_probes`. Neither has a sun.
    let mut worst: Vec<String> = Vec::new();
    for (name, tol) in [
        ("dry sand", 0.10),
        ("porcelain", 0.10),
        ("granite", 0.10),
        ("brass", 0.22),
        ("N-BK7", 0.40),
    ] {
        let material = kosm::material::named(name).unwrap_or_else(|| panic!("{name} is a name"));
        let reference = Ball { material: material.clone(), width: size.0, height: size.1, spp: 64 }
            .see(&world);

        let (mut s, by_name) = Scene::new(Sun::from_rgb([0.0, 0.0, 1.0], [0.0; 3], 0.01), probes.clone());
        s.bounds = ([-0.16, -0.16, -0.03], [0.16, 0.16, 0.09]);
        let subject = by_name[name];
        // `Ball`'s plate, as its own `Pbr::plastic` states it
        let plate = s.push_material(rgb_material([0.30, 0.30, 0.31], 0.55, 0.0));
        let mut p = slab([-0.15, -0.15, -0.02], [0.15, 0.15, 0.0]);
        p.instances.push(Instance { material: plate, ..Default::default() });
        s.push_mesh(p);
        let mut b = sphere_mesh([0.0, 0.0, BALL_RADIUS], BALL_RADIUS, 96, 48);
        b.instances.push(Instance { material: subject, ..Default::default() });
        s.push_mesh(b);

        let mut f = Frame::new(cam, size);
        f.exposure = 0.7; // `Ball::see`'s own `to_srgb8(0.7, false)`
        let got = draw(ctx, &s, &f);

        // Over the ball alone: the plate is the same neutral plastic on both
        // tiers and averaging it in would flatter every material equally.
        let mut sum = 0.0f64;
        let mut n = 0.0f64;
        for (x, y, px) in got.enumerate_pixels() {
            let q = reference.get_pixel(x, y);
            // the ball's own disc, in the frame `Ball` composes: it fills the
            // middle, and the plate is the bottom corners and the top edge
            let (dx, dy) = (
                (x as f64 - size.0 as f64 * 0.5) / size.0 as f64,
                (y as f64 - size.1 as f64 * 0.52) / size.1 as f64,
            );
            if dx * dx + dy * dy > 0.16 * 0.16 {
                continue;
            }
            for c in 0..3 {
                sum += (px[c] as f64 - q[c] as f64).abs() / 255.0;
                n += 1.0;
            }
        }
        let mad = sum / n.max(1.0);
        eprintln!("datasheet parity  {name:>10}: mad {mad:.4}  (tolerance {tol:.2})");
        if mad > tol {
            worst.push(format!("{name}: {mad:.4} over {tol:.2}"));
        }
    }
    assert!(worst.is_empty(), "the two tiers disagree about {}", worst.join(", "));
}
