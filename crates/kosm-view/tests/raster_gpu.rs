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

// ── the pace ──────────────────────────────────────────────────────────────

/// **What the whole chain costs at the window's own size.**
///
/// Eight passes now, where there were two: a depth prepass, a half-resolution
/// occlusion and two blurs over it, a full-screen sky, the shading, and a
/// bloom in three. The design's line is that a walking frame is drawn in
/// single-digit milliseconds — the tier exists because the tracer could not —
/// so this draws a level-sized scene at 1280×720 until the queue has caught
/// up and divides.
///
/// It is a *wall-clock* number on whatever adapter is running the test, so it
/// is reported always and only asserted loosely: the assert is there to catch
/// a pass that went quadratic, not to pin a machine's speed.
#[test]
fn the_whole_chain_draws_a_window_frame_in_single_digit_milliseconds() {
    let Some(ctx) = ctx_or_skip("the_whole_chain_draws_a_window_frame") else { return };
    use kosm_render::env::SkyEnv;

    let sun_dir = Vec3::new(-0.32, -0.87, 0.375).normalize();
    let build = |effects: bool| {
        let probes = uniform_probes([-24.0, -24.0, -4.0], 0.5, [97, 97, 33], 0.16);
        let (mut s, _) = Scene::new(
            Sun::from_rgb([sun_dir.x, sun_dir.y, sun_dir.z], [6.2, 4.8, 2.9], 0.02),
            probes,
        );
        s.bounds = ([-24.0, -24.0, -4.0], [24.0, 24.0, 12.0]);
        s.exposure = 0.7;
        if effects {
            s.sky = Some(SkyEnv::new(sun_dir, 2.5, [0.42, 0.36, 0.24], 0.163, 0.02));
            s.air = kosm_render::post::Aerial { density: 0.0045, scale_h: 60.0 };
            s.ao_strength = 1.0;
            s.vignette = 0.35;
            s.bloom_threshold = 1.05;
            s.bloom_strength = 0.1;
        }
        let rock = s.push_material(rgb_material([0.30, 0.29, 0.27], 0.85, 0.15));
        let sand = s.push_material(rgb_material([0.62, 0.52, 0.35], 0.95, 0.12));
        let mut floor = floor_mesh();
        floor.instances.push(Instance { material: sand, ..Default::default() });
        s.push_mesh(floor);
        // ~90 k triangles of scattered geometry, which is the cove's own order
        let mut b = sphere_mesh([0.0, 0.0, 0.0], 0.6, 24, 12);
        for k in 0..160 {
            let a = k as f64 * 2.39996;
            let r = 0.6 * (k as f64).sqrt();
            b.instances.push(Instance::rigid(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [r * a.cos(), r * a.sin(), 0.5],
                rock,
            ));
        }
        s.push_mesh(b);
        s
    };
    let size = (1280u32, 720u32);
    let cam = kosm_render::Camera::look_at(
        Point3::new(0.0, -9.0, 1.8),
        Point3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        55.0,
    );
    let mut f = Frame::new(cam, size);
    f.exposure = 0.7;

    for (name, effects) in [("plain", false), ("with the light", true)] {
        let s = build(effects);
        let mut r = Raster::new(&ctx.device, &ctx.queue, &s, wgpu::TextureFormat::Rgba8Unorm)
            .expect("the raster tier builds");
        // warm the pipeline caches and the allocator
        for _ in 0..8 {
            let _ = r.draw(&ctx.device, &ctx.queue, &s, &f);
        }
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let n = 40;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = r.draw(&ctx.device, &ctx.queue, &s, &f);
        }
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        eprintln!("pace {name:>15}: {ms:.2} ms a frame at {}×{} ({} tris)", size.0, size.1, s.tris());
        assert!(ms < 25.0, "{name} draws a window frame in {ms:.2} ms, which is not a live tier");
    }
}

// ── what the probes cannot see ────────────────────────────────────────────

/// **The occlusion darkens a crease and leaves the open floor alone.**
///
/// The indirect half of every pixel is nine coefficients off a lattice metres
/// across, which has nothing to say about the centimetre where a wall meets a
/// floor. The reference tracer resolves that exactly, by tracing the
/// hemisphere. This puts a box on the floor with **no sun at all** — so every
/// photon in the frame is the probe term and the occlusion is the whole of
/// what is being measured — and asks whether the floor beside the box went
/// darker while the floor across the room did not.
#[test]
fn the_occlusion_darkens_the_foot_of_a_wall() {
    let Some(ctx) = ctx_or_skip("the_occlusion_darkens_the_foot_of_a_wall") else { return };
    let build = |strength: f32| {
        let probes = uniform_probes([-12.0, -12.0, -1.0], 4.0, [7, 7, 4], 0.25);
        // no sun: the whole picture is the probe read, times the occlusion
        let (mut s, _) = Scene::new(Sun::from_rgb([0.0, 0.0, 1.0], [0.0; 3], 0.01), probes);
        s.bounds = ([-11.0, -11.0, -0.5], [11.0, 11.0, 4.0]);
        s.exposure = 1.0;
        s.ao_radius_m = 0.6;
        s.ao_strength = strength;
        let white = s.push_material(rgb_material([0.8, 0.8, 0.8], 1.0, 0.0));
        let mut floor = floor_mesh();
        floor.instances.push(Instance { material: white, ..Default::default() });
        s.push_mesh(floor);
        // a wall standing on the floor, its foot along y = 0
        let mut wall = box_mesh(4.0, 0.15, 1.5);
        wall.instances.push(Instance::rigid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 1.5],
            white,
        ));
        s.push_mesh(wall);
        s
    };
    // low and to one side, so the floor in front of the wall fills the frame
    let cam = kosm_render::Camera::look_at(
        Point3::new(0.0, -4.0, 1.2),
        Point3::new(0.0, 0.4, 0.4),
        Vec3::new(0.0, 0.0, 1.0),
        55.0,
    );
    let size = (256u32, 144u32);
    let mut f = Frame::new(cam, size);
    f.exposure = 1.0;
    let off = draw(ctx, &build(0.0), &f);
    let on = draw(ctx, &build(1.0), &f);

    // the floor a hand's breadth from the wall's foot, and the floor by the
    // camera's own feet, both projected rather than guessed at
    let vp = kosm_view::raster::pipeline::view_proj(&cam, size.0 as f32 / size.1 as f32);
    let at = |p: [f32; 3]| -> (u32, u32) {
        let (x, y, w) = (
            vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12],
            vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13],
            vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15],
        );
        (
            ((x / w * 0.5 + 0.5) * size.0 as f32) as u32,
            ((0.5 - y / w * 0.5) * size.1 as f32) as u32,
        )
    };
    let crease = at([0.0, -0.25, 0.0]);
    let open = at([0.0, -3.0, 0.0]);
    let d_crease = patch(&off, crease.0, crease.1, 2) - patch(&on, crease.0, crease.1, 2);
    let d_open = patch(&off, open.0, open.1, 3) - patch(&on, open.0, open.1, 3);
    eprintln!("ao: the crease lost {d_crease:.4}, the open floor {d_open:.4}");
    assert!(d_crease > 0.04, "the occlusion did not find the crease: {d_crease:.4}");
    assert!(
        d_open < d_crease * 0.4,
        "the occlusion is a wash: the open floor lost {d_open:.4} against {d_crease:.4}"
    );
}

// ── the sky, and the film it goes through ─────────────────────────────────

/// **The WGSL sky is the Rust sky, and the WGSL film is the Rust film.**
///
/// The settle blend fades the raster frame into the traced one, and the sky
/// fills most of a frame looking out to sea — so if the two tiers' Preetham
/// disagreed anywhere, or their vignette did, a standing player would watch
/// the horizon change colour as the reference arrived. This draws a frame with
/// **no geometry in it at all**, so every pixel is `fs_sky` and the post pass,
/// and compares it against `SkyEnv::radiance` put through
/// `kosm_render::post::Post` on the CPU.
///
/// The tolerance is a code and a half of 255, which is `f16` in the HDR target
/// plus the difference between an `exp` on a GPU and one in libm.
#[test]
fn the_sky_and_the_film_agree_with_the_reference() {
    let Some(ctx) = ctx_or_skip("the_sky_and_the_film_agree_with_the_reference") else { return };
    use kosm_render::env::SkyEnv;
    use kosm_render::post::Post;

    let size = (192u32, 108u32);
    let sun_dir = Vec3::new(-0.35, -0.45, 0.42).normalize();
    let sky = SkyEnv::new(sun_dir, 2.5, [0.42, 0.36, 0.24], 0.163, 0.02);

    // An empty scene, so nothing is drawn over the sky. The probes are still
    // bound — an empty storage buffer is not a binding — but nothing reads
    // them once a sky model is set.
    let probes = uniform_probes([-4.0, -4.0, -1.0], 4.0, [2, 2, 2], 0.0);
    let (mut s, _) = Scene::new(
        Sun::from_rgb([sun_dir.x, sun_dir.y, sun_dir.z], [0.0; 3], 0.02),
        probes,
    );
    s.bounds = ([-4.0, -4.0, -1.0], [4.0, 4.0, 4.0]);
    s.exposure = 0.7;
    s.sky = Some(sky);
    s.vignette = 0.35;

    // Level and looking a little off the sun's azimuth, so the frame has the
    // horizon, the zenith and the aureole in it rather than one of them.
    let cam = kosm_render::Camera::look_at(
        Point3::new(0.0, 0.0, 1.6),
        Point3::new(-1.0, -1.0, 1.9),
        Vec3::new(0.0, 0.0, 1.0),
        50.0,
    );
    let mut f = Frame::new(cam, size);
    f.exposure = s.exposure;
    let got = draw(ctx, &s, &f);

    // the same picture, in Rust: a film whose radiance is the sky along each
    // pixel's own ray, through the same post chain
    let mut film = kosm_render::Film::new(size.0, size.1);
    let half_h = (cam.fov_deg.to_radians() * 0.5).tan();
    let half_w = half_h * size.0 as f64 / size.1 as f64;
    for j in 0..size.1 {
        let sy = 1.0 - 2.0 * (j as f64 + 0.5) / size.1 as f64;
        for i in 0..size.0 {
            let sx = 2.0 * (i as f64 + 0.5) / size.0 as f64 - 1.0;
            let d = (cam.forward + cam.right * (sx * half_w) + cam.up * (sy * half_h)).normalize();
            let c = sky.radiance(d);
            let k = ((j * size.0 + i) * 3) as usize;
            film.rgb[k..k + 3].copy_from_slice(&c);
        }
    }
    let want = Post { exposure: f.exposure, vignette: 0.35, ..Post::default() }
        .apply(&film, &cam, false);

    let mut worst = 0u8;
    let mut sum = 0.0f64;
    for (k, (g, w)) in got.as_raw().iter().zip(want.iter()).enumerate() {
        if k % 4 == 3 {
            continue;
        }
        worst = worst.max(g.abs_diff(*w));
        sum += g.abs_diff(*w) as f64;
    }
    let mean = sum / (got.as_raw().len() as f64 * 0.75);
    eprintln!("sky parity: worst {worst} codes, mean {mean:.3}");
    assert!(worst <= 2, "the two skies differ by {worst} codes");
    assert!(mean < 0.5, "the two skies differ by {mean:.3} codes on average");

    // and the vignette is actually doing something, or the test above would
    // pass with both tiers ignoring it
    let plain = Post { exposure: f.exposure, ..Post::default() }.apply(&film, &cam, false);
    let corner = |v: &[u8]| v[((size.1 - 1) * size.0) as usize * 4] as i32;
    assert!(
        corner(&plain) - corner(&want) > 8,
        "the vignette darkened the corner by {} codes, which is not a vignette",
        corner(&plain) - corner(&want)
    );
}

/// **Aerial perspective: the far headland hazes and the near sand does not.**
///
/// The exact closed form is `kosm_render::post::Aerial`'s own unit test and
/// the WGSL is a line-for-line port of it; what a GPU can be asked is whether
/// the term is wired to the right things — that it grows with distance, that
/// it pulls toward the *sky's* colour and not toward grey, and that the sky
/// itself is left alone (a background pixel is already the sky, and hazing it
/// again would double the term at the horizon, which is where it would show).
#[test]
fn the_haze_grows_with_distance_and_leaves_the_sky_alone() {
    let Some(ctx) = ctx_or_skip("the_haze_grows_with_distance") else { return };
    use kosm_render::env::SkyEnv;

    let sun_dir = Vec3::new(-0.32, -0.87, 0.375).normalize();
    let sky = SkyEnv::new(sun_dir, 2.5, [0.42, 0.36, 0.24], 0.163, 0.02);
    let build = |density: f32| {
        let probes = uniform_probes([-200.0, -20.0, -4.0], 100.0, [5, 3, 3], 0.16);
        let (mut s, _) = Scene::new(
            Sun::from_rgb([sun_dir.x, sun_dir.y, sun_dir.z], [0.0; 3], 0.02),
            probes,
        );
        s.bounds = ([-200.0, -20.0, -4.0], [200.0, 180.0, 60.0]);
        s.exposure = 0.7;
        s.sky = Some(sky);
        s.air = kosm_render::post::Aerial { density, scale_h: 60.0 };
        // a dark wall, so the haze has somewhere to go
        let dark = s.push_material(rgb_material([0.06, 0.06, 0.06], 0.9, 0.0));
        // near, on the left of the frame; far, on the right — same material,
        // same normal, so the only difference between them is the distance
        // Wide enough that each fills its own half of the frame right out to
        // the edge — a wall that ran out before the patch did was the sky —
        // and **low enough that the sky is still visible above both**, which
        // is what the last assertion needs. The tops are different heights
        // because the two are at different distances and the same height
        // would put one horizon line at the top of the frame.
        for (y, x0, x1, top) in [
            (8.0f64, -900.0f64, -0.02f64, 3.0f64),
            (160.0, 0.02, 900.0, 60.0),
        ] {
            let mut w = Mesh::from_m(
                &[[x0, y, -400.0], [x1, y, -400.0], [x1, y, top], [x0, y, top]],
                &[[0.0, -1.0, 0.0]; 4],
                &[0, 1, 2, 0, 2, 3],
            );
            w.instances.push(Instance { material: dark, ..Default::default() });
            s.push_mesh(w);
        }
        s
    };
    let size = (256u32, 144u32);
    let cam = kosm_render::Camera::look_at(
        Point3::new(0.0, 0.0, 1.6),
        Point3::new(0.0, 20.0, 1.6),
        Vec3::new(0.0, 0.0, 1.0),
        50.0,
    );
    let mut f = Frame::new(cam, size);
    f.exposure = 0.7;
    let off = draw(ctx, &build(0.0), &f);
    let on = draw(ctx, &build(0.006), &f);

    // Which half of the frame world `+x` lands on is `Camera::look_at`'s own
    // handedness, so the two are told apart by the haze itself: the far wall
    // is the one that moved.
    let (lx, rx) = (size.0 / 4, size.0 * 3 / 4);
    let l = (patch(&off, lx, size.1 / 2, 8), patch(&on, lx, size.1 / 2, 8));
    let r = (patch(&off, rx, size.1 / 2, 8), patch(&on, rx, size.1 / 2, 8));
    let (near, far, fx) = if r.1 - r.0 > l.1 - l.0 { (l, r, rx) } else { (r, l, lx) };
    let (near_off, near_on) = near;
    let (far_off, far_on) = far;
    eprintln!(
        "haze: the near wall {near_off:.4} → {near_on:.4}, the far one {far_off:.4} → {far_on:.4}"
    );
    assert!(far_on - far_off > 0.05, "the far wall did not haze: {far_off:.4} → {far_on:.4}");
    assert!(
        far_on - far_off > 3.0 * (near_on - near_off),
        "the haze does not grow with distance: near +{:.4}, far +{:.4}",
        near_on - near_off,
        far_on - far_off
    );
    // **And it pulls toward the sky, not toward grey.** Not toward the sky at
    // the top of the frame — toward the sky *along the wall's own view ray*,
    // which at this pixel is the anti-solar horizon of a 22° afternoon and is
    // warm rather than blue. So the target is computed here, from the same
    // `SkyEnv` the shader was handed, and the wall has to have moved toward
    // it.
    let hue = |img: &image::RgbaImage| {
        let p = img.get_pixel(fx, size.1 / 2);
        p[2] as i32 - p[0] as i32
    };
    let half_h = (cam.fov_deg.to_radians() * 0.5).tan();
    let half_w = half_h * size.0 as f64 / size.1 as f64;
    let sx = 2.0 * (fx as f64 + 0.5) / size.0 as f64 - 1.0;
    let d = (cam.forward + cam.right * (sx * half_w)).normalize();
    let c = sky.radiance(d);
    let code = |v: f32| {
        kosm_render::cpu::film::linear_to_srgb(kosm_render::cpu::film::tonemap_aces(v * 0.7))
            * 255.0
    };
    let target = (code(c[2]) - code(c[0])) as i32;
    let (was, now) = (hue(&off), hue(&on));
    assert!(
        (now - was).signum() == (target - was).signum() && (now - was).abs() >= 3,
        "the haze is grey and not sky: the wall went {was} → {now}, and the sky along \
         that ray is {target}"
    );
    // the sky is a background pixel and is not hazed twice
    for (x, y) in [(8u32, 4u32), (size.0 - 8, 4)] {
        let (a, b) = (off.get_pixel(x, y), on.get_pixel(x, y));
        assert_eq!(a, b, "the sky at ({x}, {y}) was hazed on top of itself");
    }
}

/// **Bloom, on the device, spreads a bright pixel the way the Rust does.**
///
/// The sun's own disc is the brightest thing the tier ever draws, so this puts
/// it in frame and asks whether the sky *beside* it brightened when bloom was
/// switched on — which is the one thing a bloom has to do and the one thing a
/// threshold set wrong would silently not.
#[test]
fn the_bloom_bleeds_off_the_sun() {
    let Some(ctx) = ctx_or_skip("the_bloom_bleeds_off_the_sun") else { return };
    use kosm_render::env::SkyEnv;

    let size = (160u32, 120u32);
    let sun_dir = Vec3::new(0.0, -0.6, 0.8).normalize();
    let probes = uniform_probes([-4.0, -4.0, -1.0], 4.0, [2, 2, 2], 0.0);
    let build = |bloom: f32| {
        let (mut s, _) = Scene::new(
            Sun::from_rgb([sun_dir.x, sun_dir.y, sun_dir.z], [6.0, 4.8, 3.0], 0.03),
            probes.clone(),
        );
        s.bounds = ([-4.0, -4.0, -1.0], [4.0, 4.0, 4.0]);
        s.exposure = 0.7;
        s.sky = Some(SkyEnv::new(sun_dir, 2.5, [0.4; 3], 0.163, 0.03));
        s.bloom_threshold = 1.0;
        s.bloom_strength = bloom;
        s.bloom_radius_px = 8.0;
        s
    };
    let cam = kosm_render::Camera::look_at(
        Point3::new(0.0, 0.0, 1.6),
        Point3::new(0.0, 0.0, 1.6) + Vec3::new(sun_dir.x, sun_dir.y, sun_dir.z),
        Vec3::new(0.0, 0.0, 1.0),
        60.0,
    );
    let f = Frame::new(cam, size);
    let off = draw(ctx, &build(0.0), &f);
    let on = draw(ctx, &build(0.25), &f);

    // Twelve pixels off the disc, which subtends three. **The kernel reaches
    // four quarter-resolution texels, so sixteen full pixels is its whole
    // extent** — a bloom asked about further out than that is asking about a
    // tail the nine-tap Gaussian does not have, on either tier.
    let ring = patch(&on, size.0 / 2 + 12, size.1 / 2, 3) - patch(&off, size.0 / 2 + 12, size.1 / 2, 3);
    eprintln!("bloom: the sky 12 px off the sun gained {ring:.4}");
    assert!(ring > 0.01, "the bloom did not bleed off the disc: {ring:.4}");
    // and the far corner is untouched, or it is a wash and not a bloom
    let corner = patch(&on, 6, 6, 4) - patch(&off, 6, 6, 4);
    assert!(corner.abs() < ring * 0.5, "the bloom is a wash: corner {corner:.4} against {ring:.4}");
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
