//! The pool, handed to `kosm-render`.
//!
//! The bespoke tracer in `pool.rs` knows what a pool is: it has a function
//! for the tiles, a function for the sky, a marcher for the surface and a
//! precomputed [`Caustic`] grid that it looks up when it shades the floor.
//! That grid is the interesting cheat. It is a *forward* light transport
//! step — sun rays pushed through the surface by Snell's law and binned
//! where they land — bolted onto a backward tracer, and it exists because a
//! backward tracer cannot find the sun through moving water.
//!
//! This module builds the same pool out of nothing but geometry and
//! materials and gives it to the general path tracer, to see whether the
//! caustic comes back for free. It does not. See `### the pool` in the
//! README, and the note on `WATER` below.
//!
//! What is here:
//!
//! * The basin, deck, coping and grandstand as boxes; the tiles as a
//!   two-material checker (a texture would be one image, but there are no
//!   textures, so a checker is 125k quads and the lane lines are a third
//!   material). Grout lines are skipped: a `linear-pattern` of thin boxes
//!   over every tile edge would multiply the triangle count by six for a
//!   half-pixel feature.
//! * The melon as a unit-sphere mesh under a non-uniform `Transform`. The
//!   TLAS maps the ray into local space and the normal back by the inverse
//!   transpose, so a scaled instance of a sphere *is* an ellipsoid and is
//!   correct at any anisotropy. (`kosm-render` has no analytic sphere;
//!   `TriMesh` and `HeightField` are the only geometries, so the ellipsoid
//!   is tessellated either way.)
//! * The water as [`HeightField`]s: one fine field over the splash region
//!   and four coarse ones tiling the rest of the pool around it, all sampled
//!   from [`Surface::height`] so the fine grid, the far field and the blend
//!   between them arrive already resolved. Per frame they are
//!   `update_heights` + `Bvh::refit`, which is the whole point of the
//!   height-field primitive.
//! * Foam is skipped. It is a coverage field in the reference renderer — a
//!   whitening of the surface shade, not geometry — and there is nothing in
//!   `Pbr` to drive per-point from a client. It would need a texture, or a
//!   third water material and a per-cell material index.

use std::sync::Arc;

use kosm_render::geometry::Geometry;
use kosm_render::heightfield::HeightField;
use kosm_render::math::{Aabb, Point3, Transform, Vec3};
use kosm_render::caustics::{self, CausticMap, CausticOptions};
use kosm_render::pathtrace::{
    Camera, Environment, GradientEnv, Object, PathTraceOptions, Pbr, Scene, Sun,
};
use kosm_render::{Bvh, Hit, Ray, TriMesh};
use tang::Vec3 as V;

use super::{
    ABSORB, COPING, N_WATER, PoolGeometry, PoolSnapshot, STAND_RISE, STAND_ROWS, STAND_TREAD,
    SUN_IRRADIANCE, Surface, View, box_half, sun_dir,
};

/// Which renderer draws the pool.
///
/// `Legacy` is the bespoke tracer in `pool.rs` — the one with the
/// precomputed caustic grid. `Kosm` is this module, which has no caustic
/// grid and must find the sun through the water on its own. Legacy is the
/// default until the new one looks right; see `### the pool` in the README
/// for how far that is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolRenderer {
    /// The bespoke tracer in `pool.rs`.
    #[default]
    Legacy,
    /// `kosm-render`'s path tracer, driven from this module.
    Kosm,
}

impl PoolRenderer {
    /// Parse the `--pool-render` argument.
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "legacy" => Ok(Self::Legacy),
            "kosm" => Ok(Self::Kosm),
            other => Err(anyhow::anyhow!(
                "--pool-render takes `legacy` or `kosm`, not `{other}`"
            )),
        }
    }
}

// ---- the one geometry ------------------------------------------------------

/// `kosm_render::Scene` is generic over a single geometry type, so a pool
/// that is both meshes and water needs one enum that is both. Dispatch is
/// static and the BVH is built per object, so this costs a branch per
/// primitive test and nothing else.
pub enum PoolGeom {
    /// Basin, deck, stand, melon, droplets.
    Mesh(TriMesh),
    /// A patch of water.
    Water(HeightField),
}

impl Geometry for PoolGeom {
    fn len(&self) -> usize {
        match self {
            PoolGeom::Mesh(m) => m.len(),
            PoolGeom::Water(w) => w.len(),
        }
    }
    fn bounds(&self, i: usize) -> Aabb {
        match self {
            PoolGeom::Mesh(m) => m.bounds(i),
            PoolGeom::Water(w) => w.bounds(i),
        }
    }
    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        match self {
            PoolGeom::Mesh(m) => m.intersect(ray, i, t_min, t_max),
            PoolGeom::Water(w) => w.intersect(ray, i, t_min, t_max),
        }
    }
    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) {
        match self {
            PoolGeom::Mesh(m) => m.intersect_all(ray, i, out),
            PoolGeom::Water(w) => w.intersect_all(ray, i, out),
        }
    }
    fn occludes(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> bool {
        match self {
            PoolGeom::Mesh(m) => m.occludes(ray, i, t_min, t_max),
            PoolGeom::Water(w) => w.occludes(ray, i, t_min, t_max),
        }
    }
}

// ---- materials -------------------------------------------------------------

/// Water. `ABSORB` is absorption per metre, RGB; `attenuation_color` is what
/// one `attenuation_distance` of interior transmits, so with the distance
/// pinned at a metre the colour is `exp(-a)` component-wise — the same
/// Beer–Lambert law the reference tracer applies by hand along its refracted
/// segment.
///
/// **This is where the port stops being free.** `Pbr::transmission` makes the
/// surface refract camera paths correctly, but the integrator's shadow ray
/// (`Scene::occluded`) is a material-blind any-hit test: water is an opaque
/// blocker to next-event estimation. Every point on the tiles is therefore
/// in shadow as far as the sun's NEE strategy is concerned, and the only
/// path that carries sunlight to the floor is a BSDF-sampled bounce that
/// leaves the floor, refracts back up through the surface and lands inside a
/// 0.6°-radius disc — about one ray in 10^4.5. See the README.
pub fn water() -> Pbr {
    Pbr::glass(N_WATER as f32, 0.02).with_attenuation(
        [
            (-ABSORB[0]).exp() as f32,
            (-ABSORB[1]).exp() as f32,
            (-ABSORB[2]).exp() as f32,
        ],
        1.0,
    )
}

fn matte(rgb: [f32; 3], roughness: f32) -> Pbr {
    Pbr {
        base_color: rgb,
        roughness,
        ..Default::default()
    }
}

/// The two blues of the checker, the dark lane line, the pale wall, the deck
/// and the concrete of the stand: the reference tracer's palette, as
/// materials rather than as a function of position.
fn tile_a() -> Pbr {
    matte([0.58, 0.78, 0.86], 0.35)
}
fn tile_b() -> Pbr {
    matte([0.50, 0.72, 0.84], 0.35)
}
fn lane_line() -> Pbr {
    matte([0.05, 0.09, 0.18], 0.35)
}
fn wall_tile() -> Pbr {
    matte([0.54, 0.75, 0.85], 0.4)
}
fn deck_mat() -> Pbr {
    matte([0.80, 0.68, 0.52], 0.7)
}
fn stand_mat() -> Pbr {
    matte([0.72, 0.70, 0.66], 0.8)
}
fn melon_mat() -> Pbr {
    matte([0.10, 0.30, 0.12], 0.3)
}

// ---- mesh building ---------------------------------------------------------

/// Positions and indices under construction; several boxes and quads share
/// one mesh so they share one BVH.
#[derive(Default)]
struct MeshBuild {
    positions: Vec<Point3>,
    indices: Vec<u32>,
}

impl MeshBuild {
    fn quad(&mut self, a: Point3, b: Point3, c: Point3, d: Point3) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&[a, b, c, d]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// An axis-aligned box, outward-facing.
    fn boxed(&mut self, lo: V<f64>, hi: V<f64>) {
        let p = |x: f64, y: f64, z: f64| Point3::new(x, y, z);
        let (x0, y0, z0) = (lo.x, lo.y, lo.z);
        let (x1, y1, z1) = (hi.x, hi.y, hi.z);
        // -z, +z
        self.quad(p(x0, y0, z0), p(x0, y1, z0), p(x1, y1, z0), p(x1, y0, z0));
        self.quad(p(x0, y0, z1), p(x1, y0, z1), p(x1, y1, z1), p(x0, y1, z1));
        // -y, +y
        self.quad(p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), p(x0, y0, z1));
        self.quad(p(x0, y1, z0), p(x0, y1, z1), p(x1, y1, z1), p(x1, y1, z0));
        // -x, +x
        self.quad(p(x0, y0, z0), p(x0, y0, z1), p(x0, y1, z1), p(x0, y1, z0));
        self.quad(p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1), p(x1, y0, z1));
    }

    /// A horizontal quad at `z`, facing up.
    fn floor_quad(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, z: f64) {
        self.quad(
            Point3::new(x0, y0, z),
            Point3::new(x1, y0, z),
            Point3::new(x1, y1, z),
            Point3::new(x0, y1, z),
        );
    }

    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn finish(self) -> TriMesh {
        TriMesh::new(self.positions, Vec::new(), &self.indices)
    }
}

/// A unit sphere, smooth-shaded, for the melon and the droplets. Placed under
/// a non-uniform `Transform` it is an ellipsoid: the TLAS renormalises the
/// local ray (so `t` stays comparable) and maps the normal by the inverse
/// transpose (so the ellipsoid's normal is not the sphere's).
fn unit_sphere(segments: usize, rings: usize) -> TriMesh {
    let mut positions = Vec::with_capacity((rings + 1) * (segments + 1));
    let mut normals = Vec::with_capacity((rings + 1) * (segments + 1));
    for j in 0..=rings {
        let theta = std::f64::consts::PI * j as f64 / rings as f64;
        let (st, ct) = theta.sin_cos();
        for i in 0..=segments {
            let phi = std::f64::consts::TAU * i as f64 / segments as f64;
            let (sp, cp) = phi.sin_cos();
            let n = V::new(st * cp, st * sp, ct);
            positions.push(Point3::new(n.x, n.y, n.z));
            normals.push(Vec3::new(n.x, n.y, n.z));
        }
    }
    let row = (segments + 1) as u32;
    let mut indices = Vec::with_capacity(rings * segments * 6);
    for j in 0..rings as u32 {
        for i in 0..segments as u32 {
            let (a, b, c, d) = (j * row + i, j * row + i + 1, (j + 1) * row + i + 1, (j + 1) * row + i);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    TriMesh::new(positions, normals, &indices)
}

/// Object → world for an ellipsoid: columns are the scaled body axes, the
/// translation is the centre. `Transform`'s matrix takes column vectors on
/// the right, so this is exactly `T · R · S`.
fn ellipsoid_placement(centre: V<f64>, axis: V<f64>, semi: [f64; 3]) -> Transform {
    let a = axis.normalize();
    let up = V::new(0.0, 0.0, 1.0);
    let b = if up.cross(&a).norm() > 1e-9 {
        up.cross(&a).normalize()
    } else {
        V::new(1.0, 0.0, 0.0)
    };
    let c = a.cross(&b);
    let col = |v: V<f64>, s: f64| tang::Vec4::new(v.x * s, v.y * s, v.z * s, 0.0);
    Transform::from_matrix(tang::Mat4::from_cols(
        col(a, semi[0]),
        col(b, semi[1]),
        col(c, semi[2]),
        tang::Vec4::new(centre.x, centre.y, centre.z, 1.0),
    ))
}

// ---- the water -------------------------------------------------------------

/// The footprint of one height field: a rectangle and a cell size.
#[derive(Clone, Copy)]
struct Patch {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    cell: f64,
}

impl Patch {
    fn lattice(&self) -> (usize, usize, f64, f64) {
        let nx = (((self.x1 - self.x0) / self.cell).ceil() as usize + 1).max(2);
        let ny = (((self.y1 - self.y0) / self.cell).ceil() as usize + 1).max(2);
        let dx = (self.x1 - self.x0) / (nx - 1) as f64;
        let dy = (self.y1 - self.y0) / (ny - 1) as f64;
        (nx, ny, dx, dy)
    }

    /// Sample `Surface::height` over the lattice. Going through `height`
    /// rather than the raw `HeightGrid` means the fine grid, the far field,
    /// the ambient ripple and the C1 blend between them are all already
    /// resolved — the renderer sees the surface the simulator does.
    fn sample(&self, surface: &Surface) -> Vec<f64> {
        let (nx, ny, dx, dy) = self.lattice();
        let mut z = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            let y = self.y0 + j as f64 * dy;
            for i in 0..nx {
                z.push(surface.height(self.x0 + i as f64 * dx, y));
            }
        }
        z
    }

    fn field(&self, surface: &Surface) -> HeightField {
        let (nx, ny, dx, dy) = self.lattice();
        HeightField::new(
            nx,
            ny,
            Point3::new(self.x0, self.y0, 0.0),
            (dx, dy),
            self.sample(surface),
        )
    }
}

/// Fine cell size for the splash region (metres). `KOSM_WATER_CELL` overrides.
fn fine_cell() -> f64 {
    std::env::var("KOSM_WATER_CELL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02)
}

/// Coarse cell size for the rest of the pool.
const FAR_CELL: f64 = 0.25;

/// The water's footprints: a fine square over the splash region, and four
/// coarse rectangles tiling the rest of the pool around it.
///
/// They must not overlap. The surface is an *open* height field, so the
/// tracer decides inside-from-outside by which way the normal faces; two
/// fields stacked over the same `(x, y)` would refract a ray twice on the
/// way in and leave it permanently underwater.
fn patches(geometry: PoolGeometry) -> Vec<Patch> {
    let [px, py] = geometry.half_extents;
    let fine = fine_cell();
    // The fine field covers the splash region plus the blend band around it.
    let b = (box_half() + 0.5).min(px.min(py));
    vec![
        Patch { x0: -b, y0: -b, x1: b, y1: b, cell: fine },
        Patch { x0: -px, y0: -py, x1: px, y1: -b, cell: FAR_CELL },
        Patch { x0: -px, y0: b, x1: px, y1: py, cell: FAR_CELL },
        Patch { x0: -px, y0: -b, x1: -b, y1: b, cell: FAR_CELL },
        Patch { x0: b, y0: -b, x1: px, y1: b, cell: FAR_CELL },
    ]
}

// ---- the scene -------------------------------------------------------------

/// Everything that does not change between frames, built once.
pub struct PoolRenderScene {
    statics: Vec<(Arc<Bvh<PoolGeom>>, Pbr)>,
    sphere: Arc<Bvh<PoolGeom>>,
    /// One BVH per water patch, refit in place every frame. Held as `Arc`s
    /// because `Object` wants one; between frames the scene that borrowed
    /// them is gone, the count is back to one, and `Arc::get_mut` hands the
    /// tree back for the refit without copying it.
    water: Vec<Arc<Bvh<PoolGeom>>>,
    patches: Vec<Patch>,
    geometry: PoolGeometry,
}

impl PoolRenderScene {
    /// Build the basin, the deck, the stand, the sphere and the water
    /// lattices for a pool of this geometry. The water's heights are
    /// whatever `surface` says now; `update` replaces them.
    pub fn build(geometry: PoolGeometry, surface: &Surface) -> Self {
        let [px, py] = geometry.half_extents;
        let depth = geometry.depth;

        // The tiles. Same lane layout as the reference tracer: ten lanes of
        // 2.5 m across the width, a 25 cm dark line down each, and the T two
        // metres from the end walls.
        let mut a = MeshBuild::default();
        let mut b = MeshBuild::default();
        let mut lane = MeshBuild::default();
        let t = 0.1;
        let nx = (2.0 * px / t).round() as i64;
        let ny = (2.0 * py / t).round() as i64;
        for j in 0..ny {
            let y0 = -py + j as f64 * t;
            let y = y0 + 0.5 * t;
            let centre = ((y + py) / 2.5).floor() * 2.5 - py + 1.25;
            for i in 0..nx {
                let x0 = -px + i as f64 * t;
                let x = x0 + 0.5 * t;
                let on_line = (y - centre).abs() < 0.125;
                let on_t = (x.abs() - (px - 2.0)).abs() < 0.125 && (y - centre).abs() < 0.5;
                let mesh = if on_line || on_t {
                    &mut lane
                } else if ((i + j) % 2) == 0 {
                    &mut a
                } else {
                    &mut b
                };
                mesh.floor_quad(x0, y0, x0 + t, y0 + t, -depth);
            }
        }

        // The basin's walls, the deck around it, the coping.
        let mut walls = MeshBuild::default();
        let w = 0.3;
        walls.boxed(V::new(-px - w, -py - w, -depth), V::new(-px, py + w, COPING));
        walls.boxed(V::new(px, -py - w, -depth), V::new(px + w, py + w, COPING));
        walls.boxed(V::new(-px, -py - w, -depth), V::new(px, -py, COPING));
        walls.boxed(V::new(-px, py, -depth), V::new(px, py + w, COPING));

        // The deck is a *ring* around the basin, not a lid over it: four
        // slabs outside the wall boxes. (A single rectangle covering the
        // whole footprint is the obvious first draft and it renders a
        // perfectly convincing concrete floor where the water should be.)
        let mut deck = MeshBuild::default();
        let reach = 4.0;
        let (ox, oy) = (px + w, py + w);
        let z = (COPING - 0.12, COPING);
        deck.boxed(V::new(-ox - reach, -oy - reach, z.0), V::new(ox + reach, -oy, z.1));
        deck.boxed(V::new(-ox - reach, oy, z.0), V::new(ox + reach, oy + reach, z.1));
        deck.boxed(V::new(-ox - reach, -oy, z.0), V::new(-ox, oy, z.1));
        deck.boxed(V::new(ox, -oy, z.0), V::new(ox + reach, oy, z.1));

        // The grandstand: stepped rows along the far long side.
        let mut stand = MeshBuild::default();
        let y0 = geometry.stand_y0();
        let y_back = y0 + STAND_ROWS as f64 * STAND_TREAD;
        for r in 0..STAND_ROWS {
            stand.boxed(
                V::new(-px, y0 + r as f64 * STAND_TREAD, COPING),
                V::new(px, y_back, COPING + (r + 1) as f64 * STAND_RISE),
            );
        }

        let mut statics = Vec::new();
        for (mesh, pbr) in [
            (a, tile_a()),
            (b, tile_b()),
            (lane, lane_line()),
            (walls, wall_tile()),
            (deck, deck_mat()),
            (stand, stand_mat()),
        ] {
            if mesh.is_empty() {
                continue;
            }
            statics.push((
                Arc::new(Bvh::build(PoolGeom::Mesh(mesh.finish()))),
                pbr,
            ));
        }

        let sphere = Arc::new(Bvh::build(PoolGeom::Mesh(unit_sphere(32, 16))));
        let patches = patches(geometry);
        let water = patches
            .iter()
            .map(|p| Arc::new(Bvh::build(PoolGeom::Water(p.field(surface)))))
            .collect();

        Self { statics, sphere, water, patches, geometry }
    }

    /// Re-sample the water and refit its trees. Returns the elapsed
    /// milliseconds, split into sampling and refit — the number the port was
    /// meant to measure.
    pub fn update(&mut self, surface: &Surface) -> (f64, f64) {
        let mut sample_ms = 0.0;
        let mut refit_ms = 0.0;
        for (patch, bvh) in self.patches.iter().zip(self.water.iter_mut()) {
            let t = std::time::Instant::now();
            let z = patch.sample(surface);
            sample_ms += t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            let tree = Arc::get_mut(bvh)
                .expect("the previous frame's scene is dropped before the next update");
            match tree.geometry_mut() {
                PoolGeom::Water(field) => field.update_heights(&z),
                PoolGeom::Mesh(_) => unreachable!("water patch holds a height field"),
            }
            tree.refit();
            refit_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        (sample_ms, refit_ms)
    }

    /// Assemble the frame's `Scene`. The statics are shared `Arc`s; the
    /// melon and the droplets are placements of the one sphere; the water is
    /// whatever `update` last put in the trees.
    pub fn scene(&self, snapshot: &PoolSnapshot) -> Scene<PoolGeom> {
        let mut objects: Vec<Object<PoolGeom>> = self
            .statics
            .iter()
            .map(|(bvh, pbr)| Object::new(bvh.clone(), *pbr))
            .collect();

        let melon = &snapshot.melon;
        objects.push(Object::placed(
            self.sphere.clone(),
            melon_mat(),
            ellipsoid_placement(melon.centre, melon.axis, melon.axes),
        ));

        // Droplets: little balls of the same water. They are the only part of
        // the splash that survives the port intact — they are positions, and
        // a position is a placement.
        let drop_water = water();
        for d in &snapshot.droplets {
            let r = 0.004 + 0.010 * d.crowd.clamp(0.0, 1.0);
            objects.push(Object::placed(
                self.sphere.clone(),
                drop_water,
                ellipsoid_placement(d.pos, V::new(1.0, 0.0, 0.0), [r, r, r]),
            ));
        }

        for bvh in &self.water {
            objects.push(Object::new(bvh.clone(), water()));
        }

        Scene {
            objects,
            lights: Vec::new(),
            env: Environment::Gradient(GradientEnv {
                // The reference tracer's sky, as an environment: a pale
                // horizon washing up into a deep blue zenith.
                zenith: [0.22, 0.44, 0.88],
                horizon: [0.66, 0.80, 0.94],
                ground: [0.30, 0.30, 0.30],
                intensity: 1.0,
            }),
            sun: Some(sun()),
            ground: None,
        }
    }

    /// The pool this scene was built for.
    pub fn geometry(&self) -> PoolGeometry {
        self.geometry
    }
}

/// The sun, at the pool's own direction and angular size.
///
/// `SUN_IRRADIANCE` is the reference tracer's scalar multiplier on a
/// normalised sun colour; `Sun` wants irradiance on a square-on surface, in
/// the same arbitrary units the environment is in. The environment here is
/// the reference sky verbatim (a zenith of 0.88 in blue), and the reference
/// tracer draws the sun's disc at 12.0, so the disc is about 13× the sky. An
/// irradiance of `SUN_IRRADIANCE * 3` over a 0.6° disc lands in the same
/// neighbourhood without the reference's filmic curve to roll it off.
pub fn sun() -> Sun {
    let d = sun_dir();
    Sun::new(
        Vec3::new(d.x, d.y, d.z),
        0.6f64.to_radians(),
        [
            (SUN_IRRADIANCE * 3.0) as f32,
            (SUN_IRRADIANCE * 2.9) as f32,
            (SUN_IRRADIANCE * 2.7) as f32,
        ],
    )
}

/// The pool's `View`, as a `kosm_render::Camera`.
pub fn camera(view: &View) -> Camera {
    Camera::look_at(
        Point3::new(view.eye.x, view.eye.y, view.eye.z),
        Point3::new(view.target.x, view.target.y, view.target.z),
        Vec3::new(0.0, 0.0, 1.0),
        view.vfov.to_degrees(),
    )
}

/// Path-trace options for a pool frame. `KOSM_SPP` overrides the sample
/// count, which is the knob the caustic question turns on.
pub fn options(seed: u64) -> PathTraceOptions {
    PathTraceOptions {
        spp: std::env::var("KOSM_SPP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256),
        // Camera → surface → water → floor → surface → sky is five events,
        // and the melon under the water adds two more.
        max_depth: std::env::var("KOSM_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12),
        rr_start: 6,
        seed,
        // The sun's *radiance* is its irradiance over a 0.6° disc: about
        // 9000 in these units. The default firefly clamp is 12. Every path
        // that does find the sun through the water — the ones that would
        // have been the caustic — is therefore scaled down by ~750x before
        // it is averaged in, so the caustic is not merely noisy, it is
        // clamped to nothing. `KOSM_CLAMP=0` turns the clamp off, which is
        // the only setting under which the caustic can converge at all.
        firefly_clamp: match std::env::var("KOSM_CLAMP").ok().and_then(|v| v.parse::<f32>().ok()) {
            Some(v) if v <= 0.0 => None,
            Some(v) => Some(v),
            None => Some(12.0),
        },
        // The caustic is a rare, enormous sample; the edge-avoiding filter
        // will happily smear one across a neighbourhood and call it a
        // gradient. Judge convergence undenoised.
        denoise: std::env::var("KOSM_DENOISE").map(|v| v != "0").unwrap_or(true),
        ..Default::default()
    }
}

/// How the caustic pass is shot for a pool frame, or `None` to skip it.
///
/// This is the answer to the question this module was written to ask. A
/// backward tracer cannot find the sun through moving water — that is not a
/// sample-count problem, it is a geometry problem, and `### the pool` in the
/// README spent a while establishing it. The caustic pass goes the other way:
/// sun photons through the surface by the same Snell's law the reference
/// tracer's [`Caustic`](crate::pool::Caustic) grid uses, deposited where they
/// land. Same physics, general machinery.
///
/// The gather radius is 6 cm, a little coarser than the reference grid's
/// cells, and the photon count is what buys the rings their contrast.
/// `KOSM_PHOTONS=0` turns the pass off and restores the caustic-free render
/// the README describes.
pub fn caustic_options(seed: u64) -> Option<CausticOptions> {
    let photons = std::env::var("KOSM_PHOTONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000_000usize);
    if photons == 0 {
        return None;
    }
    Some(CausticOptions {
        photons,
        radius: Some(
            std::env::var("KOSM_CAUSTIC_RADIUS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.06),
        ),
        seed: seed ^ 0xca05_71c5,
        ..Default::default()
    })
}

/// Shoot the caustic pass over a frame's scene, if it is enabled.
pub fn caustics_for(scene: &Scene<PoolGeom>, seed: u64) -> Option<CausticMap> {
    let opts = caustic_options(seed)?;
    Some(caustics::trace(scene, &opts))
}

/// Render one snapshot. Tonemapping is `kosm-render`'s ACES, not the pool's
/// filmic curve: ACES desaturates and darkens the bright blues a little
/// relative to the reference, so the two stills are not pixel-comparable
/// even where the transport agrees.
pub fn render_snapshot(
    scene: &PoolRenderScene,
    snapshot: &PoolSnapshot,
    view: &View,
    seed: u64,
) -> image::RgbaImage {
    let picture = scene.scene(snapshot);
    let map = caustics_for(&picture, seed);
    let film = kosm_render::pathtrace::render_with_caustics(
        &picture,
        &camera(view),
        view.width,
        view.height,
        &options(seed),
        map.as_ref(),
    );
    // The sky is the reference tracer's sky at full strength and the sun is
    // a real sun, so linear radiance off the deck runs well over 1. ACES
    // clips it to white without a stop or so of headroom; 0.35 puts the
    // water where the reference's filmic curve puts it.
    let exposure = std::env::var("KOSM_EXPOSURE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.35f32);
    image::RgbaImage::from_raw(view.width, view.height, film.to_srgb8(exposure, false))
        .expect("film is width * height * 4")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::MELON_AXES;

    #[test]
    fn water_absorption_matches_the_reference_coefficients() {
        let m = water();
        // One metre of water transmits exp(-a); red goes first.
        assert!((m.attenuation_color[0] as f64 - (-ABSORB[0]).exp()).abs() < 1e-6);
        assert!(m.attenuation_color[0] < m.attenuation_color[1]);
        assert!(m.attenuation_color[1] < m.attenuation_color[2]);
        assert_eq!(m.attenuation_distance, 1.0);
        assert_eq!(m.transmission, 1.0);
        assert!((m.ior as f64 - N_WATER).abs() < 1e-6);
    }

    #[test]
    fn the_water_patches_tile_the_pool_without_overlapping() {
        let g = PoolGeometry::reference();
        let [px, py] = g.half_extents;
        let ps = patches(g);
        let area: f64 = ps.iter().map(|p| (p.x1 - p.x0) * (p.y1 - p.y0)).sum();
        assert!((area - 4.0 * px * py).abs() < 1e-9, "patches must cover the pool exactly");
        for (i, p) in ps.iter().enumerate() {
            for q in &ps[i + 1..] {
                let overlap = (p.x1.min(q.x1) - p.x0.max(q.x0)).max(0.0)
                    * (p.y1.min(q.y1) - p.y0.max(q.y0)).max(0.0);
                assert_eq!(overlap, 0.0, "water patches must not overlap");
            }
        }
    }

    #[test]
    fn a_scaled_sphere_is_the_melon_ellipsoid() {
        // The placement maps the unit sphere's axes onto the melon's semi-axes.
        let t = ellipsoid_placement(V::new(0.0, 0.0, 1.0), V::new(1.0, 0.0, 0.0), MELON_AXES);
        let tip = t.apply_point(&Point3::new(1.0, 0.0, 0.0));
        assert!((tip.x - MELON_AXES[0]).abs() < 1e-12);
        assert!((tip.z - 1.0).abs() < 1e-12);
        // With the long axis along +x, the second body axis is z x a = +y and
        // the third is a x b = +z, each scaled by its own semi-axis. A melon
        // is not a sphere, and the placement is what makes it one.
        let beam = t.apply_point(&Point3::new(0.0, 1.0, 0.0));
        assert!((beam.y - MELON_AXES[1]).abs() < 1e-12);
        let side = t.apply_point(&Point3::new(0.0, 0.0, 1.0));
        assert!((side.z - (1.0 + MELON_AXES[2])).abs() < 1e-12);
        assert!(side.y.abs() < 1e-12);
    }

    #[test]
    fn the_camera_carries_the_views_framing() {
        let view = View {
            eye: V::new(-3.2, -2.6, 0.9),
            target: V::new(0.0, 0.1, -0.1),
            width: 64,
            height: 36,
            vfov: 0.9,
        };
        let cam = camera(&view);
        assert!((cam.fov_deg - 0.9f64.to_degrees()).abs() < 1e-9);
        assert!(cam.forward.dot(Vec3::new(0.0, 0.0, 1.0)) < 0.0, "looking down at the water");
    }
}
