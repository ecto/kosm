//! The court, lit: a path tracer.
//!
//! Where `frame.rs` is a ray caster that is its own derivative, this is the
//! other end of the same idea — the reference tier for light on the court,
//! written plainly on `f64` and thrown at every core. Geometry is the same
//! derived colliders the physics stands on (the slab, the backboard, the
//! bracket, the pole, one box each; the balls as spheres with their contact
//! pose) plus a gym the level describes: walls, a ceiling, and rows of light
//! panels that are the only light there is. The rim is the one place the
//! picture and the physics disagree on purpose — the physics has 24 box
//! segments, the picture has the torus they approximate.
//!
//! Light transport is unidirectional path tracing with next-event estimation
//! on the panels, multiple importance sampling between the panel and the
//! BSDF, Russian roulette after the third bounce, and one sample stream per
//! pixel seeded by pixel, sample and frame so a frame is a pure function of
//! the state. Surfaces: a Lambertian base under a dielectric coat where a
//! coat belongs (lacquered maple, painted steel, rubber), the backboard as a
//! thin dielectric sheet with its painted square, procedural maple planks
//! and court markings, and a ball whose seams turn with its body frame.


//! The picture is split by ownership: `camera` (the ray for a film point),
//! `geometry` (shapes and intersections), `surface` (materials and the
//! procedural textures), `transport` (sampling and the path), and this file
//! (the scene, assembled from the court, and the frame).

mod camera;
mod geometry;
mod surface;
mod transport;

use phyz_math::Mat3;
use phyz_model::{GeomInstance, Geometry};
use rayon::prelude::*;
use tang::Vec3 as V3;

use super::{Court, CourtScene};
pub use camera::Camera;
pub use surface::Material;
use geometry::{arr, pv, Hit, Prim, Shape};
use transport::{tonemap, Rng};

type V = V3<f64>;

// ---- the scene --------------------------------------------------------------

pub struct Scene {
    pub(super) prims: Vec<Prim>,
    /// Indices of the light panels.
    pub(super) lights: Vec<usize>,
    pub(super) light_radiance: f64,
    pub(super) hoop: super::Hoop,
    /// Where the court's lines are painted from: the baseline, 4 ft behind the board.
    pub(super) baseline_x: f64,
}

impl Scene {
    /// The scene at the court's current state.
    pub fn new(scene: &CourtScene, court: &Court) -> anyhow::Result<Self> {
        let mut prims = Vec::new();
        let a = &scene.authored;
        let mm = |k: &str| a.millimetres(k);

        // the level's parts, by their root material
        for (material, instances) in super::parts(scene)? {
            let mat = match material.as_str() {
                "maple" => Material::Floor,
                "glass" => Material::Glass,
                "rim" => continue, // drawn as a torus below
                _ => Material::Steel,
            };
            for g in instances {
                prims.push(Prim { shape: shape_of(&g), mat });
            }
        }
        let rod = mm("rim_rod_mm")?;
        prims.push(Prim {
            shape: Shape::Torus {
                c: court.hoop.rim_centre - V::new(0.0, 0.0, 0.5 * rod),
                big_r: court.hoop.rim_r + 0.5 * rod,
                small_r: 0.5 * rod,
            },
            mat: Material::Rim,
        });

        // the balls
        for k in 0..court.bodies() {
            prims.push(Prim {
                shape: Shape::Sphere { c: pv(court.centre(k)), r: scene.ball_r, rot: arr(&court.rotation(k)) },
                mat: Material::Ball,
            });
        }

        // the gym: walls a margin outside the slab, a ceiling, light panels
        let (cx, cy) = (0.5 * mm("court_x_mm")?, 0.5 * mm("court_y_mm")?);
        let margin = mm("gym_margin_mm")?;
        let h = mm("gym_h_mm")?;
        let (wx, wy) = (cx + margin, cy + margin);
        let wall = |c: V, half: V| Prim { shape: Shape::Box { c, half, rot: arr(&Mat3::identity()) }, mat: Material::Wall };
        let thick = 0.1;
        prims.push(wall(V::new(wx + thick, 0.0, 0.5 * h), V::new(thick, wy + 2.0 * thick, 0.5 * h + thick)));
        prims.push(wall(V::new(-wx - thick, 0.0, 0.5 * h), V::new(thick, wy + 2.0 * thick, 0.5 * h + thick)));
        prims.push(wall(V::new(0.0, wy + thick, 0.5 * h), V::new(wx + 2.0 * thick, thick, 0.5 * h + thick)));
        prims.push(wall(V::new(0.0, -wy - thick, 0.5 * h), V::new(wx + 2.0 * thick, thick, 0.5 * h + thick)));
        // the floor beyond the slab, and the ceiling
        prims.push(Prim {
            shape: Shape::Box { c: V::new(0.0, 0.0, -0.021), half: V::new(wx, wy, 0.02), rot: arr(&Mat3::identity()) },
            mat: Material::Wall,
        });
        prims.push(Prim {
            shape: Shape::Box { c: V::new(0.0, 0.0, h + thick), half: V::new(wx, wy, thick), rot: arr(&Mat3::identity()) },
            mat: Material::Ceiling,
        });
        let (rows, cols) = (a.parameter("light_rows")?.max(1.0) as usize, a.parameter("light_cols")?.max(1.0) as usize);
        let (lw, ll) = (mm("light_w_mm")?, mm("light_l_mm")?);
        let mut lights = Vec::new();
        for i in 0..cols {
            for j in 0..rows {
                let x = (i as f64 + 0.5) / cols as f64 * 2.0 * wx - wx;
                let y = (j as f64 + 0.5) / rows as f64 * 2.0 * wy - wy;
                lights.push(prims.len());
                prims.push(Prim {
                    shape: Shape::Panel {
                        c: V::new(x - 0.5 * lw, y - 0.5 * ll, h - 0.005),
                        u: V::new(lw, 0.0, 0.0),
                        v: V::new(0.0, ll, 0.0),
                    },
                    mat: Material::Light,
                });
            }
        }

        Ok(Self {
            prims,
            lights,
            light_radiance: a.parameter_or("light_radiance", 18.0),
            hoop: court.hoop,
            baseline_x: court.hoop.board_x + 1.219,
        })
    }

    pub(super) fn nearest(&self, o: V, d: V, t_max: f64) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (i, prim) in self.prims.iter().enumerate() {
            let limit = best.as_ref().map_or(t_max, |h| h.t);
            if let Some((t, n)) = prim.shape.hit(o, d, limit) {
                best = Some(Hit { t, p: o + d * t, n, prim: i });
            }
        }
        best
    }

    /// What the camera's centre ray sees, for a sanity check: distance and material.
    pub fn probe(&self, cam: &Camera) -> Option<(f64, Material)> {
        let (o, d) = cam.ray(0.5 * cam.width as f64, 0.5 * cam.height as f64);
        self.nearest(o, d, f64::INFINITY).map(|h| (h.t, self.prims[h.prim].mat))
    }

    pub fn prim_count(&self) -> usize {
        self.prims.len()
    }

    pub(super) fn occluded(&self, o: V, d: V, t_max: f64) -> bool {
        self.prims.iter().enumerate().any(|(i, prim)| {
            // glass is not an occluder for the panels: the sheet is thin and
            // its transmission is booked on the path through it
            prim.mat != Material::Glass && !self.lights.contains(&i) && prim.shape.hit(o, d, t_max).is_some()
        })
    }
}

fn shape_of(g: &GeomInstance) -> Shape {
    let c = pv(g.origin.pos);
    match g.geometry {
        Geometry::Box { half_extents } => Shape::Box { c, half: pv(half_extents), rot: arr(&g.origin.rot) },
        Geometry::Sphere { radius } => Shape::Sphere { c, r: radius, rot: arr(&g.origin.rot) },
        Geometry::Cylinder { radius, height } => Shape::Cylinder { c, r: radius, half_h: 0.5 * height },
        _ => Shape::Sphere { c, r: 0.0, rot: arr(&Mat3::identity()) },
    }
}

/// Render the scene: `spp` samples per pixel, one stream per pixel and frame.
pub fn render(scene: &Scene, cam: &Camera, spp: usize, frame: u64) -> image::RgbImage {
    let (w, h) = (cam.width, cam.height);
    let rows: Vec<Vec<[u8; 3]>> = (0..h)
        .into_par_iter()
        .map(|y| {
            (0..w)
                .map(|x| {
                    let mut sum = V::zero();
                    for s in 0..spp {
                        let mut rng = Rng::new(((frame << 40) ^ ((y as u64) << 20) ^ (x as u64)) * 0x2545_F491_4F6C_DD1D + s as u64);
                        let (o, d) = cam.ray(x as f64 + rng.next(), y as f64 + rng.next());
                        sum += scene.trace(o, d, &mut rng);
                    }
                    tonemap(sum / spp as f64, cam.exposure)
                })
                .collect()
        })
        .collect();
    let mut img = image::RgbImage::new(w, h);
    for (y, row) in rows.iter().enumerate() {
        for (x, px) in row.iter().enumerate() {
            img.put_pixel(x as u32, y as u32, image::Rgb(*px));
        }
    }
    img
}

/// The camera the level asks for, at a given picture size.
pub fn camera(scene: &CourtScene, width: u32, height: u32) -> anyhow::Result<Camera> {
    let a = &scene.authored;
    Ok(Camera {
        eye: V::new(a.millimetres("cam_x_mm")?, a.millimetres("cam_y_mm")?, a.millimetres("cam_z_mm")?),
        target: V::new(a.millimetres("cam_at_x_mm")?, a.millimetres("cam_at_y_mm")?, a.millimetres("cam_at_z_mm")?),
        vfov: a.parameter_or("cam_vfov_deg", 42.0).to_radians(),
        width,
        height,
        exposure: a.parameter_or("exposure", 1.0),
    })
}
