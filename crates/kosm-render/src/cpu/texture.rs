//! Surface textures: spatially varying materials.
//!
//! A [`Texture`] rewrites an object's [`Pbr`] at the shading point. Meshes and
//! B-reps arrive here without UVs, so every texture is addressed by world
//! position and normal: the procedural ones are solid textures (a wood block
//! has rings all the way through, so a cut face shows end grain), and an
//! image is projected triplanar, blended by the normal so seams fall where
//! the surface turns.
//!
//! Everything is a pure function of `(point, normal)`, deterministic and
//! thread-safe, so a texture costs nothing in variance and renders the same
//! on every machine.

use super::*;

/// A spatially varying material, attached to an [`Object`].
#[derive(Debug, Clone)]
pub enum Texture {
    /// Procedural wood grain.
    Wood(WoodTexture),
    /// Procedural plain-weave fabric.
    Fabric(FabricTexture),
    /// A colour image projected triplanar.
    Image(ImageTexture),
}

impl Texture {
    /// The material at `p` (world space) with surface normal `n`.
    pub fn shade(&self, base: Pbr, p: Point3, n: Vec3) -> Pbr {
        match self {
            Texture::Wood(w) => w.shade(base, p),
            Texture::Fabric(f) => f.shade(base, p, n),
            Texture::Image(i) => i.shade(base, p, n),
        }
    }
}

/// Wood: concentric growth rings around a grain axis, distorted by noise,
/// with fine pores streaked along the grain.
#[derive(Debug, Clone, Copy)]
pub struct WoodTexture {
    /// Linear early-wood (light) colour.
    pub light: [f32; 3],
    /// Linear late-wood (ring) colour.
    pub dark: [f32; 3],
    /// Distance between growth rings, in scene units.
    pub ring_spacing: f64,
    /// Grain direction (the trunk's axis). Need not be unit length.
    pub axis: Vec3,
    /// A point on the trunk's axis; rings are centred on the line through it.
    pub origin: Point3,
    /// How strongly noise wobbles the rings, as a fraction of a ring.
    pub turbulence: f64,
    /// Roughness added in the dark rings (late wood is denser and shinier
    /// in reverse: a small negative value makes rings glossier).
    pub ring_roughness: f32,
}

impl WoodTexture {
    /// White-oak-like defaults along `axis`, rings `ring_spacing` apart.
    pub fn oak(axis: Vec3, origin: Point3, ring_spacing: f64) -> Self {
        Self {
            light: [0.50, 0.33, 0.18],
            dark: [0.27, 0.15, 0.07],
            ring_spacing,
            axis,
            origin,
            turbulence: 0.35,
            ring_roughness: 0.08,
        }
    }

    fn shade(&self, mut base: Pbr, p: Point3) -> Pbr {
        let a = self.axis.normalize();
        let d = p - self.origin;
        let along = d.dot(a);
        let radial = d - a * along;
        let r = radial.norm();
        let s = self.ring_spacing.max(1e-9);
        // Low-frequency wobble, stretched along the grain like a real trunk.
        let q = Vec3::new(radial.x / s, radial.y / s, radial.z / s) + a * (along / (8.0 * s));
        let wobble = self.turbulence * fbm(q * 0.6, 3);
        let ring = (r / s + wobble).rem_euclid(1.0);
        // Sharp late-wood band at the end of each ring.
        let band = smoothstep(0.55, 0.85, ring) * (1.0 - smoothstep(0.92, 1.0, ring));
        // Pores: high frequency across the grain, long along it.
        let pq = Vec3::new(
            radial.x * 14.0 / s,
            radial.y * 14.0 / s,
            radial.z * 14.0 / s,
        ) + a * (along * 0.8 / s);
        let pores = 0.5 + 0.5 * value_noise(pq);
        let t = (0.75 * band + 0.25 * pores * band.max(0.3)) as f32;
        let tint = (0.92 + 0.16 * value_noise(q * 0.25)) as f32;
        for k in 0..3 {
            base.base_color[k] = lerp(self.light[k], self.dark[k], t) * tint;
        }
        base.roughness = (base.roughness + self.ring_roughness * t).clamp(0.02, 1.0);
        base
    }
}

/// Fabric: a plain weave of two thread colours, projected triplanar.
#[derive(Debug, Clone, Copy)]
pub struct FabricTexture {
    /// Linear colour of the warp threads.
    pub warp: [f32; 3],
    /// Linear colour of the weft threads.
    pub weft: [f32; 3],
    /// Thread pitch, in scene units.
    pub thread_spacing: f64,
    /// Sheen weight applied to the fabric (soft fuzz at grazing angles).
    pub sheen: f32,
}

impl FabricTexture {
    /// A felt-like mat in one colour, threads `thread_spacing` apart.
    pub fn felt(color: [f32; 3], thread_spacing: f64) -> Self {
        let dim = [color[0] * 0.8, color[1] * 0.8, color[2] * 0.8];
        Self {
            warp: color,
            weft: dim,
            thread_spacing,
            sheen: 0.6,
        }
    }

    fn shade(&self, mut base: Pbr, p: Point3, n: Vec3) -> Pbr {
        let s = self.thread_spacing.max(1e-9);
        let w = triplanar_weights(n);
        let mut col = [0.0f32; 3];
        for (axis, weight) in w.iter().enumerate() {
            if *weight <= 0.0 {
                continue;
            }
            let (u, v) = plane_coords(p, axis);
            let (u, v) = (u / s, v / s);
            let (iu, iv) = (u.floor(), v.floor());
            let (fu, fv) = (u - iu, v - iv);
            // Over/under: alternate which thread is on top per cell.
            let warp_on_top = ((iu + iv) as i64).rem_euclid(2) == 0;
            // Round thread profile: dark in the gaps between threads.
            let (prof, c) = if warp_on_top {
                ((std::f64::consts::PI * fu).sin(), self.warp)
            } else {
                ((std::f64::consts::PI * fv).sin(), self.weft)
            };
            let fuzz = 0.9 + 0.2 * value_noise(Vec3::new(u * 3.0, v * 3.0, 0.0));
            let shade = ((0.55 + 0.45 * prof) * fuzz) as f32;
            for k in 0..3 {
                col[k] += weight * c[k] * shade;
            }
        }
        base.base_color = col;
        base.roughness = base.roughness.max(0.8);
        base.sheen = base.sheen.max(self.sheen);
        base
    }
}

/// A colour image projected triplanar onto the surface.
#[derive(Debug, Clone)]
pub struct ImageTexture {
    width: usize,
    height: usize,
    /// Linear RGB, row-major from the top-left.
    pixels: Arc<Vec<[f32; 3]>>,
    /// Scene units covered by one repeat of the image.
    pub tile_size: f64,
}

impl ImageTexture {
    /// From linear RGB pixels, row-major. Errors on a size mismatch.
    pub fn from_linear(
        width: usize,
        height: usize,
        pixels: Vec<[f32; 3]>,
        tile_size: f64,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 || pixels.len() != width * height {
            return Err(format!(
                "image texture: {} pixels for {width}x{height}",
                pixels.len()
            ));
        }
        Ok(Self {
            width,
            height,
            pixels: Arc::new(pixels),
            tile_size,
        })
    }

    /// From 8-bit sRGB RGB bytes (3 per pixel), decoded to linear.
    pub fn from_srgb8(
        width: usize,
        height: usize,
        rgb: &[u8],
        tile_size: f64,
    ) -> Result<Self, String> {
        if rgb.len() != width * height * 3 {
            return Err(format!(
                "image texture: {} bytes for {width}x{height} RGB",
                rgb.len()
            ));
        }
        let dec = |b: u8| {
            let c = b as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        let px = rgb
            .chunks(3)
            .map(|c| [dec(c[0]), dec(c[1]), dec(c[2])])
            .collect();
        Self::from_linear(width, height, px, tile_size)
    }

    fn sample(&self, u: f64, v: f64) -> [f32; 3] {
        // Bilinear with wrap.
        let x = u.rem_euclid(1.0) * self.width as f64 - 0.5;
        let y = v.rem_euclid(1.0) * self.height as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = ((x - x0) as f32, (y - y0) as f32);
        let at = |xi: f64, yi: f64| {
            let xi = (xi as i64).rem_euclid(self.width as i64) as usize;
            let yi = (yi as i64).rem_euclid(self.height as i64) as usize;
            self.pixels[yi * self.width + xi]
        };
        let (a, b, c, d) = (
            at(x0, y0),
            at(x0 + 1.0, y0),
            at(x0, y0 + 1.0),
            at(x0 + 1.0, y0 + 1.0),
        );
        std::array::from_fn(|k| lerp(lerp(a[k], b[k], tx), lerp(c[k], d[k], tx), ty))
    }

    fn shade(&self, mut base: Pbr, p: Point3, n: Vec3) -> Pbr {
        let s = self.tile_size.max(1e-9);
        let w = triplanar_weights(n);
        let mut col = [0.0f32; 3];
        for (axis, weight) in w.iter().enumerate() {
            if *weight <= 0.0 {
                continue;
            }
            let (u, v) = plane_coords(p, axis);
            let c = self.sample(u / s, 1.0 - v / s);
            for k in 0..3 {
                col[k] += weight * c[k];
            }
        }
        base.base_color = col;
        base
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Triplanar blend weights (x, y, z planes), sharpened so the projection
/// switches quickly where the surface turns.
fn triplanar_weights(n: Vec3) -> [f32; 3] {
    let w = [n.x.abs().powi(4), n.y.abs().powi(4), n.z.abs().powi(4)];
    let sum = (w[0] + w[1] + w[2]).max(1e-12);
    [
        (w[0] / sum) as f32,
        (w[1] / sum) as f32,
        (w[2] / sum) as f32,
    ]
}

/// Coordinates of `p` in the plane perpendicular to `axis` (0 = x, ...).
fn plane_coords(p: Point3, axis: usize) -> (f64, f64) {
    match axis {
        0 => (p.y, p.z),
        1 => (p.x, p.z),
        _ => (p.x, p.y),
    }
}

fn hash3(x: i64, y: i64, z: i64) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Smooth 3D value noise in [-1, 1].
fn value_noise(p: Vec3) -> f64 {
    let (x0, y0, z0) = (p.x.floor(), p.y.floor(), p.z.floor());
    let (fx, fy, fz) = (p.x - x0, p.y - y0, p.z - z0);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy, sz) = (s(fx), s(fy), s(fz));
    let (xi, yi, zi) = (x0 as i64, y0 as i64, z0 as i64);
    let c = |dx: i64, dy: i64, dz: i64| hash3(xi + dx, yi + dy, zi + dz);
    let l = |a: f64, b: f64, t: f64| a + (b - a) * t;
    l(
        l(
            l(c(0, 0, 0), c(1, 0, 0), sx),
            l(c(0, 1, 0), c(1, 1, 0), sx),
            sy,
        ),
        l(
            l(c(0, 0, 1), c(1, 0, 1), sx),
            l(c(0, 1, 1), c(1, 1, 1), sx),
            sy,
        ),
        sz,
    )
}

/// Fractal sum of `octaves` value-noise layers, roughly in [-1, 1].
fn fbm(p: Vec3, octaves: u32) -> f64 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for _ in 0..octaves {
        sum += amp * value_noise(p * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grey() -> Pbr {
        Pbr {
            base_color: [0.5; 3],
            roughness: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn wood_has_rings_and_is_deterministic() {
        let w = Texture::Wood(WoodTexture::oak(
            Vec3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -200.0),
            3.0,
        ));
        let n = Vec3::new(0.0, 0.0, 1.0);
        let samples: Vec<f32> = (0..200)
            .map(|i| {
                w.shade(grey(), Point3::new(10.0, i as f64 * 0.37, 0.0), n)
                    .base_color[0]
            })
            .collect();
        let (lo, hi) = samples
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(hi - lo > 0.1, "rings must vary the colour: {lo}..{hi}");
        assert!(
            samples.iter().all(|&c| c > 0.05 && c < 0.7),
            "wood stays in a wood range"
        );
        let again = w.shade(grey(), Point3::new(10.0, 3.7, 0.0), n);
        assert_eq!(again, w.shade(grey(), Point3::new(10.0, 3.7, 0.0), n));
    }

    #[test]
    fn fabric_alternates_threads_and_adds_sheen() {
        let f = Texture::Fabric(FabricTexture {
            warp: [0.8, 0.1, 0.1],
            weft: [0.1, 0.1, 0.8],
            thread_spacing: 1.0,
            sheen: 0.5,
        });
        let n = Vec3::new(0.0, 0.0, 1.0);
        let a = f.shade(grey(), Point3::new(0.5, 0.5, 0.0), n);
        let b = f.shade(grey(), Point3::new(1.5, 0.5, 0.0), n);
        assert!(
            a.base_color[0] > a.base_color[2],
            "warp cell is warp-coloured"
        );
        assert!(
            b.base_color[2] > b.base_color[0],
            "next cell shows the weft"
        );
        assert!(a.sheen >= 0.5 && a.roughness >= 0.8);
    }

    #[test]
    fn image_texture_samples_and_rejects_bad_sizes() {
        assert!(ImageTexture::from_linear(2, 2, vec![[0.0; 3]; 3], 10.0).is_err());
        let img = ImageTexture::from_srgb8(2, 1, &[255, 0, 0, 0, 0, 255], 10.0).unwrap();
        let t = Texture::Image(img);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let left = t.shade(grey(), Point3::new(2.5, 5.0, 0.0), n).base_color;
        let right = t.shade(grey(), Point3::new(7.5, 5.0, 0.0), n).base_color;
        assert!(
            left[0] > left[2] && right[2] > right[0],
            "{left:?} {right:?}"
        );
    }

    /// A textured object renders differently from the same object untextured,
    /// and an untextured object renders exactly as before.
    #[test]
    fn texture_reaches_the_rendered_image() {
        use crate::cpu::testing::{test_camera, test_scene};
        let render_it = |tex: Option<Texture>| {
            let mut scene = test_scene();
            if let Some(t) = tex {
                scene.objects[0].texture = Some(Arc::new(t));
            }
            render(
                &scene,
                &test_camera(),
                32,
                32,
                &PathTraceOptions {
                    spp: 4,
                    ..Default::default()
                },
            )
            .rgb
        };
        let plain = render_it(None);
        assert_eq!(plain, render_it(None), "untextured render is deterministic");
        let fabric = render_it(Some(Texture::Fabric(FabricTexture {
            warp: [0.9, 0.05, 0.05],
            weft: [0.05, 0.05, 0.9],
            thread_spacing: 0.2,
            sheen: 0.3,
        })));
        assert_ne!(plain, fabric, "the texture never reached the image");
    }
}
