//! Solar optical-depth LUT, integrated in f64; distances and coefficients use km.
use super::wgpu;
const R: f64 = 6378.137;
pub fn extinction(h: f64) -> [f64; 3] {
    let r = (-h.max(0.) / 8.).exp();
    let m = (-h.max(0.) / 1.2).exp();
    let ozone = (1. - (h - 25.).abs() / 15.).max(0.);
    let br = [0.005802, 0.013558, 0.0331];
    let bo = [0.000650, 0.001881, 0.000085];
    std::array::from_fn(|i| br[i] * r + 0.00444 * m + bo[i] * ozone)
}
pub fn transmittance(h: f64, mu: f64, steps: usize) -> [f64; 3] {
    let radius = R + h;
    let b = radius * mu;
    let ground = b * b - radius * radius + R * R;
    if mu < 0. && ground >= 0. {
        return [0.; 3];
    }
    let end = -b
        + (b * b - radius * radius + (R + 100.).powi(2))
            .max(0.)
            .sqrt();
    let ds = end / steps as f64;
    let mut tau = [0.; 3];
    for j in 0..steps {
        let t = (j as f64 + 0.5) * ds;
        let alt = (radius * radius + t * t + 2. * b * t).sqrt() - R;
        let e = extinction(alt);
        for i in 0..3 {
            tau[i] += e[i] * ds;
        }
    }
    tau.map(|t| (-t).exp())
}
pub fn texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let (w, h) = (512u32, 128u32);
    let mut values = Vec::<f32>::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let altitude = (y as f64 / (h - 1) as f64).powi(2) * 100.;
        for x in 0..w {
            let mu = x as f64 / (w - 1) as f64 * 2. - 1.;
            let t = transmittance(altitude, mu, 128);
            values.extend([t[0] as f32, t[1] as f32, t[2] as f32, 1.]);
        }
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Rayleigh Mie ozone solar transmission"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&values),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 16),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    tex
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zenith_matches_integrated_column() {
        let t = transmittance(0., 1., 4096);
        let br = [0.005802, 0.013558, 0.0331];
        let bo = [0.000650, 0.001881, 0.000085];
        for i in 0..3 {
            let reference: f64 = (-(br[i] * 8. + 0.00444 * 1.2 + bo[i] * 15.) as f64).exp();
            assert!((t[i] - reference).abs() < 0.00001);
        }
    }
    #[test]
    fn solar_shadow_and_tangent() {
        assert_eq!(transmittance(10., -1., 128), [0.; 3]);
        let t = transmittance(0., 0., 512);
        assert!(t[0] > t[1] && t[1] > t[2]);
    }
    #[test]
    fn quadrature_converges() {
        for h in [0., 2., 10., 50.] {
            for mu in [0., 0.1, 0.5, 1.] {
                let a = transmittance(h, mu, 128);
                let b = transmittance(h, mu, 2048);
                for i in 0..3 {
                    assert!((a[i] - b[i]).abs() < 0.006, "h={h} mu={mu}");
                }
            }
        }
    }
}
