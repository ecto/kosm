//! Bounded, demand-loaded Earth imagery. Decoding never runs in the frame callback.
use super::{Uniforms, wgpu};
use std::sync::mpsc::{self, Receiver, SyncSender};
const SIDE: u32 = 1028;
const SLOTS: usize = 32;
pub fn root() -> anyhow::Result<std::path::PathBuf> {
    let root = std::env::var_os("KOSM_EARTH_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/earth")
        });
    for relative in [
        "earth-low.png",
        "clouds-8k.png",
        "cloud-noise.raw",
        "surface/manifest.json",
    ] {
        if !root.join(relative).is_file() {
            anyhow::bail!(
                "Earth assets are not installed at {}. Run `python3 scripts/fetch-earth-assets.py`, or set KOSM_EARTH_ASSETS to an existing asset directory.",
                root.display()
            );
        }
    }
    Ok(root)
}
pub struct Atlas {
    pub texture: wgpu::Texture,
    pub map: wgpu::Buffer,
    tx: SyncSender<usize>,
    rx: std::sync::Mutex<Receiver<(usize, Vec<image::RgbaImage>)>>,
    slots: [Option<usize>; SLOTS],
    pending: Vec<usize>,
    desired: Vec<usize>,
    failed: Vec<usize>,
}
fn mips(mut img: image::RgbaImage) -> Vec<image::RgbaImage> {
    let mut out = Vec::new();
    loop {
        let (w, h) = img.dimensions();
        out.push(img.clone());
        if w == 1 && h == 1 {
            break;
        }
        // Average radiance in linear RGB; the alpha ocean mask is already linear.
        let mut next = image::RgbaImage::new((w / 2).max(1), (h / 2).max(1));
        for (x, y, p) in next.enumerate_pixels_mut() {
            let mut sum = [0f32; 4];
            for dy in 0..2 {
                for dx in 0..2 {
                    let q = img.get_pixel((x * 2 + dx).min(w - 1), (y * 2 + dy).min(h - 1));
                    for c in 0..4 {
                        let v = q[c] as f32 / 255.;
                        sum[c] += if c == 3 {
                            v
                        } else if v <= 0.04045 {
                            v / 12.92
                        } else {
                            ((v + 0.055) / 1.055).powf(2.4)
                        };
                    }
                }
            }
            for c in 0..4 {
                let v = sum[c] * 0.25;
                p[c] = ((if c == 3 {
                    v
                } else if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1. / 2.4) - 0.055
                }) * 255.)
                    .round() as u8;
            }
        }
        img = next;
    }
    out
}
pub fn texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &std::path::Path,
    label: &str,
) -> anyhow::Result<wgpu::Texture> {
    let img = image::open(path)?.to_rgba8();
    let levels = mips(img);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: levels[0].width(),
            height: levels[0].height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    upload(queue, &tex, 0, &levels);
    Ok(tex)
}
fn upload(queue: &wgpu::Queue, tex: &wgpu::Texture, layer: u32, levels: &[image::RgbaImage]) {
    for (i, img) in levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: i as u32,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(img.width() * 4),
                rows_per_image: Some(img.height()),
            },
            wgpu::Extent3d {
                width: img.width(),
                height: img.height(),
                depth_or_array_layers: 1,
            },
        );
    }
}
impl Atlas {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, root: std::path::PathBuf) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Earth bounded tile cache"),
            size: wgpu::Extent3d {
                width: SIDE,
                height: SIDE,
                depth_or_array_layers: SLOTS as u32,
            },
            mip_level_count: 11,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let map = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Earth resident tiles"),
            size: 800,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&map, 0, bytemuck::cast_slice(&[-1i32; 200]));
        let (tx, requests) = mpsc::sync_channel::<usize>(8);
        let (results, rx) = mpsc::sync_channel(4);
        std::thread::Builder::new()
            .name("earth imagery".into())
            .spawn(move || {
                while let Ok(id) = requests.recv() {
                    let path = root.join("surface").join(format!("{id:03}.png"));
                    let levels = match image::open(path) {
                        Ok(i) => mips(i.to_rgba8()),
                        Err(e) => {
                            log::warn!("Earth tile {id}: {e}");
                            Vec::new()
                        }
                    };
                    if results.send((id, levels)).is_err() {
                        break;
                    }
                }
            })
            .expect("Earth image worker");
        Self {
            texture,
            map,
            tx,
            rx: std::sync::Mutex::new(rx),
            slots: [None; SLOTS],
            pending: Vec::new(),
            desired: Vec::new(),
            failed: Vec::new(),
        }
    }
    pub fn resident_count(&self) -> usize {
        self.slots.iter().filter(|v| v.is_some()).count()
    }
    pub fn update(&mut self, queue: &wgpu::Queue, u: &Uniforms) {
        let o = [u.orbit[0], u.orbit[1], u.orbit[2]];
        let r = u.orbit[3];
        let mut ranked = Vec::<(usize, f32)>::new();
        for y in -4i32..=4 {
            for x in -6i32..=6 {
                let mut d = [0.; 3];
                for j in 0..3 {
                    d[j] = u.forward[j]
                        + u.right[j] * x as f32 / 6. * u.eye[3] * u.forward[3]
                        + u.up[j] * y as f32 / 4. * u.forward[3];
                }
                let norm = d.iter().map(|v| v * v).sum::<f32>().sqrt();
                for v in &mut d {
                    *v /= norm;
                }
                let b = (0..3).map(|j| o[j] * d[j]).sum::<f32>();
                let disc = b * b - o.iter().map(|v| v * v).sum::<f32>() + r * r;
                if disc < 0. {
                    continue;
                }
                let t = -b - disc.sqrt();
                if t <= 0. {
                    continue;
                }
                let p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
                let (s, c) = u.sun[3].sin_cos();
                let lon = (c * p[1] - s * p[0]).atan2(c * p[0] + s * p[1]);
                let lat = (p[2] / r).clamp(-1., 1.).asin();
                let col = (((lon / std::f32::consts::TAU + 0.5) * 20.).floor() as usize).min(19);
                let row = (((0.5 - lat / std::f32::consts::PI) * 10.).floor() as usize).min(9);
                let id = row * 20 + col;
                let rank = (x * x + y * y) as f32;
                if let Some(v) = ranked.iter_mut().find(|v| v.0 == id) {
                    v.1 = v.1.min(rank);
                } else {
                    ranked.push((id, rank));
                }
            }
        }
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        self.desired = ranked.iter().take(SLOTS).map(|v| v.0).collect();
        for _ in 0..2 {
            let Ok((id, levels)) = self
                .rx
                .get_mut()
                .expect("exclusive tile receiver")
                .try_recv()
            else {
                break;
            };
            self.pending.retain(|v| *v != id);
            if levels.is_empty() {
                self.failed.push(id);
                continue;
            }
            if !self.desired.contains(&id) {
                continue;
            }
            let slot = self.slots.iter().position(Option::is_none).or_else(|| {
                self.slots
                    .iter()
                    .position(|v| !self.desired.contains(&v.unwrap()))
            });
            if let Some(slot) = slot {
                upload(queue, &self.texture, slot as u32, &levels);
                self.slots[slot] = Some(id);
            }
        }
        let mut map = [-1i32; 200];
        for (slot, id) in self.slots.iter().enumerate() {
            if let Some(id) = id {
                map[*id] = slot as i32;
            }
        }
        queue.write_buffer(&self.map, 0, bytemuck::cast_slice(&map));
        for id in &self.desired {
            if !self.slots.contains(&Some(*id))
                && !self.pending.contains(id)
                && !self.failed.contains(id)
            {
                if self.tx.try_send(*id).is_ok() {
                    self.pending.push(*id);
                } else {
                    break;
                }
            }
        }
    }
}
pub fn clouds(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &std::path::Path,
) -> anyhow::Result<wgpu::Texture> {
    let mut img = image::open(path)?.to_luma8();
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("8K cloud optical density"),
        size: wgpu::Extent3d {
            width: img.width(),
            height: img.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 14,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for level in 0..14 {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(img.width()),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: img.width(),
                height: img.height(),
                depth_or_array_layers: 1,
            },
        );
        img = image::imageops::resize(
            &img,
            (img.width() / 2).max(1),
            (img.height() / 2).max(1),
            image::imageops::FilterType::Triangle,
        );
    }
    Ok(tex)
}
pub fn cloud_noise(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &std::path::Path,
) -> anyhow::Result<wgpu::Texture> {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Periodic cloud erosion volume"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 64,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
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
        &std::fs::read(path)?,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(64),
            rows_per_image: Some(64),
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 64,
        },
    );
    Ok(tex)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mip_filter_averages_light_and_preserves_water_fraction() {
        let img = image::RgbaImage::from_raw(
            2,
            2,
            vec![
                0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255,
            ],
        )
        .unwrap();
        let chain = mips(img);
        let p = chain[1].get_pixel(0, 0);
        assert!((p[0] as i32 - 188).abs() <= 1);
        assert_eq!(p[3], 128);
    }
    #[test]
    fn mip_chain_reaches_one_texel_for_guttered_tiles() {
        let chain = mips(image::RgbaImage::from_pixel(
            SIDE,
            SIDE,
            image::Rgba([128, 64, 32, 255]),
        ));
        assert_eq!(chain.len(), 11);
        assert_eq!(chain.last().unwrap().dimensions(), (1, 1));
    }
}
