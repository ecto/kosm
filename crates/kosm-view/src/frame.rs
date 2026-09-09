//! What a window and a still share: the camera, and reading a texture back.

use eframe::egui_wgpu::wgpu;
use tang::Vec3 as V;

/// Camera shared by the live and reference tiers.
#[derive(Clone, Copy)]
pub struct Camera {
    pub eye: V<f64>,
    pub target: V<f64>,
    pub vfov: f64,
}

impl Camera {
    pub fn view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        let f = (self.target - self.eye).normalize();
        let r = f.cross(&V::new(0.0, 0.0, 1.0)).normalize();
        let u = r.cross(&f);
        let e = self.eye;
        // view: rows r, u, -f (right-handed, looking down -z in view space)
        let view = [
            [r.x, u.x, -f.x, 0.0],
            [r.y, u.y, -f.y, 0.0],
            [r.z, u.z, -f.z, 0.0],
            [-r.dot(&e), -u.dot(&e), f.dot(&e), 1.0],
        ];
        let (near, far) = (0.05f64, 60.0f64);
        let t = 1.0 / (0.5 * self.vfov).tan();
        // wgpu clip z in [0, 1]
        let proj = [
            [t / aspect as f64, 0.0, 0.0, 0.0],
            [0.0, t, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), -1.0],
            [0.0, 0.0, near * far / (near - far), 0.0],
        ];
        // column-major product proj * view
        let mut out = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for rr in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += proj[k][rr] * view[c][k];
                }
                out[c][rr] = s as f32;
            }
        }
        out
    }
}

/// What one frame needs from the recording.

pub fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, (w, h): (u32, u32)) -> image::RgbaImage {
    let row = ((4 * w + 255) / 256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv().expect("map").expect("map ok");
    let data = slice.get_mapped_range().expect("mapped");
    let mut px = Vec::with_capacity((4 * w * h) as usize);
    for y in 0..h {
        px.extend_from_slice(&data[(y * row) as usize..(y * row + 4 * w) as usize]);
    }
    drop(data);
    buf.unmap();
    image::RgbaImage::from_raw(w, h, px).expect("image")
}

