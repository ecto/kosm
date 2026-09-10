//! The settle blend without the readback.
//!
//! [`settle::present`](super::settle::present) mixes the two tiers in memory,
//! which means the raster frame has to *be* in memory: a 1280×720 frame is
//! 3.7 MB off the device, a million-pixel lerp on one thread, and 3.7 MB back
//! up — every frame, for as long as the player is standing still. Measured, at
//! sixty frames a second: five to eight milliseconds a frame, which took a
//! window that walks at sixty down to fifty when it stopped.
//!
//! Nothing about that mix needs a CPU. The raster frame is already a texture on
//! the viewport's own device and the reference is two megabytes that arrive a
//! few times a second, so this is one full-screen pass over the two of them and
//! the result never leaves the device at all — the same "nothing crosses the
//! bus" the walking frame has always had, extended to the standing one.
//!
//! The CPU version stays, and stays tested: `--shot --tier raster --settle`
//! blends a still that is *going* to a file, where the bytes are the point.

use std::sync::Arc;

/// The blend pass: two textures in, one out.
pub struct Blend {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    knobs: wgpu::Buffer,
    /// The reference frame on the device, and what size it is.
    reference: Option<(wgpu::Texture, wgpu::TextureView, (u32, u32))>,
    /// Bumped every time a new reference is uploaded, so the bind group is
    /// remade when it must be and not when it need not.
    generation: u64,
    /// The frame this pass writes: the raster's size, `Rgba8Unorm` holding
    /// already-sRGB bytes, with `Rgba8UnormSrgb` among its view formats so it
    /// can go straight to [`crate::viewport::Image::Texture`].
    out: Option<(Arc<wgpu::Texture>, wgpu::TextureView, (u32, u32))>,
    bind: Option<wgpu::BindGroup>,
    /// What that bind group was built from: the raster texture it samples and
    /// the reference generation it was made at.
    bound: Option<(Arc<wgpu::Texture>, u64)>,
}

/// The one format both tiers' bytes live in.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

impl Blend {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("settle blend"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blend.wgsl").into()),
        });
        let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("settle blend"),
            entries: &[
                texture(0),
                texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("settle blend"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("settle blend"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            layout,
            // Linear and clamped: the reference is smaller than the raster, and
            // this is the bilinear upscale on pixel centres `present` does by
            // hand.
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("settle blend"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            knobs: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("settle blend"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            reference: None,
            generation: 0,
            out: None,
            bind: None,
            bound: None,
        }
    }

    /// Whether there is a reference to blend toward.
    pub fn has_reference(&self) -> bool {
        self.reference.is_some()
    }

    /// The reference has moved on: what is held is of a pose that is gone.
    pub fn forget(&mut self) {
        self.reference = None;
        self.bind = None;
        self.bound = None;
    }

    /// Put a resolved reference frame on the device. `rgba` is
    /// already-sRGB RGBA8, `4 · w · h` bytes.
    pub fn set_reference(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &[u8],
        size: (u32, u32),
    ) {
        let (w, h) = size;
        if w == 0 || h == 0 || rgba.len() < (4 * w as usize * h as usize) {
            return;
        }
        let fresh = self.reference.as_ref().map(|(_, _, s)| *s) != Some(size);
        if fresh {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("settle reference"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.reference = Some((tex, view, size));
            self.bind = None;
            self.bound = None;
        }
        let Some((tex, _, _)) = &self.reference else { return };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba[..4 * w as usize * h as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.generation += 1;
    }

    /// `mix(raster, reference, blend)`, into a texture of the raster's size.
    ///
    /// [`None`] when there is nothing to blend — no reference yet — in which
    /// case the caller hands the raster's own texture over as it is.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Arc<wgpu::Texture>,
        blend: f32,
    ) -> Option<Arc<wgpu::Texture>> {
        self.reference.as_ref()?;
        let size = (raster.width(), raster.height());
        if size.0 == 0 || size.1 == 0 {
            return None;
        }
        self.ensure_out(device, size);
        self.ensure_bind(device, raster);
        queue.write_buffer(&self.knobs, 0, bytemuck::bytes_of(&[blend, 0.0, 0.0, 0.0f32]));
        let (tex, view, _) = self.out.as_ref()?;
        let out = tex.clone();
        let bind = self.bind.as_ref()?;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("settle blend"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("settle blend"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..4, 0..1);
        }
        queue.submit([enc.finish()]);
        Some(out)
    }

    fn ensure_out(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.out.as_ref().map(|(_, _, s)| *s) == Some(size) {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("settle blend out"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            // the viewport blits an already-sRGB image through an sRGB view
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(FORMAT),
            ..Default::default()
        });
        self.out = Some((Arc::new(tex), view, size));
    }

    fn ensure_bind(&mut self, device: &wgpu::Device, raster: &Arc<wgpu::Texture>) {
        let same = self
            .bound
            .as_ref()
            .is_some_and(|(t, g)| Arc::ptr_eq(t, raster) && *g == self.generation);
        if same && self.bind.is_some() {
            return;
        }
        let Some((_, reference, _)) = &self.reference else { return };
        // The raster target is `Rgba8Unorm` carrying already-sRGB bytes, and it
        // is sampled *as* `Rgba8Unorm`: the mix is in code space, which is
        // where `present` does it.
        let rv = raster.create_view(&wgpu::TextureViewDescriptor {
            format: Some(FORMAT),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });
        self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("settle blend"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&rv) },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(reference),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.knobs.as_entire_binding(),
                },
            ],
        }));
        self.bound = Some((raster.clone(), self.generation));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A texture of one colour, as the pass's raster input.
    fn flat(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        value: u8,
    ) -> (Arc<wgpu::Texture>, Vec<u8>) {
        let bytes = vec![value; 4 * size.0 as usize * size.1 as usize];
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test raster"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
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
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size.0),
                rows_per_image: Some(size.1),
            },
            wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        );
        (Arc::new(tex), bytes)
    }

    /// **The device's mix is the bytes' mix.** The same two pictures, the same
    /// blend, through [`super::super::settle::present`] and through this pass:
    /// the answers agree to a code, which is what makes replacing the readback
    /// a performance change and not a picture change.
    ///
    /// The reference is *smaller* than the raster, so the bilinear upscale is
    /// in the comparison too — a linear sampler on pixel centres against
    /// `present`'s own loop.
    #[test]
    fn the_device_blend_is_the_bytes_blend() -> anyhow::Result<()> {
        let Ok(ctx) = kosm_render::gpu::GpuContext::init_blocking() else {
            eprintln!("skipping the_device_blend_is_the_bytes_blend: no GPU");
            return Ok(());
        };
        let (device, queue) = (&ctx.device, &ctx.queue);
        let size = (64u32, 32u32);
        let small = (16u32, 8u32);
        let (raster, raster_bytes) = flat(device, queue, size, 40);
        let traced = vec![210u8; 4 * small.0 as usize * small.1 as usize];

        let mut blend = Blend::new(device);
        assert!(!blend.has_reference());
        assert!(blend.draw(device, queue, &raster, 0.5).is_none(), "a blend with no reference");
        blend.set_reference(device, queue, &traced, small);
        assert!(blend.has_reference());

        for b in [0.0f32, 0.25, 0.5, 1.0] {
            let out = blend.draw(device, queue, &raster, b).expect("the pass drew");
            let got = crate::frame::read_back(device, queue, &out, size).into_raw();
            let want = super::super::settle::present(&raster_bytes, size, &traced, small, b);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                // Colour only. `present` copies the raster's own alpha through
                // at a blend of zero and writes 255 above it; the pass always
                // writes 255, which is what an opaque frame is and what the
                // window's raster target carries anyway.
                if i % 4 == 3 {
                    assert_eq!(*g, 255, "the blended frame is not opaque");
                    continue;
                }
                assert!(
                    g.abs_diff(*w) <= 1,
                    "at blend {b}, byte {i} is {g} on the device and {w} in memory"
                );
            }
        }

        // and a move throws the reference away, so the next frame is the
        // raster's alone rather than a picture of a pose that is gone
        blend.forget();
        assert!(!blend.has_reference());
        assert!(blend.draw(device, queue, &raster, 1.0).is_none());
        Ok(())
    }
}
