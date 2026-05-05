//! FontSystem — TTF → GPU Texture Atlas

use std::collections::HashMap;
use ab_glyph::{Font as AbFont, FontRef, PxScale, ScaleFont};
use crate::core::AssetRegistry;
use crate::ecs::object::mesh::Vertex;
use crate::ecs::ui::text::Text;

pub const MODE_FONT: f32 = 1.0;

pub struct FontTexture {
    pub texture:     wgpu::Texture,
    pub view:        wgpu::TextureView,
    pub sampler:     wgpu::Sampler,
    pub glyph_uvs:   HashMap<char, [f32; 4]>,
    pub advances:    HashMap<char, f32>,
    pub atlas_scale: f32,
}

pub struct FontSystem {
    pub cache: HashMap<String, FontTexture>,
}

impl FontSystem {
    pub fn new() -> Self { Self { cache: HashMap::new() } }

    pub fn ensure_font(&mut self, font_key: &str, registry: &AssetRegistry,
                       device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.cache.contains_key(font_key) { return; }
        let bytes = match registry.font(font_key) {
            Some(b) => b,
            None => { eprintln!("[FontSystem] not found: {font_key}"); return; }
        };
        match Self::build(bytes, device, queue) {
            Ok(ft) => { self.cache.insert(font_key.to_string(), ft); }
            Err(e) => eprintln!("[FontSystem] build error: {e}"),
        }
    }

    fn build(font_bytes: &[u8], device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<FontTexture> {
        let font        = FontRef::try_from_slice(font_bytes)?;
        let atlas_scale = 48.0f32;
        let px_scale    = PxScale::from(atlas_scale);
        let scaled      = font.as_scaled(px_scale);

        let chars: Vec<char> = (0x20u32..0x7Fu32)
            .chain(0x3040u32..0x30A0u32)
            .chain(0x30A0u32..0x3100u32)
            .filter_map(char::from_u32)
            .collect();

        let cell_w  = 54u32;
        let cell_h  = 58u32;
        let cols    = 32u32;
        let rows    = (chars.len() as u32 + cols - 1) / cols;
        let atlas_w = cols * cell_w;
        let atlas_h = rows * cell_h;

        let mut atlas     = vec![0u8; (atlas_w * atlas_h) as usize];
        let mut glyph_uvs = HashMap::new();
        let mut advances  = HashMap::new();

        for (i, &c) in chars.iter().enumerate() {
            let col = (i as u32) % cols;
            let row = (i as u32) / cols;
            let ox  = col * cell_w;
            let oy  = row * cell_h;

            let gid   = font.glyph_id(c);
            let glyph = gid.with_scale_and_position(px_scale, ab_glyph::point(2.0, atlas_scale * 0.8));
            advances.insert(c, scaled.h_advance(gid));

            if let Some(og) = font.outline_glyph(glyph) {
                og.draw(|gx, gy, v| {
                    let px = ox + gx;
                    let py = oy + gy;
                    if px < atlas_w && py < atlas_h {
                        let idx = (py * atlas_w + px) as usize;
                        atlas[idx] = atlas[idx].saturating_add((v * 255.0) as u8);
                    }
                });
            }

            glyph_uvs.insert(c, [
                ox as f32 / atlas_w as f32,
                oy as f32 / atlas_h as f32,
                (ox + cell_w) as f32 / atlas_w as f32,
                (oy + cell_h) as f32 / atlas_h as f32,
            ]);
        }

        let rgba: Vec<u8> = atlas.iter().flat_map(|&a| [255u8, 255, 255, a]).collect();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("font_atlas"),
            size:            wgpu::Extent3d { width: atlas_w, height: atlas_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &rgba,
            wgpu::TexelCopyBufferLayout { offset: 0,
                bytes_per_row: Some(atlas_w * 4), rows_per_image: Some(atlas_h) },
            wgpu::Extent3d { width: atlas_w, height: atlas_h, depth_or_array_layers: 1 },
        );

        let view    = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // ★ fix: mipmap_filter は MipmapFilterMode
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("font_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::MipmapFilterMode::Nearest,  // ← 修正
            ..Default::default()
        });

        Ok(FontTexture { texture, view, sampler, glyph_uvs, advances, atlas_scale })
    }

    pub fn build_mesh(&self, text: &Text, origin_x: f32, origin_y: f32) -> Vec<Vertex> {
        let ft = match self.cache.get(&text.font_key) { Some(ft) => ft, None => return vec![] };
        let scale_ratio = text.size.0 / ft.atlas_scale;
        let cell_h_px   = ft.atlas_scale * scale_ratio;
        let mut verts   = Vec::new();
        let mut cx      = origin_x;
        let col         = text.color;

        for c in text.content.chars() {
            let uv     = match ft.glyph_uvs.get(&c) { Some(u) => *u, None => continue };
            let adv_px = ft.advances.get(&c).copied().unwrap_or(ft.atlas_scale * 0.6) * scale_ratio;
            let [u0, v0, u1, v1] = uv;
            let (x0, y0, x1, y1) = (cx, origin_y, cx + adv_px, origin_y + cell_h_px);

            verts.extend_from_slice(&[
                Vertex { position: [x0, y0, MODE_FONT], uv: [u0, v0], color: col },
                Vertex { position: [x1, y0, MODE_FONT], uv: [u1, v0], color: col },
                Vertex { position: [x1, y1, MODE_FONT], uv: [u1, v1], color: col },
                Vertex { position: [x0, y0, MODE_FONT], uv: [u0, v0], color: col },
                Vertex { position: [x1, y1, MODE_FONT], uv: [u1, v1], color: col },
                Vertex { position: [x0, y1, MODE_FONT], uv: [u0, v1], color: col },
            ]);
            cx += adv_px;
        }
        verts
    }
}

impl Default for FontSystem { fn default() -> Self { Self::new() } }
