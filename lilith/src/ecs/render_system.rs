//! render_system.rs — ECS全体を走査してGPUに頂点を流す
//!
//! ## 重要: 座標系
//! 頂点はスクリーンピクセル座標 (左上原点, Y下向き) で持つ。
//! シェーダのvs_mainでscreen_size Uniformを使ってNDC変換する。
//!
//! ## 描画モード (position.z に埋め込む)
//!   0.0 = solid color (UI背景)
//!   1.0 = font texture (文字)
//!   2.0 = progress bar (アニメーション)

use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::core::AssetRegistry;
use crate::ecs::object::{ObjectManager, Vertex};
use crate::ecs::shader::FontSystem;

pub const MODE_SOLID:    f32 = 0.0;
pub const MODE_FONT:     f32 = 1.0;
pub const MODE_PROGRESS: f32 = 2.0;

pub struct RenderSystem {
    pub font_system: FontSystem,
}

impl RenderSystem {
    pub fn new() -> Self {
        Self { font_system: FontSystem::new() }
    }

    pub fn prepare(
        &mut self,
        world: &ObjectManager,
        registry: &AssetRegistry,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        for (_, text) in world.texts.iter() {
            self.font_system.ensure_font(&text.font_key, registry, device, queue);
        }
    }

    /// 全エンティティを走査して頂点バッファを生成
    pub fn build_vertex_buffer(
        &self,
        world: &ObjectManager,
        device: &wgpu::Device,
    ) -> Option<(wgpu::Buffer, u32)> {
        let mut all_verts: Vec<Vertex> = Vec::new();

        // ── UiElement (solid rect) — 背景を最初に描画 ────────────
        for (_, ui) in world.ui_elements.iter() {
            let r = &ui.rect;
            let mode = if ui.shader_key == "progress" { MODE_PROGRESS } else { MODE_SOLID };
            let verts = quad_px(r.x, r.y, r.width, r.height, ui.color, mode);
            all_verts.extend_from_slice(&verts);
        }

        // ── Mesh (solid geometry) ──────────────────────────────────
        for (id, mesh) in world.meshes.iter() {
            all_verts.extend_from_slice(&mesh.vertices);
        }

        // ── Text エンティティ — テキストを最後に描画（最前面）──────
        for (id, text) in world.texts.iter() {
            let (ox, oy) = if let Some(t) = world.transforms.get(id) {
                (t.position[0], t.position[1])
            } else {
                (0.0, 0.0)
            };
            let verts = self.font_system.build_mesh(text, ox, oy);
            all_verts.extend(verts);
        }

        if all_verts.is_empty() { return None; }

        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("dynamic_vbuf"),
            contents: bytemuck::cast_slice(&all_verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let count = all_verts.len() as u32;
        Some((buf, count))
    }

    /// FontTexture の view/sampler を返す (BindGroup構築用)
    pub fn font_texture(&self, key: &str) -> Option<(&wgpu::TextureView, &wgpu::Sampler)> {
        let ft = self.font_system.cache.get(key)?;
        Some((&ft.view, &ft.sampler))
    }
}

impl Default for RenderSystem {
    fn default() -> Self { Self::new() }
}

/// ピクセル座標でクワッド6頂点を生成 (mode = position.z)
pub fn quad_px(x: f32, y: f32, w: f32, h: f32, color: [f32; 4], mode: f32) -> [Vertex; 6] {
    let (x0, y0, x1, y1) = (x, y, x + w, y + h);
    [
        Vertex { position: [x0, y0, mode], uv: [0.0, 0.0], color },
        Vertex { position: [x1, y0, mode], uv: [1.0, 0.0], color },
        Vertex { position: [x1, y1, mode], uv: [1.0, 1.0], color },
        Vertex { position: [x0, y0, mode], uv: [0.0, 0.0], color },
        Vertex { position: [x1, y1, mode], uv: [1.0, 1.0], color },
        Vertex { position: [x0, y1, mode], uv: [0.0, 1.0], color },
    ]
}
