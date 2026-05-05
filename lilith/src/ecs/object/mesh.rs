//! Mesh コンポーネント — 頂点データへの参照
//!
//! 実際のバイト列は AssetRegistry に置き、
//! Mesh はキー文字列だけを保持する（データ指向）。

use crate::ecs::component::Component;

/// 頂点フォーマット: position(xyz) + uv(xy) + color(rgba)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv:       [f32; 2],
    pub color:    [f32; 4],
}

impl Vertex {
    /// 2D クワッド (x,y,w,h) を NDC 正規化して頂点6枚生成
    pub fn quad_2d(x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) -> [Vertex; 6] {
        let (x0, y0, x1, y1) = (x, y, x + w, y + h);
        [
            Vertex { position: [x0, y0, 0.0], uv: [0.0, 1.0], color },
            Vertex { position: [x1, y0, 0.0], uv: [1.0, 1.0], color },
            Vertex { position: [x1, y1, 0.0], uv: [1.0, 0.0], color },
            Vertex { position: [x0, y0, 0.0], uv: [0.0, 1.0], color },
            Vertex { position: [x1, y1, 0.0], uv: [1.0, 0.0], color },
            Vertex { position: [x0, y1, 0.0], uv: [0.0, 0.0], color },
        ]
    }
}

/// メッシュコンポーネント
#[derive(Debug, Clone)]
pub struct Mesh {
    /// AssetRegistry のキー ("quad", "sprite" 等)
    /// None なら頂点を inline で持つ
    pub asset_key: Option<String>,
    /// インライン頂点 (動的生成テキスト等)
    pub vertices: Vec<Vertex>,
    /// 対応するシェーダキー
    pub shader_key: String,
}

impl Mesh {
    pub fn from_asset(asset_key: impl Into<String>, shader_key: impl Into<String>) -> Self {
        Self {
            asset_key: Some(asset_key.into()),
            vertices: vec![],
            shader_key: shader_key.into(),
        }
    }

    pub fn from_vertices(vertices: Vec<Vertex>, shader_key: impl Into<String>) -> Self {
        Self {
            asset_key: None,
            vertices,
            shader_key: shader_key.into(),
        }
    }
}

impl Component for Mesh {}
