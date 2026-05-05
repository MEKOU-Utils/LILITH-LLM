//! Text コンポーネント
//!
//! テキスト文字列 + フォントキー（AssetRegistry参照）を保持。
//! 実際のラスタライズは FontSystem（GPU System）が行う。

use crate::ecs::component::Component;

/// フォントサイズ単位
#[derive(Debug, Clone, Copy)]
pub struct FontSize(pub f32);

/// テキストの水平揃え
#[derive(Debug, Clone, Copy)]
pub enum Align {
    Left,
    Center,
    Right,
}

/// テキストコンポーネント — データのみ
#[derive(Debug, Clone)]
pub struct Text {
    /// 表示文字列
    pub content: String,
    /// AssetRegistry のフォントキー (例: "genkai-mincho")
    pub font_key: String,
    /// フォントサイズ (px)
    pub size: FontSize,
    /// 色 RGBA [0.0 - 1.0]
    pub color: [f32; 4],
    /// 文字揃え
    pub align: Align,
    /// レイアウト幅制限 (None = 無制限)
    pub max_width: Option<f32>,
}

impl Text {
    pub fn new(content: impl Into<String>, font_key: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            font_key: font_key.into(),
            size: FontSize(16.0),
            color: [1.0, 1.0, 1.0, 1.0],
            align: Align::Left,
            max_width: None,
        }
    }

    pub fn with_size(mut self, px: f32) -> Self {
        self.size = FontSize(px);
        self
    }

    pub fn with_color(mut self, rgba: [f32; 4]) -> Self {
        self.color = rgba;
        self
    }
}

impl Component for Text {}
