//! Canvas / UiElement コンポーネント
//!
//! UI の矩形領域とインタラクション種別を保持。
//! イベント処理ロジックは System 側。

use crate::ecs::component::Component;

/// 矩形 (スクリーン座標 px)
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width
            && py >= self.y && py <= self.y + self.height
    }
}

/// UI が受け付けるアクション種別
#[derive(Debug, Clone)]
pub enum Action {
    /// クリック / タップ
    Click,
    /// テキスト入力
    KeyInput,
    /// ホバー
    Hover,
    /// なし (静的要素)
    None,
}

/// UI 要素コンポーネント
#[derive(Debug, Clone)]
pub struct UiElement {
    pub rect:      Rect,
    pub z_index:   u32,
    /// 対応する描画シェーダキー
    pub shader_key: String,
    /// このUIが受けるアクション
    pub action:    Action,
    /// 背景色 RGBA
    pub color:     [f32; 4],
}

impl UiElement {
    pub fn new(rect: Rect, shader_key: impl Into<String>) -> Self {
        Self {
            rect,
            z_index: 0,
            shader_key: shader_key.into(),
            action: Action::None,
            color: [0.15, 0.15, 0.15, 1.0],
        }
    }
}

impl Component for UiElement {}
