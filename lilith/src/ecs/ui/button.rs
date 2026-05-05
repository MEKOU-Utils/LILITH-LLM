//! Button コンポーネント
//!
//! UiElement + Text の組み合わせを1つのコンポーネントで表現。
//! GPU描画時は UiElement(背景) + Text(ラベル) に分解される。

use crate::ecs::component::Component;
use crate::ecs::ui::canvas::{Rect, UiElement, Action};
use crate::ecs::ui::text::Text;

#[derive(Debug, Clone)]
pub struct Button {
    pub background: UiElement,
    pub label:      Text,
    /// ホバー時の色変化
    pub hover_color: [f32; 4],
    pub is_hovered:  bool,
    pub is_pressed:  bool,
}

impl Button {
    pub fn new(
        rect: Rect,
        label_text: impl Into<String>,
        font_key: impl Into<String>,
        shader_key: impl Into<String>,
    ) -> Self {
        let mut bg = UiElement::new(rect, shader_key);
        bg.action = Action::Click;

        Self {
            background:  bg,
            label:       Text::new(label_text, font_key),
            hover_color: [0.25, 0.25, 0.25, 1.0],
            is_hovered:  false,
            is_pressed:  false,
        }
    }
}

impl Component for Button {}
