//! InputText コンポーネント
//!
//! テキスト入力フィールド。
//! カーソル位置とバッファだけを保持。描画は Text + UiElement に委譲。

use crate::ecs::component::Component;
use crate::ecs::ui::canvas::{Rect, UiElement};
use crate::ecs::ui::text::Text;

#[derive(Debug, Clone)]
pub struct InputText {
    pub background:   UiElement,
    pub display_text: Text,
    /// 実際の入力バッファ
    pub buffer:       String,
    /// カーソル位置 (バイトインデックス)
    pub cursor:       usize,
    pub is_focused:   bool,
    pub placeholder:  String,
}

impl InputText {
    pub fn new(
        rect: Rect,
        font_key: impl Into<String>,
        shader_key: impl Into<String>,
        placeholder: impl Into<String>,
    ) -> Self {
        let font_key = font_key.into();
        let placeholder = placeholder.into();

        Self {
            background:   UiElement::new(rect, shader_key),
            display_text: Text::new("", &font_key),
            buffer:       String::new(),
            cursor:       0,
            is_focused:   false,
            placeholder,
        }
    }

    pub fn push_char(&mut self, c: char) {
        self.buffer.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.display_text.content = self.buffer.clone();
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.buffer[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.buffer.remove(prev);
            self.cursor = prev;
            self.display_text.content = self.buffer.clone();
        }
    }
}

impl Component for InputText {}
