//! Engine — データ指向のコアループ
//!
//! Engine 自体はロジックを持たない。
//! ECS コンポーネント（Shader, Vertex, Font）と
//! AssetRegistry を束ねるだけ。

use crate::core::asset_loader::AssetRegistry;

/// エンジンコア
pub struct Engine {
    pub registry: AssetRegistry,
}

impl Engine {
    pub fn new(registry: AssetRegistry) -> Self {
        Self { registry }
    }

    pub fn update(&mut self) {
        // ECS システム呼び出しをここに積む
    }
}
