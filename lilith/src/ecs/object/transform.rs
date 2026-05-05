//! Transform — 2D/3D 空間位置データ
//!
//! GPU に流す行列を生成する責務だけを持つ。
//! ロジックはシステム側が担う。

use crate::ecs::component::Component;

/// スクリーン空間 or ワールド空間の位置・サイズ
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    /// 位置 (x, y, z)  z = depth / z-index
    pub position: [f32; 3],
    /// スケール (w, h)
    pub scale: [f32; 2],
    /// Z 軸回転 (ラジアン)
    pub rotation: f32,
}

impl Transform {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            scale: [1.0, 1.0],
            rotation: 0.0,
        }
    }

    /// NDC 空間への 4x4 モデル行列 (column-major)
    pub fn to_matrix(&self) -> [[f32; 4]; 4] {
        let (sx, sy) = (self.scale[0], self.scale[1]);
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        [
            [sx * cos, sx * sin, 0.0, 0.0],
            [-sy * sin, sy * cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [self.position[0], self.position[1], self.position[2], 1.0],
        ]
    }
}

impl Component for Transform {}
