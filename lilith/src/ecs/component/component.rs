//! ECS コアの型定義
//!
//! エンティティは u32 ID だけ。
//! コンポーネントは純粋データ構造体 — ロジックを持たない。
//! システムが &[Component] を受け取って処理する。

/// エンティティID — u32 一枚
pub type EntityId = u32;

/// 全コンポーネントが実装するマーカー trait
/// serialize / deserialize はここに生やす（将来）
pub trait Component: Send + Sync + 'static {}
