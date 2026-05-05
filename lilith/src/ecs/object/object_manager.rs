//! ObjectManager — エンティティのライフサイクル管理
//!
//! spawn / despawn / query を提供する。
//! 全コンポーネントストアをここが所有する。

use std::collections::HashSet;
use crate::ecs::component::EntityId;
use crate::ecs::object::object::ComponentStore;
use crate::ecs::object::transform::Transform;
use crate::ecs::object::mesh::Mesh;
use crate::ecs::ui::text::Text;
use crate::ecs::ui::canvas::UiElement;

/// エンティティ生成カウンタ
static NEXT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

fn new_id() -> EntityId {
    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// シーン全体のコンポーネントストアを束ねる
pub struct ObjectManager {
    pub alive: HashSet<EntityId>,

    // ─── コンポーネントストア ─────────────────────
    pub transforms:  ComponentStore<Transform>,
    pub meshes:      ComponentStore<Mesh>,
    pub texts:       ComponentStore<Text>,
    pub ui_elements: ComponentStore<UiElement>,
}

impl ObjectManager {
    pub fn new() -> Self {
        Self {
            alive:       HashSet::new(),
            transforms:  ComponentStore::new(),
            meshes:      ComponentStore::new(),
            texts:       ComponentStore::new(),
            ui_elements: ComponentStore::new(),
        }
    }

    /// 空エンティティを生成して ID を返す
    pub fn spawn(&mut self) -> EntityId {
        let id = new_id();
        self.alive.insert(id);
        id
    }

    /// エンティティと全コンポーネントを削除
    pub fn despawn(&mut self, id: EntityId) {
        self.alive.remove(&id);
        self.transforms.remove(id);
        self.meshes.remove(id);
        self.texts.remove(id);
        self.ui_elements.remove(id);
    }

    pub fn is_alive(&self, id: EntityId) -> bool {
        self.alive.contains(&id)
    }
}

impl Default for ObjectManager {
    fn default() -> Self { Self::new() }
}
