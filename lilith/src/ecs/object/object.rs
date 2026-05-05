//! Object / Scene グラフ
//!
//! GameObject = EntityId + Component の SoA (Structure of Arrays) 管理
//! 全コンポーネントは HashMap<EntityId, T> で保持。

use std::collections::HashMap;
use crate::ecs::component::{Component, EntityId};

/// ゲームオブジェクト本体は ID だけ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GameObject {
    pub id: EntityId,
}

/// コンポーネントストレージ: 型ごとに HashMap
pub struct ComponentStore<T: Component> {
    pub data: HashMap<EntityId, T>,
}

impl<T: Component> ComponentStore<T> {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }
    pub fn insert(&mut self, id: EntityId, component: T) {
        self.data.insert(id, component);
    }
    pub fn get(&self, id: EntityId) -> Option<&T> {
        self.data.get(&id)
    }
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut T> {
        self.data.get_mut(&id)
    }
    pub fn remove(&mut self, id: EntityId) {
        self.data.remove(&id);
    }
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &T)> {
        self.data.iter().map(|(&id, c)| (id, c))
    }
}

impl<T: Component> Default for ComponentStore<T> {
    fn default() -> Self { Self::new() }
}
