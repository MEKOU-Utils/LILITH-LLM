//! gltf_loader.rs — glTF / GLB ローダー
//!
//! gltf クレートを使って .glb / .gltf を読み込み、
//! wgpu 用の Vertex 配列に変換する。
//!
//! ## 使い方
//! ```rust
//! let scene = load_gltf("assets/meshes/model.glb")?;
//! eprintln!("meshes={} nodes={}", scene.meshes.len(), scene.nodes.len());
//! // scene.meshes[i].vertices を wgpu Buffer に流す
//! ```

use anyhow::{Context, Result};
use crate::ecs::object::mesh::Vertex;

// ─────────────────────────────────────────────────────────────────
// 公開型
// ─────────────────────────────────────────────────────────────────

/// ロード済みメッシュ (描画可能な頂点リスト)
#[derive(Debug, Clone)]
pub struct GltfMesh {
    pub name:     String,
    pub vertices: Vec<Vertex>,
}

/// シーンノード (階層は簡略化して配列で管理)
#[derive(Debug, Clone)]
pub struct GltfNode {
    pub name:      String,
    pub mesh_idx:  Option<usize>,
    /// TRS 行列 (列優先 4×4, 単位行列デフォルト)
    pub transform: [[f32; 4]; 4],
}

/// ロード済み glTF シーン
pub struct GltfScene {
    pub meshes: Vec<GltfMesh>,
    pub nodes:  Vec<GltfNode>,
}

// ─────────────────────────────────────────────────────────────────
// ローダー
// ─────────────────────────────────────────────────────────────────

/// glTF / GLB ファイルを読み込んで GltfScene を返す
pub fn load_gltf(path: &str) -> Result<GltfScene> {
    let (doc, buffers, _images) = gltf::import(path)
        .with_context(|| format!("glTF import failed: {path}"))?;

    let mut meshes: Vec<GltfMesh> = Vec::new();

    for mesh in doc.meshes() {
        let mut verts: Vec<Vertex> = Vec::new();

        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| buffers.get(buf.index()).map(|b| b.0.as_slice()));

            // ── 位置 ──────────────────────────────────────────────
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|it| it.collect())
                .unwrap_or_default();

            // ── 法線 (あれば) ─────────────────────────────────────
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|it| it.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            // ── UV (あれば) ───────────────────────────────────────
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|it| it.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            // ── インデックス ──────────────────────────────────────
            let indices: Vec<usize> = if let Some(idx) = reader.read_indices() {
                idx.into_u32().map(|i| i as usize).collect()
            } else {
                (0..positions.len()).collect()
            };

            // ── Vertex 変換 ───────────────────────────────────────
            for &i in &indices {
                let pos = if i < positions.len() { positions[i] } else { [0.0; 3] };
                let uv  = if i < uvs.len()       { uvs[i]       } else { [0.0; 2] };
                let n   = if i < normals.len()    { normals[i]   } else { [0.0, 1.0, 0.0] };

                // 法線をカラーとして利用 (ノーマルビジュアライゼーション)
                let color = [
                    n[0] * 0.5 + 0.5,
                    n[1] * 0.5 + 0.5,
                    n[2] * 0.5 + 0.5,
                    1.0,
                ];
                verts.push(Vertex {
                    position: pos,
                    uv,
                    color,
                });
            }
        }

        meshes.push(GltfMesh {
            name: mesh.name().unwrap_or("mesh").to_string(),
            vertices: verts,
        });
    }

    // ── ノード ────────────────────────────────────────────────────
    let nodes: Vec<GltfNode> = doc.nodes().map(|node| {
        let transform = node_transform(&node);
        GltfNode {
            name:     node.name().unwrap_or("node").to_string(),
            mesh_idx: node.mesh().map(|m| m.index()),
            transform,
        }
    }).collect();

    eprintln!("[glTF] loaded: {} meshes, {} nodes from {}", meshes.len(), nodes.len(), path);
    Ok(GltfScene { meshes, nodes })
}

/// ノードの変換行列を取得 (TRS → 4×4 列優先)
fn node_transform(node: &gltf::Node) -> [[f32; 4]; 4] {
    let m = node.transform().matrix();
    m  // gltf クレートは既に [[f32;4];4] を返す
}

// ─────────────────────────────────────────────────────────────────
// 3D 描画用ユーティリティ
// ─────────────────────────────────────────────────────────────────

/// シーンの全メッシュを結合した頂点バッファを返す
pub fn scene_vertices(scene: &GltfScene) -> Vec<Vertex> {
    scene.meshes.iter().flat_map(|m| m.vertices.iter().copied()).collect()
}
