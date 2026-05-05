//! AssetLoader — GPU に流す全アセットの走査・登録・取得
//!
//! ## 設計思想
//! GPUパイプラインに必要なものは3種類だけ：
//!   - Shader  : 描画ロジック (.wgsl)
//!   - Vertex  : ジオメトリ   (.bin / .vert)
//!   - Font    : テキスト描画 (.ttf / .otf) → 最終的に vertex + texture に分解
//!
//! AssetRegistry が assets/ ディレクトリを走査し、
//! HashMap<String, GpuAsset> として登録する。
//! キーはファイル名（拡張子なし）。例: "default" → shader.wgsl

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use anyhow::{Context, Result};

// ─────────────────────────────────────────────
// アセット種別
// ─────────────────────────────────────────────

/// GPU に渡せるアセット
#[derive(Debug, Clone)]
pub enum GpuAsset {
    /// WGSL ソースコード文字列
    Shader(String),
    /// 頂点バイナリ (interleaved f32 など、フォーマットは呼び出し側で解釈)
    Vertex(Vec<u8>),
    /// フォントバイナリ (TTF / OTF)
    Font(Vec<u8>),
    ///texture
    Texture(Vec<u8>),
}

impl GpuAsset {
    pub fn as_shader(&self) -> Option<&str> {
        if let GpuAsset::Shader(s) = self { Some(s) } else { None }
    }
    pub fn as_vertex(&self) -> Option<&[u8]> {
        if let GpuAsset::Vertex(v) = self { Some(v) } else { None }
    }
    pub fn as_font(&self) -> Option<&[u8]> {
        if let GpuAsset::Font(f) = self { Some(f) } else { None }
    }
    pub fn as_texture(&self) -> Option<&[u8]> {
        if let GpuAsset::Texture(f) = self { Some(f) } else { None }
    }
}

// ─────────────────────────────────────────────
// ローダ trait (WASM / native 両対応)
// ─────────────────────────────────────────────

pub trait Loader {
    fn load_shader<P: AsRef<Path>>(&self, path: P) -> Result<String>;
    fn load_binary<P: AsRef<Path>>(&self, path: P) -> Result<Vec<u8>>;
}

/// デフォルト実装: native = fs, WASM = stub
pub struct FileLoader;

impl Loader for FileLoader {
    fn load_shader<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::read_to_string(path.as_ref())
                .with_context(|| format!("shader load failed: {:?}", path.as_ref()))
        }
        #[cfg(target_arch = "wasm32")]
        {
            // WASM では rust-embed 等で埋め込む想定
            Ok(String::from("/* WASM: embed shader here */"))
        }
    }

    fn load_binary<P: AsRef<Path>>(&self, path: P) -> Result<Vec<u8>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::read(path.as_ref())
                .with_context(|| format!("binary load failed: {:?}", path.as_ref()))
        }
        #[cfg(target_arch = "wasm32")]
        {
            Ok(vec![])
        }
    }
}

// ─────────────────────────────────────────────
// AssetRegistry — dir 全体をスキャンして登録
// ─────────────────────────────────────────────

/// アセットの一元管理レジストリ
pub struct AssetRegistry {
    assets: HashMap<String, GpuAsset>,
}

impl AssetRegistry {
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
        }
    }

    /// `root_dir` 以下を再帰的にスキャンしてアセットを登録する
    ///
    /// ```
    /// assets/
    ///   shaders/default.wgsl   → key: "default"
    ///   meshes/quad.bin        → key: "quad"
    ///   fonts/noto.ttf         → key: "noto"
    /// ```
    pub fn load_from_dir<P: AsRef<Path>>(
        &mut self,
        root_dir: P,
        loader: &impl Loader,
    ) -> Result<()> {
        self.scan_dir(root_dir.as_ref(), loader)
    }

    fn scan_dir(&mut self, dir: &Path, loader: &impl Loader) -> Result<()> {
        // native 専用。WASM では embed マクロを使う
        #[cfg(not(target_arch = "wasm32"))]
        {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("cannot read dir: {:?}", dir))?
            {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    // 再帰
                    self.scan_dir(&path, loader)?;
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let key = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let asset = match ext.to_lowercase().as_str() {
                        "wgsl" => {
                            let src = loader.load_shader(&path)?;
                            Some(GpuAsset::Shader(src))
                        }
                        "bin" | "vert" => {
                            let bytes = loader.load_binary(&path)?;
                            Some(GpuAsset::Vertex(bytes))
                        }
                        "ttf" | "otf" => {
                            let bytes = loader.load_binary(&path)?;
                            Some(GpuAsset::Font(bytes))
                        }
                        _ => None, // 未知の拡張子はスキップ
                    };

                    if let Some(asset) = asset {
                        self.assets.insert(key, asset);
                    }
                }
            }
        }
        Ok(())
    }

    // ─── 取得 ─────────────────────────────────

    pub fn get(&self, key: &str) -> Option<&GpuAsset> {
        self.assets.get(key)
    }

    pub fn shader(&self, key: &str) -> Option<&str> {
        self.assets.get(key)?.as_shader()
    }

    pub fn vertex(&self, key: &str) -> Option<&[u8]> {
        self.assets.get(key)?.as_vertex()
    }

    pub fn font(&self, key: &str) -> Option<&[u8]> {
        self.assets.get(key)?.as_font()
    }

    // ─── 手動登録（テスト・WASM埋め込み用）─────

    pub fn insert(&mut self, key: impl Into<String>, asset: GpuAsset) {
        self.assets.insert(key.into(), asset);
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.assets.keys()
    }

    pub fn len(&self) -> usize {
        self.assets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }
}

impl Default for AssetRegistry {
    fn default() -> Self {
        Self::new()
    }
}
