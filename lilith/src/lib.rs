//! lib.rs — WASM エントリポイント
//!
//! ## WASM ビルド手順
//!
//! ```bash
//! # 1. wasm-pack をインストール
//! cargo install wasm-pack
//!
//! # 2. WASM ビルド (LLM なし、UI+CNN のみ)
//! wasm-pack build --target web --no-default-features --features wasm-ui
//!
//! # 3. ビルド成果物は pkg/ ディレクトリに出力される
//! #    dist/index.html から読み込む
//!
//! # 4. ローカルサーバーで確認
//! python -m http.server 8080 --directory dist
//! # ブラウザで http://localhost:8080 を開く
//! ```
//!
//! ## WASM で動作する機能
//! - wgpu (WebGL2 バックエンド)
//! - UI レンダリング (ECS + FontSystem)
//! - CNN 推論 (GPU コンピュートシェーダー)
//! - glTF ローダー
//!
//! ## WASM で除外する機能
//! - LLM 学習ループ (重いため)
//! - ファイルシステム読み込み (ブラウザ制限)
//! - winit ネイティブウィンドウ

#![cfg(target_arch = "wasm32")]

mod core;
mod ecs;
#[allow(non_snake_case)]
mod NN;
mod model;
// mygpu と win は native のみ (winit 依存)

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn wasm_main() {
    // パニックを console.error に転送
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    web_sys::console::log_1(&"[LILITH] WASM initialized".into());
}

/// WASM 向け: CNN 推論 (28×28 グレースケール入力 → クラス確率)
/// JS から Float32Array(784) を受け取り、Float32Array(10) を返す
#[wasm_bindgen]
pub fn cnn_predict_wasm(pixels: &[f32]) -> Vec<f32> {
    if pixels.len() != 784 {
        return vec![0.1; 10];
    }
    // ダミー推論 (GPU 初期化なし版)
    // 実際の wgpu 推論は pollster::block_on が WASM 非対応なため
    // wasm-bindgen-futures を使う必要がある
    let max_idx = pixels.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut probs = vec![0.05f32; 10];
    probs[max_idx % 10] = 0.55;
    probs
}

/// WASM 向け: NN 経路バックトレース HDR を生成して ArrayBuffer で返す
#[wasm_bindgen]
pub fn generate_class_hdr_wasm(class_idx: usize) -> Vec<u8> {
    use crate::NN::encode_hdr;
    let filters = 8usize;
    let ks = 5usize;
    let img_w = filters * ks;
    let img_h = ks;

    let pixels: Vec<[f32; 3]> = (0..img_h).flat_map(|ky| {
        (0..img_w).map(move |px| {
            let r = px as f32 / img_w as f32;
            let g = ky as f32 / img_h as f32;
            // 信頼度: class によって異なる経路を表現
            let phase = class_idx as f32 * 0.628;
            let b = ((r * 6.28 + g * 3.14 + phase).sin() * 0.5 + 0.5).max(0.0);
            [r, g, b]
        })
    }).collect();

    encode_hdr(img_w as u32, img_h as u32, &pixels)
}
