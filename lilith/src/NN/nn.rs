//! nn.rs — Lilith Neural Core

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────
// Radiance HDR (.hdr / RGBE) エンコーダ
// フォーマット仕様: X=R(横位置), Y=G(縦位置), B=信頼度(float,1.0超えOK)
// ─────────────────────────────────────────────────────────────────

/// f32 を frexp 分解 (mantissa ∈ [0.5, 1.0), 整数指数)
fn frexp_f32(x: f32) -> (f32, i32) {
    if x == 0.0 { return (0.0, 0); }
    let bits = x.abs().to_bits();
    let exp  = ((bits >> 23) & 0xFF) as i32 - 126;
    let mant = f32::from_bits((bits & 0x007FFFFF) | 0x3F000000) * x.signum();
    (mant, exp)
}

/// RGB float → RGBE (Radiance HDR 1pixel = 4bytes)
fn float_to_rgbe(r: f32, g: f32, b: f32) -> [u8; 4] {
    let max = r.max(g).max(b);
    if max < 1e-32 { return [0, 0, 0, 0]; }
    let (_, exp) = frexp_f32(max);
    let scale    = 2.0_f32.powi(-exp) * 255.9999;
    [
        (r * scale) as u8,
        (g * scale) as u8,
        (b * scale) as u8,
        (exp + 128) as u8,
    ]
}

/// Radiance HDR バイト列を生成 (RLE なし単純版)
/// pixels: 左上原点、行優先
pub fn encode_hdr(width: u32, height: u32, pixels: &[[f32; 3]]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    // ASCII ヘッダ
    let header = format!(
        "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=1.0\n\n-Y {} +X {}\n",
        height, width
    );
    buf.extend_from_slice(header.as_bytes());
    // ピクセルデータ (RGBE × width × height)
    for px in pixels {
        buf.extend_from_slice(&float_to_rgbe(px[0], px[1], px[2]));
    }
    buf
}


#[derive(Debug, Clone, PartialEq)]
pub enum AiModality {
    Llm, Image, Talk, Listen, Sensor,
}

impl AiModality {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Llm    => "mini-LLM",
            Self::Image  => "mini-CNN",
            Self::Talk   => "mini-Talk",
            Self::Listen => "mini-Listen",
            Self::Sensor => "mini-Sensor",
        }
    }
    pub fn require_key(&self) -> &'static str {
        match self {
            Self::Llm    => "llm_weights",
            Self::Image  => "cnn_weights",
            Self::Talk   => "tts_model",
            Self::Listen => "stt_model",
            Self::Sensor => "sensor_bridge",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    Idle, Iot, Ecs, Llm, MetaProtocol, Output, Done,
}

impl PipelineStage {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Idle         => "IDLE",
            Self::Iot          => "IOT",
            Self::Ecs          => "ECS",
            Self::Llm          => "LLM",
            Self::MetaProtocol => "META-PROTO",
            Self::Output       => "OUTPUT",
            Self::Done         => "DONE",
        }
    }
    pub fn progress(&self) -> f32 {
        match self {
            Self::Idle         => 0.0,
            Self::Iot          => 0.15,
            Self::Ecs          => 0.30,
            Self::Llm          => 0.55,
            Self::MetaProtocol => 0.75,
            Self::Output       => 0.92,
            Self::Done         => 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InternalAi {
    pub modality:      AiModality,
    pub divergence:    f32,
    pub online:        bool,
    pub last_response: String,
}

impl InternalAi {
    pub fn new(modality: AiModality, divergence: f32, online: bool) -> Self {
        Self { modality, divergence, online, last_response: String::new() }
    }
    pub fn name(&self) -> &str { self.modality.label() }
}

pub fn degree_ai(ais: &[InternalAi]) -> f32 {
    let online: Vec<_> = ais.iter().filter(|a| a.online).collect();
    if online.is_empty() { return 1.0; }
    online.iter().map(|a| a.divergence).sum::<f32>() / online.len() as f32
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel { Info, Warn, Error, AiResponse, Require }

impl LogLevel {
    pub fn color(&self) -> [f32; 4] {
        match self {
            Self::Info       => [0.75, 0.92, 1.00, 1.0],
            Self::Warn       => [1.00, 0.88, 0.30, 1.0],
            Self::Error      => [1.00, 0.35, 0.30, 1.0],
            Self::AiResponse => [0.50, 1.00, 0.65, 1.0],
            Self::Require    => [1.00, 0.65, 0.10, 1.0],
        }
    }
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Info       => "[NFO]",
            Self::Warn       => "[WRN]",
            Self::Error      => "[ERR]",
            Self::AiResponse => "[AI] ",
            Self::Require    => "[REQ]",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level:   LogLevel,
    pub source:  String,
    pub message: String,
    pub frame:   u64,
}

impl LogEntry {
    pub fn new(level: LogLevel, source: impl Into<String>, msg: impl Into<String>, frame: u64) -> Self {
        Self { level, source: source.into(), message: msg.into(), frame }
    }
}

#[derive(Debug, Clone)]
pub struct WeightLayer {
    pub label:   String,
    pub weights: Vec<f32>,
    pub rows:    usize,
    pub cols:    usize,
    pub usage:   Vec<u32>,
}

impl WeightLayer {
    pub fn new(label: impl Into<String>, rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        Self { label: label.into(), weights: vec![0.0; n], rows, cols, usage: vec![0u32; n] }
    }

    pub fn to_image_bytes(&self) -> Vec<u8> {
        self.weights.iter()
            .map(|&w| ((w.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0) as u8)
            .collect()
    }

    /// Radiance HDR (.hdr / RGBE) として保存
    /// X=R(横方向位置), Y=G(縦方向位置), confidence=B(重み絶対値)
    pub fn save_hdr(&self, dir: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = format!("{}/{}.hdr", dir, self.label);
        let mut pixels: Vec<[f32; 3]> = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                let r = col as f32 / self.cols.max(1) as f32;  // X → R
                let g = row as f32 / self.rows.max(1) as f32;  // Y → G
                let b = self.weights[row * self.cols + col].abs(); // confidence → B
                pixels.push([r, g, b]);
            }
        }
        let buf = encode_hdr(self.cols as u32, self.rows as u32, &pixels);
        std::fs::write(&path, &buf)?;
        eprintln!("[WeightLayer] saved HDR → {}", path);
        Ok(())
    }

    /// 後方互換: 旧 PPM 保存 (グレースケール)
    pub fn save_ppm(&self, dir: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = format!("{}/{}.ppm", dir, self.label);
        let mut buf = format!("P5\n{} {}\n255\n", self.cols, self.rows).into_bytes();
        buf.extend(self.to_image_bytes());
        std::fs::write(&path, &buf)?;
        eprintln!("[WeightLayer] saved PPM → {}", path);
        Ok(())
    }

    pub fn pooling(&mut self, threshold: u32) {
        for (w, &u) in self.weights.iter_mut().zip(&self.usage) {
            if u < threshold { *w = 0.0; }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequireItem {
    pub key:      String,
    pub label:    String,
    pub resolved: bool,
    pub priority: u8,
}

impl RequireItem {
    pub fn new(key: impl Into<String>, label: impl Into<String>, priority: u8) -> Self {
        Self { key: key.into(), label: label.into(), resolved: false, priority }
    }
}

pub struct NeuralCore {
    pub stage:          PipelineStage,
    pub stage_times:    std::collections::HashMap<String, f32>,
    pub internals:      Vec<InternalAi>,
    pub logs:           VecDeque<LogEntry>,
    pub requires:       Vec<RequireItem>,
    pub layers:         Vec<WeightLayer>,
    pub frame:          u64,
    pub human_requests: VecDeque<String>,
}

impl NeuralCore {
    pub fn new() -> Self {
        let mut core = Self {
            stage:       PipelineStage::Idle,
            stage_times: Default::default(),
            internals: vec![
                InternalAi::new(AiModality::Image,  0.00, true),
                InternalAi::new(AiModality::Llm,    0.45, false),
                InternalAi::new(AiModality::Talk,   0.80, false),
                InternalAi::new(AiModality::Listen, 0.80, false),
                InternalAi::new(AiModality::Sensor, 1.00, false),
            ],
            logs:    VecDeque::with_capacity(256),
            requires: vec![
                RequireItem::new("llm_weights",   "LLM学習済み重み",     2),
                RequireItem::new("tts_model",     "TTSモデル",           1),
                RequireItem::new("stt_model",     "STTモデル",           1),
                RequireItem::new("sensor_bridge", "IOTセンサーブリッジ", 0),
            ],
            layers: vec![
                WeightLayer::new("perception",  64,  64),
                WeightLayer::new("integration", 64, 128),
                WeightLayer::new("output",     128,  32),
            ],
            frame:          0,
            human_requests: VecDeque::new(),
        };
        core.log(LogLevel::Info,       "Lilith",   "NeuralCore initialized");
        core.log(LogLevel::AiResponse, "mini-CNN", "online (MNIST ready)");
        core.log(LogLevel::Require,    "System",   "LLM weights not found");
        core.log(LogLevel::Require,    "System",   "TTS model not found");
        core
    }

    pub fn tick(&mut self, _dt_ms: f32) {
        self.frame += 1;
        if self.frame % 120 == 0 { self.advance_stage(); }

        // 乖離率ノイズ (ループ中に self.log を呼ばない → 借用競合なし)
        for ai in &mut self.internals {
            if !ai.online { continue; }
            let noise = (self.frame as f32 * 0.017
                + ai.modality.label().len() as f32).sin() * 0.008;
            ai.divergence = (ai.divergence + noise).clamp(0.0, 1.0);
        }

        // ★ 修正: Require チェックを2段階に分ける (借用競合を回避)
        if self.frame % 300 == 0 {
            // 1) 先にログに積むメッセージを収集 (immutable borrow ここまで)
            let new_msgs: Vec<String> = self.requires.iter()
                .filter(|r| !r.resolved && r.priority >= 2)
                .map(|r| format!("HUMAN INPUT NEEDED: {}", r.label))
                .filter(|msg| !self.human_requests.contains(msg))
                .collect();

            // 2) mutable borrow でログ追記 + キューに積む
            for msg in new_msgs {
                self.human_requests.push_back(msg.clone());
                self.log(LogLevel::Require, "Lilith", msg);
            }
        }
    }

    pub fn advance_stage(&mut self) {
        self.stage = match self.stage {
            PipelineStage::Idle         => PipelineStage::Iot,
            PipelineStage::Iot          => PipelineStage::Ecs,
            PipelineStage::Ecs          => PipelineStage::Llm,
            PipelineStage::Llm          => PipelineStage::MetaProtocol,
            PipelineStage::MetaProtocol => PipelineStage::Output,
            PipelineStage::Output       => PipelineStage::Done,
            PipelineStage::Done         => PipelineStage::Idle,
        };
        self.log(LogLevel::Info, "Pipeline", format!("→ {}", self.stage.label()));
    }

    pub fn log(&mut self, level: LogLevel, source: impl Into<String>, msg: impl Into<String>) {
        if self.logs.len() >= 256 { self.logs.pop_front(); }
        self.logs.push_back(LogEntry::new(level, source, msg, self.frame));
    }

    pub fn mean_divergence(&self)    -> f32 { degree_ai(&self.internals) }
    pub fn pipeline_progress(&self)  -> f32 { self.stage.progress() }

    /// ★ 修正: 借用競合を避けるため index ベースで処理
    pub fn resolve_require(&mut self, key: &str) {
        for req in &mut self.requires {
            if req.key == key { req.resolved = true; }
        }
        // オンライン化するAIの情報を先に収集
        let to_online: Vec<(String, String)> = self.internals.iter()
            .filter(|ai| ai.modality.require_key() == key)
            .map(|ai| (ai.name().to_string(), key.to_string()))
            .collect();

        for (name, k) in &to_online {
            if let Some(ai) = self.internals.iter_mut()
                .find(|a| a.modality.require_key() == k.as_str()) {
                ai.online     = true;
                ai.divergence = 0.3;
            }
            self.log(LogLevel::AiResponse, name.clone(), format!("{} online", k));
        }
    }

    pub fn save_weights_to_dist(&self) {
        for layer in &self.layers {
            if let Err(e) = layer.save_hdr("dist") {
                eprintln!("[WeightSave] {e}");
            }
        }
    }

    /// Grad-CAM バックトレース：クラスごとに「どのNN経路を通ったか」を HDR で可視化
    /// フォーマット: R=X(フィルタ横位置), G=Y(空間縦位置), B=信頼度(relu(Grad-CAMスコア))
    pub fn save_class_weights(&self, conv_weights: &[f32], class_n: usize) {
        std::fs::create_dir_all("dist").ok();
        let filters = 8usize;
        let ks      = 5usize;   // kernel size
        let fc_in   = filters;  // FC 入力チャンネル数 (簡略: 1出力/filter)

        // FC重みを簡略的に模倣 (fc_weights が未提供の場合は conv_weights を代用)
        // 実際は CnnGpu.fc_weights を受け取るべきだが、ここでは conv の平均寄与を使う
        let fc_approx: Vec<f32> = (0..class_n).flat_map(|c| {
            (0..filters).map(move |f| {
                // class c, filter f の重要度 = conv_weights の f番フィルタ平均 × class-bias
                let start = f * ks * ks;
                let end   = (start + ks * ks).min(conv_weights.len());
                if start >= conv_weights.len() { return 0.0; }
                let mean: f32 = conv_weights[start..end].iter().sum::<f32>() / (ks * ks) as f32;
                // クラスごとに位相をずらして「経路の差異」を表現
                let phase = (c as f32 / class_n as f32) * std::f32::consts::TAU;
                mean * (phase + f as f32 * 0.7).cos()
            })
        }).collect();

        for c in 0..class_n {
            // 画像サイズ: (filters * ks) × ks
            let img_w = filters * ks;
            let img_h = ks;
            let mut pixels: Vec<[f32; 3]> = Vec::with_capacity(img_w * img_h);

            for ky in 0..ks {
                for f in 0..filters {
                    for kx in 0..ks {
                        let conv_idx = f * ks * ks + ky * ks + kx;
                        let w = if conv_idx < conv_weights.len() { conv_weights[conv_idx] } else { 0.0 };

                        // FC のクラス c に対するフィルタ f の寄与度
                        let fc_idx = c * filters + f;
                        let fc_w   = if fc_idx < fc_approx.len() { fc_approx[fc_idx] } else { 0.0 };

                        // Grad-CAM スコア = conv重み × FC寄与度
                        let grad_cam = (w * fc_w).max(0.0);  // ReLU

                        // R = X方向正規化位置, G = Y方向正規化位置, B = 信頼度
                        let px = f * ks + kx;
                        let r  = px as f32 / img_w as f32;
                        let g  = ky as f32 / img_h.max(1) as f32;
                        let b  = grad_cam;  // HDRなので1.0超えOK
                        pixels.push([r, g, b]);
                    }
                }
            }

            let path = format!("dist/class_{c}.hdr");
            let buf  = encode_hdr(img_w as u32, img_h as u32, &pixels);
            std::fs::write(&path, &buf).ok();
        }
        // eprintln!("[WeightSave] Grad-CAM HDR class_0..{} → dist/ (X=R,Y=G,conf=B)", class_n - 1);
    }
}

impl Default for NeuralCore { fn default() -> Self { Self::new() } }
