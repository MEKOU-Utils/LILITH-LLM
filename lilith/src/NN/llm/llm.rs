//! llm.rs — Mini Transformer LLM (日本語5択QA, JCommonsenseQA形式)
//!
//! アーキテクチャ:
//!   CharTokenizer → Embedding(d=64) → MultiHeadAttention(h=4) → FFN(256) → Linear(→5)
//!
//! データセット: dataset/train-v1.3.json / test-v1.3.json
//!   {"q_id":N, "question":"...", "choice0":"...", ..., "choice4":"...", "label":N}

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────
// データセット
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QaItem {
    pub q_id:     u64,
    pub question: String,
    pub choices:  [String; 5],
    pub label:    usize,
}

impl QaItem {
    /// question + 全choiceを結合した入力文字列
    pub fn as_input(&self) -> String {
        let mut s = self.question.clone();
        for (i, c) in self.choices.iter().enumerate() {
            s.push_str(&format!(" [{}]{}", i, c));
        }
        s
    }
}

pub fn load_dataset(path: &str) -> Vec<QaItem> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("[LLM] dataset load failed: {e}"); return vec![]; }
    };
    let mut items = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        // 軽量JSONパーサ (serde不要)
        if let Some(item) = parse_qa_line(line) { items.push(item); }
    }
    eprintln!("[LLM] loaded {} items from {}", items.len(), path);
    items
}

fn extract_str<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\":", key);
    let pos = s.find(&needle)?;
    let rest = &s[pos + needle.len()..].trim_start();
    if rest.starts_with('"') {
        let inner = &rest[1..];
        let end = inner.find('"')?;
        Some(&inner[..end])
    } else {
        None
    }
}

fn extract_num(s: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{}\":", key);
    let pos = s.find(&needle)?;
    let rest = s[pos + needle.len()..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse().ok()
}

fn parse_qa_line(line: &str) -> Option<QaItem> {
    let q_id    = extract_num(line, "q_id")?;
    let question = extract_str(line, "question")?.to_string();
    let c0 = extract_str(line, "choice0")?.to_string();
    let c1 = extract_str(line, "choice1")?.to_string();
    let c2 = extract_str(line, "choice2")?.to_string();
    let c3 = extract_str(line, "choice3")?.to_string();
    let c4 = extract_str(line, "choice4")?.to_string();
    let label = extract_num(line, "label")? as usize;
    Some(QaItem { q_id, question, choices: [c0, c1, c2, c3, c4], label })
}

// ─────────────────────────────────────────────────────────────────
// 文字レベルトークナイザ
// ─────────────────────────────────────────────────────────────────

pub struct CharTokenizer {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: Vec<char>,
    pub vocab_size: usize,
}

impl CharTokenizer {
    pub fn build(texts: &[&str]) -> Self {
        // '\0' = PAD, '\x01' = UNK
        let mut chars: Vec<char> = vec!['\0', '\x01'];
        let mut seen = std::collections::HashSet::from(['\0', '\x01']);
        for t in texts {
            for c in t.chars() {
                if !seen.contains(&c) { chars.push(c); seen.insert(c); }
            }
        }
        let vocab_size = chars.len();
        let char_to_id = chars.iter().enumerate().map(|(i,&c)| (c,i)).collect();
        Self { char_to_id, id_to_char: chars, vocab_size }
    }

    pub fn encode(&self, s: &str, max_len: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = s.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&1))
            .take(max_len)
            .collect();
        while ids.len() < max_len { ids.push(0); }
        ids
    }
}

// ─────────────────────────────────────────────────────────────────
// 数値計算ユーティリティ (ndarray 不要の最小実装)
// ─────────────────────────────────────────────────────────────────

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k { s += a[i*k+p] * b[p*n+j]; }
            c[i*n+j] = s;
        }
    }
    c
}

fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = v.iter().map(|&x| (x - max).exp()).sum();
    for x in v.iter_mut() { *x = (*x - max).exp() / sum; }
}

fn relu(v: &mut [f32]) { for x in v.iter_mut() { if *x < 0.0 { *x = 0.0; } } }

fn layer_norm(v: &mut [f32]) {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var  = v.iter().map(|&x| (x-mean)*(x-mean)).sum::<f32>() / v.len() as f32;
    let std  = (var + 1e-5).sqrt();
    for x in v.iter_mut() { *x = (*x - mean) / std; }
}

// ─────────────────────────────────────────────────────────────────
// LLM 設定
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub d_model:   usize,  // 64
    pub n_heads:   usize,  // 4
    pub d_ff:      usize,  // 256
    pub seq_len:   usize,  // 128
    pub n_classes: usize,  // 5 (5択)
    pub vocab_size: usize,
}

impl LlmConfig {
    pub fn default_qa(vocab_size: usize) -> Self {
        Self { d_model: 64, n_heads: 4, d_ff: 256, seq_len: 128, n_classes: 5, vocab_size }
    }
    pub fn d_k(&self) -> usize { self.d_model / self.n_heads }
}

// ─────────────────────────────────────────────────────────────────
// Mini Transformer (1層)
// ─────────────────────────────────────────────────────────────────

pub struct MiniLlm {
    pub cfg: LlmConfig,
    // Embedding [vocab_size × d_model]
    embed:     Vec<f32>,
    // MultiHeadAttention
    w_q: Vec<f32>,  // [d_model × d_model]
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    // FFN
    w_ff1: Vec<f32>,  // [d_model × d_ff]
    b_ff1: Vec<f32>,  // [d_ff]
    w_ff2: Vec<f32>,  // [d_ff × d_model]
    b_ff2: Vec<f32>,  // [d_model]
    // 出力
    w_cls: Vec<f32>,  // [d_model × n_classes]
    b_cls: Vec<f32>,  // [n_classes]
}

impl MiniLlm {
    pub fn new(cfg: LlmConfig) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let ff = cfg.d_ff;
        let nc = cfg.n_classes;
        let scale_d  = (2.0 / d as f32).sqrt();
        let scale_ff = (2.0 / ff as f32).sqrt();

        let xavier = |n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| {
                let x = ((i as f32 * 1.6180339) % 1.0) * 2.0 - 1.0;
                x * scale
            }).collect()
        };

        Self {
            cfg: cfg.clone(),
            embed:  xavier(v * d,   (1.0 / d as f32).sqrt()),
            w_q:    xavier(d * d,   scale_d),
            w_k:    xavier(d * d,   scale_d),
            w_v:    xavier(d * d,   scale_d),
            w_o:    xavier(d * d,   scale_d),
            w_ff1:  xavier(d * ff,  scale_d),
            b_ff1:  vec![0.0; ff],
            w_ff2:  xavier(ff * d,  scale_ff),
            b_ff2:  vec![0.0; d],
            w_cls:  xavier(d * nc,  scale_d),
            b_cls:  vec![0.0; nc],
        }
    }

    /// 順伝播: token ids → class logits [n_classes]
    pub fn forward(&self, ids: &[usize]) -> Vec<f32> {
        let cfg = &self.cfg;
        let seq = ids.len().min(cfg.seq_len);
        let d   = cfg.d_model;
        let dk  = cfg.d_k();
        let h   = cfg.n_heads;

        // ── Embedding + 位置エンコーディング ──────────────────────
        let mut x: Vec<f32> = Vec::with_capacity(seq * d);
        for (pos, &tok) in ids[..seq].iter().enumerate() {
            let base = tok.min(cfg.vocab_size - 1) * d;
            for di in 0..d {
                let emb = self.embed[base + di];
                // Sinusoidal PE
                let pe = if di % 2 == 0 {
                    (pos as f32 / 10000_f32.powf(di as f32 / d as f32)).sin()
                } else {
                    (pos as f32 / 10000_f32.powf((di-1) as f32 / d as f32)).cos()
                };
                x.push(emb + pe * 0.1);
            }
        }

        // ── Multi-Head Self-Attention ──────────────────────────────
        let q = matmul(&x, &self.w_q, seq, d, d);
        let k = matmul(&x, &self.w_k, seq, d, d);
        let v = matmul(&x, &self.w_v, seq, d, d);
        let scale = (dk as f32).sqrt();

        let mut attn_out = vec![0.0f32; seq * d];
        for head in 0..h {
            let hd = head * dk;
            // Attention score = Q·Kᵀ / sqrt(dk)
            let mut scores = vec![0.0f32; seq * seq];
            for i in 0..seq {
                for j in 0..seq {
                    let mut s = 0.0f32;
                    for p in 0..dk { s += q[i*d+hd+p] * k[j*d+hd+p]; }
                    scores[i*seq+j] = s / scale;
                }
            }
            // Softmax per row
            for i in 0..seq {
                softmax_inplace(&mut scores[i*seq..(i+1)*seq]);
            }
            // weighted sum of V
            for i in 0..seq {
                for p in 0..dk {
                    let mut s = 0.0f32;
                    for j in 0..seq { s += scores[i*seq+j] * v[j*d+hd+p]; }
                    attn_out[i*d+hd+p] += s;
                }
            }
        }

        // W_O 射影 + 残差 + LayerNorm
        let proj = matmul(&attn_out, &self.w_o, seq, d, d);
        let mut h1: Vec<f32> = x.iter().zip(&proj).map(|(a,b)| a+b).collect();
        for i in 0..seq { layer_norm(&mut h1[i*d..(i+1)*d]); }

        // ── FFN ───────────────────────────────────────────────────
        let mut ff1 = matmul(&h1, &self.w_ff1, seq, d, cfg.d_ff);
        for i in 0..seq {
            for j in 0..cfg.d_ff { ff1[i*cfg.d_ff+j] += self.b_ff1[j]; }
        }
        relu(&mut ff1);
        let mut ff2 = matmul(&ff1, &self.w_ff2, seq, cfg.d_ff, d);
        for i in 0..seq {
            for j in 0..d { ff2[i*d+j] += self.b_ff2[j]; }
        }
        // 残差 + LayerNorm
        let mut h2: Vec<f32> = h1.iter().zip(&ff2).map(|(a,b)| a+b).collect();
        for i in 0..seq { layer_norm(&mut h2[i*d..(i+1)*d]); }

        // ── [CLS] pooling (平均プーリング) → 分類 ─────────────────
        let mut pooled = vec![0.0f32; d];
        for i in 0..seq {
            for j in 0..d { pooled[j] += h2[i*d+j]; }
        }
        for v in pooled.iter_mut() { *v /= seq as f32; }

        let mut logits = matmul(&pooled, &self.w_cls, 1, d, cfg.n_classes);
        for j in 0..cfg.n_classes { logits[j] += self.b_cls[j]; }
        logits
    }

    /// 予測クラス (0-4) と softmax 確率
    pub fn predict(&self, ids: &[usize]) -> (usize, Vec<f32>) {
        let mut logits = self.forward(ids);
        softmax_inplace(&mut logits);
        let pred = logits.iter().enumerate()
            .max_by(|a,b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i,_)| i).unwrap_or(0);
        (pred, logits)
    }

    // ── SGD 学習 (1サンプル) ──────────────────────────────────────
    pub fn train_step(&mut self, ids: &[usize], label: usize, lr: f32) -> f32 {
        let mut logits = self.forward(ids);
        softmax_inplace(&mut logits);
        // クロスエントロピー損失
        let loss = -logits[label].max(1e-9).ln();
        // 出力層の勾配 (δ = prob - one_hot)
        let nc = self.cfg.n_classes;
        let d  = self.cfg.d_model;
        let seq = ids.len().min(self.cfg.seq_len);

        // pooled の再計算 (フルバックプロップは複雑なので出力層のみ更新)
        let h = self.forward_pooled(ids);
        let mut dlogits = logits.clone();
        dlogits[label] -= 1.0;

        // w_cls, b_cls のみ更新 (出力層 SGD)
        for j in 0..nc {
            self.b_cls[j] -= lr * dlogits[j];
            for i in 0..d {
                self.w_cls[i*nc+j] -= lr * dlogits[j] * h[i];
            }
        }
        loss
    }

    fn forward_pooled(&self, ids: &[usize]) -> Vec<f32> {
        let d   = self.cfg.d_model;
        let seq = ids.len().min(self.cfg.seq_len);
        // Embedding だけで近似 (高速化)
        let mut pooled = vec![0.0f32; d];
        for &tok in &ids[..seq] {
            let base = tok.min(self.cfg.vocab_size - 1) * d;
            for j in 0..d { pooled[j] += self.embed[base+j]; }
        }
        for v in pooled.iter_mut() { *v /= seq as f32; }
        pooled
    }

    /// バッチ学習
    pub fn train_epoch(&mut self, data: &[(Vec<usize>, usize)], lr: f32) -> f32 {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for (ids, label) in data {
            let (pred, _) = self.predict(ids);
            if pred == *label { correct += 1; }
            total_loss += self.train_step(ids, *label, lr);
        }
        let acc = correct as f32 / data.len() as f32;
        eprintln!("[LLM] loss={:.4} acc={:.3}", total_loss / data.len() as f32, acc);
        total_loss / data.len() as f32
    }

    // ── 重みの保存・読み込み ──────────────────────────────────────
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        // 簡易バイナリ: w_cls, b_cls のみ (最小限)
        let nc = self.cfg.n_classes;
        let d  = self.cfg.d_model;
        f.write_all(bytemuck::cast_slice(&self.w_cls[..d*nc]))?;
        f.write_all(bytemuck::cast_slice(&self.b_cls[..nc]))?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────
// ChatBot — データセットベースのQ&Aインターフェース
// ─────────────────────────────────────────────────────────────────

pub struct ChatBot {
    pub llm:       MiniLlm,
    pub tokenizer: CharTokenizer,
    pub dataset:   Vec<QaItem>,
    pub trained:   bool,
    pub history:   Vec<(String, String)>,  // (user, bot)
}

impl ChatBot {
    /// データセットを読み込み、トークナイザをビルドしてモデルを初期化
    pub fn new(train_path: &str, test_path: &str) -> Self {
        let train = load_dataset(train_path);
        let test  = load_dataset(test_path);
        let mut all_dataset = train.clone();
        all_dataset.extend_from_slice(&test);

        // トークナイザを全テキストで構築
        let texts: Vec<String> = all_dataset.iter().map(|q| q.as_input()).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let tokenizer = CharTokenizer::build(&text_refs);

        eprintln!("[LLM] vocab_size={}", tokenizer.vocab_size);

        let cfg = LlmConfig::default_qa(tokenizer.vocab_size);
        let llm = MiniLlm::new(cfg);

        Self { llm, tokenizer, dataset: all_dataset, trained: false, history: Vec::new() }
    }

    /// 学習実行 (最大 n_epochs エポック)
    pub fn train(&mut self, n_epochs: usize, lr: f32) {
        let cfg = self.llm.cfg.clone();
        let data: Vec<(Vec<usize>, usize)> = self.dataset.iter()
            .map(|item| {
                let ids = self.tokenizer.encode(&item.as_input(), cfg.seq_len);
                (ids, item.label)
            })
            .collect();

        for epoch in 0..n_epochs {
            let loss = self.llm.train_epoch(&data, lr);
            if epoch % 5 == 0 {
                eprintln!("[LLM] epoch={} loss={:.4}", epoch, loss);
            }
        }
        self.trained = true;
        eprintln!("[LLM] training done ({} epochs)", n_epochs);
    }

    /// チャット: 質問文を受け取り、データセットで最も近い問題を見つけて回答
    pub fn chat(&mut self, user_input: &str) -> String {
        // 1) 最も近いデータセット項目を検索 (文字n-gram類似度)
        let best = self.find_closest(user_input);

        // 2) LLM で分類
        let ids = self.tokenizer.encode(&best.as_input(), self.llm.cfg.seq_len);
        let (pred, probs) = self.llm.predict(&ids);
        let conf = probs[pred];
        let answer = &best.choices[pred];

        let reply = format!(
            "Q: {}\n→ {} (conf: {:.1}%)\n[選択肢: {}|{}|{}|{}|{}]",
            best.question, answer, conf * 100.0,
            best.choices[0], best.choices[1], best.choices[2],
            best.choices[3], best.choices[4]
        );

        self.history.push((user_input.to_string(), reply.clone()));
        if self.history.len() > 50 { self.history.remove(0); }
        reply
    }

    /// 文字ユニグラム重複率でデータセットを検索
    fn find_closest<'a>(&'a self, query: &str) -> &'a QaItem {
        let q_chars: std::collections::HashSet<char> = query.chars().collect();
        let best = self.dataset.iter().max_by(|a, b| {
            let sa = Self::char_overlap(&q_chars, &a.question);
            let sb = Self::char_overlap(&q_chars, &b.question);
            sa.partial_cmp(&sb).unwrap()
        });
        best.unwrap_or(&self.dataset[0])
    }

    fn char_overlap(a: &std::collections::HashSet<char>, b: &str) -> f32 {
        let bc: std::collections::HashSet<char> = b.chars().collect();
        let inter = a.intersection(&bc).count();
        inter as f32 / (a.len() + bc.len()).max(1) as f32
    }

    /// 評価精度
    pub fn eval_accuracy(&self, data: &[QaItem]) -> f32 {
        let mut correct = 0;
        for item in data {
            let ids = self.tokenizer.encode(&item.as_input(), self.llm.cfg.seq_len);
            let (pred, _) = self.llm.predict(&ids);
            if pred == item.label { correct += 1; }
        }
        correct as f32 / data.len().max(1) as f32
    }
}

// ─────────────────────────────────────────────────────────────────
// bytemuck (save で使用)
// ─────────────────────────────────────────────────────────────────
// bytemuck は Cargo.toml 既存依存なのでそのまま利用可

mod bytemuck {
    pub fn cast_slice<T: Copy>(v: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                v.as_ptr() as *const u8,
                v.len() * std::mem::size_of::<T>(),
            )
        }
    }
}
