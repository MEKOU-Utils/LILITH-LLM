//! ui_system.rs — NeuralCore 状態 → ECS エンティティ変換 + リサイズ対応
//!
//! ## レイアウト (画面幅に追従)
//!
//!  ┌──────────────────────────────────────────────────────────┐
//!  │█ LILITH NEURAL CORE v0.1          AVG DIV: 0.28  [IDLE]│  header
//!  ├─────────────────────┬────────────────────────────────────┤
//!  │ PIPELINE            │ INTERNAL AI                        │
//!  │ ● IOT  ▓▓▓▓░░░░░░  │ ■ mini-CNN    ████░░  0.12 online│
//!  │ ● ECS               │ □ mini-LLM    ░░░░░░  0.45 off   │
//!  │ ○ LLM               │ □ mini-Talk   ░░░░░░  0.80 off   │
//!  │ ○ META-PROTO        │ □ mini-Listen ░░░░░░  0.80 off   │
//!  │ ○ OUTPUT            │ □ mini-Sensor ░░░░░░  1.00 off   │
//!  │ ○ DONE              │                                    │
//!  │ [████████░░ 55%]    │                                    │
//!  ├─────────────────────┴────────────────────────────────────┤
//!  │ CNN WEIGHT VISUALIZER                                     │
//!  │ Filters[f0][f1][f2][f3][f4][f5][f6][f7]                 │
//!  │ Class 0▓ 1▓ 2▓ 3▓ 4▓ 5▓ 6▓ 7██ 8▓ 9▓                  │
//!  │ pred:7(0.93) KL vs mini-CNN: 0.021                       │
//!  ├──────────────────────────────────────────────────────────┤
//!  │ LOG                                                       │
//!  │ [NFO] Lilith        > NeuralCore initialized             │
//!  │ [REQ] System        > LLM weights not found              │
//!  │ [AI ] mini-CNN      > online (MNIST ready)               │
//!  ├──────────────────────────────────────────────────────────┤
//!  │ REQUIRE (未解決): llm_weights | tts_model                │
//!  │ [▶ REQUEST TO HUMAN]   [▶ SAVE WEIGHTS]                 │  ← ボタン
//!  └──────────────────────────────────────────────────────────┘

use crate::ecs::component::EntityId;
use crate::ecs::object::{ObjectManager, Transform};
use crate::ecs::ui::{Rect, UiElement, Text};
use crate::ecs::ui::text::FontSize;
use crate::NN::{NeuralCore, LogLevel};

// ─────────────────────────────────────────────────────────────────
// 色定数 (視認性重視)
// ─────────────────────────────────────────────────────────────────
pub mod col {
    pub const BG:        [f32; 4] = [0.02, 0.02, 0.04, 1.00];
    pub const PANEL:     [f32; 4] = [0.06, 0.06, 0.10, 1.00];
    pub const TRACK:     [f32; 4] = [0.12, 0.12, 0.18, 1.00];
    pub const BORDER:    [f32; 4] = [0.20, 0.22, 0.30, 1.00];
    // テキスト — 白ベース高コントラスト
    pub const TEXT:      [f32; 4] = [0.95, 0.97, 1.00, 1.00];  // ほぼ白
    pub const TEXT_DIM:  [f32; 4] = [0.60, 0.65, 0.75, 1.00];  // グレー (でも十分明るい)
    pub const ACCENT:    [f32; 4] = [0.30, 0.65, 1.00, 1.00];  // 明るいブルー
    pub const GREEN:     [f32; 4] = [0.25, 1.00, 0.50, 1.00];  // 鮮やか緑
    pub const YELLOW:    [f32; 4] = [1.00, 0.92, 0.20, 1.00];  // 黄 (アクティブ)
    pub const ORANGE:    [f32; 4] = [1.00, 0.65, 0.10, 1.00];  // Require
    pub const RED:       [f32; 4] = [1.00, 0.30, 0.25, 1.00];  // エラー / 高乖離
    pub const BTN:       [f32; 4] = [0.15, 0.35, 0.60, 1.00];  // ボタン背景
    pub const BTN_REQ:   [f32; 4] = [0.50, 0.20, 0.05, 1.00];  // Requireボタン
}

pub fn lerp_c(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t,
     a[2]+(b[2]-a[2])*t, a[3]+(b[3]-a[3])*t]
}

// ─────────────────────────────────────────────────────────────────
// レイアウト計算 (画面幅から動的に決定)
// ─────────────────────────────────────────────────────────────────
pub struct Layout {
    pub x0:      f32,
    pub y0:      f32,
    pub w:       f32,   // パネル全体幅 = screen_w - margin*2
    pub pad:     f32,
    pub line_h:  f32,
    pub font_sz: f32,
}

impl Layout {
    pub fn from_screen(sw: f32, sh: f32) -> Self {
        let margin = 8.0;
        let w = (sw - margin * 2.0).max(300.0);
        // 画面幅に応じてフォントサイズを調整
        let font_sz = if sw < 600.0 { 10.0 } else if sw < 900.0 { 12.0 } else { 13.0 };
        Self { x0: margin, y0: margin, w, pad: 8.0, line_h: font_sz + 4.0, font_sz }
    }
}

// ─────────────────────────────────────────────────────────────────
// ヘルパ
// ─────────────────────────────────────────────────────────────────
fn mk_ui(world: &mut ObjectManager, rect: Rect, color: [f32; 4], key: &str) -> EntityId {
    let id = world.spawn();
    let mut e = UiElement::new(rect, key);
    e.color = color;
    world.ui_elements.insert(id, e);
    id
}

fn mk_txt(world: &mut ObjectManager, x: f32, y: f32,
          s: impl Into<String>, fk: &str, sz: f32, color: [f32; 4]) -> EntityId {
    let id = world.spawn();
    world.transforms.insert(id, Transform::new(x, y, 0.0));
    let mut t = Text::new(s, fk);
    t.size  = FontSize(sz);
    t.color = color;
    world.texts.insert(id, t);
    id
}

fn set_txt(world: &mut ObjectManager, id: EntityId, s: impl Into<String>, c: [f32; 4]) {
    if let Some(t) = world.texts.get_mut(id) { t.content = s.into(); t.color = c; }
}
fn set_col(world: &mut ObjectManager, id: EntityId, c: [f32; 4]) {
    if let Some(u) = world.ui_elements.get_mut(id) { u.color = c; }
}
fn set_w(world: &mut ObjectManager, id: EntityId, w: f32) {
    if let Some(u) = world.ui_elements.get_mut(id) { u.rect.width = w; }
}
fn despawn_all(world: &mut ObjectManager, ids: &[EntityId]) {
    for &id in ids { world.despawn(id); }
}

// ─────────────────────────────────────────────────────────────────
// ボタンの状態
// ─────────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ButtonAction {
    RequestToHuman,
    SaveWeights,
}

pub struct UiButton {
    pub bg:     EntityId,
    pub label:  EntityId,
    pub action: ButtonAction,
    pub rect:   Rect,
}

// ─────────────────────────────────────────────────────────────────
// NeuralUi
// ─────────────────────────────────────────────────────────────────
pub struct NeuralUi {
    pub font_key:     String,
    screen_w:     f32,
    screen_h:     f32,

    // 全エンティティ (rebuild時に全 despawn)
    all_entities: Vec<EntityId>,

    // 動的更新対象 ID
    e_div_text:    EntityId,
    e_stage_label: EntityId,
    e_prog_fill:   EntityId,
    e_prog_track_w: f32,    // バー最大幅キャッシュ

    e_stage_texts: Vec<EntityId>,
    e_stage_dots:  Vec<EntityId>,

    // Internal AI
    e_ai_texts:  Vec<EntityId>,
    e_ai_bars:   Vec<EntityId>,
    e_ai_bar_max: f32,

    // CNN
    e_cnn_pred:      EntityId,
    e_cnn_kl:        EntityId,
    e_class_bars:    Vec<EntityId>,
    e_class_y_base:  Vec<f32>,    // 各クラスバーの基準Y
    e_class_max_h:   f32,
    e_class_labels:  Vec<EntityId>,
    e_filter_cells:  Vec<EntityId>,

    // Log
    e_log_lines: Vec<EntityId>,

    // Require
    e_req_text: EntityId,

    // Buttons
    pub buttons: Vec<UiButton>,
}

const LOG_N:    usize = 8;
const AI_N:     usize = 5;
const STAGE_N:  usize = 6;
const CLASS_N:  usize = 10;
const FILTER_N: usize = 8;

impl NeuralUi {
    pub fn build(world: &mut ObjectManager, fk: &str, sw: f32, sh: f32) -> Self {
        let mut ui = Self::empty(fk, sw, sh);
        ui.rebuild(world, fk, sw, sh);
        ui
    }

    fn empty(fk: &str, sw: f32, sh: f32) -> Self {
        let dummy = 0u32; // EntityId placeholder (despawnされる前提)
        Self {
            font_key: fk.to_string(),
            screen_w: sw, screen_h: sh,
            all_entities:   vec![],
            e_div_text:     dummy,
            e_stage_label:  dummy,
            e_prog_fill:    dummy,
            e_prog_track_w: 0.0,
            e_stage_texts:  vec![],
            e_stage_dots:   vec![],
            e_ai_texts:     vec![],
            e_ai_bars:      vec![],
            e_ai_bar_max:   0.0,
            e_cnn_pred:     dummy,
            e_cnn_kl:       dummy,
            e_class_bars:   vec![],
            e_class_y_base: vec![],
            e_class_max_h:  0.0,
            e_class_labels: vec![],
            e_filter_cells: vec![],
            e_log_lines:    vec![],
            e_req_text:     dummy,
            buttons:        vec![],
        }
    }

    /// リサイズ時: 全エンティティを作り直す
    pub fn rebuild(&mut self, world: &mut ObjectManager, fk: &str, sw: f32, sh: f32) {
        // 旧エンティティを全削除
        let old = std::mem::take(&mut self.all_entities);
        despawn_all(world, &old);
        self.buttons.clear();
        self.e_stage_texts.clear();
        self.e_stage_dots.clear();
        self.e_ai_texts.clear();
        self.e_ai_bars.clear();
        self.e_class_bars.clear();
        self.e_class_y_base.clear();
        self.e_class_labels.clear();
        self.e_filter_cells.clear();
        self.e_log_lines.clear();

        self.screen_w = sw;
        self.screen_h = sh;

        let lo = Layout::from_screen(sw, sh);
        let mut ids: Vec<EntityId> = Vec::new();
        let mut y = lo.y0;

        macro_rules! ui {
            ($r:expr, $c:expr, $k:expr) => {{ let id = mk_ui(world,$r,$c,$k); ids.push(id); id }};
        }
        macro_rules! txt {
            ($x:expr,$yy:expr,$s:expr,$sz:expr,$c:expr) => {{
                let id = mk_txt(world,$x,$yy,$s,fk,$sz,$c); ids.push(id); id
            }};
        }
        let sz  = lo.font_sz;
        let sz_s = (sz - 1.0).max(8.0);
        let p   = lo.pad;

        // ── HEADER ─────────────────────────────────────────────────
        let hdr_h = sz + p * 2.0;
        ui!(Rect::new(lo.x0, y, lo.w, hdr_h), col::BG, "solid");
        // 左タイトル
        txt!(lo.x0 + p, y + p, "LILITH NEURAL CORE v0.1", sz, col::ACCENT);
        // 右: 乖離率 + ステージ
        let e_div = txt!(lo.x0 + lo.w * 0.55, y + p, "AVG DIV: 0.000", sz, col::YELLOW);
        let e_stg = txt!(lo.x0 + lo.w * 0.82, y + p, "[IDLE]", sz, col::TEXT_DIM);
        self.e_div_text    = e_div;
        self.e_stage_label = e_stg;
        y += hdr_h + 1.0;

        // ── PIPELINE + INTERNAL AI (横並び) ────────────────────────
        let mid_h = lo.line_h * (STAGE_N as f32 + 2.5) + p * 2.0;
        let pipe_w = lo.w * 0.44;
        let ai_w   = lo.w - pipe_w - 1.0;
        let ai_x   = lo.x0 + pipe_w + 1.0;

        // 背景
        ui!(Rect::new(lo.x0, y, pipe_w, mid_h), col::PANEL, "solid");
        ui!(Rect::new(ai_x,  y, ai_w,   mid_h), col::PANEL, "solid");
        // 区切り線
        ui!(Rect::new(lo.x0, y, lo.w, 1.0), col::BORDER, "solid");
        ui!(Rect::new(ai_x-1.0, y, 1.0, mid_h), col::BORDER, "solid");

        txt!(lo.x0 + p, y + p, "PIPELINE", sz_s, col::ACCENT);
        txt!(ai_x  + p, y + p, "INTERNAL AI", sz_s, col::ACCENT);

        // ステージ一覧
        let stages = ["IOT","ECS","LLM","META-PROTO","OUTPUT","DONE"];
        for (i, &s) in stages.iter().enumerate() {
            let ty = y + p + lo.line_h * (1.5 + i as f32);
            let dot = ui!(Rect::new(lo.x0 + p, ty + sz*0.2, sz*0.55, sz*0.55), col::TEXT_DIM, "solid");
            let lbl = txt!(lo.x0 + p + sz*0.8, ty, s, sz_s, col::TEXT_DIM);
            self.e_stage_dots.push(dot);
            self.e_stage_texts.push(lbl);
        }

        // プログレスバー
        let bar_y  = y + mid_h - lo.line_h - p;
        let bar_w  = pipe_w - p * 2.0;
        ui!(Rect::new(lo.x0 + p, bar_y, bar_w, lo.line_h * 0.6), col::TRACK, "solid");
        let e_pf = ui!(Rect::new(lo.x0 + p, bar_y, 4.0, lo.line_h * 0.6), col::ACCENT, "progress");
        self.e_prog_fill    = e_pf;
        self.e_prog_track_w = bar_w;

        // 内部AI一覧
        let ai_bar_max = ai_w - p * 2.0 - 50.0;
        self.e_ai_bar_max = ai_bar_max;
        for i in 0..AI_N {
            let ty  = y + p + lo.line_h * (1.5 + i as f32);
            let lbl = txt!(ai_x + p, ty, "---", sz_s, col::TEXT_DIM);
            let bar = ui!(Rect::new(ai_x + p + 90.0, ty + sz * 0.45,
                          4.0, lo.line_h * 0.35), col::TEXT_DIM, "solid");
            self.e_ai_texts.push(lbl);
            self.e_ai_bars.push(bar);
        }
        y += mid_h + 1.0;

        // ── CNN / WEIGHT VISUALIZER ─────────────────────────────────
        let filter_cell = 5.0;
        let filter_gap  = 3.0;
        let filter_h    = filter_cell * 5.0;
        let class_bar_h = 30.0;
        let cnn_h       = filter_h + class_bar_h + lo.line_h * 3.5 + p * 2.0;

        ui!(Rect::new(lo.x0, y, lo.w, 1.0), col::BORDER, "solid");
        ui!(Rect::new(lo.x0, y, lo.w, cnn_h), col::PANEL, "solid");
        txt!(lo.x0 + p, y + p, "CNN / WEIGHT VISUALIZER", sz_s, col::ACCENT);

        // フィルタセル (8×25)
        let fc_x0  = lo.x0 + p;
        let fc_y0  = y + p + lo.line_h * 1.3;
        for f in 0..FILTER_N {
            let fx = fc_x0 + f as f32 * (filter_cell * 5.0 + filter_gap);
            for ky in 0..5usize {
                for kx in 0..5usize {
                    let cx = fx + kx as f32 * filter_cell;
                    let cy = fc_y0 + ky as f32 * filter_cell;
                    let cell = ui!(Rect::new(cx, cy, filter_cell-0.5, filter_cell-0.5),
                                  [0.15, 0.15, 0.20, 1.0], "solid");
                    self.e_filter_cells.push(cell);
                }
            }
        }

        // 推論情報
        let info_y = fc_y0 + filter_h + 2.0;
        let e_pred = txt!(lo.x0 + p, info_y, "pred: -  conf: --", sz_s, col::TEXT);
        let e_kl   = txt!(lo.x0 + p + lo.w * 0.45, info_y, "KL(CNN): --", sz_s, col::TEXT_DIM);
        self.e_cnn_pred = e_pred;
        self.e_cnn_kl   = e_kl;

        // クラスバー (0-9)
        let cb_y    = info_y + lo.line_h + 2.0;
        let cell_cw = (lo.w - p * 2.0) / CLASS_N as f32;
        self.e_class_max_h = class_bar_h;
        for i in 0..CLASS_N {
            let cx = lo.x0 + p + i as f32 * cell_cw;
            // トラック
            ui!(Rect::new(cx + 1.0, cb_y, cell_cw - 2.0, class_bar_h), col::TRACK, "solid");
            // バー (初期高さ1)
            let bar = ui!(Rect::new(cx + 1.0, cb_y + class_bar_h - 1.0,
                          cell_cw - 2.0, 1.0), col::TEXT_DIM, "solid");
            let lbl = txt!(cx + 2.0, cb_y + class_bar_h + 1.0,
                          i.to_string(), sz_s - 1.0, col::TEXT_DIM);
            self.e_class_bars.push(bar);
            self.e_class_y_base.push(cb_y + class_bar_h); // バー下端固定Y
            self.e_class_labels.push(lbl);
        }
        y += cnn_h + 1.0;

        // ── LOG ────────────────────────────────────────────────────
        let log_h = lo.line_h * (LOG_N as f32 + 2.0) + p;
        ui!(Rect::new(lo.x0, y, lo.w, 1.0), col::BORDER, "solid");
        ui!(Rect::new(lo.x0, y, lo.w, log_h), col::PANEL, "solid");
        txt!(lo.x0 + p, y + p, "LOG", sz_s, col::ACCENT);
        for i in 0..LOG_N {
            let ly  = y + p + lo.line_h * (1.4 + i as f32);
            let lid = txt!(lo.x0 + p, ly, "", sz_s, col::TEXT_DIM);
            self.e_log_lines.push(lid);
        }
        y += log_h + 1.0;

        // ── REQUIRE + ボタン ───────────────────────────────────────
        let req_h = lo.line_h * 3.0 + p * 2.0;
        ui!(Rect::new(lo.x0, y, lo.w, 1.0), col::BORDER, "solid");
        ui!(Rect::new(lo.x0, y, lo.w, req_h), [0.08, 0.04, 0.02, 1.0], "solid");
        let e_req = txt!(lo.x0 + p, y + p, "REQUIRE: ---", sz_s, col::ORANGE);
        self.e_req_text = e_req;

        // ボタン2個
        let btn_y  = y + p + lo.line_h * 1.5;
        let btn_h  = lo.line_h + p;
        let btn_w  = (lo.w * 0.45).min(180.0);

        // [▶ REQUEST TO HUMAN]
        let b1_rect = Rect::new(lo.x0 + p, btn_y, btn_w, btn_h);
        let b1_bg   = ui!(b1_rect, col::BTN_REQ, "solid");
        let b1_lbl  = txt!(lo.x0 + p * 2.0, btn_y + p * 0.5,
                           "▶ REQUEST TO HUMAN", sz_s, col::TEXT);
        self.buttons.push(UiButton {
            bg: b1_bg, label: b1_lbl,
            action: ButtonAction::RequestToHuman,
            rect: b1_rect,
        });

        // [▶ SAVE WEIGHTS]
        let b2_x    = lo.x0 + p + btn_w + p;
        let b2_rect = Rect::new(b2_x, btn_y, btn_w, btn_h);
        let b2_bg   = ui!(b2_rect, col::BTN, "solid");
        let b2_lbl  = txt!(b2_x + p * 2.0, btn_y + p * 0.5,
                           "▶ SAVE WEIGHTS", sz_s, col::TEXT);
        self.buttons.push(UiButton {
            bg: b2_bg, label: b2_lbl,
            action: ButtonAction::SaveWeights,
            rect: b2_rect,
        });

        self.all_entities = ids;
    }

    // ─────────────────────────────────────────────────────────────
    // フレーム更新
    // ─────────────────────────────────────────────────────────────

    pub fn update(&self, world: &mut ObjectManager, core: &NeuralCore) {
        // ヘッダ
        let div = core.mean_divergence();
        set_txt(world, self.e_div_text,
            format!("AVG DIV: {:.3}", div),
            lerp_c(col::GREEN, col::RED, div * 2.0));
        set_txt(world, self.e_stage_label,
            format!("[{}]", core.stage.label()), col::YELLOW);

        // Pipeline バー
        let prog = core.pipeline_progress();
        set_w(world, self.e_prog_fill, (self.e_prog_track_w * prog).max(4.0));

        // ステージ ドット + テキスト
        let stage_keys = ["IOT","ECS","LLM","META-PROTO","OUTPUT","DONE"];
        let cur = core.stage.label();
        let prog_idx = stage_keys.iter().position(|&s| s == cur).unwrap_or(0);
        for i in 0..STAGE_N.min(self.e_stage_texts.len()) {
            let done   = i < prog_idx;
            let active = i == prog_idx;
            let c = if active { col::YELLOW } else if done { col::GREEN } else { col::TEXT_DIM };
            set_txt(world, self.e_stage_texts[i], stage_keys[i], c);
            set_col(world, self.e_stage_dots[i],
                if active { col::YELLOW } else if done { col::GREEN } else { col::TRACK });
        }

        // Internal AI
        for (i, ai) in core.internals.iter().enumerate().take(AI_N.min(self.e_ai_texts.len())) {
            let status = if ai.online { "ON " } else { "off" };
            let c = if ai.online {
                lerp_c(col::GREEN, col::RED, ai.divergence * 1.5)
            } else { col::TEXT_DIM };
            set_txt(world, self.e_ai_texts[i],
                format!("{:<12} {:.2} {}", ai.name(), ai.divergence, status), c);
            let bar_w = (self.e_ai_bar_max * ai.divergence).max(2.0);
            set_w(world, self.e_ai_bars[i], bar_w);
            set_col(world, self.e_ai_bars[i], c);
        }

        // LOG
        let logs: Vec<_> = core.logs.iter().rev().take(LOG_N).collect();
        for (i, &id) in self.e_log_lines.iter().enumerate() {
            if let Some(e) = logs.get(i) {
                set_txt(world, id,
                    format!("{} {:>12} > {}", e.level.tag(), e.source, e.message),
                    e.level.color());
            } else {
                set_txt(world, id, "", col::TEXT_DIM);
            }
        }

        // Require テキスト
        let unresolved: Vec<_> = core.requires.iter()
            .filter(|r| !r.resolved)
            .map(|r| r.key.as_str())
            .collect();
        if unresolved.is_empty() {
            set_txt(world, self.e_req_text, "REQUIRE: (all resolved)", col::GREEN);
        } else {
            set_txt(world, self.e_req_text,
                format!("REQUIRE: {}", unresolved.join(" | ")), col::ORANGE);
        }
    }

    pub fn update_cnn(
        &self,
        world:        &mut ObjectManager,
        class_probs:  &[f32; 10],
        pred_class:   usize,
        kl:           f32,
        conv_weights: &[f32],
    ) {
        let conf = class_probs[pred_class];
        set_txt(world, self.e_cnn_pred,
            format!("pred:{} conf:{:.2}", pred_class, conf),
            if conf > 0.8 { col::GREEN } else { col::YELLOW });
        set_txt(world, self.e_cnn_kl,
            format!("KL(CNN):{:.4}", kl),
            lerp_c(col::GREEN, col::RED, (kl * 5.0).min(1.0)));

        // クラスバー (下揃え固定)
        for i in 0..CLASS_N.min(self.e_class_bars.len()) {
            let p  = class_probs[i];
            let h  = (p * self.e_class_max_h).max(1.0);
            let c  = if i == pred_class { col::ACCENT } else { col::TEXT_DIM };
            if let Some(u) = world.ui_elements.get_mut(self.e_class_bars[i]) {
                u.rect.height = h;
                u.rect.y      = self.e_class_y_base[i] - h;  // 下揃え
                u.color       = c;
            }
        }

        // Convフィルタ
        for f in 0..FILTER_N {
            for ky in 0..5usize {
                for kx in 0..5usize {
                    let idx = f * 25 + ky * 5 + kx;
                    if idx >= conv_weights.len() { break; }
                    let w = conv_weights[idx];
                    let v = (w.clamp(-1.0, 1.0) + 1.0) * 0.5;
                    let c = if w >= 0.0 { [0.1, v * 0.9 + 0.1, v * 0.4, 1.0] }
                            else        { [v * 0.9 + 0.1, 0.1, 0.1, 1.0] };
                    let ci = f * 25 + ky * 5 + kx;
                    if let Some(u) = world.ui_elements.get_mut(self.e_filter_cells[ci]) {
                        u.color = c;
                    }
                }
            }
        }
    }

    /// マウスクリック座標からボタンヒットを返す
    pub fn hit_button(&self, mx: f32, my: f32) -> Option<ButtonAction> {
        for btn in &self.buttons {
            let r = &btn.rect;
            if mx >= r.x && mx <= r.x + r.width && my >= r.y && my <= r.y + r.height {
                return Some(btn.action);
            }
        }
        None
    }
}
