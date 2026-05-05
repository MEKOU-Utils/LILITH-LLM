//! game.rs — プレイヤー・カメラ・入力・物理

use std::collections::HashSet;
use winit::keyboard::KeyCode;
use crate::chunk::{Block, World, CHUNK_W, CHUNK_D, CHUNK_H};

// ─────────────────────────────────────────────────────────────────
// 数学ユーティリティ
// ─────────────────────────────────────────────────────────────────
pub type Vec3 = [f32; 3];
pub type Mat4 = [[f32; 4]; 4];

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
pub fn dot(a: Vec3, b: Vec3) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
pub fn normalize(v: Vec3) -> Vec3 {
    let l = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt().max(1e-9);
    [v[0]/l, v[1]/l, v[2]/l]
}
pub fn add3(a: Vec3, b: Vec3) -> Vec3 { [a[0]+b[0],a[1]+b[1],a[2]+b[2]] }
pub fn sub3(a: Vec3, b: Vec3) -> Vec3 { [a[0]-b[0],a[1]-b[1],a[2]-b[2]] }
pub fn scale3(a: Vec3, s: f32) -> Vec3 { [a[0]*s,a[1]*s,a[2]*s] }

/// 列優先 4×4 行列乗算 (WGSL mat4x4 と同じレイアウト)
pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0f32;4];4];
    for col in 0..4 {
        for row in 0..4 {
            let mut s = 0.0f32;
            for k in 0..4 { s += a[k][row] * b[col][k]; }
            out[col][row] = s;
        }
    }
    out
}

/// wgpu 向け透視投影行列 (右手系, depth [0,1], 列優先)
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far/(near-far),      -1.0],
        [0.0, 0.0, near*far/(near-far),  0.0],
    ]
}

/// ビュー行列 (カメラ位置 + yaw/pitch から構築, 列優先)
pub fn view_matrix(pos: Vec3, yaw: f32, pitch: f32) -> Mat4 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    // 右手系: -Z が前
    let fwd   = [sy*cp, -sp, -cy*cp];
    let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
    let up    = cross(right, fwd);

    // 列優先ビュー行列
    [
        [right[0], up[0], -fwd[0], 0.0],
        [right[1], up[1], -fwd[1], 0.0],
        [right[2], up[2], -fwd[2], 0.0],
        [-dot(right,pos), -dot(up,pos), dot(fwd,pos), 1.0],
    ]
}

// ─────────────────────────────────────────────────────────────────
// プレイヤー
// ─────────────────────────────────────────────────────────────────

pub const PLAYER_H: f32 = 1.8;
pub const PLAYER_W: f32 = 0.6;
const GRAVITY: f32 = -22.0;
const JUMP_V:  f32 =  8.0;
const SPEED:   f32 =  4.5;

pub struct Player {
    pub pos:       Vec3,   // 足元座標
    pub vel:       Vec3,
    pub yaw:       f32,    // ラジアン, 水平回転
    pub pitch:     f32,    // ラジアン, 垂直回転 [-pi/2, pi/2]
    pub on_ground: bool,
    pub selected:  Block,  // 置くブロック種類
}

impl Player {
    pub fn new(spawn: Vec3) -> Self {
        Self {
            pos: spawn, vel: [0.0;3],
            yaw: 0.0, pitch: 0.0,
            on_ground: false,
            selected: Block::Grass,
        }
    }

    /// カメラ位置 (目線の高さ)
    pub fn eye_pos(&self) -> Vec3 {
        [self.pos[0], self.pos[1] + PLAYER_H * 0.85, self.pos[2]]
    }

    /// 視線方向ベクトル
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [sy*cp, -sp, -cy*cp]
    }

    /// 物理更新 (dt 秒)
    pub fn update(&mut self, world: &World, pressed: &HashSet<KeyCode>, dt: f32) {
        // ── 水平移動 ─────────────────────────────────────────────
        let (sy, cy) = self.yaw.sin_cos();
        let fwd_h  = normalize([sy, 0.0, -cy]);
        let right_h = [fwd_h[2], 0.0, -fwd_h[0]];

        let mut move_dir = [0.0f32; 3];
        if pressed.contains(&KeyCode::KeyW) { move_dir = add3(move_dir, fwd_h); }
        if pressed.contains(&KeyCode::KeyS) { move_dir = sub3(move_dir, fwd_h); }
        if pressed.contains(&KeyCode::KeyD) { move_dir = add3(move_dir, right_h); }
        if pressed.contains(&KeyCode::KeyA) { move_dir = sub3(move_dir, right_h); }

        let move_len = (move_dir[0]*move_dir[0]+move_dir[2]*move_dir[2]).sqrt();
        if move_len > 0.01 {
            let speed = if pressed.contains(&KeyCode::ShiftLeft) { SPEED * 2.0 } else { SPEED };
            self.vel[0] = move_dir[0] / move_len * speed;
            self.vel[2] = move_dir[2] / move_len * speed;
        } else {
            self.vel[0] *= 0.15f32.powf(dt);
            self.vel[2] *= 0.15f32.powf(dt);
        }

        // ── ジャンプ ──────────────────────────────────────────────
        if pressed.contains(&KeyCode::Space) && self.on_ground {
            self.vel[1] = JUMP_V;
            self.on_ground = false;
        }

        // ── 重力 ──────────────────────────────────────────────────
        if !self.on_ground { self.vel[1] += GRAVITY * dt; }

        // ── AABB コリジョン (軸ごと) ──────────────────────────────
        let hw = PLAYER_W * 0.5;
        for axis in 0..3 {
            let mut np = self.pos;
            np[axis] += self.vel[axis] * dt;
            if !collides(world, np, hw) {
                self.pos[axis] = np[axis];
            } else {
                if axis == 1 {
                    self.on_ground = self.vel[1] < 0.0;
                    self.vel[1] = 0.0;
                } else {
                    self.vel[axis] = 0.0;
                }
            }
        }

        // 落下限界
        if self.pos[1] < -20.0 {
            self.pos[1] = 30.0;
            self.vel = [0.0;3];
        }
    }
}

/// AABB がブロックと衝突するか
fn collides(world: &World, feet: Vec3, hw: f32) -> bool {
    let x0 = (feet[0] - hw).floor() as i32;
    let x1 = (feet[0] + hw).ceil()  as i32;
    let y0 = feet[1].floor() as i32;
    let y1 = (feet[1] + PLAYER_H).ceil() as i32;
    let z0 = (feet[2] - hw).floor() as i32;
    let z1 = (feet[2] + hw).ceil()  as i32;
    for bx in x0..=x1 { for by in y0..=y1 { for bz in z0..=z1 {
        if world.get_block(bx, by, bz).is_solid() { return true; }
    }}}
    false
}
