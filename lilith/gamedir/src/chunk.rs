//! chunk.rs — ブロック・チャンク・ワールド管理

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────
// ブロック定義
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Block {
    Air    = 0,
    Grass  = 1,
    Dirt   = 2,
    Stone  = 3,
    Sand   = 4,
    Wood   = 5,
    Leaves = 6,
    Water  = 7,
}

impl Block {
    pub fn is_solid(self) -> bool { self != Block::Air && self != Block::Water }
    pub fn is_transparent(self) -> bool { matches!(self, Block::Air | Block::Leaves | Block::Water) }

    /// 面ごとのカラー (RGBA, 0〜1)  face: 0=+Y 1=-Y 2=side
    pub fn color(self, face: u8) -> [f32; 4] {
        match self {
            Block::Grass => match face {
                0 => [0.30, 0.70, 0.22, 1.0],   // 上: 草
                1 => [0.48, 0.35, 0.18, 1.0],   // 下: 土
                _ => [0.45, 0.58, 0.25, 1.0],   // 側: 草土
            },
            Block::Dirt   => [0.48, 0.35, 0.18, 1.0].into_arr(),
            Block::Stone  => [0.50, 0.50, 0.52, 1.0].into_arr(),
            Block::Sand   => [0.88, 0.83, 0.55, 1.0].into_arr(),
            Block::Wood   => match face {
                0 | 1 => [0.55, 0.40, 0.20, 1.0],
                _     => [0.60, 0.45, 0.25, 1.0],
            },
            Block::Leaves => [0.20, 0.55, 0.15, 0.85],
            Block::Water  => [0.20, 0.40, 0.85, 0.70],
            Block::Air    => [0.0; 4],
        }
    }

    /// 面の明度係数 (0=+Y top, 1=-Y bottom, 2=N/S, 3=E/W)
    pub fn face_light(face_dir: usize) -> f32 {
        match face_dir {
            0 => 1.00,  // 上 (日照)
            1 => 0.40,  // 下 (暗)
            2 | 3 => 0.75,  // N/S
            _ => 0.65,  // E/W
        }
    }
}

trait IntoArr { fn into_arr(self) -> [f32; 4]; }
impl IntoArr for [f32; 4] { fn into_arr(self) -> [f32; 4] { self } }

// ─────────────────────────────────────────────────────────────────
// チャンク
// ─────────────────────────────────────────────────────────────────

pub const CHUNK_W: usize = 16;
pub const CHUNK_H: usize = 64;
pub const CHUNK_D: usize = 16;

pub struct Chunk {
    blocks: Box<[Block; CHUNK_W * CHUNK_H * CHUNK_D]>,
    pub dirty: bool,
}

impl Chunk {
    pub fn empty() -> Self {
        Self {
            blocks: Box::new([Block::Air; CHUNK_W * CHUNK_H * CHUNK_D]),
            dirty: true,
        }
    }

    fn idx(x: usize, y: usize, z: usize) -> usize {
        y * CHUNK_W * CHUNK_D + z * CHUNK_W + x
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Block {
        if x >= CHUNK_W || y >= CHUNK_H || z >= CHUNK_D { Block::Air }
        else { self.blocks[Self::idx(x, y, z)] }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, b: Block) {
        if x < CHUNK_W && y < CHUNK_H && z < CHUNK_D {
            self.blocks[Self::idx(x, y, z)] = b;
            self.dirty = true;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// 地形生成 (ハッシュベース疑似ノイズ)
// ─────────────────────────────────────────────────────────────────

fn hash2(x: i32, z: i32) -> f32 {
    let h = (x.wrapping_mul(374761393)).wrapping_add(z.wrapping_mul(668265263)) as u32;
    let h = h ^ (h >> 13);
    let h = h.wrapping_mul(1274126177) ^ (h >> 16);
    (h as f32) / (u32::MAX as f32)
}

fn smooth(x: f32, z: f32, scale: f32) -> f32 {
    let xi = (x / scale).floor() as i32;
    let zi = (z / scale).floor() as i32;
    let fx = (x / scale).fract();
    let fz = (z / scale).fract();
    let fx = fx * fx * (3.0 - 2.0 * fx);
    let fz = fz * fz * (3.0 - 2.0 * fz);
    let a = hash2(xi,   zi);
    let b = hash2(xi+1, zi);
    let c = hash2(xi,   zi+1);
    let d = hash2(xi+1, zi+1);
    a*(1.0-fx)*(1.0-fz) + b*fx*(1.0-fz) + c*(1.0-fx)*fz + d*fx*fz
}

pub fn terrain_height(wx: i32, wz: i32) -> usize {
    let x = wx as f32;
    let z = wz as f32;
    let h = smooth(x, z, 24.0) * 14.0
          + smooth(x, z, 8.0)  * 4.0
          + smooth(x, z, 3.0)  * 1.5;
    (8.0 + h).clamp(2.0, CHUNK_H as f32 - 4.0) as usize
}

pub fn generate_chunk(cx: i32, cz: i32) -> Chunk {
    let mut chunk = Chunk::empty();
    for lx in 0..CHUNK_W {
        for lz in 0..CHUNK_D {
            let wx = cx * CHUNK_W as i32 + lx as i32;
            let wz = cz * CHUNK_D as i32 + lz as i32;
            let h  = terrain_height(wx, wz);

            for ly in 0..h.min(CHUNK_H) {
                let b = if ly == h - 1 {
                    if h < 12 { Block::Sand } else { Block::Grass }
                } else if ly > h.saturating_sub(4) {
                    if h < 12 { Block::Sand } else { Block::Dirt }
                } else {
                    Block::Stone
                };
                chunk.set(lx, ly, lz, b);
            }
            // 水面 (y<10 は水)
            if h < 10 {
                for ly in h..10 { chunk.set(lx, ly, lz, Block::Water); }
            }
            // 木 (確率)
            if h >= 12 && hash2(wx * 7, wz * 13) > 0.97 {
                let ty = h;
                for dy in 0..5 {
                    if ty + dy < CHUNK_H { chunk.set(lx, ty+dy, lz, Block::Wood); }
                }
                for dx in 0usize..3 { for dz in 0usize..3 {
                    for dy in 3..6usize {
                        let lx2 = lx + dx;
                        let lz2 = lz + dz;
                        if lx2 < CHUNK_W && lz2 < CHUNK_D && ty + dy < CHUNK_H {
                            if chunk.get(lx2, ty+dy, lz2) == Block::Air {
                                chunk.set(lx2, ty+dy, lz2, Block::Leaves);
                            }
                        }
                    }
                }}
            }
        }
    }
    chunk
}

// ─────────────────────────────────────────────────────────────────
// ワールド
// ─────────────────────────────────────────────────────────────────

pub struct World {
    pub chunks: HashMap<(i32, i32), Chunk>,
}

impl World {
    pub fn new() -> Self { Self { chunks: HashMap::new() } }

    pub fn ensure_chunk(&mut self, cx: i32, cz: i32) {
        self.chunks.entry((cx, cz)).or_insert_with(|| generate_chunk(cx, cz));
    }

    /// ワールド座標でブロック取得
    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> Block {
        if wy < 0 || wy >= CHUNK_H as i32 { return Block::Air; }
        let (cx, lx) = floor_div(wx, CHUNK_W as i32);
        let (cz, lz) = floor_div(wz, CHUNK_D as i32);
        self.chunks.get(&(cx, cz))
            .map(|c| c.get(lx as usize, wy as usize, lz as usize))
            .unwrap_or(Block::Air)
    }

    /// ワールド座標でブロックをセット
    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, b: Block) {
        if wy < 0 || wy >= CHUNK_H as i32 { return; }
        let (cx, lx) = floor_div(wx, CHUNK_W as i32);
        let (cz, lz) = floor_div(wz, CHUNK_D as i32);
        if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
            chunk.set(lx as usize, wy as usize, lz as usize, b);
        }
    }

    /// 視線レイキャスト → ヒットしたブロック座標と面法線を返す
    pub fn raycast(&self, pos: [f32;3], dir: [f32;3], max_dist: f32)
        -> Option<([i32;3], [i32;3])>
    {
        let step = 0.05f32;
        let steps = (max_dist / step) as usize;
        let mut prev = [pos[0] as i32, pos[1] as i32, pos[2] as i32];
        for i in 1..=steps {
            let t = i as f32 * step;
            let px = pos[0] + dir[0]*t;
            let py = pos[1] + dir[1]*t;
            let pz = pos[2] + dir[2]*t;
            let bx = px.floor() as i32;
            let by = py.floor() as i32;
            let bz = pz.floor() as i32;
            if self.get_block(bx, by, bz).is_solid() {
                let normal = [prev[0]-bx, prev[1]-by, prev[2]-bz];
                return Some(([bx, by, bz], normal));
            }
            prev = [bx, by, bz];
        }
        None
    }
}

fn floor_div(a: i32, b: i32) -> (i32, i32) {
    let d = a.div_euclid(b);
    let r = a.rem_euclid(b);
    (d, r)
}
