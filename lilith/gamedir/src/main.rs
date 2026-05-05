mod chunk;
mod game;
mod render;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::chunk::{World, CHUNK_W, CHUNK_D, Block};
use crate::game::{Player, mat4_mul, perspective, view_matrix};
use crate::render::{Renderer, RenderMesh, build_chunk_mesh};

struct App {
    window:     Option<Arc<Window>>,
    renderer:   Option<Renderer>,
    world:      World,
    player:     Player,
    meshes:     HashMap<(i32, i32), RenderMesh>,
    keys:       HashSet<KeyCode>,
    last_frame: Instant,
    pointer_locked: bool,
}

impl App {
    fn new() -> Self {
        let mut world = World::new();
        // 初期チャンク生成 (3x3チャンク)
        for cx in -1..=1 {
            for cz in -1..=1 {
                world.ensure_chunk(cx, cz);
            }
        }

        Self {
            window: None,
            renderer: None,
            world,
            player: Player::new([8.0, 30.0, 8.0]),
            meshes: HashMap::new(),
            keys: HashSet::new(),
            last_frame: Instant::now(),
            pointer_locked: false,
        }
    }

    fn rebuild_dirty_chunks(&mut self) {
        let Some(renderer) = &self.renderer else { return };
        // Dirtyチャンクの座標を収集
        let mut dirty_coords = Vec::new();
        for cx in -2..=2 {
            for cz in -2..=2 {
                if let Some(chunk) = self.world.chunks.get(&(cx, cz)) {
                    if chunk.dirty { dirty_coords.push((cx, cz)); }
                } else {
                    dirty_coords.push((cx, cz));
                }
            }
        }
        
        // メッシュ再構築
        for (cx, cz) in dirty_coords {
            self.world.ensure_chunk(cx, cz);
            if let Some(mesh) = build_chunk_mesh(renderer.device(), &self.world, cx, cz) {
                self.meshes.insert((cx, cz), mesh);
            }
            if let Some(chunk) = self.world.chunks.get_mut(&(cx, cz)) {
                chunk.dirty = false;
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win = event_loop.create_window(Window::default_attributes().with_title("Minecraft + Lilith")).unwrap();
            let win = Arc::new(win);
            self.window = Some(win.clone());

            let renderer = pollster::block_on(Renderer::new(win.clone()));
            self.renderer = Some(renderer);
            
            // 全チャンク強制メッシュ化
            for (_, chunk) in &mut self.world.chunks {
                chunk.dirty = true;
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer { r.resize(size.width, size.height); }
            }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state, .. }, .. } => {
                if let PhysicalKey::Code(code) = physical_key {
                    if state == ElementState::Pressed {
                        self.keys.insert(code);
                        if code == KeyCode::Escape { self.pointer_locked = false; }
                    } else {
                        self.keys.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                if !self.pointer_locked {
                    self.pointer_locked = true;
                    return;
                }
                
                // ブロック破壊 / 設置
                if let Some((bp, normal)) = self.world.raycast(self.player.eye_pos(), self.player.forward(), 6.0) {
                    if button == winit::event::MouseButton::Left {
                        self.world.set_block(bp[0], bp[1], bp[2], Block::Air);
                    } else if button == winit::event::MouseButton::Right {
                        let np = [bp[0]+normal[0], bp[1]+normal[1], bp[2]+normal[2]];
                        // プレイヤーと重ならないか簡易チェック
                        let p_feet = [self.player.pos[0].floor() as i32, self.player.pos[1].floor() as i32, self.player.pos[2].floor() as i32];
                        let p_head = [p_feet[0], p_feet[1]+1, p_feet[2]];
                        if np != p_feet && np != p_head {
                            self.world.set_block(np[0], np[1], np[2], self.player.selected);
                        }
                    }
                    
                    // 周辺チャンクの更新フラグを立てる
                    let cx = (bp[0] as f32 / CHUNK_W as f32).floor() as i32;
                    let cz = (bp[2] as f32 / CHUNK_D as f32).floor() as i32;
                    for dx in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(c) = self.world.chunks.get_mut(&(cx+dx, cz+dz)) {
                                c.dirty = true;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: winit::event::DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.pointer_locked {
                self.player.yaw -= (dx as f32) * 0.003;
                self.player.pitch -= (dy as f32) * 0.003;
                self.player.pitch = self.player.pitch.clamp(-1.57, 1.57);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        if let Some(win) = &self.window {
            win.set_cursor_grab(if self.pointer_locked { winit::window::CursorGrabMode::Confined } else { winit::window::CursorGrabMode::None }).ok();
            win.set_cursor_visible(!self.pointer_locked);
            win.request_redraw();
        }

        self.player.update(&self.world, &self.keys, dt);
        self.rebuild_dirty_chunks();

        if let (Some(renderer), Some(win)) = (&mut self.renderer, &self.window) {
            let size = win.inner_size();
            let aspect = size.width as f32 / size.height.max(1) as f32;
            let proj = perspective(1.2, aspect, 0.1, 1000.0);
            let view = view_matrix(self.player.eye_pos(), self.player.yaw, self.player.pitch);
            let view_proj = mat4_mul(proj, view); // wgsl mat4 is column-major

            let mesh_refs: Vec<&RenderMesh> = self.meshes.values().collect();
            renderer.draw(view_proj, &mesh_refs);
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
