use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, KeyEvent, MouseButton, ElementState},
    event_loop::ActiveEventLoop,
    keyboard::PhysicalKey,
    window::{Window, WindowId},
};
use pollster;

use crate::mygpu::State;

pub struct AppMain {
    pub state:    Option<State>,
    mouse_pos:    (f32, f32),
}

impl AppMain {
    pub fn new() -> Self { Self { state: None, mouse_pos: (0.0, 0.0) } }
}

impl ApplicationHandler for AppMain {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Lilith Neural Core")
            .with_inner_size(winit::dpi::LogicalSize::new(760u32, 820u32));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(s) => {
                state.resize(s.width, s.height);
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    logical_key,
                    text,
                    state: key_state, ..
                }, ..
            } => {
                state.handle_key(event_loop, code, key_state.is_pressed());
                if key_state.is_pressed() {
                    if let winit::keyboard::Key::Named(winit::keyboard::NamedKey::Backspace) = logical_key {
                        state.handle_text("\x08".into()); // Backspace
                    } else if let winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter) = logical_key {
                        state.handle_text("\n".into()); // Enter
                    } else if let Some(t) = text {
                        state.handle_text(t.as_str());
                    }
                }
            }

            // マウス移動
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x as f32, position.y as f32);
            }

            // マウスクリック
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: ElementState::Pressed, ..
            } => {
                let (mx, my) = self.mouse_pos;
                state.handle_click(mx, my);
            }

            WindowEvent::RedrawRequested => {
                state.update();
                if let Err(e) = state.render() {
                    eprintln!("[render] {e}");
                }
            }

            _ => {}
        }
    }
}
