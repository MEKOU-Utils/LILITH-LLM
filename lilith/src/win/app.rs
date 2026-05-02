use std::sync::Arc;

use winit::{ application::ApplicationHandler, event_loop::ActiveEventLoop, window::{ Window, WindowId }, event::WindowEvent};
use pollster;

use crate::mygpu::State;
pub struct AppMain {
    pub state: Option<State>,
}

impl ApplicationHandler for AppMain {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("LiLith");
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        
        let state = pollster::block_on(State::new(window.clone())).unwrap();
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => {
                println!("LiLith engine");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size.width, physical_size.height);
            }
            WindowEvent::RedrawRequested => {
                state.render().expect("render failed");
            }
            _ => {},
        }
    }
}

