use winit::event_loop::EventLoop;

mod core;
mod ecs;
#[allow(non_snake_case)]
mod NN;
mod win;
mod mygpu;
mod model;

use win::AppMain;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let mut app = AppMain::new();
    let _ = event_loop.run_app(&mut app);
    Ok(())
}
