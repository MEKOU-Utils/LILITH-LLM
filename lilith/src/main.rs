use anyhow::Ok;
use winit::event_loop::{ EventLoop};

mod win;
use win::AppMain;

mod mygpu;
use mygpu::State;

fn main() -> anyhow::Result<()> {
    
    let event_loop = EventLoop::new()?;

    let mut app = AppMain {state: None};
    event_loop.run_app(&mut app);
    Ok(())
}