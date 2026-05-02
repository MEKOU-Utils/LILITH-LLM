    //!
    //! WGPUのテストファイルここを更新していくが全然構造が失敗するなら壊しまくるぜ！( ^)o(^ )
    //! 
    //! wgpu = "29.0.3"


    ///import群
    use std::{ sync::Arc};
    use anyhow::Ok;
    use wgpu::{Dx12BackendOptions, GlBackendOptions, NoopBackendOptions };
    use winit::window::Window;

    /* */
    /*
    struct State{
        surface: 画面への転送窓口
        device:  GPUの仮想化本体
        queue:   GPUへの命令ベルトコンベヤー
        config:  画面の設計図
        window:  OSのwindow実体
    }
    */

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
}


///ここを参照
/// https://docs.rs/wgpu-types/29.0.3/src/wgpu_types/instance.rs.html#67

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds { for_resource_creation: None, for_device_loss: None },
            backend_options: wgpu::BackendOptions { gl:GlBackendOptions::default(), dx12: Dx12BackendOptions::default(), noop: NoopBackendOptions::default() },
            display: None,
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await?;
        
        let (device, queue) = adapter
            .request_device(&wgpu::wgt::DeviceDescriptor { 
                label: None, 
                required_features: wgpu::Features::empty(), 
                required_limits: if cfg!(target_arch = "wasm32"){
                        wgpu::Limits::downlevel_defaults()
                    } else {
                        wgpu::Limits::defaults()
                    },
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off, 
            }).await?;

        let surface_caps = surface.get_capabilities(&adapter);
        
        //HDRを有効にしておくことにする
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba16Float )
            .unwrap_or(
                surface_caps.formats.iter()
                .copied()
                .find(|f| f.is_srgb())
                .unwrap_or(surface_caps.formats[0])
            );

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        let _modes = &surface_caps.present_modes;

        surface.configure(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: true,
            window
        })
    }

    pub fn resize(&mut self, width: u32, height: u32){
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }


}