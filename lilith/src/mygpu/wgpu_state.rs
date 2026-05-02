    //!
    //! WGPUのテストファイルここを更新していくが全然構造が失敗するなら壊しまくるぜ！( ^)o(^ )
    //! 
    //! wgpu = "29.0.3"


    ///import群
    use std::{ sync::Arc};
    use anyhow::Ok;
    use wgpu::{
        Dx12BackendOptions,
        GlBackendOptions,
        NoopBackendOptions,
        PipelineLayoutDescriptor,

    };
    use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::{ActivationToken, Window}};


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
    render_pipeline: wgpu::RenderPipeline,
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

        // State::new の中に追加
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // 頂点バッファを自前で作るまでは空でOK
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            // ... その他（ラスタライザ設定など）
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: true,
            window,
            render_pipeline,
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

    pub fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }

    pub fn update() {
        todo!();
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }
            let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.surface.configure(&self.device, &self.config);
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Skip this frame
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // You could recreate the devices and all resources
                // created with it here, but we'll just bail
                anyhow::bail!("Lost device");
            }
        };

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            // ここで「何を描くか」のブロックを開始
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 }), // 背景色
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            // さっき作ったパイプラインをセット！
            render_pass.set_pipeline(&self.render_pipeline);
            // 3つの頂点（三角形）を描画しろ！という命令
            render_pass.draw(0..3, 0..1); 
        } // ここで render_pass がドロップされ、命令が確定する

        // 命令をベルトコンベヤー（Queue）に流す
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

}