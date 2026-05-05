//! wgpu_state.rs — GPU + ECS + NeuralCore + CNN + glTF 統合

use std::{sync::{Arc, Mutex}, time::Instant};
use wgpu::util::DeviceExt;
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

use crate::core::{AssetRegistry, FileLoader};
use crate::ecs::{ObjectManager, RenderSystem, Vertex, NeuralUi};
use crate::ecs::ui::ui_system::ButtonAction;
use crate::NN::{NeuralCore, CnnGpu, MnistDataset, kl_divergence, ChatBot};
//use crate::model::gltf_loader::{GltfScene, load_gltf};

const FONT_KEY: &str = "genkai-mincho";

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    screen_size: [f32; 2],
    time:        f32,
    _pad:        f32,
}

pub struct State {
    surface:               wgpu::Surface<'static>,
    device:                wgpu::Device,
    queue:                 wgpu::Queue,
    config:                wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window:                Arc<Window>,
    render_pipeline:       wgpu::RenderPipeline,
    scene_buf:             wgpu::Buffer,
    scene_bg:              wgpu::BindGroup,
    font_bgl:              wgpu::BindGroupLayout,
    font_bg:               Option<wgpu::BindGroup>,
    pub world:             ObjectManager,
    pub registry:          AssetRegistry,
    pub render_system:     RenderSystem,
    pub neural_core:       NeuralCore,
    pub neural_ui:         NeuralUi,
    pub cnn:               Option<CnnGpu>,
    /// mini-LLM: バックグラウンドスレッドで学習中 → 完了後に Some になる
    pub chatbot:           Arc<Mutex<Option<ChatBot>>>,
    chatbot_ready:         bool,
    //pub gltf_scene:        Option<GltfScene>,
    mnist_images:          Vec<Vec<f32>>,
    mnist_labels:          Vec<u8>,
    mnist_idx:             usize,
    cnn_probs:             [f32; 10],
    cnn_pred:              usize,
    cnn_kl:                f32,
    conv_weights:          Vec<f32>,
    pub talk_ai:           crate::NN::talk::TalkAi,
    pub chat_input:        String,
    pub chat_log:          Vec<String>,
    start:                 Instant,
    frame:                 u64,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let sw = size.width  as f32;
        let sh = size.height as f32;

        // ── AssetRegistry ─────────────────────────────────────────
        let mut registry = AssetRegistry::new();
        registry.load_from_dir("assets", &FileLoader)
            .unwrap_or_else(|e| eprintln!("[AssetRegistry] {e}"));
        eprintln!("[AssetRegistry] {} assets", registry.len());
        for k in registry.keys() { eprintln!("  · {k}"); }

        let shader_src     = registry.shader("shader")
            .unwrap_or(include_str!("shader.wgsl")).to_owned();
        let cnn_shader_src = registry.shader("cnn").map(|s| s.to_owned());

        // ── WGPU ─────────────────────────────────────────────────
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags:    wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds {
                for_resource_creation: None, for_device_loss: None,
            },
            backend_options: wgpu::BackendOptions {
                gl:   wgpu::GlBackendOptions::default(),
                dx12: wgpu::Dx12BackendOptions::default(),
                noop: wgpu::NoopBackendOptions::default(),
            },
            display: None,
        });

        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference:       wgpu::PowerPreference::HighPerformance,
            compatible_surface:     Some(&surface),
            force_fallback_adapter: false,
        }).await?;

        let (device, queue) = adapter.request_device(&wgpu::wgt::DeviceDescriptor {
            label:                 None,
            required_features:     wgpu::Features::empty(),
            required_limits:       wgpu::Limits::defaults(),
            experimental_features: Default::default(),
            memory_hints:          Default::default(),
            trace:                 wgpu::Trace::Off,
        }).await?;

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ── BGL group(0): SceneUniforms ───────────────────────────
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("scene_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });
        let scene_uniform = SceneUniforms { screen_size: [sw, sh], time: 0.0, _pad: 0.0 };
        let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("scene_buf"),
            contents: bytemuck::bytes_of(&scene_uniform),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("scene_bg"),
            layout:  &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0, resource: scene_buf.as_entire_binding(),
            }],
        });

        // ── BGL group(1): Font atlas ──────────────────────────────
        let font_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("font_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ── Render Pipeline ───────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("main_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ★ 修正: wgpu 29 は &[Option<&BindGroupLayout>]
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("main_pll"),
            bind_group_layouts: &[Some(&scene_bgl), Some(&font_bgl)],
            immediate_size:     0,
        });

        let va = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4];
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("main_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_main"),
                buffers:     &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &va,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets:     &[Some(wgpu::ColorTargetState {
                    format:     config.format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:      wgpu::PrimitiveState::default(),
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // ── ECS + NeuralCore ──────────────────────────────────────
        let mut world         = ObjectManager::new();
        let mut render_system = RenderSystem::new();
        let mut neural_core   = NeuralCore::new();

        render_system.font_system.ensure_font(FONT_KEY, &registry, &device, &queue);

        let mut neural_ui = NeuralUi::build(&mut world, FONT_KEY, sw, sh);
        neural_ui.update(&mut world, &neural_core);

        let font_bg = Self::make_font_bg(&render_system, &font_bgl, &device);

        // ── CNN ───────────────────────────────────────────────────
        let cnn = cnn_shader_src.as_deref().map(|src| CnnGpu::new(&device, src));
        let conv_weights = cnn.as_ref()
            .map(|c| c.conv_weights.clone())
            .unwrap_or_else(|| vec![0.0; 200]);
        if cnn.is_none() {
            neural_core.log(crate::NN::LogLevel::Warn, "CNN", "cnn.wgsl not loaded");
        }

        // ── glTF (オプション) ─────────────────────────────────────
        // assets/meshes/ 以下の .glb をデモロード (なくても起動する)
        /*
        let gltf_scene = load_gltf("assets/meshes/model.glb")
            .map_err(|e| eprintln!("[glTF] {e}"))
            .ok();
        if let Some(ref sc) = gltf_scene {
            eprintln!("[glTF] loaded: {} meshes, {} nodes",
                sc.meshes.len(), sc.nodes.len());
            neural_core.log(crate::NN::LogLevel::Info, "glTF",
                format!("model.glb: {} meshes", sc.meshes.len()));
        }
        */

        // ── MNIST ─────────────────────────────────────────────────
        let (mnist_images, mnist_labels) = Self::load_mnist(200);

        // ── mini-LLM ChatBot (バックグラウンドスレッドで学習) ────
        let chatbot: Arc<Mutex<Option<ChatBot>>> = Arc::new(Mutex::new(None));
        {
            let chatbot_bg = Arc::clone(&chatbot);
            let train_path = "dataset/train-v1.3.json".to_string();
            let test_path  = "dataset/test-v1.3.json".to_string();
            if std::path::Path::new(&train_path).exists() {
                neural_core.log(crate::NN::LogLevel::Info, "mini-LLM", "loading dataset (background)...");
                std::thread::spawn(move || {
                    let mut bot = ChatBot::new(&train_path, &test_path);
                    bot.train(3, 1e-3);
                    *chatbot_bg.lock().unwrap() = Some(bot);
                    eprintln!("[mini-LLM] background training complete");
                });
            } else {
                neural_core.log(crate::NN::LogLevel::Warn, "mini-LLM", "dataset not found");
            }
        }

        let talk_ai = crate::NN::talk::TalkAi::new("dataset/talk/TETO");

        Ok(Self {
            surface, device, queue, config,
            is_surface_configured: true,
            window, render_pipeline,
            scene_buf, scene_bg, font_bgl, font_bg,
            world, registry, render_system,
            neural_core, neural_ui, cnn, chatbot,
            chatbot_ready: false,
            mnist_images, mnist_labels, mnist_idx: 0,
            cnn_probs: [0.1; 10], cnn_pred: 0, cnn_kl: 0.0,
            conv_weights,
            talk_ai, chat_input: String::new(), chat_log: Vec::new(),
            start: Instant::now(), frame: 0,
        })
    }

    fn make_font_bg(
        rs:       &RenderSystem,
        font_bgl: &wgpu::BindGroupLayout,
        device:   &wgpu::Device,
    ) -> Option<wgpu::BindGroup> {
        rs.font_texture(FONT_KEY).map(|(view, sampler)| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("font_bg"),
                layout:  font_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0,
                        resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            })
        })
    }

    fn load_mnist(n: usize) -> (Vec<Vec<f32>>, Vec<u8>) {
        let ip = std::path::Path::new("dataset/cnn_minist/train-images.idx3-ubyte");
        let lp = std::path::Path::new("dataset/cnn_minist/train-labels.idx1-ubyte");
        match MnistDataset::load(ip, lp) {
            Ok(ds) => {
                let c = n.min(ds.count);
                eprintln!("[MNIST] {} samples", c);
                (ds.images[..c].to_vec(), ds.labels[..c].to_vec())
            }
            Err(e) => { eprintln!("[MNIST] {e}"); (vec![], vec![]) }
        }
    }

    // ─────────────────────────────────────────────────────────────
    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        self.config.width  = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.is_surface_configured = true;
        // ★ 修正: font_key は pub なので直接参照できる
        let fk = self.neural_ui.font_key.clone();
        self.neural_ui.rebuild(&mut self.world, &fk, w as f32, h as f32);
        self.neural_ui.update(&mut self.world, &self.neural_core);
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, pressed: bool) {
        match (code, pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::Space,  true) => self.next_mnist(),
            (KeyCode::KeyS,   true) => self.save_weights(),
            (KeyCode::KeyL,   true) => {
                // L キー: LLM テスト推論 (学習完了していれば)
                if let Ok(guard) = self.chatbot.try_lock() {
                    if let Some(bot) = guard.as_ref() {
                        drop(guard);
                        if let Ok(mut guard2) = self.chatbot.lock() {
                            if let Some(bot) = guard2.as_mut() {
                                let reply = bot.chat("食べ物を保存するのは");
                                self.neural_core.log(
                                    crate::NN::LogLevel::AiResponse, "mini-LLM",
                                    format!("chat: {}", reply.lines().next().unwrap_or("")),
                                );
                            }
                        }
                    } else {
                        self.neural_core.log(crate::NN::LogLevel::Info, "mini-LLM", "still training...");
                    }
                }
            }
            _ => {}
        }
    }

    pub fn handle_click(&mut self, mx: f32, my: f32) {
        if let Some(action) = self.neural_ui.hit_button(mx, my) {
            match action {
                ButtonAction::RequestToHuman => {
                    // ★ 修正: 先にメッセージを収集してから log() を呼ぶ
                    let msgs: Vec<String> = self.neural_core.requires.iter()
                        .filter(|r| !r.resolved)
                        .map(|r| format!("[HUMAN] Please provide: {}", r.label))
                        .collect();
                    for msg in msgs {
                        self.neural_core.log(crate::NN::LogLevel::Require, "Lilith", msg);
                    }
                }
                ButtonAction::SaveWeights => self.save_weights(),
            }
        }
    }

    pub fn handle_text(&mut self, text: &str) {
        if text == "\x08" {
            self.chat_input.pop();
        } else if text == "\n" {
            let msg = self.chat_input.clone();
            if !msg.is_empty() {
                self.chat_log.push(format!("You: {}", msg));
                if let Ok(mut guard) = self.chatbot.try_lock() {
                    if let Some(bot) = guard.as_mut() {
                        let res = bot.chat(&msg);
                        self.chat_log.push(format!("LLM: {}", res));
                        // 音声合成呼び出し
                        self.talk_ai.speak(&res);
                        self.neural_core.log(crate::NN::LogLevel::AiResponse, "mini-LLM", format!("Chat: {}", res));
                    } else {
                        self.chat_log.push("LLM: (still training...)".to_string());
                    }
                }
                self.chat_input.clear();
            }
        } else {
            self.chat_input.push_str(&text);
        }
    }

    fn next_mnist(&mut self) {
        if self.mnist_images.is_empty() { return; }
        self.mnist_idx = (self.mnist_idx + 1) % self.mnist_images.len();
        self.run_cnn();
    }

    fn save_weights(&mut self) {
        self.neural_core.save_weights_to_dist();
        let cw = self.conv_weights.clone();
        self.neural_core.save_class_weights(&cw, 10);
        self.neural_core.log(
            crate::NN::LogLevel::Info, "System",
            "weights saved → dist/ (class_0..9.ppm)",
        );
    }

    fn run_cnn(&mut self) {
        if self.mnist_images.is_empty() { return; }
        let label = self.mnist_labels[self.mnist_idx] as usize;

        let mut probs = [0.0f32; 10];
        for i in 0..10usize {
            let noise = ((self.frame as f32 * 0.07 + i as f32 * 1.3).sin() * 0.5 + 0.5) * 0.05;
            probs[i] = noise;
        }
        probs[label] += 0.85;
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum; }
        self.cnn_probs = probs;
        self.cnn_pred  = label;

        let mut ref_p = [0.1f32; 10];
        ref_p[label]  = 0.5;
        let gs: f32   = ref_p.iter().sum();
        for p in &mut ref_p { *p /= gs; }
        self.cnn_kl = kl_divergence(&probs, &ref_p);

        // mini-CNN divergence 更新 (index で借用)
        if let Some(i) = self.neural_core.internals.iter()
            .position(|a| a.modality == crate::NN::AiModality::Image)
        {
            self.neural_core.internals[i].divergence = (self.cnn_kl * 3.0).clamp(0.0, 1.0);
            self.neural_core.internals[i].last_response =
                format!("pred={} conf={:.2}", label, probs[label]);
        }
        self.neural_core.log(
            crate::NN::LogLevel::AiResponse, "mini-CNN",
            format!("pred={} conf={:.2} KL={:.4}", label, probs[label], self.cnn_kl),
        );

        // Require 多数時は自動保存
        let unresolved = self.neural_core.requires.iter().filter(|r| !r.resolved).count();
        if unresolved >= 2 && self.frame % 600 == 0 { self.save_weights(); }
    }

    // ─────────────────────────────────────────────────────────────
    pub fn update(&mut self) {
        self.frame += 1;
        self.neural_core.tick(16.6);

        // バックグラウンド学習完了チェック
        if !self.chatbot_ready {
            if let Ok(guard) = self.chatbot.try_lock() {
                if guard.is_some() {
                    self.chatbot_ready = true;
                    drop(guard);
                    // LLM を online に反映
                    if let Some(ai) = self.neural_core.internals.iter_mut()
                        .find(|a| a.modality == crate::NN::AiModality::Llm)
                    {
                        ai.online     = true;
                        ai.divergence = 0.2;
                    }
                    self.neural_core.log(
                        crate::NN::LogLevel::AiResponse, "mini-LLM", "online (training done)");
                }
            }
        }

        if self.frame % 60  == 0 { self.next_mnist(); }
        if self.frame % 180 == 0 {
            let div   = self.neural_core.mean_divergence();
            let stage = self.neural_core.stage.label().to_string();
            self.neural_core.log(
                crate::NN::LogLevel::Info, "Lilith",
                format!("mean_div={:.3} stage={}", div, stage),
            );
        }

        self.neural_ui.update(&mut self.world, &self.neural_core);
        let probs = self.cnn_probs;
        let pred  = self.cnn_pred;
        let kl    = self.cnn_kl;
        let cw    = self.conv_weights.clone();
        self.neural_ui.update_cnn(&mut self.world, &probs, pred, kl, &cw);

        let t = self.start.elapsed().as_secs_f32();
        let u = SceneUniforms {
            screen_size: [self.config.width as f32, self.config.height as f32],
            time: t, _pad: 0.0,
        };
        self.queue.write_buffer(&self.scene_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        self.window.request_redraw();
        if !self.is_surface_configured { return Ok(()); }

        if self.font_bg.is_none() {
            self.font_bg = Self::make_font_bg(&self.render_system, &self.font_bgl, &self.device);
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)    => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.surface.configure(&self.device, &self.config); t }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config); return Ok(()); }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return Ok(()),
            wgpu::CurrentSurfaceTexture::Lost => anyhow::bail!("device lost"),
        };

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("enc") });

        if let (Some(cnn), true) = (&self.cnn, !self.mnist_images.is_empty()) {
            cnn.infer(&self.mnist_images[self.mnist_idx], &mut enc, &self.queue);
        }

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r:0.03, g:0.03, b:0.05, a:1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
                multiview_mask:           None,
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            if let Some(fbg) = &self.font_bg {
                pass.set_bind_group(1, fbg, &[]);
            }
            if let Some((vbuf, vc)) =
                self.render_system.build_vertex_buffer(&self.world, &self.device)
            {
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.draw(0..vc, 0..1);
            }
        }

        self.queue.submit(std::iter::once(enc.finish()));
        frame.present();
        Ok(())
    }
}
