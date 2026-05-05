//! render.rs — wgpu レンダリング・ECS連携

use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;
use crate::chunk::{World, CHUNK_W, CHUNK_D, CHUNK_H, Block};

// ─────────────────────────────────────────────────────────────────
// ECS: コンポーネント群
// ─────────────────────────────────────────────────────────────────
pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale:    [f32; 3],
}

pub struct RenderMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer:  Option<wgpu::Buffer>,
    pub num_vertices:  u32,
    pub num_indices:   u32,
}

// ─────────────────────────────────────────────────────────────────
// 頂点フォーマット
// ─────────────────────────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color:    [f32; 4],
    pub normal:   [f32; 3],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// カメラ (Uniform)
// ─────────────────────────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

// ─────────────────────────────────────────────────────────────────
// メッシュビルダ
// ─────────────────────────────────────────────────────────────────
pub fn build_chunk_mesh(device: &wgpu::Device, world: &World, cx: i32, cz: i32) -> Option<RenderMesh> {
    let mut vertices: Vec<Vertex> = Vec::new();
    let Some(chunk) = world.chunks.get(&(cx, cz)) else { return None; };

    let wx0 = cx * CHUNK_W as i32;
    let wz0 = cz * CHUNK_D as i32;

    for y in 0..CHUNK_H {
        for z in 0..CHUNK_D {
            for x in 0..CHUNK_W {
                let block = chunk.get(x, y, z);
                if block == Block::Air { continue; }

                let wx = wx0 + x as i32;
                let wy = y as i32;
                let wz = wz0 + z as i32;

                // 面チェック: 0=+Y, 1=-Y, 2=-Z, 3=+Z, 4=-X, 5=+X
                let checks = [
                    (wx, wy + 1, wz, 0),
                    (wx, wy - 1, wz, 1),
                    (wx, wy, wz - 1, 2),
                    (wx, wy, wz + 1, 3),
                    (wx - 1, wy, wz, 4),
                    (wx + 1, wy, wz, 5),
                ];

                for (nx, ny, nz, face) in checks {
                    let nb = world.get_block(nx, ny, nz);
                    if nb.is_transparent() && !(block == Block::Water && nb == Block::Water) {
                        add_face(&mut vertices, wx, wy, wz, block, face);
                    }
                }
            }
        }
    }

    if vertices.is_empty() { return None; }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("Chunk Mesh ({},{})", cx, cz)),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    Some(RenderMesh {
        vertex_buffer,
        index_buffer: None,
        num_vertices: vertices.len() as u32,
        num_indices: 0,
    })
}

fn add_face(verts: &mut Vec<Vertex>, x: i32, y: i32, z: i32, b: Block, face: usize) {
    let x = x as f32; let y = y as f32; let z = z as f32;
    let mut c = b.color(face as u8);
    let l = Block::face_light(face);
    c[0] *= l; c[1] *= l; c[2] *= l;

    let (v0, v1, v2, v3, n) = match face {
        0 => ([x,y+1.,z+1.], [x+1.,y+1.,z+1.], [x+1.,y+1.,z], [x,y+1.,z], [0.,1.,0.]),
        1 => ([x,y,z], [x+1.,y,z], [x+1.,y,z+1.], [x,y,z+1.], [0.,-1.,0.]),
        2 => ([x+1.,y,z], [x,y,z], [x,y+1.,z], [x+1.,y+1.,z], [0.,0.,-1.]),
        3 => ([x,y,z+1.], [x+1.,y,z+1.], [x+1.,y+1.,z+1.], [x,y+1.,z+1.], [0.,0.,1.]),
        4 => ([x,y,z], [x,y,z+1.], [x,y+1.,z+1.], [x,y+1.,z], [-1.,0.,0.]),
        5 => ([x+1.,y,z+1.], [x+1.,y,z], [x+1.,y+1.,z], [x+1.,y+1.,z+1.], [1.,0.,0.]),
        _ => unreachable!(),
    };

    verts.push(Vertex { position: v0, color: c, normal: n });
    verts.push(Vertex { position: v1, color: c, normal: n });
    verts.push(Vertex { position: v2, color: c, normal: n });
    verts.push(Vertex { position: v2, color: c, normal: n });
    verts.push(Vertex { position: v3, color: c, normal: n });
    verts.push(Vertex { position: v0, color: c, normal: n });
}

// ─────────────────────────────────────────────────────────────────
// Wgpu レンダラコア
// ─────────────────────────────────────────────────────────────────
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device:  wgpu::Device,
    queue:   wgpu::Queue,
    config:  wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    depth_texture: wgpu::TextureView,
    camera_buf: wgpu::Buffer,
    camera_bg:  wgpu::BindGroup,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        
        let config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
        surface.configure(&device, &config);

        // カメラ
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let depth_texture = Self::create_depth_texture(&device, &config);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&camera_bgl)],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            surface, device, queue, config, pipeline, depth_texture, camera_buf, camera_bg
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
        }
    }

    fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
        let size = wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn device(&self) -> &wgpu::Device { &self.device }

    pub fn draw(&self, view_proj: [[f32; 4]; 4], meshes: &[&RenderMesh]) {
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&CameraUniform { view_proj }));

        // 戻り値の型が CurrentSurfaceTexture で、内部に texture を持つ
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.5, g: 0.7, b: 1.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.camera_bg, &[]);

            for mesh in meshes {
                rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                if let Some(ib) = &mesh.index_buffer {
                    rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                } else {
                    rpass.draw(0..mesh.num_vertices, 0..1);
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}
