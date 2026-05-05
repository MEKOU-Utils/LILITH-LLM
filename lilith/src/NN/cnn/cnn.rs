//! cnn.rs — CNN on GPU (MNIST物体検出 → 乖離率計算)
//!
//! ## アーキテクチャ
//!   Input(28×28) → Conv2D(5×5,8f) → LeakyReLU → MaxPool(2×2)
//!                → FC(8×14×14 → 128) → FC(128 → 10) → Softmax
//!
//! ## 重み可視化
//!   WeightVisualizer が Conv フィルタを ECS テクスチャとして描画する。
//!   「記号として見える」フィルタ = Lilith の知覚シンボル。
//!
//! ## 他AI乖離
//!   同じ入力を他AIの重みで推論 → KL発散 → NeuralCore.divergence に反映

use std::io::{self, Read};
use std::path::Path;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ─────────────────────────────────────────────────────────────────
// MNIST データローダー
// ─────────────────────────────────────────────────────────────────

/// MNIST IDX フォーマットから画像を読む
pub struct MnistDataset {
    /// 画像データ (N × 28 × 28), 値域 [0.0, 1.0]
    pub images: Vec<Vec<f32>>,
    /// ラベル (0-9)
    pub labels: Vec<u8>,
    pub count:  usize,
}

impl MnistDataset {
    /// `images_path`: train-images-idx3-ubyte
    /// `labels_path`: train-labels-idx1-ubyte
    pub fn load(images_path: &Path, labels_path: &Path) -> anyhow::Result<Self> {
        let img_bytes   = std::fs::read(images_path)?;
        let label_bytes = std::fs::read(labels_path)?;

        // IDX3 ヘッダ: magic(4) count(4) rows(4) cols(4)
        let count = u32::from_be_bytes(img_bytes[4..8].try_into()?) as usize;
        let rows  = u32::from_be_bytes(img_bytes[8..12].try_into()?) as usize;
        let cols  = u32::from_be_bytes(img_bytes[12..16].try_into()?) as usize;
        let px_per_img = rows * cols;

        let mut images = Vec::with_capacity(count);
        for i in 0..count {
            let start = 16 + i * px_per_img;
            let end   = start + px_per_img;
            let img: Vec<f32> = img_bytes[start..end]
                .iter()
                .map(|&b| b as f32 / 255.0)
                .collect();
            images.push(img);
        }

        // IDX1 ヘッダ: magic(4) count(4)
        let labels: Vec<u8> = label_bytes[8..8 + count].to_vec();

        Ok(Self { images, labels, count })
    }

    pub fn sample(&self, idx: usize) -> (&[f32], u8) {
        (&self.images[idx], self.labels[idx])
    }
}

// ─────────────────────────────────────────────────────────────────
// GPU Params (Uniform)
// ─────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CnnParams {
    pub input_w:      u32,  // 28
    pub input_h:      u32,  // 28
    pub num_class:    u32,  // 10
    pub conv_filters: u32,  // 8
}

// ─────────────────────────────────────────────────────────────────
// CnnGpu — GPU上のCNNパイプライン
// ─────────────────────────────────────────────────────────────────

pub struct CnnGpu {
    // バッファ
    pub input_buf:       wgpu::Buffer,
    pub output_buf:      wgpu::Buffer,
    pub conv_weight_buf: wgpu::Buffer,
    pub fc_weight_buf:   wgpu::Buffer,
    pub params_buf:      wgpu::Buffer,

    // パイプライン
    pub conv_pipeline:    wgpu::ComputePipeline,
    pub softmax_pipeline: wgpu::ComputePipeline,
    pub bind_group:       wgpu::BindGroup,

    // 重み (CPU側コピー, 可視化用)
    pub conv_weights:     Vec<f32>,  // 8 filters × 5×5
    pub fc_weights:       Vec<f32>,

    pub params: CnnParams,
}

impl CnnGpu {
    /// WGSL ソースを AssetRegistry 経由で渡す
    pub fn new(device: &wgpu::Device, shader_src: &str) -> Self {
        let params = CnnParams {
            input_w:      28,
            input_h:      28,
            num_class:    10,
            conv_filters: 8,
        };

        let px  = 28 * 28;
        let out = 10 + 8 * 28 * 28; // softmax 10 + feature map scratch
        let cw  = 8 * 5 * 5;        // 8 filters × 5×5
        let fcw = 128 * 10;         // FC: 128→10

        // ランダム初期化 (Xavier 近似)
        let conv_weights: Vec<f32> = (0..cw).map(|i| {
            let x = (i as f32 * 1.6180339) % 1.0;
            (x - 0.5) * 2.0 * (2.0 / (25.0f32)).sqrt()
        }).collect();
        let fc_weights: Vec<f32> = (0..fcw).map(|i| {
            let x = (i as f32 * 2.7182818) % 1.0;
            (x - 0.5) * 2.0 * (2.0 / 128.0f32).sqrt()
        }).collect();

        let mk_buf = |data: &[f32], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    None,
                contents: bytemuck::cast_slice(data),
                usage,
            })
        };
        let mk_zeros = |n: usize, usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    None,
                contents: bytemuck::cast_slice(&vec![0.0f32; n]),
                usage,
            })
        };

        let s = wgpu::BufferUsages::STORAGE;
        let su = s | wgpu::BufferUsages::COPY_DST;
        let u = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;

        let input_buf       = mk_zeros(px, su);
        let output_buf      = mk_zeros(out, s | wgpu::BufferUsages::COPY_SRC);
        let conv_weight_buf = mk_buf(&conv_weights, s);
        let fc_weight_buf   = mk_buf(&fc_weights, s);
        let params_buf      = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cnn_params"),
            contents: bytemuck::bytes_of(&params),
            usage:    u,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("cnn_bgl"),
            entries: &[
                Self::storage_entry(0, false),
                Self::storage_entry(1, true),
                Self::storage_entry(2, false),
                Self::storage_entry(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("cnn_bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: conv_weight_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: fc_weight_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("cnn_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("cnn_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let conv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("conv_pass"),
            layout:      Some(&pl_layout),
            module:      &shader,
            entry_point: Some("conv_pass"),
            compilation_options: Default::default(),
            cache:       None,
        });
        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("softmax_pass"),
            layout:      Some(&pl_layout),
            module:      &shader,
            entry_point: Some("softmax_pass"),
            compilation_options: Default::default(),
            cache:       None,
        });

        Self {
            input_buf, output_buf, conv_weight_buf, fc_weight_buf, params_buf,
            conv_pipeline, softmax_pipeline, bind_group,
            conv_weights, fc_weights, params,
        }
    }

    fn storage_entry(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if read_write {
                    wgpu::BufferBindingType::Storage { read_only: false }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: true }
                },
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        }
    }

    /// 入力画像をGPUにアップロードして推論キューに積む
    pub fn infer(&self, image: &[f32], encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        queue.write_buffer(&self.input_buf, 0, bytemuck::cast_slice(image));

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:              Some("cnn_pass"),
            timestamp_writes:   None,
        });
        cpass.set_pipeline(&self.conv_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        // 28×28 をworkgroup(8,8,8)で分割
        cpass.dispatch_workgroups(4, 4, self.params.conv_filters);

        cpass.set_pipeline(&self.softmax_pipeline);
        cpass.dispatch_workgroups(1, 1, 1);
    }
}

// ─────────────────────────────────────────────────────────────────
// WeightVisualizer — Conv フィルタを RGBA テクスチャ化
// ─────────────────────────────────────────────────────────────────

/// 8個のConvフィルタ(5×5)を並べた可視化テクスチャ
/// → ECS の Mesh テクスチャとして扱う (40×8 px atlas)
pub struct WeightVisualizer {
    pub texture: wgpu::Texture,
    pub view:    wgpu::TextureView,
    pub width:   u32,  // 5*8 = 40
    pub height:  u32,  // 5
}

impl WeightVisualizer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, conv_weights: &[f32]) -> Self {
        let filters = 8usize;
        let ks      = 5usize;  // kernel size
        let w = (ks * filters) as u32;
        let h = ks as u32;

        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for f in 0..filters {
            for ky in 0..ks {
                for kx in 0..ks {
                    let w_val = conv_weights[f * ks * ks + ky * ks + kx];
                    // -1〜1 → 0〜255
                    let v = ((w_val.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0) as u8;
                    let px = f * ks + kx;
                    let py = ky;
                    let idx = (py * w as usize + px) * 4;
                    // 重みの正 = 緑、負 = 赤 で記号的に表現
                    if w_val > 0.0 {
                        rgba[idx]     = 0;
                        rgba[idx + 1] = v;
                        rgba[idx + 2] = 0;
                    } else {
                        rgba[idx]     = v;
                        rgba[idx + 1] = 0;
                        rgba[idx + 2] = 0;
                    }
                    rgba[idx + 3] = 255;
                }
            }
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("weight_vis"),
            size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &rgba,
            wgpu::TexelCopyBufferLayout { offset: 0,
                bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view, width: w, height: h }
    }

    /// 重みが更新されたらテクスチャを再upload
    pub fn refresh(&self, queue: &wgpu::Queue, conv_weights: &[f32]) {
        let filters = 8usize;
        let ks      = 5usize;
        let w       = (ks * filters) as u32;
        let h       = ks as u32;
        let mut rgba = vec![0u8; (w * h * 4) as usize];

        for f in 0..filters {
            for ky in 0..ks {
                for kx in 0..ks {
                    let w_val = conv_weights[f * ks * ks + ky * ks + kx];
                    let v = ((w_val.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0) as u8;
                    let px  = f * ks + kx;
                    let idx = (ky * w as usize + px) * 4;
                    if w_val > 0.0 { rgba[idx] = 0; rgba[idx+1] = v; rgba[idx+2] = 0; }
                    else           { rgba[idx] = v; rgba[idx+1] = 0; rgba[idx+2] = 0; }
                    rgba[idx+3] = 255;
                }
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &rgba,
            wgpu::TexelCopyBufferLayout { offset: 0,
                bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// KL発散 — CPU側乖離率計算
// ─────────────────────────────────────────────────────────────────

/// KL(P || Q) — Lilith の出力 P と他AI の出力 Q の乖離
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 1e-9 && qi > 1e-9 {
            pi * (pi / qi).ln()
        } else {
            0.0
        }
    }).sum()
}
