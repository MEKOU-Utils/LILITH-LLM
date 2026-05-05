// ── 構造体 ──────────────────────────────────────────────────────
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) color:    vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv:    vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) mode:  f32,
}

// ── Uniform: group(0) ────────────────────────────────────────────
struct SceneUniforms {
    screen_size: vec2<f32>,
    time:        f32,
    _pad:        f32,
}
@group(0) @binding(0) var<uniform> scene: SceneUniforms;

// ── Font texture: group(1) ───────────────────────────────────────
@group(1) @binding(0) var font_tex: texture_2d<f32>;
@group(1) @binding(1) var font_smp: sampler;

// ── Vertex shader ────────────────────────────────────────────────
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // position.z = draw mode (0=solid, 1=font, 2=progress)
    // xy をスクリーン px → NDC 変換
    let ndc_x =  (in.position.x / scene.screen_size.x) * 2.0 - 1.0;
    let ndc_y = -((in.position.y / scene.screen_size.y) * 2.0 - 1.0);

    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv    = in.uv;
    out.color = in.color;
    out.mode  = in.position.z;

    return out;
}

// ── Fragment shader ──────────────────────────────────────────────
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let mode = i32(in.mode + 0.5);

    // mode 1: フォントアトラスサンプリング
    if mode == 1 {
        let alpha = textureSample(font_tex, font_smp, in.uv).a;
        if alpha < 0.05 {
            discard;
        }
        return vec4<f32>(in.color.rgb, in.color.a * alpha);
    }

    // mode 2: プログレスバー (グロウアニメーション)
    if mode == 2 {
        let glow = sin(in.uv.x * 3.14159 + scene.time * 2.0) * 0.15 + 0.85;
        return vec4<f32>(in.color.rgb * glow, in.color.a);
    }

    // mode 0: ソリッドカラー
    return in.color;
}
