struct CameraUniform {
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    out.normal = model.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 簡易ライティング (環境光 + 指向性)
    let ambient = 0.3;
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.4));
    let diff = max(dot(in.normal, light_dir), 0.0);
    
    // 面の明るさは頂点カラーに焼き込み済みなので、基本カラーをそのまま出力
    // 追加のディレクショナルライトを少しだけ足す
    let lighting = ambient + diff * 0.2;
    
    let final_color = vec3<f32>(in.color.rgb * lighting);
    return vec4<f32>(final_color, in.color.a);
}
