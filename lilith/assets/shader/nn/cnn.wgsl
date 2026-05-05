// cnn.wgsl — CNN Forward Pass (GPU Compute)
//
// パイプライン:
//   Conv2D → LeakyReLU → MaxPool → Flatten → FC → Softmax
//
// binding構成:
//   group(0) binding(0): input  (28×28 グレースケール, f32)
//   group(0) binding(1): output (10クラス確率, f32)
//   group(0) binding(2): conv_weights (5×5×1×8 フィルタ, f32)
//   group(0) binding(3): fc_weights   (FC層重み, f32)
//   group(0) binding(4): params       (Params uniform)

struct Params {
    input_w:  u32,   // 28
    input_h:  u32,   // 28
    num_class: u32,  // 10
    conv_filters: u32, // 8
};

@group(0) @binding(0) var<storage, read>       input_img:    array<f32>;
@group(0) @binding(1) var<storage, read_write> output_class: array<f32>;
@group(0) @binding(2) var<storage, read>       conv_weights: array<f32>;
@group(0) @binding(3) var<storage, read>       fc_weights:   array<f32>;
@group(0) @binding(4) var<uniform>             params:       Params;

// ── LeakyReLU ───────────────────────────────────────────────────
fn leaky_relu(x: f32) -> f32 {
    return select(x * 0.01, x, x >= 0.0);
}

// ── Conv2D 5×5, stride=1, same padding ──────────────────────────
// workgroup: 各スレッドが出力マップの1ピクセル×1フィルタを担当
@compute @workgroup_size(8, 8, 1)
fn conv_pass(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let filter_id = gid.z;
    let ox = gid.x;
    let oy = gid.y;
    let W  = params.input_w;
    let H  = params.input_h;
    let KS = 5u;  // kernel size

    if ox >= W || oy >= H { return; }

    var acc: f32 = 0.0;
    for (var ky = 0u; ky < KS; ky++) {
        for (var kx = 0u; kx < KS; kx++) {
            let ix = i32(ox) + i32(kx) - 2;
            let iy = i32(oy) + i32(ky) - 2;
            var px: f32 = 0.0;
            if ix >= 0 && iy >= 0 && u32(ix) < W && u32(iy) < H {
                px = input_img[u32(iy) * W + u32(ix)];
            }
            let w_idx = filter_id * KS * KS + ky * KS + kx;
            acc += px * conv_weights[w_idx];
        }
    }

    // LeakyReLU して feature map に書く (output を一時バッファとして使用)
    let out_idx = filter_id * W * H + oy * W + ox;
    output_class[out_idx] = leaky_relu(acc);
}

// ── Softmax (10クラス、1スレッドで実行) ─────────────────────────
@compute @workgroup_size(1)
fn softmax_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.num_class;
    var max_val: f32 = output_class[0];
    for (var i = 1u; i < N; i++) {
        max_val = max(max_val, output_class[i]);
    }
    var sum: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let e = exp(output_class[i] - max_val);
        output_class[i] = e;
        sum += e;
    }
    for (var i = 0u; i < N; i++) {
        output_class[i] = output_class[i] / sum;
    }
}
