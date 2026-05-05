// llm.wgsl — Token Embedding + Simple Attention (GPU Compute)
//
// 簡易Transformer: Embedding → QKV Attention → FC → LayerNorm
//
// group(0) binding(0): token_ids (u32[seq_len])
// group(0) binding(1): embeddings (f32[vocab × dim])
// group(0) binding(2): attn_weights (f32[dim × dim × 3], QKV)
// group(0) binding(3): output (f32[seq_len × dim])
// group(0) binding(4): params (LlmParams)

struct LlmParams {
    seq_len:   u32,
    vocab_size: u32,
    embed_dim:  u32,
    num_heads:  u32,
};

@group(0) @binding(0) var<storage, read>       token_ids:   array<u32>;
@group(0) @binding(1) var<storage, read>       embeddings:  array<f32>;
@group(0) @binding(2) var<storage, read>       attn_w:      array<f32>;
@group(0) @binding(3) var<storage, read_write> out_hidden:  array<f32>;
@group(0) @binding(4) var<uniform>             params:      LlmParams;

// ── Embedding lookup: 各スレッド = 1トークン ────────────────────
@compute @workgroup_size(64)
fn embed_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tok_idx = gid.x;
    if tok_idx >= params.seq_len { return; }

    let token_id = token_ids[tok_idx];
    let D        = params.embed_dim;

    for (var d = 0u; d < D; d++) {
        let emb_val = embeddings[token_id * D + d];
        out_hidden[tok_idx * D + d] = emb_val;
    }
}

// ── Scaled Dot-Product Attention (simplified, single-head) ──────
@compute @workgroup_size(8, 8)
fn attention_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;  // query token index
    let ki = gid.y;  // key token index
    let S  = params.seq_len;
    let D  = params.embed_dim;
    if qi >= S || ki >= S { return; }

    // Q・K dot product
    var dot: f32 = 0.0;
    for (var d = 0u; d < D; d++) {
        let q = out_hidden[qi * D + d];
        let k = out_hidden[ki * D + d];
        dot += q * k;
    }
    dot /= sqrt(f32(D));

    // causal mask: qi < ki は -inf
    if ki > qi { dot = -1e9; }

    // attn score を out_hidden の後半に一時書き込み (簡易)
    out_hidden[S * D + qi * S + ki] = dot;
}
