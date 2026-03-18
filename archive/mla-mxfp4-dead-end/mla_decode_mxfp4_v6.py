"""
MLA decode v6 — MXFP4 K with FP8 V (hybrid approach).

The insight: dot_scaled gives us hardware-accelerated MXFP4 QK^T,
but V dequant from fp4 is too expensive in software.

Instead: store K as MXFP4 (for dot_scaled QK^T) and V as fp8 (for cheap cast + tl.dot P@V).
For MLA's shared KV buffer, this means:
- KV_fp4: [total_kv, qk_dim//2] uint8 — used for QK^T via dot_scaled
- KV_fp8: [total_kv, v_dim] float8 — used for P@V (just cast to bf16)
- KV_fp4_scale: [total_kv, qk_dim//32] uint8 — e8m0 scales for K

This halves K bandwidth and V stays at fp8 (same as baseline).
Net savings: K goes from fp8 to fp4 = 50% K bandwidth reduction.
K is 576 dims, V is 512 dims. K bandwidth: 576 bytes/token (fp8) -> 288+18 = 306 bytes.
V bandwidth: 512 bytes/token (fp8, unchanged).
Total: 1088 -> 818 bytes/token = 25% reduction.

Not as good as full MXFP4 but avoids the dequant nightmare.
"""

import torch
import triton
import triton.language as tl

SCALE_GROUP_SIZE = tl.constexpr(32)


@triton.jit
def _mxfp4_quant_inline(x, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_K // SCALE_GROUP_SIZE

    x = x.to(tl.float32)
    x = x.reshape(BLOCK_M, NUM_QUANT_BLOCKS, SCALE_GROUP_SIZE)
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
    quant_scale = tl.exp2(-scale_e8m0_unbiased)
    qx = x * quant_scale
    qx = qx.to(tl.uint32, bitcast=True)
    s = qx & 0x80000000
    qx = qx ^ s
    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= 6
    denormal_mask = (not saturate_mask) & (qx_fp32 < 1)
    normal_mask = not (saturate_mask | denormal_mask)
    denorm_exp: tl.constexpr = (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)
    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(e2m1_value, [BLOCK_M, NUM_QUANT_BLOCKS, SCALE_GROUP_SIZE // 2, 2])
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_M, BLOCK_K // 2)
    return x_fp4, bs_e8m0.reshape(BLOCK_M, NUM_QUANT_BLOCKS)


@triton.jit
def _mla_stage1_hybrid(
    Q,              # [total_q, NHEADS, QK_DIM_PAD] bf16
    K_fp4,          # [total_kv, QK_DIM_PAD // 2] uint8
    K_scale,        # [total_kv, NUM_QK_GROUPS_PAD] uint8
    V_fp8,          # [total_kv, V_DIM] float8_e4m3fnuz
    V_scale,        # [1] float32 (per-tensor scale for V)
    Partial_O,      # [total_q, NUM_SPLITS, NHEADS, V_DIM] float32
    Partial_LSE,    # [total_q, NUM_SPLITS, NHEADS] float32
    qo_indptr, kv_indptr,
    sm_scale,
    stride_q_tok: tl.int64, stride_q_h: tl.int64,
    stride_k_tok: tl.int64, stride_ks_tok: tl.int64,
    stride_v_tok: tl.int64,
    stride_po_tok: tl.int64, stride_po_split: tl.int64, stride_po_h: tl.int64,
    stride_plse_tok: tl.int64, stride_plse_split: tl.int64,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    split_idx = tl.program_id(1)

    q_start = tl.load(qo_indptr + batch_idx)
    q_end = tl.load(qo_indptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)
    seqlen_kv = kv_end - kv_start

    kv_per_split = tl.cdiv(seqlen_kv, NUM_SPLITS)
    split_start = kv_start + split_idx * kv_per_split
    split_end = tl.minimum(kv_start + (split_idx + 1) * kv_per_split, kv_end)
    split_len = tl.maximum(split_end - split_start, 0)

    QK_HALF: tl.constexpr = QK_DIM // 2
    NUM_QK_GROUPS: tl.constexpr = QK_DIM // SCALE_GROUP_SIZE

    offs_h = tl.arange(0, NHEADS)
    offs_qk = tl.arange(0, QK_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    offs_v = tl.arange(0, V_DIM)

    v_scale_val = tl.load(V_scale)

    for q_pos in range(q_start, q_end):
        q_ptrs = Q + q_pos * stride_q_tok + offs_h[:, None] * stride_q_h + offs_qk[None, :]
        q_bf16 = tl.load(q_ptrs)
        # Use Q as bf16 in dot_scaled (no quantization loss)
        NUM_Q_SCALE_GROUPS: tl.constexpr = QK_DIM // SCALE_GROUP_SIZE
        q_scale = tl.full([NHEADS, NUM_Q_SCALE_GROUPS], 127, dtype=tl.uint8)  # scale=1.0

        m_i = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([NHEADS], dtype=tl.float32)
        acc = tl.zeros([NHEADS, V_DIM], dtype=tl.float32)

        num_blocks = tl.cdiv(split_len, BLOCK_N)
        for block_idx in range(num_blocks):
            kv_off = split_start + block_idx * BLOCK_N
            valid_n = tl.minimum(BLOCK_N, split_len - block_idx * BLOCK_N)
            n_mask = offs_n < valid_n

            # K: [QK_HALF, BLOCK_N] transposed load
            k_ptrs = (K_fp4 + (kv_off + offs_n[None, :]) * stride_k_tok
                      + tl.arange(0, QK_HALF)[:, None])
            k_u8 = tl.load(k_ptrs, mask=n_mask[None, :], other=0)

            ks_ptrs = (K_scale + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_QK_GROUPS)[None, :])
            k_sc = tl.load(ks_ptrs, mask=n_mask[:, None], other=0)

            # QK^T: Q as bf16 (no quant loss), K as MXFP4
            qk = tl.dot_scaled(q_bf16, q_scale, "bf16", k_u8, k_sc, "e2m1")
            qk *= sm_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # Online softmax
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            m_i = m_new

            # V: load as fp8, cast to bf16 (cheap!)
            v_ptrs = (V_fp8 + (kv_off + offs_n[:, None]) * stride_v_tok
                      + offs_v[None, :])
            v_fp8_block = tl.load(v_ptrs, mask=n_mask[:, None])
            # Dequant: fp8 * scale -> bf16
            v_bf16 = (v_fp8_block.to(tl.float32) * v_scale_val).to(tl.bfloat16)

            p_bf16 = p.to(tl.bfloat16)
            acc += tl.dot(p_bf16, v_bf16, out_dtype=tl.float32)

        lse = m_i + tl.log2(l_i + 1e-10)
        inv_l = 1.0 / (l_i + 1e-10)
        acc = acc * inv_l[:, None]

        po_base = Partial_O + q_pos * stride_po_tok + split_idx * stride_po_split
        tl.store(po_base + offs_h[:, None] * stride_po_h + offs_v[None, :], acc)
        plse_ptrs = Partial_LSE + q_pos * stride_plse_tok + split_idx * stride_plse_split + offs_h
        tl.store(plse_ptrs, lse)


@triton.jit
def _mla_reduce(
    Partial_O, Partial_LSE, Out,
    stride_po_tok: tl.int64, stride_po_split: tl.int64, stride_po_h: tl.int64,
    stride_plse_tok: tl.int64, stride_plse_split: tl.int64,
    stride_o_tok: tl.int64, stride_o_h: tl.int64,
    NHEADS: tl.constexpr, V_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    q_idx = tl.program_id(0)
    offs_h = tl.arange(0, NHEADS)
    offs_v = tl.arange(0, V_DIM)

    global_m = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
    for s in range(NUM_SPLITS):
        lse_s = tl.load(Partial_LSE + q_idx * stride_plse_tok + s * stride_plse_split + offs_h)
        global_m = tl.maximum(global_m, lse_s)

    acc = tl.zeros([NHEADS, V_DIM], dtype=tl.float32)
    total_w = tl.zeros([NHEADS], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        lse_s = tl.load(Partial_LSE + q_idx * stride_plse_tok + s * stride_plse_split + offs_h)
        w = tl.exp2(lse_s - global_m)
        total_w += w
        po_base = Partial_O + q_idx * stride_po_tok + s * stride_po_split
        po = tl.load(po_base + offs_h[:, None] * stride_po_h + offs_v[None, :])
        acc += w[:, None] * po

    acc *= (1.0 / (total_w + 1e-10))[:, None]
    o_base = Out + q_idx * stride_o_tok
    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_v[None, :], acc.to(Out.type.element_ty))


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def mla_decode_hybrid(
    q, k_fp4, k_scale, v_fp8, v_scale,
    qo_indptr, kv_indptr,
    nheads, qk_dim, v_dim, sm_scale,
    num_splits=16, BLOCK_N=64,
):
    total_q = q.shape[0]
    batch_size = qo_indptr.shape[0] - 1

    qk_dim_pad = _next_pow2(qk_dim)
    qk_half = qk_dim // 2
    qk_half_pad = qk_dim_pad // 2
    nqg = qk_dim // 32
    nqg_pad = _next_pow2(nqg)

    # Pad Q
    if qk_dim_pad != qk_dim:
        q_pad = torch.zeros(total_q, nheads, qk_dim_pad, dtype=q.dtype, device=q.device)
        q_pad[:, :, :qk_dim] = q
    else:
        q_pad = q
    # Pad K fp4 data
    if qk_half_pad != qk_half:
        k_pad = torch.zeros(k_fp4.shape[0], qk_half_pad, dtype=k_fp4.dtype, device=k_fp4.device)
        k_pad[:, :qk_half] = k_fp4
    else:
        k_pad = k_fp4
    # Pad K scale
    if nqg_pad != nqg:
        ks_pad = torch.zeros(k_scale.shape[0], nqg_pad, dtype=k_scale.dtype, device=k_scale.device)
        ks_pad[:, :nqg] = k_scale
    else:
        ks_pad = k_scale

    partial_o = torch.empty(total_q, num_splits, nheads, v_dim, dtype=torch.float32, device=q.device)
    partial_lse = torch.empty(total_q, num_splits, nheads, dtype=torch.float32, device=q.device)

    _mla_stage1_hybrid[(batch_size, num_splits)](
        q_pad, k_pad, ks_pad, v_fp8, v_scale,
        partial_o, partial_lse,
        qo_indptr, kv_indptr, sm_scale,
        q_pad.stride(0), q_pad.stride(1),
        k_pad.stride(0), ks_pad.stride(0),
        v_fp8.stride(0),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        NHEADS=nheads, QK_DIM=qk_dim_pad, V_DIM=v_dim,
        NUM_SPLITS=num_splits, BLOCK_N=BLOCK_N,
    )

    out = torch.empty(total_q, nheads, v_dim, dtype=torch.bfloat16, device=q.device)
    _mla_reduce[(total_q,)](
        partial_o, partial_lse, out,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        out.stride(0), out.stride(1),
        NHEADS=nheads, V_DIM=v_dim, NUM_SPLITS=num_splits,
    )
    return out


# ── Helpers ──

from aiter import dtypes as aiter_dtypes
FP8_DTYPE = aiter_dtypes.fp8


def quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8 = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8, scale.to(torch.float32).reshape(1)


def _quantize_mxfp4_torch(tensor):
    FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device)
    N, D = tensor.shape
    t = tensor.float().reshape(N, D // 32, 32)
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    log2_amax = torch.floor(torch.log2(amax)) + 2
    log2_amax = log2_amax.clamp(-127, 127)
    scale_e8m0 = (log2_amax.long() + 127).clamp(0, 254).to(torch.uint8)
    scale_f32 = torch.pow(2.0, log2_amax)
    t_scaled = t / scale_f32
    signs = (t_scaled < 0).to(torch.uint8)
    t_abs = t_scaled.abs()
    diffs = (t_abs.unsqueeze(-1) - FP4_VALUES).abs()
    mag_idx = diffs.argmin(dim=-1).to(torch.uint8)
    fp4_vals = (signs << 3) | mag_idx
    fp4_vals = fp4_vals.reshape(N, D // 2, 2)
    fp4x2 = fp4_vals[:, :, 0] | (fp4_vals[:, :, 1] << 4)
    return fp4x2.to(torch.uint8), scale_e8m0.reshape(N, D // 32)


def _dequant_mxfp4_torch(data_u8, scale_u8, dim):
    FP4_LUT_T = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=data_u8.device)
    lo = data_u8 & 0x0F
    hi = (data_u8 >> 4) & 0x0F
    lo_sign = ((lo >> 3) & 1).float() * (-2.0) + 1.0
    lo_mag = (lo & 0x07).long()
    hi_sign = ((hi >> 3) & 1).float() * (-2.0) + 1.0
    hi_mag = (hi & 0x07).long()
    lo_val = lo_sign * FP4_LUT_T[lo_mag]
    hi_val = hi_sign * FP4_LUT_T[hi_mag]
    scale_f32 = torch.pow(2.0, scale_u8.float() - 127.0)
    n_groups = scale_f32.shape[1]
    scale_expanded = scale_f32.unsqueeze(2).expand(-1, n_groups, 16).reshape(-1, dim // 2)
    N = data_u8.shape[0]
    result = torch.empty(N, dim, device=data_u8.device, dtype=torch.float32)
    result[:, 0::2] = lo_val * lo_sign * scale_expanded
    result[:, 1::2] = hi_val * hi_sign * scale_expanded
    return result


def mla_decode_reference(q, k_fp4, k_scale, v_fp8, v_scale_val,
                          qo_indptr, kv_indptr, nheads, qk_dim, v_dim, sm_scale):
    batch_size = qo_indptr.shape[0] - 1
    total_kv = k_fp4.shape[0]
    K_f32 = _dequant_mxfp4_torch(k_fp4, k_scale[:total_kv], qk_dim)
    V_f32 = v_fp8.float() * v_scale_val.item()
    outputs = []
    for b in range(batch_size):
        qs, qe = qo_indptr[b].item(), qo_indptr[b + 1].item()
        ks, ke = kv_indptr[b].item(), kv_indptr[b + 1].item()
        q_b = q[qs:qe].float()
        k_b, v_b = K_f32[ks:ke], V_f32[ks:ke]
        for qi in range(q_b.shape[0]):
            scores = torch.matmul(q_b[qi], k_b.T) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            outputs.append(torch.matmul(attn, v_b))
    return torch.stack(outputs).to(torch.bfloat16)


if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = "cuda"

    NHEADS = 16
    QK_DIM = 576
    V_DIM = 512

    # Also compare against aiter fp8 baseline
    from aiter.mla import mla_decode_fwd
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

    for BATCH, KV_SL in [(1, 64), (1, 256), (4, 256), (4, 1024), (61, 4096)]:
        print(f"\n=== batch={BATCH}, kv_seqlen={KV_SL} ===")
        tq = BATCH
        tkv = BATCH * KV_SL
        q = torch.randn(tq, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
        qo_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
        kv_bf16 = torch.randn(tkv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
        kv_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device) * KV_SL

        # Prepare K as MXFP4
        k_fp4, k_scale = _quantize_mxfp4_torch(kv_bf16)

        # Prepare V as FP8 (first V_DIM dims)
        v_bf16 = kv_bf16[:, :V_DIM]
        v_fp8, v_scale = quantize_fp8(v_bf16)

        sm_scale = 1.0 / (QK_DIM ** 0.5)

        # Reference
        out_ref = mla_decode_reference(q, k_fp4, k_scale, v_fp8, v_scale,
                                        qo_indptr, kv_indptr, NHEADS, QK_DIM, V_DIM, sm_scale)

        # Hybrid kernel
        out_hyb = mla_decode_hybrid(q, k_fp4, k_scale, v_fp8, v_scale,
                                     qo_indptr, kv_indptr, NHEADS, QK_DIM, V_DIM, sm_scale,
                                     num_splits=16, BLOCK_N=64)

        cos_sim = torch.nn.functional.cosine_similarity(
            out_ref.float().reshape(-1), out_hyb.float().reshape(-1), dim=0).item()
        max_diff = (out_ref.float() - out_hyb.float()).abs().max().item()
        print(f"  cos_sim={cos_sim:.6f}  max_diff={max_diff:.6f}")
        if cos_sim < 0.98:
            print("  FAIL (cos_sim < 0.98)"); sys.exit(1)
        print("  PASS")

    # Benchmark
    print("\n=== Benchmark ===")
    for BATCH, KV_SL, n_splits in [(1, 4096, 16), (4, 4096, 16), (32, 4096, 16),
                                     (61, 4096, 16), (61, 4096, 32), (128, 4096, 16)]:
        tq = BATCH
        tkv = BATCH * KV_SL
        q = torch.randn(tq, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
        qo = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
        kv_bf16 = torch.randn(tkv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
        kv_ind = torch.arange(BATCH + 1, dtype=torch.int32, device=device) * KV_SL
        k_fp4, k_sc = _quantize_mxfp4_torch(kv_bf16)
        v_fp8, v_sc = quantize_fp8(kv_bf16[:, :V_DIM])
        sm = 1.0 / (QK_DIM ** 0.5)

        # Pre-pad
        qk_dim_pad = _next_pow2(QK_DIM)
        qk_half_pad = qk_dim_pad // 2
        nqg_pad = _next_pow2(QK_DIM // 32)
        q_pad = torch.zeros(tq, NHEADS, qk_dim_pad, dtype=q.dtype, device=device)
        q_pad[:, :, :QK_DIM] = q
        k_pad = torch.zeros(tkv, qk_half_pad, dtype=k_fp4.dtype, device=device)
        k_pad[:, :QK_DIM // 2] = k_fp4
        ks_pad = torch.zeros(tkv, nqg_pad, dtype=k_sc.dtype, device=device)
        ks_pad[:, :QK_DIM // 32] = k_sc

        # Warmup
        for _ in range(5):
            partial_o = torch.empty(tq, n_splits, NHEADS, V_DIM, dtype=torch.float32, device=device)
            partial_lse = torch.empty(tq, n_splits, NHEADS, dtype=torch.float32, device=device)
            _mla_stage1_hybrid[(BATCH, n_splits)](
                q_pad, k_pad, ks_pad, v_fp8, v_sc,
                partial_o, partial_lse, qo, kv_ind, sm,
                q_pad.stride(0), q_pad.stride(1),
                k_pad.stride(0), ks_pad.stride(0), v_fp8.stride(0),
                partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
                partial_lse.stride(0), partial_lse.stride(1),
                NHEADS=NHEADS, QK_DIM=qk_dim_pad, V_DIM=V_DIM,
                NUM_SPLITS=n_splits, BLOCK_N=64,
            )
            out = torch.empty(tq, NHEADS, V_DIM, dtype=torch.bfloat16, device=device)
            _mla_reduce[(tq,)](
                partial_o, partial_lse, out,
                partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
                partial_lse.stride(0), partial_lse.stride(1),
                out.stride(0), out.stride(1),
                NHEADS=NHEADS, V_DIM=V_DIM, NUM_SPLITS=n_splits,
            )
        torch.cuda.synchronize()

        # Time stage1 only (the main kernel)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        NI = 20
        start.record()
        for _ in range(NI):
            _mla_stage1_hybrid[(BATCH, n_splits)](
                q_pad, k_pad, ks_pad, v_fp8, v_sc,
                partial_o, partial_lse, qo, kv_ind, sm,
                q_pad.stride(0), q_pad.stride(1),
                k_pad.stride(0), ks_pad.stride(0), v_fp8.stride(0),
                partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
                partial_lse.stride(0), partial_lse.stride(1),
                NHEADS=NHEADS, QK_DIM=qk_dim_pad, V_DIM=V_DIM,
                NUM_SPLITS=n_splits, BLOCK_N=64,
            )
        end.record()
        torch.cuda.synchronize()
        us_s1 = start.elapsed_time(end) * 1000.0 / NI

        # Total (stage1 + reduce)
        start.record()
        for _ in range(NI):
            _mla_stage1_hybrid[(BATCH, n_splits)](
                q_pad, k_pad, ks_pad, v_fp8, v_sc,
                partial_o, partial_lse, qo, kv_ind, sm,
                q_pad.stride(0), q_pad.stride(1),
                k_pad.stride(0), ks_pad.stride(0), v_fp8.stride(0),
                partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
                partial_lse.stride(0), partial_lse.stride(1),
                NHEADS=NHEADS, QK_DIM=qk_dim_pad, V_DIM=V_DIM,
                NUM_SPLITS=n_splits, BLOCK_N=64,
            )
            _mla_reduce[(tq,)](
                partial_o, partial_lse, out,
                partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
                partial_lse.stride(0), partial_lse.stride(1),
                out.stride(0), out.stride(1),
                NHEADS=NHEADS, V_DIM=V_DIM, NUM_SPLITS=n_splits,
            )
        end.record()
        torch.cuda.synchronize()
        us_total = start.elapsed_time(end) * 1000.0 / NI

        k_bytes = tkv * (QK_DIM // 2 + QK_DIM // 32)  # fp4 + scale
        v_bytes = tkv * V_DIM  # fp8
        q_bytes = tq * NHEADS * QK_DIM * 2
        total = q_bytes + k_bytes + v_bytes
        bw = total / (us_total * 1e-6) / 1e12

        print(f"  bs={BATCH:4d} kv={KV_SL} splits={n_splits:2d}: "
              f"stage1={us_s1:7.1f}μs  total={us_total:7.1f}μs  "
              f"BW={bw:.2f} TB/s  ({total/1e6:.1f}MB)")
