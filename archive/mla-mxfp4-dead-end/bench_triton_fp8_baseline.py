"""
Triton fp8 MLA decode baseline — same architecture as our MXFP4 kernel
but using fp8 Q/KV and tl.dot instead of dot_scaled.

This gives an apples-to-apples comparison: same Triton framework,
same grid/schedule, just different data formats.
"""
import os
os.environ.pop("HIP_VISIBLE_DEVICES", None)
import torch
torch.cuda.set_device(4)

import triton
import triton.language as tl
from aiter import dtypes as aiter_dtypes

FP8_DTYPE = aiter_dtypes.fp8


@triton.jit
def _mla_decode_fp8_stage1(
    Q_fp8,         # [total_q, NHEADS, QK_DIM] fp8
    Q_scale,       # [1] float32
    KV_fp8,        # [total_kv, QK_DIM] fp8
    KV_scale,      # [1] float32
    Partial_O,     # [total_q, NUM_SPLITS, NHEADS, V_DIM] float32
    Partial_LSE,   # [total_q, NUM_SPLITS, NHEADS] float32
    qo_indptr, kv_indptr,
    sm_scale,
    stride_q_tok: tl.int64, stride_q_h: tl.int64,
    stride_kv_tok: tl.int64,
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

    offs_h = tl.arange(0, NHEADS)
    offs_qk = tl.arange(0, QK_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    offs_v = tl.arange(0, V_DIM)

    q_scale_val = tl.load(Q_scale)
    kv_scale_val = tl.load(KV_scale)
    combined_scale = q_scale_val * kv_scale_val

    for q_pos in range(q_start, q_end):
        # Load Q as fp8, cast to bf16
        q_ptrs = Q_fp8 + q_pos * stride_q_tok + offs_h[:, None] * stride_q_h + offs_qk[None, :]
        q_fp8_data = tl.load(q_ptrs)
        q_bf16 = q_fp8_data.to(tl.bfloat16)  # [NHEADS, QK_DIM] bf16

        m_i = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([NHEADS], dtype=tl.float32)
        acc = tl.zeros([NHEADS, V_DIM], dtype=tl.float32)

        num_blocks = tl.cdiv(split_len, BLOCK_N)
        for block_idx in range(num_blocks):
            kv_off = split_start + block_idx * BLOCK_N
            valid_n = tl.minimum(BLOCK_N, split_len - block_idx * BLOCK_N)
            n_mask = offs_n < valid_n

            # Load K transposed: [QK_DIM, BLOCK_N] fp8
            k_ptrs = (KV_fp8 + (kv_off + offs_n[None, :]) * stride_kv_tok
                      + offs_qk[:, None])
            k_fp8_data = tl.load(k_ptrs, mask=n_mask[None, :])
            k_bf16 = k_fp8_data.to(tl.bfloat16)

            # QK^T: [NHEADS, QK_DIM] @ [QK_DIM, BLOCK_N] -> [NHEADS, BLOCK_N]
            qk = tl.dot(q_bf16, k_bf16, out_dtype=tl.float32)
            qk *= sm_scale * combined_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # Online softmax
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            m_i = m_new

            # Load V: [BLOCK_N, V_DIM] fp8
            v_ptrs = (KV_fp8 + (kv_off + offs_n[:, None]) * stride_kv_tok
                      + offs_v[None, :])
            v_fp8_data = tl.load(v_ptrs, mask=n_mask[:, None])
            v_bf16 = v_fp8_data.to(tl.bfloat16)

            p_bf16 = p.to(tl.bfloat16)
            acc += tl.dot(p_bf16, v_bf16, out_dtype=tl.float32)

        # Apply KV scale to V output
        acc *= kv_scale_val

        lse = m_i + tl.log2(l_i + 1e-10)
        inv_l = 1.0 / (l_i + 1e-10)
        acc *= inv_l[:, None]

        po_base = Partial_O + q_pos * stride_po_tok + split_idx * stride_po_split
        tl.store(po_base + offs_h[:, None] * stride_po_h + offs_v[None, :], acc)
        plse_ptrs = Partial_LSE + q_pos * stride_plse_tok + split_idx * stride_plse_split + offs_h
        tl.store(plse_ptrs, lse)


@triton.jit
def _mla_reduce_fp8(
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


def mla_decode_fp8_triton(q_fp8, q_scale, kv_fp8, kv_scale, qo_indptr, kv_indptr,
                           nheads, qk_dim, v_dim, sm_scale, num_splits=16, BLOCK_N=64,
                           _q_pad=None, _kv_pad=None):
    total_q = q_fp8.shape[0]
    batch_size = qo_indptr.shape[0] - 1
    device = q_fp8.device

    qk_dim_pad = _next_pow2(qk_dim)

    if _q_pad is not None:
        q_pad = _q_pad
        kv_pad = _kv_pad
    else:
        if qk_dim_pad != qk_dim:
            q_pad = torch.zeros(total_q, nheads, qk_dim_pad, dtype=q_fp8.dtype, device=device)
            q_pad[:, :, :qk_dim] = q_fp8
            kv_pad = torch.zeros(kv_fp8.shape[0], qk_dim_pad, dtype=kv_fp8.dtype, device=device)
            kv_pad[:, :qk_dim] = kv_fp8
        else:
            q_pad = q_fp8
            kv_pad = kv_fp8

    partial_o = torch.empty(total_q, num_splits, nheads, v_dim, dtype=torch.float32, device=device)
    partial_lse = torch.empty(total_q, num_splits, nheads, dtype=torch.float32, device=device)

    _mla_decode_fp8_stage1[(batch_size, num_splits)](
        q_pad, q_scale, kv_pad, kv_scale,
        partial_o, partial_lse, qo_indptr, kv_indptr, sm_scale,
        q_pad.stride(0), q_pad.stride(1),
        kv_pad.stride(0),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        NHEADS=nheads, QK_DIM=qk_dim_pad, V_DIM=v_dim,
        NUM_SPLITS=num_splits, BLOCK_N=BLOCK_N,
    )

    out = torch.empty(total_q, nheads, v_dim, dtype=torch.bfloat16, device=device)
    _mla_reduce_fp8[(total_q,)](
        partial_o, partial_lse, out,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        out.stride(0), out.stride(1),
        NHEADS=nheads, V_DIM=v_dim, NUM_SPLITS=num_splits,
    )
    return out


def quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8 = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8, scale.to(torch.float32).reshape(1)


# Import MXFP4 kernel
import sys
sys.path.insert(0, '/workspace/mla')
from mla_decode_mxfp4_v5 import mla_decode_mxfp4, _quantize_mxfp4_torch, _pad_tensors


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"

    NHEADS = 16
    QK_DIM = 576
    V_DIM = 512
    SM = 1.0 / (QK_DIM ** 0.5)

    print("=== Triton fp8 vs Triton MXFP4: apples-to-apples ===\n")

    for BATCH, KV_SL, n_splits in [(4, 4096, 16), (32, 4096, 16), (61, 4096, 16),
                                     (128, 4096, 16)]:
        tq = BATCH
        tkv = BATCH * KV_SL

        q_bf16 = torch.randn(tq, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
        kv_bf16 = torch.randn(tkv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
        qo = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
        kv_ind = torch.arange(BATCH + 1, dtype=torch.int32, device=device) * KV_SL

        # FP8
        q_fp8, q_sc = quantize_fp8(q_bf16)
        kv_fp8, kv_sc = quantize_fp8(kv_bf16)
        qk_dim_pad = _next_pow2(QK_DIM)
        q_fp8_pad = torch.zeros(tq, NHEADS, qk_dim_pad, dtype=q_fp8.dtype, device=device)
        q_fp8_pad[:, :, :QK_DIM] = q_fp8
        kv_fp8_pad = torch.zeros(tkv, qk_dim_pad, dtype=kv_fp8.dtype, device=device)
        kv_fp8_pad[:, :QK_DIM] = kv_fp8

        # MXFP4
        kv_fp4, kv_sc4 = _quantize_mxfp4_torch(kv_bf16)
        q_pad, kv_data_pad, kv_scale_pad = _pad_tensors(q_bf16, kv_fp4, kv_sc4, NHEADS, QK_DIM)

        # Warmup both
        for _ in range(3):
            mla_decode_fp8_triton(q_fp8, q_sc, kv_fp8, kv_sc, qo, kv_ind,
                                   NHEADS, QK_DIM, V_DIM, SM, n_splits, 64,
                                   q_fp8_pad, kv_fp8_pad)
            mla_decode_mxfp4(q_bf16, kv_fp4, kv_sc4, qo, kv_ind,
                             NHEADS, QK_DIM, V_DIM, SM, n_splits, 64,
                             q_pad, kv_data_pad, kv_scale_pad)
        torch.cuda.synchronize()

        # Bench FP8 Triton
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        NI = 30
        start.record()
        for _ in range(NI):
            mla_decode_fp8_triton(q_fp8, q_sc, kv_fp8, kv_sc, qo, kv_ind,
                                   NHEADS, QK_DIM, V_DIM, SM, n_splits, 64,
                                   q_fp8_pad, kv_fp8_pad)
        end.record()
        torch.cuda.synchronize()
        us_fp8 = start.elapsed_time(end) * 1000.0 / NI

        # Bench MXFP4 Triton
        start.record()
        for _ in range(NI):
            mla_decode_mxfp4(q_bf16, kv_fp4, kv_sc4, qo, kv_ind,
                             NHEADS, QK_DIM, V_DIM, SM, n_splits, 64,
                             q_pad, kv_data_pad, kv_scale_pad)
        end.record()
        torch.cuda.synchronize()
        us_fp4 = start.elapsed_time(end) * 1000.0 / NI

        fp8_kv_bytes = tkv * QK_DIM  # 1 byte per elem
        fp4_kv_bytes = tkv * (QK_DIM // 2 + QK_DIM // 32)  # fp4x2 + scales
        ratio = us_fp8 / us_fp4

        print(f"bs={BATCH:4d} kv={KV_SL} splits={n_splits}:")
        print(f"  Triton fp8:   {us_fp8:7.1f}μs  (KV={fp8_kv_bytes/1e6:.1f}MB)")
        print(f"  Triton mxfp4: {us_fp4:7.1f}μs  (KV={fp4_kv_bytes/1e6:.1f}MB)")
        print(f"  ratio:        {ratio:.2f}x {'(mxfp4 faster)' if ratio > 1 else '(fp8 faster)'}")
        print(f"  KV reduction: {(1 - fp4_kv_bytes/fp8_kv_bytes)*100:.0f}%")
        print()
