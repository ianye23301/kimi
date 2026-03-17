"""
MLA decode kernel with MXFP4 KV cache for MI355X (gfx950).

Uses tl.dot_scaled for hardware-accelerated MXFP4 dequant+matmul.

Key shapes (DeepSeek R1 / Kimi K2.5 forward_absorb MLA, TP8):
  Q: [batch, 16, 576] bf16  (16 query heads, 576 = 512 latent + 64 RoPE)
  KV: [total_kv, 1, 576] -> MXFP4: [total_kv, 1, 288] fp4x2 + scales
  Output: [batch, 16, 512] bf16

The kernel processes one batch element per program, iterating over KV blocks.
All 16 Q heads share 1 KV head (GQA ratio = 16).

QK^T: [16, 576] × [BLOCK_N, 576]^T using tl.dot_scaled (Q as bf16, K as mxfp4)
V:    softmax_scores [16, BLOCK_N] × V [BLOCK_N, 512] using tl.dot (bf16)

For V, we dequant MXFP4 to bf16 in registers since tl.dot_scaled produces
scores in fp32 and we need to multiply by V.
"""

import torch
import triton
import triton.language as tl
from aiter.utility.fp4_utils import dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32


SCALE_GROUP_SIZE = 32


@triton.jit
def _mla_decode_mxfp4_kernel(
    Q,           # [total_q, nheads, qk_dim] bf16
    KV_fp4,      # [total_kv, 1, qk_dim // 2] fp4x2 (packed)
    KV_scale,    # [padded_kv, num_scale_blocks] e8m0
    Out,         # [total_q, nheads, v_dim] bf16
    qo_indptr,   # [batch + 1] int32
    kv_indptr,   # [batch + 1] int32
    sm_scale: tl.constexpr,
    stride_qb,   # stride for Q batch dim
    stride_qh,   # stride for Q head dim
    stride_kv_n,  # stride for KV token dim (in fp4x2 units)
    stride_ks_n,  # stride for KV_scale token dim
    stride_ob,   # stride for Out batch dim
    stride_oh,   # stride for Out head dim
    NHEADS: tl.constexpr,      # 16
    QK_DIM: tl.constexpr,      # 576
    V_DIM: tl.constexpr,       # 512
    BLOCK_N: tl.constexpr,     # KV tokens per block (e.g., 64)
    BLOCK_DV: tl.constexpr,    # V dim block (512 or padded to 512)
):
    batch_idx = tl.program_id(0)

    # Get this batch's Q and KV ranges
    q_start = tl.load(qo_indptr + batch_idx)
    q_end = tl.load(qo_indptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)
    seqlen_kv = kv_end - kv_start

    # For decode, q_seq_len = 1 typically
    # Process each query position
    for q_pos in range(q_start, q_end):
        # Load Q: [NHEADS, QK_DIM] bf16
        # We need Q in a format compatible with tl.dot_scaled
        # For Q as bf16: need q_descale (all ones for bf16, or quantize Q to mxfp4 too)
        #
        # Actually, tl.dot_scaled requires BOTH operands to have scales.
        # For bf16 Q, we can pass ones as scale, but the format string matters.
        # Let's quantize Q to fp4 on the fly for hardware acceleration.

        offs_h = tl.arange(0, NHEADS)
        offs_qk = tl.arange(0, QK_DIM // 2)  # fp4x2 packed dim

        # Load Q as bf16, will need to handle this carefully
        # Actually for dot_scaled we need the data in fp4x2 or fp8 format
        # Let's use regular tl.dot for QK^T and see if that's fast enough first
        # Then optimize to mxfp4 Q if needed

        # Load Q: [NHEADS, QK_DIM] bf16
        q_ptrs = Q + q_pos * stride_qb + offs_h[:, None] * stride_qh + tl.arange(0, QK_DIM)[None, :]
        q = tl.load(q_ptrs)  # [NHEADS, QK_DIM] bf16

        # Online softmax accumulators
        m_i = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([NHEADS], dtype=tl.float32)
        acc = tl.zeros([NHEADS, BLOCK_DV], dtype=tl.float32)

        # Iterate over KV blocks
        num_kv_blocks = tl.cdiv(seqlen_kv, BLOCK_N)
        for block_idx in range(num_kv_blocks):
            kv_block_start = kv_start + block_idx * BLOCK_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_N, kv_end)
            valid_n = kv_block_end - kv_block_start

            offs_n = tl.arange(0, BLOCK_N)
            n_mask = offs_n < valid_n

            # ── Load K block as MXFP4 ──
            # KV_fp4: [total_kv, 1, QK_DIM//2], stride_kv_n is per-token
            # K tile: [BLOCK_N, QK_DIM//2] fp4x2
            k_ptrs = KV_fp4 + (kv_block_start + offs_n[:, None]) * stride_kv_n + tl.arange(0, QK_DIM // 2)[None, :]
            k_fp4 = tl.load(k_ptrs, mask=n_mask[:, None], other=0)

            # K scales: [BLOCK_N, num_scale_groups]
            num_scale_groups = QK_DIM // SCALE_GROUP_SIZE  # 576/32 = 18
            k_scale_ptrs = KV_scale + (kv_block_start + offs_n[:, None]) * stride_ks_n + tl.arange(0, num_scale_groups)[None, :]
            k_scale = tl.load(k_scale_ptrs, mask=n_mask[:, None], other=0)

            # ── QK^T: [NHEADS, QK_DIM] bf16 × [BLOCK_N, QK_DIM//2] fp4x2 ──
            # Need Q in mxfp4 format for tl.dot_scaled, or use software path
            # For now, dequant K to bf16 and use regular tl.dot
            # TODO: quantize Q to mxfp4 on the fly for tl.dot_scaled

            # Software dequant K: fp4x2 -> bf16
            # This is the slow path — we'll optimize later
            # For now, just get correctness
            # k_bf16 = dequant_mxfp4_inline(k_fp4, k_scale)  # [BLOCK_N, QK_DIM]

            # Actually, let's try tl.dot_scaled with Q in "e2m1" format
            # We need to quantize Q to fp4 on the fly
            # This is tricky — let's first try the simpler approach:
            # Load KV as bf16 (dequant on host) and do standard attention
            # Then profile to see if bandwidth is actually the bottleneck

            # For the prototype, let's skip inline dequant and just measure
            # the kernel launch overhead + tl.dot_scaled pattern
            pass

        # Store output
        # acc = acc / l_i[:, None]
        # out_ptrs = Out + q_pos * stride_ob + offs_h[:, None] * stride_oh + tl.arange(0, BLOCK_DV)[None, :]
        # tl.store(out_ptrs, acc.to(Out.type.element_ty))


# ──────────────────────────────────────────────────────────────────────
# Host-side: quantize KV to MXFP4 and run decode
# ──────────────────────────────────────────────────────────────────────

def quantize_kv_mxfp4(kv_bf16: torch.Tensor):
    """Quantize KV buffer from bf16 to MXFP4.

    Args:
        kv_bf16: [total_kv, 1, 576] bf16

    Returns:
        kv_fp4: [total_kv, 1, 288] fp4x2
        kv_scale: [padded_kv, padded_blocks] e8m0
    """
    total_kv, nkv, dim = kv_bf16.shape
    assert nkv == 1 and dim == 576

    # dynamic_mxfp4_quant expects 2D
    kv_2d = kv_bf16.view(total_kv, dim)
    kv_fp4_2d, kv_scale = dynamic_mxfp4_quant(kv_2d)
    kv_fp4 = kv_fp4_2d.view(total_kv, 1, dim // 2)
    return kv_fp4, kv_scale


def mla_decode_mxfp4_reference(
    q: torch.Tensor,         # [total_q, nheads, 576] bf16
    kv_fp4: torch.Tensor,    # [total_kv, 1, 288] fp4x2
    kv_scale: torch.Tensor,  # [padded_kv, padded_blocks] e8m0
    qo_indptr: torch.Tensor, # [batch+1] int32
    kv_indptr: torch.Tensor, # [batch+1] int32
    sm_scale: float,
    v_dim: int = 512,
) -> torch.Tensor:
    """Reference implementation: dequant KV to bf16, then standard attention."""
    total_kv = int(kv_indptr[-1].item())
    dim = 576

    # Dequant KV to bf16
    kv_fp4_2d = kv_fp4.view(-1, dim // 2)
    float_vals = mxfp4_to_f32(kv_fp4_2d)  # [total_kv, 576]

    num_blocks = dim // 32  # 18
    scale_f32 = e8m0_to_f32(kv_scale)
    scale_f32 = scale_f32[:total_kv, :num_blocks]

    float_blocked = float_vals.view(total_kv, num_blocks, 32)
    kv_bf16 = (float_blocked * scale_f32.unsqueeze(-1)).view(total_kv, dim).to(torch.bfloat16)

    # Standard attention
    batch_size = qo_indptr.shape[0] - 1
    nheads = q.shape[1]
    out_list = []

    for i in range(batch_size):
        qs, qe = int(qo_indptr[i]), int(qo_indptr[i+1])
        kvs, kve = int(kv_indptr[i]), int(kv_indptr[i+1])

        qi = q[qs:qe].float()             # [seq_q, nheads, 576]
        ki = kv_bf16[kvs:kve].float()      # [seq_kv, 576]
        vi = ki[:, :v_dim]                  # [seq_kv, 512]

        # [nheads, seq_q, 576] @ [576, seq_kv] -> [nheads, seq_q, seq_kv]
        qi_t = qi.permute(1, 0, 2)
        scores = torch.matmul(qi_t * sm_scale, ki.T)
        scores = torch.softmax(scores, dim=-1)

        # [nheads, seq_q, seq_kv] @ [seq_kv, 512] -> [nheads, seq_q, 512]
        oi = torch.matmul(scores, vi)
        out_list.append(oi.permute(1, 0, 2).to(torch.bfloat16))

    return torch.cat(out_list, dim=0)


if __name__ == "__main__":
    import sys
    torch.cuda.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 4)

    # Test correctness of MXFP4 quantize -> dequant -> attention
    bs, kvsl, nheads = 32, 4096, 16
    dim, v_dim = 576, 512
    sm_scale = 1.0 / (dim ** 0.5)

    q = torch.randn(bs, nheads, dim, dtype=torch.bfloat16, device="cuda") * 0.02
    kv_bf16 = torch.randn(bs * kvsl, 1, dim, dtype=torch.bfloat16, device="cuda") * 0.02

    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda") * kvsl

    # Quantize
    kv_fp4, kv_scale = quantize_kv_mxfp4(kv_bf16)
    print(f"KV bf16: {kv_bf16.shape} = {kv_bf16.nelement() * 2 / 1e6:.1f} MB")
    print(f"KV fp4:  {kv_fp4.shape} = {kv_fp4.nelement() / 1e6:.1f} MB")
    print(f"KV scale: {kv_scale.shape} = {kv_scale.nelement() / 1e6:.1f} MB")
    print(f"Compression: {kv_bf16.nelement() * 2 / (kv_fp4.nelement() + kv_scale.nelement()):.1f}x")

    # Reference with bf16 KV
    import torch.nn.functional as F
    out_bf16_list = []
    for i in range(bs):
        qi = q[i:i+1].float().permute(1, 0, 2)  # [16, 1, 576]
        ki = kv_bf16[i*kvsl:(i+1)*kvsl, 0].float()  # [kvsl, 576]
        vi = ki[:, :v_dim]
        scores = torch.matmul(qi * sm_scale, ki.T)
        scores = F.softmax(scores, dim=-1)
        oi = torch.matmul(scores, vi)
        out_bf16_list.append(oi.permute(1, 0, 2).to(torch.bfloat16))
    out_ref = torch.cat(out_bf16_list, dim=0)

    # Reference with MXFP4 KV (dequant then attention)
    out_mxfp4 = mla_decode_mxfp4_reference(q, kv_fp4, kv_scale, qo_indptr, kv_indptr, sm_scale, v_dim)

    # Compare
    diff = (out_ref.float() - out_mxfp4.float()).abs()
    print(f"\nbf16 vs mxfp4 dequant reference:")
    print(f"  max diff:  {diff.max().item():.6f}")
    print(f"  mean diff: {diff.mean().item():.6f}")
    print(f"  cosine sim: {F.cosine_similarity(out_ref.float().reshape(-1), out_mxfp4.float().reshape(-1), dim=0).item():.6f}")

    # Benchmark: bf16 attention vs mxfp4-dequant attention
    warmup, iters = 10, 50

    # bf16 path (using aiter)
    from aiter.mla import mla_decode_fwd, get_meta_param
    from aiter import dtypes

    FP8 = dtypes.fp8
    finfo = torch.finfo(FP8)
    q_amax = q.abs().amax().clamp(min=1e-12)
    q_scale_fp8 = (q_amax / finfo.max).float().reshape(1)
    q_fp8 = (q / q_scale_fp8).clamp(finfo.min, finfo.max).to(FP8)
    kv_amax = kv_bf16.abs().amax().clamp(min=1e-12)
    kv_scale_fp8 = (kv_amax / finfo.max).float().reshape(1)
    kv_fp8 = (kv_bf16 / kv_scale_fp8).clamp(finfo.min, finfo.max).to(FP8)

    kv_last_page_len = torch.full((bs,), kvsl, dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(bs * kvsl, dtype=torch.int32, device="cuda")
    kv_4d_fp8 = kv_fp8.view(-1, 1, 1, dim)

    ns, nsi = get_meta_param(None, bs, bs * kvsl, nheads, 1, FP8)
    o_aiter = torch.empty(bs, nheads, v_dim, dtype=torch.bfloat16, device="cuda")

    for _ in range(warmup):
        mla_decode_fwd(q_fp8, kv_4d_fp8, o_aiter, qo_indptr, kv_indptr, kv_indices,
                       kv_last_page_len, 1, page_size=1, nhead_kv=1, sm_scale=sm_scale,
                       logit_cap=0.0, num_kv_splits=ns, num_kv_splits_indptr=nsi,
                       q_scale=q_scale_fp8, kv_scale=kv_scale_fp8)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        mla_decode_fwd(q_fp8, kv_4d_fp8, o_aiter, qo_indptr, kv_indptr, kv_indices,
                       kv_last_page_len, 1, page_size=1, nhead_kv=1, sm_scale=sm_scale,
                       logit_cap=0.0, num_kv_splits=ns, num_kv_splits_indptr=nsi,
                       q_scale=q_scale_fp8, kv_scale=kv_scale_fp8)
    e.record()
    torch.cuda.synchronize()
    fp8_us = s.elapsed_time(e) * 1000 / iters

    # mxfp4 dequant path (host dequant, then bf16 attention — measures overhead)
    for _ in range(warmup):
        _ = mla_decode_mxfp4_reference(q, kv_fp4, kv_scale, qo_indptr, kv_indptr, sm_scale, v_dim)
    torch.cuda.synchronize()

    s.record()
    for _ in range(iters):
        _ = mla_decode_mxfp4_reference(q, kv_fp4, kv_scale, qo_indptr, kv_indptr, sm_scale, v_dim)
    e.record()
    torch.cuda.synchronize()
    mxfp4_ref_us = s.elapsed_time(e) * 1000 / iters

    print(f"\n=== Benchmark (bs={bs}, kvsl={kvsl}, nheads={nheads}) ===")
    print(f"  aiter fp8 ASM:       {fp8_us:.1f} us")
    print(f"  mxfp4 dequant ref:   {mxfp4_ref_us:.1f} us (torch, not optimized)")
    print(f"  Theoretical speedup from mxfp4 bandwidth: ~1.8x")
    print(f"  KV bytes/token: fp8={dim} B, mxfp4={dim//2 + dim//32}={dim//2+dim//32} B")
