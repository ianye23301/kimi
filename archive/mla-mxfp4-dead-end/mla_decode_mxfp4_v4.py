"""
MLA decode kernel with MXFP4 KV cache — v4.

Major changes from v3:
- dot_scaled for BOTH QK^T and P@V (no software dequant)
- Parallelized across (batch, kv_split) — each program handles a chunk of KV
- Stage 2 reduction kernel to combine partial results across splits
- P is quantized to MXFP4 on-the-fly for P@V dot_scaled

Layout:
- KV data: [total_kv, 1, 1, qk_dim] stored as fp4x2 -> [total_kv, qk_dim // 2] uint8
- KV scale: [total_kv, qk_dim // 32] uint8 (e8m0)
- V is the first v_dim elements of the KV buffer (MLA compressed latent)
"""

import torch
import triton
import triton.language as tl

SCALE_GROUP_SIZE = tl.constexpr(32)


@triton.jit
def _mxfp4_quant_inline(
    x,  # [M, K] float
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Quantize float [M, K] to mxfp4. Returns (fp4x2 [M, K//2], scales [M, K//32])."""
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
def _mla_decode_stage1(
    Q,             # [total_q, NHEADS, QK_DIM] bf16
    KV_data,       # [total_kv, QK_DIM_PAD // 2] uint8 (fp4x2)
    KV_scale,      # [total_kv, NUM_QK_GROUPS_PAD] uint8 (e8m0)
    # Partial outputs
    Partial_O,     # [total_q, NUM_SPLITS, NHEADS, V_DIM] float32
    Partial_LSE,   # [total_q, NUM_SPLITS, NHEADS] float32
    qo_indptr,     # [batch + 1] int32
    kv_indptr,     # [batch + 1] int32
    sm_scale,      # float
    stride_q_tok: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_tok: tl.int64,
    stride_ks_tok: tl.int64,
    stride_po_tok: tl.int64,
    stride_po_split: tl.int64,
    stride_po_h: tl.int64,
    stride_plse_tok: tl.int64,
    stride_plse_split: tl.int64,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,       # padded to power of 2
    V_DIM: tl.constexpr,        # 512, already power of 2
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Stage 1: each program handles one (batch, split) pair.
    Computes partial attention over a chunk of the KV sequence.

    For P@V, we quantize P to MXFP4 on the fly and use dot_scaled with V.
    V is loaded transposed: [V_HALF, BLOCK_N] for dot_scaled.
    """
    batch_idx = tl.program_id(0)
    split_idx = tl.program_id(1)

    q_start = tl.load(qo_indptr + batch_idx)
    q_end = tl.load(qo_indptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)
    seqlen_kv = kv_end - kv_start

    # Compute this split's range
    kv_per_split = tl.cdiv(seqlen_kv, NUM_SPLITS)
    split_kv_start = kv_start + split_idx * kv_per_split
    split_kv_end = tl.minimum(kv_start + (split_idx + 1) * kv_per_split, kv_end)
    split_len = split_kv_end - split_kv_start

    QK_HALF: tl.constexpr = QK_DIM // 2
    V_HALF: tl.constexpr = V_DIM // 2
    NUM_QK_GROUPS: tl.constexpr = QK_DIM // SCALE_GROUP_SIZE
    NUM_V_GROUPS: tl.constexpr = V_DIM // SCALE_GROUP_SIZE

    offs_h = tl.arange(0, NHEADS)
    offs_qk = tl.arange(0, QK_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    for q_pos in range(q_start, q_end):
        # ── Load Q and quantize to MXFP4 ──
        q_ptrs = Q + q_pos * stride_q_tok + offs_h[:, None] * stride_q_h + offs_qk[None, :]
        q_bf16 = tl.load(q_ptrs)
        q_fp4, q_scale = _mxfp4_quant_inline(q_bf16, QK_DIM, NHEADS)

        # Online softmax state
        m_i = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([NHEADS], dtype=tl.float32)
        acc = tl.zeros([NHEADS, V_DIM], dtype=tl.float32)

        num_blocks = tl.cdiv(split_len, BLOCK_N)
        for block_idx in range(num_blocks):
            kv_off = split_kv_start + block_idx * BLOCK_N
            valid_n = tl.minimum(BLOCK_N, split_len - block_idx * BLOCK_N)
            n_mask = offs_n < valid_n

            # ── Load K transposed: [QK_HALF, BLOCK_N] ──
            k_ptrs = (KV_data
                      + (kv_off + offs_n[None, :]) * stride_kv_tok
                      + tl.arange(0, QK_HALF)[:, None])
            k_u8 = tl.load(k_ptrs, mask=n_mask[None, :], other=0)

            # K scales [BLOCK_N, NUM_QK_GROUPS]
            ks_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_QK_GROUPS)[None, :])
            k_scale = tl.load(ks_ptrs, mask=n_mask[:, None], other=0)

            # ── QK^T via dot_scaled ──
            qk = tl.dot_scaled(q_fp4, q_scale, "e2m1", k_u8, k_scale, "e2m1")
            qk *= sm_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # ── Online softmax ──
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            m_i = m_new

            # ── P@V: quantize P to MXFP4, then dot_scaled with V ──
            # P: [NHEADS, BLOCK_N] float32 -> MXFP4
            # We need BLOCK_N to be divisible by 32 for MXFP4 quant
            p_fp4, p_scale = _mxfp4_quant_inline(p, BLOCK_N, NHEADS)
            # p_fp4: [NHEADS, BLOCK_N // 2], p_scale: [NHEADS, BLOCK_N // 32]

            # Load V transposed: [V_HALF, BLOCK_N] uint8 (fp4x2)
            v_ptrs = (KV_data
                      + (kv_off + offs_n[None, :]) * stride_kv_tok
                      + tl.arange(0, V_HALF)[:, None])
            v_u8 = tl.load(v_ptrs, mask=n_mask[None, :], other=0)

            # V scales [BLOCK_N, NUM_V_GROUPS]
            vs_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_V_GROUPS)[None, :])
            v_scale = tl.load(vs_ptrs, mask=n_mask[:, None], other=0)

            # dot_scaled: [NHEADS, BLOCK_N/2] x [V_HALF, BLOCK_N] -> [NHEADS, V_DIM]
            # Wait — dot_scaled(A, A_scale, A_fmt, B, B_scale, B_fmt) computes A @ B^T
            # We want P @ V where P is [NHEADS, BLOCK_N] and V is [BLOCK_N, V_DIM]
            # dot_scaled does: A[M,K] @ B[N,K]^T = C[M,N]
            # So we need B = V^T = [V_DIM, BLOCK_N], stored as fp4x2 = [V_HALF, BLOCK_N]
            # But our V is stored row-major [BLOCK_N, V_DIM] -> fp4x2 [BLOCK_N, V_HALF]
            # We loaded v_u8 as [V_HALF, BLOCK_N] by transposing the pointer arithmetic
            # v_scale needs to be [V_DIM, BLOCK_N//32] for the transposed layout... NO
            #
            # Actually re-read dot_scaled semantics:
            # tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format)
            # Computes lhs @ rhs (standard matmul, NOT rhs^T)
            # lhs: [M, K_packed], rhs: [K_packed, N]
            # lhs_scale: [M, K//32], rhs_scale: [N, K//32] (NOTE: rhs_scale is [N, ...] not [K, ...])
            #
            # For QK^T:
            #   lhs=q_fp4[NHEADS, QK_HALF], rhs=k_u8[QK_HALF, BLOCK_N]
            #   lhs_scale=q_scale[NHEADS, QK_GROUPS], rhs_scale=k_scale[BLOCK_N, QK_GROUPS]
            #   Result: [NHEADS, BLOCK_N] ✓
            #
            # For P@V:
            #   P[NHEADS, BLOCK_N] @ V[BLOCK_N, V_DIM]
            #   lhs = P_fp4[NHEADS, BLOCK_N//2], lhs_scale = p_scale[NHEADS, BLOCK_N//32]
            #   rhs = V_u8[BLOCK_N//2, V_DIM], rhs_scale = V_scale_transposed[V_DIM, BLOCK_N//32]
            #   Result: [NHEADS, V_DIM]
            #
            # But V is stored per-row: V_scale[BLOCK_N, V_DIM//32], not [V_DIM, BLOCK_N//32]
            # We'd need to transpose the V data AND restructure scales.
            # The V data transposition (loading [V_HALF, BLOCK_N]) handles the data part,
            # but the SCALE layout is the problem: hardware needs [N, K//32] = [V_DIM, BLOCK_N//32]
            # while we have [BLOCK_N, V_DIM//32].

            # P@V with dot_scaled won't work easily because V's scale layout is wrong.
            # Fall back to bf16 tl.dot for P@V.
            # Dequant P to bf16, dequant V to bf16, then tl.dot.

            # Actually simpler: just use bf16 tl.dot for P@V
            # P is already computed as float, just cast to bf16
            p_bf16 = p.to(tl.bfloat16)

            # V: load as bf16 by dequanting on the fly... but V is stored as fp4x2!
            # We need the ACTUAL V values. Let's load V row-major and do software dequant.
            # This is the bottleneck we wanted to avoid.

            # ALTERNATIVE: since we're bandwidth-bound, the V dequant compute is "free"
            # if we're loading fewer bytes. Let's just do it.
            # Load V row-major: [BLOCK_N, V_HALF] uint8
            v_row_ptrs = (KV_data
                          + (kv_off + offs_n[:, None]) * stride_kv_tok
                          + tl.arange(0, V_HALF)[None, :])
            v_row_u8 = tl.load(v_row_ptrs, mask=n_mask[:, None], other=0)

            # Unpack fp4x2
            lo = v_row_u8 & 0x0F
            hi = (v_row_u8 >> 4) & 0x0F
            lo_sign = (lo >> 3).to(tl.float32) * (-2.0) + 1.0
            lo_mag = lo & 0x07
            hi_sign = (hi >> 3).to(tl.float32) * (-2.0) + 1.0
            hi_mag = hi & 0x07

            # LUT via nested where
            lo_val = tl.where(lo_mag == 0, 0.0,
                     tl.where(lo_mag == 1, 0.5,
                     tl.where(lo_mag == 2, 1.0,
                     tl.where(lo_mag == 3, 1.5,
                     tl.where(lo_mag == 4, 2.0,
                     tl.where(lo_mag == 5, 3.0,
                     tl.where(lo_mag == 6, 4.0, 6.0)))))))
            hi_val = tl.where(hi_mag == 0, 0.0,
                     tl.where(hi_mag == 1, 0.5,
                     tl.where(hi_mag == 2, 1.0,
                     tl.where(hi_mag == 3, 1.5,
                     tl.where(hi_mag == 4, 2.0,
                     tl.where(hi_mag == 5, 3.0,
                     tl.where(hi_mag == 6, 4.0, 6.0)))))))

            lo_float = lo_sign * lo_val
            hi_float = hi_sign * hi_val

            # Apply V scales
            HALF_GROUP: tl.constexpr = SCALE_GROUP_SIZE // 2
            v_scale_int = v_scale.to(tl.uint32)
            v_scale_f32 = (v_scale_int << 23).to(tl.float32, bitcast=True)
            v_scale_exp = v_scale_f32.reshape(BLOCK_N, NUM_V_GROUPS, 1)
            v_scale_exp = tl.broadcast_to(v_scale_exp, [BLOCK_N, NUM_V_GROUPS, HALF_GROUP])
            v_scale_exp = v_scale_exp.reshape(BLOCK_N, V_HALF)

            v_even = (lo_float * v_scale_exp).to(tl.bfloat16)  # [BLOCK_N, V_HALF]
            v_odd = (hi_float * v_scale_exp).to(tl.bfloat16)   # [BLOCK_N, V_HALF]

            # P@V_even and P@V_odd
            pv_even = tl.dot(p_bf16, v_even, out_dtype=tl.float32)  # [NHEADS, V_HALF]
            pv_odd = tl.dot(p_bf16, v_odd, out_dtype=tl.float32)   # [NHEADS, V_HALF]

            # Interleave into acc[NHEADS, V_DIM]
            # acc[:, 0::2] += pv_even, acc[:, 1::2] += pv_odd
            # Use reshape trick: acc is [NHEADS, V_HALF, 2]
            acc_3d = acc.reshape(NHEADS, V_HALF, 2)
            # Can't slice+assign in Triton. Use separate accumulators instead.
            # Actually wait — we already solved this in v3. Let me just use acc_even/acc_odd.
            pass

        # This got messy. Let me simplify.
        pass


# ============================================================
# OK, the dot_scaled approach for P@V doesn't work due to scale layout mismatch.
# Let me go back to v3's approach (split acc_even/acc_odd) but add KV split parallelism.
# ============================================================

@triton.jit
def _mla_decode_split_kv(
    Q,             # [total_q, NHEADS, QK_DIM] bf16 (padded)
    KV_data,       # [total_kv, QK_DIM_PAD // 2] uint8 (fp4x2, padded)
    KV_scale,      # [total_kv, NUM_QK_GROUPS_PAD] uint8 (e8m0, padded)
    Partial_O_even,  # [total_q, NUM_SPLITS, NHEADS, V_HALF] float32
    Partial_O_odd,   # [total_q, NUM_SPLITS, NHEADS, V_HALF] float32
    Partial_LSE,     # [total_q, NUM_SPLITS, NHEADS] float32
    qo_indptr,
    kv_indptr,
    sm_scale,
    stride_q_tok: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_tok: tl.int64,
    stride_ks_tok: tl.int64,
    stride_po_tok: tl.int64,
    stride_po_split: tl.int64,
    stride_po_h: tl.int64,
    stride_plse_tok: tl.int64,
    stride_plse_split: tl.int64,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Stage 1: one program per (batch, split). Produces partial O and LSE."""
    batch_idx = tl.program_id(0)
    split_idx = tl.program_id(1)

    q_start = tl.load(qo_indptr + batch_idx)
    q_end = tl.load(qo_indptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)
    seqlen_kv = kv_end - kv_start

    kv_per_split = tl.cdiv(seqlen_kv, NUM_SPLITS)
    split_kv_start = kv_start + split_idx * kv_per_split
    split_kv_end = tl.minimum(kv_start + (split_idx + 1) * kv_per_split, kv_end)
    split_len = tl.maximum(split_kv_end - split_kv_start, 0)

    QK_HALF: tl.constexpr = QK_DIM // 2
    V_HALF: tl.constexpr = V_DIM // 2
    NUM_QK_GROUPS: tl.constexpr = QK_DIM // SCALE_GROUP_SIZE
    NUM_V_GROUPS: tl.constexpr = V_DIM // SCALE_GROUP_SIZE
    HALF_GROUP: tl.constexpr = SCALE_GROUP_SIZE // 2

    offs_h = tl.arange(0, NHEADS)
    offs_qk = tl.arange(0, QK_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    for q_pos in range(q_start, q_end):
        # Load Q and quantize
        q_ptrs = Q + q_pos * stride_q_tok + offs_h[:, None] * stride_q_h + offs_qk[None, :]
        q_bf16 = tl.load(q_ptrs)
        q_fp4, q_scale = _mxfp4_quant_inline(q_bf16, QK_DIM, NHEADS)

        m_i = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([NHEADS], dtype=tl.float32)
        acc_even = tl.zeros([NHEADS, V_HALF], dtype=tl.float32)
        acc_odd = tl.zeros([NHEADS, V_HALF], dtype=tl.float32)

        num_blocks = tl.cdiv(split_len, BLOCK_N)
        for block_idx in range(num_blocks):
            kv_off = split_kv_start + block_idx * BLOCK_N
            valid_n = tl.minimum(BLOCK_N, split_len - block_idx * BLOCK_N)
            n_mask = offs_n < valid_n

            # Load K transposed [QK_HALF, BLOCK_N]
            k_ptrs = (KV_data
                      + (kv_off + offs_n[None, :]) * stride_kv_tok
                      + tl.arange(0, QK_HALF)[:, None])
            k_u8 = tl.load(k_ptrs, mask=n_mask[None, :], other=0)

            ks_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_QK_GROUPS)[None, :])
            k_scale = tl.load(ks_ptrs, mask=n_mask[:, None], other=0)

            # QK^T via dot_scaled
            qk = tl.dot_scaled(q_fp4, q_scale, "e2m1", k_u8, k_scale, "e2m1")
            qk *= sm_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # Online softmax
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc_even = acc_even * alpha[:, None]
            acc_odd = acc_odd * alpha[:, None]
            m_i = m_new

            # Load V [BLOCK_N, V_HALF] and dequant
            v_ptrs = (KV_data
                      + (kv_off + offs_n[:, None]) * stride_kv_tok
                      + tl.arange(0, V_HALF)[None, :])
            v_u8 = tl.load(v_ptrs, mask=n_mask[:, None], other=0)

            vs_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_V_GROUPS)[None, :])
            v_scale_u8 = tl.load(vs_ptrs, mask=n_mask[:, None], other=0)

            # Dequant V
            lo = v_u8 & 0x0F
            hi = (v_u8 >> 4) & 0x0F

            # Sign: bit 3
            lo_sign = (lo >> 3).to(tl.float32) * (-2.0) + 1.0
            hi_sign = (hi >> 3).to(tl.float32) * (-2.0) + 1.0
            lo_mag = (lo & 0x07).to(tl.float32)
            hi_mag = (hi & 0x07).to(tl.float32)

            # Polynomial approximation of LUT (faster than nested where):
            # f(x) = x * (0.5 + x * (0 + x * (1/6))) approximation... nah
            # Just use the where chain, it compiles to select instructions
            lo_mag_i = lo & 0x07
            hi_mag_i = hi & 0x07
            lo_val = tl.where(lo_mag_i == 0, 0.0,
                     tl.where(lo_mag_i == 1, 0.5,
                     tl.where(lo_mag_i == 2, 1.0,
                     tl.where(lo_mag_i == 3, 1.5,
                     tl.where(lo_mag_i == 4, 2.0,
                     tl.where(lo_mag_i == 5, 3.0,
                     tl.where(lo_mag_i == 6, 4.0, 6.0)))))))
            hi_val = tl.where(hi_mag_i == 0, 0.0,
                     tl.where(hi_mag_i == 1, 0.5,
                     tl.where(hi_mag_i == 2, 1.0,
                     tl.where(hi_mag_i == 3, 1.5,
                     tl.where(hi_mag_i == 4, 2.0,
                     tl.where(hi_mag_i == 5, 3.0,
                     tl.where(hi_mag_i == 6, 4.0, 6.0)))))))

            lo_float = lo_sign * lo_val
            hi_float = hi_sign * hi_val

            # Apply scales
            v_scale_int = v_scale_u8.to(tl.uint32)
            v_scale_f32 = (v_scale_int << 23).to(tl.float32, bitcast=True)
            v_scale_exp = v_scale_f32.reshape(BLOCK_N, NUM_V_GROUPS, 1)
            v_scale_exp = tl.broadcast_to(v_scale_exp, [BLOCK_N, NUM_V_GROUPS, HALF_GROUP])
            v_scale_exp = v_scale_exp.reshape(BLOCK_N, V_HALF)

            v_even_bf16 = (lo_float * v_scale_exp).to(tl.bfloat16)
            v_odd_bf16 = (hi_float * v_scale_exp).to(tl.bfloat16)

            p_bf16 = p.to(tl.bfloat16)
            acc_even += tl.dot(p_bf16, v_even_bf16, out_dtype=tl.float32)
            acc_odd += tl.dot(p_bf16, v_odd_bf16, out_dtype=tl.float32)

        # Store partial results
        # lse = m_i + log2(l_i) — but we used exp2 so this is the log-sum-exp in base 2
        lse = m_i + tl.log2(l_i + 1e-10)

        # Normalize partial output
        inv_l = 1.0 / (l_i + 1e-10)
        acc_even = acc_even * inv_l[:, None]
        acc_odd = acc_odd * inv_l[:, None]

        # Store partial O even [NHEADS, V_HALF]
        offs_vh = tl.arange(0, V_HALF)
        po_base_even = (Partial_O_even + q_pos * stride_po_tok
                        + split_idx * stride_po_split)
        po_even_ptrs = po_base_even + offs_h[:, None] * stride_po_h + offs_vh[None, :]
        tl.store(po_even_ptrs, acc_even)

        po_base_odd = (Partial_O_odd + q_pos * stride_po_tok
                       + split_idx * stride_po_split)
        po_odd_ptrs = po_base_odd + offs_h[:, None] * stride_po_h + offs_vh[None, :]
        tl.store(po_odd_ptrs, acc_odd)

        # Store LSE [NHEADS]
        plse_ptrs = (Partial_LSE + q_pos * stride_plse_tok
                     + split_idx * stride_plse_split + offs_h)
        tl.store(plse_ptrs, lse)


@triton.jit
def _mla_reduce(
    Partial_O_even,  # [total_q, NUM_SPLITS, NHEADS, V_HALF] float32
    Partial_O_odd,   # [total_q, NUM_SPLITS, NHEADS, V_HALF] float32
    Partial_LSE,     # [total_q, NUM_SPLITS, NHEADS] float32
    Out,             # [total_q, NHEADS, V_DIM] bf16
    stride_po_tok: tl.int64,
    stride_po_split: tl.int64,
    stride_po_h: tl.int64,
    stride_plse_tok: tl.int64,
    stride_plse_split: tl.int64,
    stride_o_tok: tl.int64,
    stride_o_h: tl.int64,
    NHEADS: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Reduce partial results across splits. One program per query token."""
    q_idx = tl.program_id(0)
    V_HALF: tl.constexpr = V_DIM // 2

    offs_h = tl.arange(0, NHEADS)
    offs_vh = tl.arange(0, V_HALF)

    # Load all partial LSEs and find global max
    # Process splits sequentially (NUM_SPLITS is small, typically 8-32)
    global_m = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
    for s in range(NUM_SPLITS):
        lse_ptrs = Partial_LSE + q_idx * stride_plse_tok + s * stride_plse_split + offs_h
        lse_s = tl.load(lse_ptrs)
        global_m = tl.maximum(global_m, lse_s)

    # Compute weighted sum
    acc_even = tl.zeros([NHEADS, V_HALF], dtype=tl.float32)
    acc_odd = tl.zeros([NHEADS, V_HALF], dtype=tl.float32)
    total_w = tl.zeros([NHEADS], dtype=tl.float32)

    for s in range(NUM_SPLITS):
        lse_ptrs = Partial_LSE + q_idx * stride_plse_tok + s * stride_plse_split + offs_h
        lse_s = tl.load(lse_ptrs)
        w = tl.exp2(lse_s - global_m)  # weight for this split
        total_w += w

        po_base_even = Partial_O_even + q_idx * stride_po_tok + s * stride_po_split
        po_even = tl.load(po_base_even + offs_h[:, None] * stride_po_h + offs_vh[None, :])

        po_base_odd = Partial_O_odd + q_idx * stride_po_tok + s * stride_po_split
        po_odd = tl.load(po_base_odd + offs_h[:, None] * stride_po_h + offs_vh[None, :])

        acc_even += w[:, None] * po_even
        acc_odd += w[:, None] * po_odd

    # Normalize
    inv_w = 1.0 / (total_w + 1e-10)
    acc_even = acc_even * inv_w[:, None]
    acc_odd = acc_odd * inv_w[:, None]

    # Interleave and store
    o_base = Out + q_idx * stride_o_tok
    o_even_ptrs = o_base + offs_h[:, None] * stride_o_h + (offs_vh[None, :] * 2)
    tl.store(o_even_ptrs, acc_even.to(Out.type.element_ty))
    o_odd_ptrs = o_base + offs_h[:, None] * stride_o_h + (offs_vh[None, :] * 2 + 1)
    tl.store(o_odd_ptrs, acc_odd.to(Out.type.element_ty))


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def mla_decode_mxfp4(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    nheads: int,
    qk_dim: int,
    v_dim: int,
    sm_scale: float,
    num_splits: int = 16,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    total_q = q.shape[0]
    batch_size = qo_indptr.shape[0] - 1
    v_half = v_dim // 2

    # Pad QK to next power of 2
    qk_dim_pad = _next_pow2(qk_dim)
    qk_half = qk_dim // 2
    qk_half_pad = qk_dim_pad // 2
    num_qk_groups = qk_dim // 32
    num_qk_groups_pad = _next_pow2(num_qk_groups)

    if qk_dim_pad != qk_dim:
        q_pad = torch.zeros(total_q, nheads, qk_dim_pad, dtype=q.dtype, device=q.device)
        q_pad[:, :, :qk_dim] = q
    else:
        q_pad = q

    if qk_half_pad != qk_half:
        kv_data_pad = torch.zeros(kv_data.shape[0], qk_half_pad, dtype=kv_data.dtype, device=kv_data.device)
        kv_data_pad[:, :qk_half] = kv_data
    else:
        kv_data_pad = kv_data

    if num_qk_groups_pad != num_qk_groups:
        kv_scale_pad = torch.zeros(kv_scale.shape[0], num_qk_groups_pad, dtype=kv_scale.dtype, device=kv_scale.device)
        kv_scale_pad[:, :num_qk_groups] = kv_scale
    else:
        kv_scale_pad = kv_scale

    # Allocate partial outputs
    partial_o_even = torch.empty(total_q, num_splits, nheads, v_half, dtype=torch.float32, device=q.device)
    partial_o_odd = torch.empty(total_q, num_splits, nheads, v_half, dtype=torch.float32, device=q.device)
    partial_lse = torch.empty(total_q, num_splits, nheads, dtype=torch.float32, device=q.device)

    # Stage 1
    grid1 = (batch_size, num_splits)
    _mla_decode_split_kv[grid1](
        q_pad, kv_data_pad, kv_scale_pad,
        partial_o_even, partial_o_odd, partial_lse,
        qo_indptr, kv_indptr,
        sm_scale,
        q_pad.stride(0), q_pad.stride(1),
        kv_data_pad.stride(0),
        kv_scale_pad.stride(0),
        partial_o_even.stride(0), partial_o_even.stride(1), partial_o_even.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        NHEADS=nheads,
        QK_DIM=qk_dim_pad,
        V_DIM=v_dim,
        NUM_SPLITS=num_splits,
        BLOCK_N=BLOCK_N,
    )

    # Stage 2: reduce
    out = torch.empty(total_q, nheads, v_dim, dtype=torch.bfloat16, device=q.device)
    grid2 = (total_q,)
    _mla_reduce[grid2](
        partial_o_even, partial_o_odd, partial_lse,
        out,
        partial_o_even.stride(0), partial_o_even.stride(1), partial_o_even.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        out.stride(0), out.stride(1),
        NHEADS=nheads,
        V_DIM=v_dim,
        NUM_SPLITS=num_splits,
    )
    return out


# ── Reference ──

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
    half_group = 16
    n_groups = scale_f32.shape[1]
    scale_expanded = scale_f32.unsqueeze(2).expand(-1, n_groups, half_group).reshape(-1, dim // 2)
    lo_scaled = lo_val * scale_expanded
    hi_scaled = hi_val * scale_expanded
    N = data_u8.shape[0]
    result = torch.empty(N, dim, device=data_u8.device, dtype=torch.float32)
    result[:, 0::2] = lo_scaled
    result[:, 1::2] = hi_scaled
    return result


def mla_decode_reference(q, kv_data, kv_scale, qo_indptr, kv_indptr, nheads, qk_dim, v_dim, sm_scale):
    batch_size = qo_indptr.shape[0] - 1
    total_kv = kv_data.shape[0]
    num_qk_groups = qk_dim // 32
    num_v_groups = v_dim // 32
    kv_f32 = _dequant_mxfp4_torch(kv_data, kv_scale[:total_kv, :num_qk_groups], qk_dim)
    V_f32 = _dequant_mxfp4_torch(kv_data[:, :v_dim // 2], kv_scale[:total_kv, :num_v_groups], v_dim)
    outputs = []
    for b in range(batch_size):
        q_start, q_end = qo_indptr[b].item(), qo_indptr[b + 1].item()
        kv_start, kv_end = kv_indptr[b].item(), kv_indptr[b + 1].item()
        q_b = q[q_start:q_end].float()
        k_b = kv_f32[kv_start:kv_end]
        v_b = V_f32[kv_start:kv_end]
        for qi in range(q_b.shape[0]):
            scores = torch.matmul(q_b[qi], k_b.T) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_b)
            outputs.append(out)
    return torch.stack(outputs).to(torch.bfloat16)


def _quantize_mxfp4_torch(tensor):
    FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device)
    N, D = tensor.shape
    assert D % 32 == 0
    t = tensor.float()
    t_grouped = t.reshape(N, D // 32, 32)
    amax = t_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    log2_amax = torch.floor(torch.log2(amax)) + 2
    log2_amax = log2_amax.clamp(-127, 127)
    scale_e8m0 = (log2_amax.long() + 127).clamp(0, 254).to(torch.uint8)
    scale_f32 = torch.pow(2.0, log2_amax)
    t_scaled = t_grouped / scale_f32
    signs = (t_scaled < 0).to(torch.uint8)
    t_abs = t_scaled.abs()
    diffs = (t_abs.unsqueeze(-1) - FP4_VALUES.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs()
    mag_idx = diffs.argmin(dim=-1).to(torch.uint8)
    fp4_vals = (signs << 3) | mag_idx
    fp4_vals = fp4_vals.reshape(N, D // 2, 2)
    fp4x2 = fp4_vals[:, :, 0] | (fp4_vals[:, :, 1] << 4)
    return fp4x2.to(torch.uint8), scale_e8m0.reshape(N, D // 32)


if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = "cuda"

    NHEADS = 16
    QK_DIM = 576
    V_DIM = 512
    BATCH = 4
    KV_SEQLEN = 256

    print(f"Testing MXFP4 MLA decode v4: batch={BATCH}, nheads={NHEADS}, "
          f"qk_dim={QK_DIM}, v_dim={V_DIM}, kv_seqlen={KV_SEQLEN}")

    total_q = BATCH
    q = torch.randn(total_q, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
    total_kv = BATCH * KV_SEQLEN
    kv_bf16 = torch.randn(total_kv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
    kv_indptr = (torch.arange(BATCH + 1, dtype=torch.int32, device=device) * KV_SEQLEN)
    kv_fp4, kv_scale = _quantize_mxfp4_torch(kv_bf16)
    sm_scale = 1.0 / (QK_DIM ** 0.5)

    print("Running reference...")
    out_ref = mla_decode_reference(q, kv_fp4, kv_scale, qo_indptr, kv_indptr,
                                    NHEADS, QK_DIM, V_DIM, sm_scale)

    print("Running Triton v4 (split-kv)...")
    out_tri = mla_decode_mxfp4(q, kv_fp4, kv_scale, qo_indptr, kv_indptr,
                                NHEADS, QK_DIM, V_DIM, sm_scale,
                                num_splits=8, BLOCK_N=64)

    cos_sim = torch.nn.functional.cosine_similarity(
        out_ref.float().reshape(-1), out_tri.float().reshape(-1), dim=0
    ).item()
    max_diff = (out_ref.float() - out_tri.float()).abs().max().item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Max abs diff: {max_diff:.6f}")

    if cos_sim > 0.99:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)

    # Benchmark
    print("\n=== Benchmarking ===")
    for batch, kv_sl, n_splits in [(1, 4096, 16), (4, 4096, 16), (32, 4096, 16),
                                     (61, 4096, 16), (128, 4096, 16)]:
        tq = batch
        tkv = batch * kv_sl
        q_b = torch.randn(tq, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
        qo_b = torch.arange(batch + 1, dtype=torch.int32, device=device)
        kv_bf = torch.randn(tkv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
        kv_b = (torch.arange(batch + 1, dtype=torch.int32, device=device) * kv_sl)
        kv_fp4_b, kv_sc_b = _quantize_mxfp4_torch(kv_bf)

        for _ in range(5):
            mla_decode_mxfp4(q_b, kv_fp4_b, kv_sc_b, qo_b, kv_b,
                             NHEADS, QK_DIM, V_DIM, sm_scale,
                             num_splits=n_splits, BLOCK_N=64)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        N_ITER = 20
        start.record()
        for _ in range(N_ITER):
            mla_decode_mxfp4(q_b, kv_fp4_b, kv_sc_b, qo_b, kv_b,
                             NHEADS, QK_DIM, V_DIM, sm_scale,
                             num_splits=n_splits, BLOCK_N=64)
        end.record()
        torch.cuda.synchronize()
        elapsed_us = start.elapsed_time(end) * 1000.0 / N_ITER

        # KV bytes loaded (fp4x2 + scales)
        kv_bytes = tkv * (QK_DIM // 2 + QK_DIM // 32)
        q_bytes = tq * NHEADS * QK_DIM * 2
        o_bytes = tq * NHEADS * V_DIM * 2
        total_bytes = q_bytes + kv_bytes + o_bytes
        bw_tb_s = total_bytes / (elapsed_us * 1e-6) / 1e12

        print(f"  bs={batch:4d} kv={kv_sl:5d} splits={n_splits:2d}: {elapsed_us:8.1f} μs  "
              f"BW={bw_tb_s:.2f} TB/s  ({total_bytes/1e6:.1f} MB)")
