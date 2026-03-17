"""
MLA decode kernel with MXFP4 KV cache — v3.

Strategy:
- QK^T: quantize Q to MXFP4 on the fly, use tl.dot_scaled(Q_fp4, K_fp4)
- P@V: dequant V from MXFP4 to bf16 in registers, use tl.dot(P_bf16, V_bf16)

Grid: one program per (batch_element, head_group) to maximize parallelism.
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


# FP4 E2M1 dequant lookup table (unsigned values 0-7)
# 0->0.0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@triton.jit
def _dequant_mxfp4_block(
    data_u8,      # [BLOCK_N, DIM_HALF] uint8 (packed fp4x2)
    scale_u8,     # [BLOCK_N, NUM_GROUPS] uint8 (e8m0)
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    """Dequant MXFP4 block to float32. Returns [BLOCK_N, DIM] float32."""
    DIM_HALF: tl.constexpr = DIM // 2
    NUM_GROUPS: tl.constexpr = DIM // SCALE_GROUP_SIZE

    # Unpack fp4x2 to two nibbles
    lo = data_u8 & 0x0F                    # [BLOCK_N, DIM_HALF] even elements
    hi = (data_u8 >> 4) & 0x0F             # [BLOCK_N, DIM_HALF] odd elements

    # Extract sign and magnitude for each nibble
    lo_sign = (lo >> 3).to(tl.float32) * (-2.0) + 1.0  # 1.0 if positive, -1.0 if negative
    lo_mag = lo & 0x07                       # magnitude 0-7
    hi_sign = (hi >> 3).to(tl.float32) * (-2.0) + 1.0
    hi_mag = hi & 0x07

    # Lookup: convert magnitude to float
    # Manually handle each case since we can't index a LUT in Triton easily
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

    lo_float = lo_sign * lo_val  # [BLOCK_N, DIM_HALF]
    hi_float = hi_sign * hi_val  # [BLOCK_N, DIM_HALF]

    # Interleave: result[i, 2j] = lo[i,j], result[i, 2j+1] = hi[i,j]
    # Reshape to [BLOCK_N, DIM_HALF, 2] then flatten
    lo_3d = lo_float.reshape(BLOCK_N, DIM_HALF, 1)
    hi_3d = hi_float.reshape(BLOCK_N, DIM_HALF, 1)
    # Can't easily interleave in Triton, so use blocked layout instead
    # Since scales are per-32-element blocks, and each fp4x2 byte covers 2 consecutive elements,
    # 16 bytes = 32 elements = 1 scale group
    # So lo covers even indices, hi covers odd indices within each byte

    # Actually, the fp4x2 packing is: byte j holds elements [2j] and [2j+1]
    # lo = element[2j], hi = element[2j+1]
    # So for scale application:
    # element[2j] belongs to group floor(2j / 32) = floor(j / 16)
    # element[2j+1] belongs to group floor((2j+1) / 32) = floor(j / 16) (for j not at boundary)
    # Both elements in the same byte belong to the same scale group (since 2j and 2j+1 differ by 1)

    # Apply scales per group
    # scale_u8: [BLOCK_N, NUM_GROUPS] e8m0 -> float32 = 2^(val - 127)
    scale_int = scale_u8.to(tl.int32)
    # Reconstruct float32: put e8m0 value as the exponent of an IEEE float
    scale_f32 = ((scale_int.to(tl.uint32)) << 23).to(tl.float32, bitcast=True)

    # Expand scales to match half-dim: each group covers 16 bytes (32 elements = 16 pairs)
    # scale_expanded: [BLOCK_N, DIM_HALF] where scale_expanded[i, j] = scale[i, j // 16]
    # Using reshape + broadcast
    HALF_GROUP: tl.constexpr = SCALE_GROUP_SIZE // 2  # 16
    scale_expanded = scale_f32.reshape(BLOCK_N, NUM_GROUPS, 1)
    scale_expanded = tl.broadcast_to(scale_expanded, [BLOCK_N, NUM_GROUPS, HALF_GROUP])
    scale_expanded = scale_expanded.reshape(BLOCK_N, DIM_HALF)

    lo_scaled = lo_float * scale_expanded  # [BLOCK_N, DIM_HALF]
    hi_scaled = hi_float * scale_expanded  # [BLOCK_N, DIM_HALF]

    return lo_scaled, hi_scaled  # Even and odd elements, both [BLOCK_N, DIM_HALF]


@triton.jit
def _mla_decode_mxfp4_v3(
    Q,             # [total_q, NHEADS, QK_DIM] bf16
    KV_data,       # [total_kv, QK_DIM // 2] uint8 (fp4x2 packed, kv_heads squeezed)
    KV_scale,      # [total_kv_padded, NUM_SCALE_GROUPS_PADDED] uint8 (e8m0)
    Out,           # [total_q, NHEADS, V_DIM] bf16
    qo_indptr,     # [batch + 1] int32
    kv_indptr,     # [batch + 1] int32
    sm_scale,      # float
    stride_q_tok: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_tok: tl.int64,
    stride_ks_tok: tl.int64,
    stride_o_tok: tl.int64,
    stride_o_h: tl.int64,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    One program per batch element. Iterates over KV sequence.

    Key insight for P@V with fp4x2 packed V:
    - fp4x2 byte j contains element[2j] (lo nibble) and element[2j+1] (hi nibble)
    - After dequant we get v_even[BLOCK_N, V_HALF] and v_odd[BLOCK_N, V_HALF]
    - P @ v_even gives output at even indices, P @ v_odd gives output at odd indices
    - We maintain acc_even[NHEADS, V_HALF] and acc_odd[NHEADS, V_HALF] separately
    - Interleave only at final store: out[h, 2j]=acc_even[h,j], out[h, 2j+1]=acc_odd[h,j]
    """
    batch_idx = tl.program_id(0)

    q_start = tl.load(qo_indptr + batch_idx)
    q_end = tl.load(qo_indptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)
    seqlen_kv = kv_end - kv_start

    QK_HALF: tl.constexpr = QK_DIM // 2         # 288
    V_HALF: tl.constexpr = V_DIM // 2           # 256
    NUM_QK_GROUPS: tl.constexpr = QK_DIM // 32  # 18
    NUM_V_GROUPS: tl.constexpr = V_DIM // 32    # 16

    offs_h = tl.arange(0, NHEADS)
    offs_qk = tl.arange(0, QK_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    for q_pos in range(q_start, q_end):
        # ── Load Q [NHEADS, QK_DIM] bf16, quantize to MXFP4 ──
        q_ptrs = Q + q_pos * stride_q_tok + offs_h[:, None] * stride_q_h + offs_qk[None, :]
        q_bf16 = tl.load(q_ptrs)
        q_fp4, q_scale = _mxfp4_quant_inline(q_bf16, QK_DIM, NHEADS)

        # Online softmax state — separate accumulators for even/odd V elements
        m_i = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([NHEADS], dtype=tl.float32)
        acc_even = tl.zeros([NHEADS, V_HALF], dtype=tl.float32)
        acc_odd = tl.zeros([NHEADS, V_HALF], dtype=tl.float32)

        num_blocks = tl.cdiv(seqlen_kv, BLOCK_N)
        for block_idx in range(num_blocks):
            kv_off = kv_start + block_idx * BLOCK_N
            valid_n = tl.minimum(BLOCK_N, seqlen_kv - block_idx * BLOCK_N)
            n_mask = offs_n < valid_n

            # ── Load K [QK_HALF, BLOCK_N] fp4x2 (transposed for dot_scaled) ──
            k_ptrs = (KV_data
                      + (kv_off + offs_n[None, :]) * stride_kv_tok
                      + tl.arange(0, QK_HALF)[:, None])
            k_u8 = tl.load(k_ptrs, mask=n_mask[None, :], other=0)

            # K scales [BLOCK_N, NUM_QK_GROUPS]
            ks_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_QK_GROUPS)[None, :])
            k_scale = tl.load(ks_ptrs, mask=n_mask[:, None], other=0)

            # ── QK^T: [NHEADS, QK_HALF] x [QK_HALF, BLOCK_N] via dot_scaled ──
            qk = tl.zeros([NHEADS, BLOCK_N], dtype=tl.float32)
            qk = tl.dot_scaled(q_fp4, q_scale, "e2m1", k_u8, k_scale, "e2m1", acc=qk)
            qk *= sm_scale

            # Mask invalid positions
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # ── Online softmax update ──
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc_even = acc_even * alpha[:, None]
            acc_odd = acc_odd * alpha[:, None]
            m_i = m_new

            # ── Load V (first V_DIM elements of KV buffer) and dequant ──
            v_ptrs = (KV_data
                      + (kv_off + offs_n[:, None]) * stride_kv_tok
                      + tl.arange(0, V_HALF)[None, :])
            v_u8 = tl.load(v_ptrs, mask=n_mask[:, None], other=0)

            # V scales [BLOCK_N, NUM_V_GROUPS]
            vs_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_V_GROUPS)[None, :])
            v_scale_u8 = tl.load(vs_ptrs, mask=n_mask[:, None], other=0)

            # Dequant V: get even/odd element arrays [BLOCK_N, V_HALF] each
            v_even, v_odd = _dequant_mxfp4_block(v_u8, v_scale_u8, BLOCK_N, V_DIM)

            # P@V split: P[NHEADS, BLOCK_N] @ V_half[BLOCK_N, V_HALF] -> [NHEADS, V_HALF]
            p_bf16 = p.to(tl.bfloat16)
            v_even_bf16 = v_even.to(tl.bfloat16)
            v_odd_bf16 = v_odd.to(tl.bfloat16)

            acc_even += tl.dot(p_bf16, v_even_bf16, out_dtype=tl.float32)
            acc_odd += tl.dot(p_bf16, v_odd_bf16, out_dtype=tl.float32)

        # ── Finalize: divide by softmax denominator ──
        inv_l = 1.0 / l_i
        acc_even = acc_even * inv_l[:, None]
        acc_odd = acc_odd * inv_l[:, None]

        # ── Interleave and store: out[h, 2j] = acc_even[h,j], out[h, 2j+1] = acc_odd[h,j] ──
        # Reshape to [NHEADS, V_HALF, 2] then flatten to [NHEADS, V_DIM]
        acc_even_3d = acc_even.reshape(NHEADS, V_HALF, 1)
        acc_odd_3d = acc_odd.reshape(NHEADS, V_HALF, 1)
        # Use scatter stores: write even and odd indices separately
        offs_v_half = tl.arange(0, V_HALF)
        o_base = Out + q_pos * stride_o_tok

        # Store even indices: out[h, 2*j]
        o_even_ptrs = o_base + offs_h[:, None] * stride_o_h + (offs_v_half[None, :] * 2)
        tl.store(o_even_ptrs, acc_even.to(Out.type.element_ty))

        # Store odd indices: out[h, 2*j + 1]
        o_odd_ptrs = o_base + offs_h[:, None] * stride_o_h + (offs_v_half[None, :] * 2 + 1)
        tl.store(o_odd_ptrs, acc_odd.to(Out.type.element_ty))


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def mla_decode_mxfp4(
    q: torch.Tensor,           # [batch, nheads, qk_dim] bf16
    kv_data: torch.Tensor,     # [total_kv, qk_dim // 2] uint8 (fp4x2)
    kv_scale: torch.Tensor,    # [total_kv_padded, num_groups_padded] uint8 (e8m0)
    qo_indptr: torch.Tensor,   # [batch + 1] int32
    kv_indptr: torch.Tensor,   # [batch + 1] int32
    nheads: int,
    qk_dim: int,
    v_dim: int,
    sm_scale: float,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """Launch the MXFP4 MLA decode kernel."""
    total_q = q.shape[0]
    batch_size = qo_indptr.shape[0] - 1

    # Pad QK dimension to next power of 2 for tl.arange compatibility
    qk_dim_pad = _next_pow2(qk_dim)
    qk_half = qk_dim // 2
    qk_half_pad = qk_dim_pad // 2
    num_qk_groups = qk_dim // 32
    num_qk_groups_pad = _next_pow2(num_qk_groups)

    # Pad Q: [total_q, nheads, qk_dim] -> [total_q, nheads, qk_dim_pad]
    if qk_dim_pad != qk_dim:
        q_pad = torch.zeros(total_q, nheads, qk_dim_pad, dtype=q.dtype, device=q.device)
        q_pad[:, :, :qk_dim] = q
    else:
        q_pad = q

    # Pad KV data: [total_kv, qk_half] -> [total_kv, qk_half_pad]
    if qk_half_pad != qk_half:
        kv_data_pad = torch.zeros(kv_data.shape[0], qk_half_pad, dtype=kv_data.dtype, device=kv_data.device)
        kv_data_pad[:, :qk_half] = kv_data
    else:
        kv_data_pad = kv_data

    # Pad KV scale: [total_kv, num_qk_groups] -> [total_kv, num_qk_groups_pad]
    if num_qk_groups_pad != num_qk_groups:
        kv_scale_pad = torch.zeros(kv_scale.shape[0], num_qk_groups_pad, dtype=kv_scale.dtype, device=kv_scale.device)
        kv_scale_pad[:, :num_qk_groups] = kv_scale
    else:
        kv_scale_pad = kv_scale

    out = torch.empty((total_q, nheads, v_dim), dtype=torch.bfloat16, device=q.device)

    grid = (batch_size,)
    _mla_decode_mxfp4_v3[grid](
        q_pad, kv_data_pad, kv_scale_pad, out,
        qo_indptr, kv_indptr,
        sm_scale,
        q_pad.stride(0), q_pad.stride(1),
        kv_data_pad.stride(0),
        kv_scale_pad.stride(0),
        out.stride(0), out.stride(1),
        NHEADS=nheads,
        QK_DIM=qk_dim_pad,
        V_DIM=v_dim,
        BLOCK_N=BLOCK_N,
    )
    return out


# ── Reference implementation for correctness testing ──

def _dequant_mxfp4_torch(data_u8: torch.Tensor, scale_u8: torch.Tensor, dim: int) -> torch.Tensor:
    """Dequant MXFP4 to float32. data_u8: [N, dim//2], scale_u8: [N, dim//32]."""
    FP4_LUT_T = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=data_u8.device)

    lo = data_u8 & 0x0F
    hi = (data_u8 >> 4) & 0x0F

    lo_sign = ((lo >> 3) & 1).float() * (-2.0) + 1.0
    lo_mag = (lo & 0x07).long()
    hi_sign = ((hi >> 3) & 1).float() * (-2.0) + 1.0
    hi_mag = (hi & 0x07).long()

    lo_val = lo_sign * FP4_LUT_T[lo_mag]
    hi_val = hi_sign * FP4_LUT_T[hi_mag]

    # Scale: e8m0 -> float32 = 2^(val - 127)
    scale_f32 = torch.pow(2.0, scale_u8.float() - 127.0)
    # Expand: each group covers 16 bytes (32 elements, 16 pairs)
    half_group = 16
    n_groups = scale_f32.shape[1]
    scale_expanded = scale_f32.unsqueeze(2).expand(-1, n_groups, half_group).reshape(-1, dim // 2)

    lo_scaled = lo_val * scale_expanded
    hi_scaled = hi_val * scale_expanded

    # Interleave: result[i, 2j] = lo[i,j], result[i, 2j+1] = hi[i,j]
    N = data_u8.shape[0]
    result = torch.empty(N, dim, device=data_u8.device, dtype=torch.float32)
    result[:, 0::2] = lo_scaled
    result[:, 1::2] = hi_scaled
    return result


def mla_decode_reference(
    q: torch.Tensor,           # [batch, nheads, qk_dim] bf16
    kv_data: torch.Tensor,     # [total_kv, qk_dim // 2] uint8 (fp4x2)
    kv_scale: torch.Tensor,    # [total_kv_padded, num_groups_padded] uint8 (e8m0)
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    nheads: int,
    qk_dim: int,
    v_dim: int,
    sm_scale: float,
) -> torch.Tensor:
    """Pure-torch reference: dequant KV to f32, then standard attention."""
    batch_size = qo_indptr.shape[0] - 1
    total_kv = kv_data.shape[0]
    num_qk_groups = qk_dim // 32
    num_v_groups = v_dim // 32

    # Dequant full KV buffer
    kv_f32 = _dequant_mxfp4_torch(kv_data, kv_scale[:total_kv, :num_qk_groups], qk_dim)
    # K is full qk_dim, V is first v_dim
    K_f32 = kv_f32  # [total_kv, qk_dim]
    V_f32 = _dequant_mxfp4_torch(
        kv_data[:, :v_dim // 2],
        kv_scale[:total_kv, :num_v_groups],
        v_dim
    )  # [total_kv, v_dim]

    outputs = []
    for b in range(batch_size):
        q_start = qo_indptr[b].item()
        q_end = qo_indptr[b + 1].item()
        kv_start = kv_indptr[b].item()
        kv_end = kv_indptr[b + 1].item()

        q_b = q[q_start:q_end].float()  # [q_len, nheads, qk_dim]
        k_b = K_f32[kv_start:kv_end]    # [kv_len, qk_dim]
        v_b = V_f32[kv_start:kv_end]    # [kv_len, v_dim]

        for qi in range(q_b.shape[0]):
            # [nheads, qk_dim] @ [qk_dim, kv_len] -> [nheads, kv_len]
            scores = torch.matmul(q_b[qi], k_b.T) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            # [nheads, kv_len] @ [kv_len, v_dim] -> [nheads, v_dim]
            out = torch.matmul(attn, v_b)
            outputs.append(out)

    return torch.stack(outputs).to(torch.bfloat16)


def _quantize_mxfp4_torch(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [N, D] float tensor to MXFP4. Returns (fp4x2 [N, D//2], scale [N, D//32])."""
    FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device)
    N, D = tensor.shape
    assert D % 32 == 0

    t = tensor.float()
    # Reshape to groups of 32
    t_grouped = t.reshape(N, D // 32, 32)
    # Compute scale: max abs per group -> e8m0
    amax = t_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    # e8m0: 2^floor(log2(amax)) rounded up to cover the range
    log2_amax = torch.floor(torch.log2(amax)) + 2  # +2 because max fp4 magnitude is 6.0
    log2_amax = log2_amax.clamp(-127, 127)
    scale_e8m0 = (log2_amax.long() + 127).clamp(0, 254).to(torch.uint8)
    scale_f32 = torch.pow(2.0, log2_amax)

    # Quantize: find nearest fp4 value
    t_scaled = t_grouped / scale_f32
    signs = (t_scaled < 0).to(torch.uint8)
    t_abs = t_scaled.abs()

    # Find nearest magnitude
    diffs = (t_abs.unsqueeze(-1) - FP4_VALUES.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs()
    mag_idx = diffs.argmin(dim=-1).to(torch.uint8)  # 0-7

    fp4_vals = (signs << 3) | mag_idx  # [N, D//32, 32] uint8, 0-15 each

    # Pack pairs into fp4x2: byte j = element[2j] | (element[2j+1] << 4)
    fp4_vals = fp4_vals.reshape(N, D // 2, 2)
    fp4x2 = fp4_vals[:, :, 0] | (fp4_vals[:, :, 1] << 4)

    return fp4x2.to(torch.uint8), scale_e8m0.reshape(N, D // 32)


if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = "cuda"

    # Kimi K2.5 MLA params (TP8)
    NHEADS = 16
    QK_DIM = 576
    V_DIM = 512
    BATCH = 4
    KV_SEQLEN = 256  # small for correctness test

    print(f"Testing MXFP4 MLA decode: batch={BATCH}, nheads={NHEADS}, "
          f"qk_dim={QK_DIM}, v_dim={V_DIM}, kv_seqlen={KV_SEQLEN}")

    # Generate random Q and KV in bf16, then quantize KV to MXFP4
    total_q = BATCH
    q = torch.randn(total_q, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device)

    total_kv = BATCH * KV_SEQLEN
    kv_bf16 = torch.randn(total_kv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
    kv_indptr = (torch.arange(BATCH + 1, dtype=torch.int32, device=device) * KV_SEQLEN)

    # Quantize KV to MXFP4
    num_qk_groups = QK_DIM // 32
    kv_fp4, kv_scale = _quantize_mxfp4_torch(kv_bf16)
    # kv_fp4: [total_kv, QK_DIM // 2] uint8
    # kv_scale: [total_kv, num_qk_groups] uint8

    sm_scale = 1.0 / (QK_DIM ** 0.5)

    # Reference
    print("Running reference...")
    out_ref = mla_decode_reference(
        q, kv_fp4, kv_scale, qo_indptr, kv_indptr,
        NHEADS, QK_DIM, V_DIM, sm_scale,
    )

    # Triton kernel
    print("Running Triton kernel...")
    out_tri = mla_decode_mxfp4(
        q, kv_fp4, kv_scale, qo_indptr, kv_indptr,
        NHEADS, QK_DIM, V_DIM, sm_scale,
        BLOCK_N=64,
    )

    # Compare
    cos_sim = torch.nn.functional.cosine_similarity(
        out_ref.float().reshape(-1), out_tri.float().reshape(-1), dim=0
    ).item()
    max_diff = (out_ref.float() - out_tri.float()).abs().max().item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Max abs diff: {max_diff:.6f}")

    if cos_sim > 0.99:
        print("PASS - correctness looks good")
    else:
        print("FAIL - cosine similarity too low")
        sys.exit(1)

    # ── Benchmark ──
    print("\n=== Benchmarking ===")
    for batch, kv_sl in [(1, 4096), (4, 4096), (32, 4096), (61, 4096), (128, 4096)]:
        total_q_b = batch
        total_kv_b = batch * kv_sl
        q_b = torch.randn(total_q_b, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
        qo_b = torch.arange(batch + 1, dtype=torch.int32, device=device)
        kv_bf16_b = torch.randn(total_kv_b, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
        kv_b = (torch.arange(batch + 1, dtype=torch.int32, device=device) * kv_sl)
        kv_fp4_b, kv_scale_b = _quantize_mxfp4_torch(kv_bf16_b)

        # Warmup
        for _ in range(5):
            mla_decode_mxfp4(q_b, kv_fp4_b, kv_scale_b, qo_b, kv_b,
                             NHEADS, QK_DIM, V_DIM, sm_scale, BLOCK_N=64)
        torch.cuda.synchronize()

        # Timed
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        N_ITER = 20
        start.record()
        for _ in range(N_ITER):
            mla_decode_mxfp4(q_b, kv_fp4_b, kv_scale_b, qo_b, kv_b,
                             NHEADS, QK_DIM, V_DIM, sm_scale, BLOCK_N=64)
        end.record()
        torch.cuda.synchronize()
        elapsed_us = start.elapsed_time(end) * 1000.0 / N_ITER

        # Bandwidth: read Q + KV_fp4 + KV_scale + write O
        q_bytes = total_q_b * NHEADS * QK_DIM * 2
        kv_bytes = total_kv_b * (QK_DIM // 2)  # fp4x2
        ks_bytes = total_kv_b * (QK_DIM // 32)  # scales
        o_bytes = total_q_b * NHEADS * V_DIM * 2
        total_bytes = q_bytes + kv_bytes + ks_bytes + o_bytes
        bw_tb_s = total_bytes / (elapsed_us * 1e-6) / 1e12

        print(f"  bs={batch:4d} kv={kv_sl:5d}: {elapsed_us:8.1f} μs  "
              f"BW={bw_tb_s:.2f} TB/s  ({total_bytes/1e6:.1f} MB)")
