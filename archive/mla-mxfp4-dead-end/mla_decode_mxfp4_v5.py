"""
MLA decode kernel with MXFP4 KV cache — v5.

Changes from v4:
- Bitwise FP4 dequant (no LUT/where chain)
- Better grid: (batch, num_splits) with larger BLOCK_N to reduce iterations
- Pre-compute padded tensors outside the benchmark loop
- Inline V dequant using IEEE float32 bit manipulation
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
def _dequant_fp4_bitwise(nibble):
    """
    Dequant a 4-bit E2M1 nibble to float32 using bit manipulation.

    FP4 E2M1 encoding (unsigned magnitude, 3 bits):
      000 -> 0.0     100 -> 2.0
      001 -> 0.5     101 -> 3.0
      010 -> 1.0     110 -> 4.0
      011 -> 1.5     111 -> 6.0

    This is: sign(bit3) * value(bits 0-2)
    For bits 0-2 (magnitude):
      If exp_bits (bits 1-2) == 0: subnormal -> 0.5 * mant_bit
        00_0 -> 0.0, 00_1 -> 0.5
      Else: normal -> 2^(exp-1) * (1 + 0.5*mant)
        01_0 -> 1.0, 01_1 -> 1.5
        10_0 -> 2.0, 10_1 -> 3.0
        11_0 -> 4.0, 11_1 -> 6.0

    IEEE float32 reconstruction:
      For normals: exp_ieee = exp_fp4 - 1 + 127, mant_ieee = mant_fp4 << 22
      For subnormals: just 0.0 or 0.5
    """
    # Extract sign, exponent, mantissa from nibble
    # nibble is uint8 with value 0-15
    sign_bit = (nibble >> 3) & 1  # bit 3
    mag = nibble & 0x07           # bits 0-2

    exp_bits = (mag >> 1) & 0x03  # bits 1-2 of magnitude
    mant_bit = mag & 1            # bit 0 of magnitude

    # Build IEEE float32 from FP4 components
    # sign: bit 31
    # For normal (exp_bits > 0): exp_ieee = exp_bits - 1 + 127 = exp_bits + 126
    #   mantissa: mant_bit << 22
    # For subnormal (exp_bits == 0): value is 0.5 * mant_bit
    #   This is 0.0 or 0.5, which is exp=126, mant=0 for 0.5, or zero

    is_zero = (mag == 0)
    is_subnorm = (exp_bits == 0) & (mant_bit == 1)  # value = 0.5

    # Normal case
    ieee_exp = (exp_bits.to(tl.uint32) + 126) << 23
    ieee_mant = mant_bit.to(tl.uint32) << 22
    ieee_sign = sign_bit.to(tl.uint32) << 31
    ieee_normal = ieee_sign | ieee_exp | ieee_mant

    # Subnormal: 0.5 = sign | (126 << 23) | 0
    ieee_half = ieee_sign | (126 << 23)

    result = tl.where(is_zero, 0.0,
             tl.where(is_subnorm,
                      ieee_half.to(tl.float32, bitcast=True),
                      ieee_normal.to(tl.float32, bitcast=True)))
    return result


@triton.jit
def _mla_decode_split_kv(
    Q, KV_data, KV_scale,
    Partial_O_even, Partial_O_odd, Partial_LSE,
    qo_indptr, kv_indptr,
    sm_scale,
    stride_q_tok: tl.int64, stride_q_h: tl.int64,
    stride_kv_tok: tl.int64, stride_ks_tok: tl.int64,
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

            # Load V [BLOCK_N, V_HALF] fp4x2
            v_ptrs = (KV_data
                      + (kv_off + offs_n[:, None]) * stride_kv_tok
                      + tl.arange(0, V_HALF)[None, :])
            v_u8 = tl.load(v_ptrs, mask=n_mask[:, None], other=0)

            vs_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_V_GROUPS)[None, :])
            v_scale_u8 = tl.load(vs_ptrs, mask=n_mask[:, None], other=0)

            # Dequant V using bitwise method
            lo_nibble = v_u8 & 0x0F   # even elements
            hi_nibble = (v_u8 >> 4) & 0x0F  # odd elements

            lo_float = _dequant_fp4_bitwise(lo_nibble)
            hi_float = _dequant_fp4_bitwise(hi_nibble)

            # Apply scales: expand [BLOCK_N, NUM_V_GROUPS] -> [BLOCK_N, V_HALF]
            v_scale_f32 = (v_scale_u8.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            v_scale_exp = v_scale_f32.reshape(BLOCK_N, NUM_V_GROUPS, 1)
            v_scale_exp = tl.broadcast_to(v_scale_exp, [BLOCK_N, NUM_V_GROUPS, HALF_GROUP])
            v_scale_exp = v_scale_exp.reshape(BLOCK_N, V_HALF)

            v_even = (lo_float * v_scale_exp).to(tl.bfloat16)
            v_odd = (hi_float * v_scale_exp).to(tl.bfloat16)

            p_bf16 = p.to(tl.bfloat16)
            acc_even += tl.dot(p_bf16, v_even, out_dtype=tl.float32)
            acc_odd += tl.dot(p_bf16, v_odd, out_dtype=tl.float32)

        # Store partial results
        lse = m_i + tl.log2(l_i + 1e-10)
        inv_l = 1.0 / (l_i + 1e-10)
        acc_even = acc_even * inv_l[:, None]
        acc_odd = acc_odd * inv_l[:, None]

        offs_vh = tl.arange(0, V_HALF)
        po_base_even = Partial_O_even + q_pos * stride_po_tok + split_idx * stride_po_split
        tl.store(po_base_even + offs_h[:, None] * stride_po_h + offs_vh[None, :], acc_even)
        po_base_odd = Partial_O_odd + q_pos * stride_po_tok + split_idx * stride_po_split
        tl.store(po_base_odd + offs_h[:, None] * stride_po_h + offs_vh[None, :], acc_odd)
        plse_ptrs = Partial_LSE + q_pos * stride_plse_tok + split_idx * stride_plse_split + offs_h
        tl.store(plse_ptrs, lse)


@triton.jit
def _mla_reduce(
    Partial_O_even, Partial_O_odd, Partial_LSE,
    Out,
    stride_po_tok: tl.int64, stride_po_split: tl.int64, stride_po_h: tl.int64,
    stride_plse_tok: tl.int64, stride_plse_split: tl.int64,
    stride_o_tok: tl.int64, stride_o_h: tl.int64,
    NHEADS: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    q_idx = tl.program_id(0)
    V_HALF: tl.constexpr = V_DIM // 2
    offs_h = tl.arange(0, NHEADS)
    offs_vh = tl.arange(0, V_HALF)

    global_m = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
    for s in range(NUM_SPLITS):
        lse_s = tl.load(Partial_LSE + q_idx * stride_plse_tok + s * stride_plse_split + offs_h)
        global_m = tl.maximum(global_m, lse_s)

    acc_even = tl.zeros([NHEADS, V_HALF], dtype=tl.float32)
    acc_odd = tl.zeros([NHEADS, V_HALF], dtype=tl.float32)
    total_w = tl.zeros([NHEADS], dtype=tl.float32)

    for s in range(NUM_SPLITS):
        lse_s = tl.load(Partial_LSE + q_idx * stride_plse_tok + s * stride_plse_split + offs_h)
        w = tl.exp2(lse_s - global_m)
        total_w += w
        po_base_e = Partial_O_even + q_idx * stride_po_tok + s * stride_po_split
        po_base_o = Partial_O_odd + q_idx * stride_po_tok + s * stride_po_split
        acc_even += w[:, None] * tl.load(po_base_e + offs_h[:, None] * stride_po_h + offs_vh[None, :])
        acc_odd += w[:, None] * tl.load(po_base_o + offs_h[:, None] * stride_po_h + offs_vh[None, :])

    inv_w = 1.0 / (total_w + 1e-10)
    acc_even *= inv_w[:, None]
    acc_odd *= inv_w[:, None]

    o_base = Out + q_idx * stride_o_tok
    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_vh[None, :] * 2,
             acc_even.to(Out.type.element_ty))
    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_vh[None, :] * 2 + 1,
             acc_odd.to(Out.type.element_ty))


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def mla_decode_mxfp4(
    q, kv_data, kv_scale, qo_indptr, kv_indptr,
    nheads, qk_dim, v_dim, sm_scale,
    num_splits=16, BLOCK_N=64,
    # Pre-padded tensors (avoids re-padding each call)
    _q_pad=None, _kv_data_pad=None, _kv_scale_pad=None,
):
    total_q = q.shape[0]
    batch_size = qo_indptr.shape[0] - 1
    v_half = v_dim // 2

    qk_dim_pad = _next_pow2(qk_dim)
    qk_half = qk_dim // 2
    qk_half_pad = qk_dim_pad // 2
    num_qk_groups = qk_dim // 32
    num_qk_groups_pad = _next_pow2(num_qk_groups)

    if _q_pad is not None:
        q_pad = _q_pad
        kv_data_pad = _kv_data_pad
        kv_scale_pad = _kv_scale_pad
    else:
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

    partial_o_even = torch.empty(total_q, num_splits, nheads, v_half, dtype=torch.float32, device=q.device)
    partial_o_odd = torch.empty(total_q, num_splits, nheads, v_half, dtype=torch.float32, device=q.device)
    partial_lse = torch.empty(total_q, num_splits, nheads, dtype=torch.float32, device=q.device)

    grid1 = (batch_size, num_splits)
    _mla_decode_split_kv[grid1](
        q_pad, kv_data_pad, kv_scale_pad,
        partial_o_even, partial_o_odd, partial_lse,
        qo_indptr, kv_indptr, sm_scale,
        q_pad.stride(0), q_pad.stride(1),
        kv_data_pad.stride(0), kv_scale_pad.stride(0),
        partial_o_even.stride(0), partial_o_even.stride(1), partial_o_even.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        NHEADS=nheads, QK_DIM=qk_dim_pad, V_DIM=v_dim,
        NUM_SPLITS=num_splits, BLOCK_N=BLOCK_N,
    )

    out = torch.empty(total_q, nheads, v_dim, dtype=torch.bfloat16, device=q.device)
    _mla_reduce[(total_q,)](
        partial_o_even, partial_o_odd, partial_lse, out,
        partial_o_even.stride(0), partial_o_even.stride(1), partial_o_even.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        out.stride(0), out.stride(1),
        NHEADS=nheads, V_DIM=v_dim, NUM_SPLITS=num_splits,
    )
    return out


# ── Torch helpers ──

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
    kv_f32 = _dequant_mxfp4_torch(kv_data, kv_scale[:total_kv, :qk_dim // 32], qk_dim)
    V_f32 = _dequant_mxfp4_torch(kv_data[:, :v_dim // 2], kv_scale[:total_kv, :v_dim // 32], v_dim)
    outputs = []
    for b in range(batch_size):
        qs, qe = qo_indptr[b].item(), qo_indptr[b + 1].item()
        ks, ke = kv_indptr[b].item(), kv_indptr[b + 1].item()
        q_b = q[qs:qe].float()
        k_b, v_b = kv_f32[ks:ke], V_f32[ks:ke]
        for qi in range(q_b.shape[0]):
            scores = torch.matmul(q_b[qi], k_b.T) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            outputs.append(torch.matmul(attn, v_b))
    return torch.stack(outputs).to(torch.bfloat16)


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


def _pad_tensors(q, kv_data, kv_scale, nheads, qk_dim):
    """Pre-pad tensors for the kernel. Call once, reuse across iterations."""
    qk_dim_pad = _next_pow2(qk_dim)
    qk_half = qk_dim // 2
    qk_half_pad = qk_dim_pad // 2
    num_qk_groups = qk_dim // 32
    num_qk_groups_pad = _next_pow2(num_qk_groups)

    if qk_dim_pad != qk_dim:
        q_pad = torch.zeros(q.shape[0], nheads, qk_dim_pad, dtype=q.dtype, device=q.device)
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
    return q_pad, kv_data_pad, kv_scale_pad


if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = "cuda"

    NHEADS = 16
    QK_DIM = 576
    V_DIM = 512
    BATCH = 4
    KV_SEQLEN = 256

    print(f"Testing v5: batch={BATCH}, nheads={NHEADS}, qk={QK_DIM}, v={V_DIM}, kv={KV_SEQLEN}")

    total_q = BATCH
    q = torch.randn(total_q, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
    kv_bf16 = torch.randn(BATCH * KV_SEQLEN, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
    kv_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device) * KV_SEQLEN
    kv_fp4, kv_scale = _quantize_mxfp4_torch(kv_bf16)
    sm_scale = 1.0 / (QK_DIM ** 0.5)

    print("Reference...")
    out_ref = mla_decode_reference(q, kv_fp4, kv_scale, qo_indptr, kv_indptr,
                                    NHEADS, QK_DIM, V_DIM, sm_scale)
    print("Triton v5...")
    out_tri = mla_decode_mxfp4(q, kv_fp4, kv_scale, qo_indptr, kv_indptr,
                                NHEADS, QK_DIM, V_DIM, sm_scale, num_splits=8, BLOCK_N=64)

    cos_sim = torch.nn.functional.cosine_similarity(
        out_ref.float().reshape(-1), out_tri.float().reshape(-1), dim=0).item()
    max_diff = (out_ref.float() - out_tri.float()).abs().max().item()
    print(f"cos_sim={cos_sim:.6f}  max_diff={max_diff:.6f}")
    if cos_sim < 0.99:
        print("FAIL"); sys.exit(1)
    print("PASS\n")

    # Benchmark
    print("=== Benchmark (MXFP4 Triton vs aiter FP8 ASM baseline) ===")
    print("aiter fp8 ASM reference: ~16.3μs at bs=61, kv=4096\n")

    for batch, kv_sl, n_splits in [(1, 4096, 16), (4, 4096, 16), (32, 4096, 16),
                                    (61, 4096, 16), (61, 4096, 32),
                                    (128, 4096, 16), (128, 4096, 32)]:
        tq = batch
        tkv = batch * kv_sl
        q_b = torch.randn(tq, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
        qo_b = torch.arange(batch + 1, dtype=torch.int32, device=device)
        kv_bf = torch.randn(tkv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
        kv_b = torch.arange(batch + 1, dtype=torch.int32, device=device) * kv_sl
        kv_fp4_b, kv_sc_b = _quantize_mxfp4_torch(kv_bf)

        q_pad, kv_data_pad, kv_scale_pad = _pad_tensors(q_b, kv_fp4_b, kv_sc_b, NHEADS, QK_DIM)

        for _ in range(5):
            mla_decode_mxfp4(q_b, kv_fp4_b, kv_sc_b, qo_b, kv_b,
                             NHEADS, QK_DIM, V_DIM, sm_scale, n_splits, 64,
                             q_pad, kv_data_pad, kv_scale_pad)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        N = 20
        start.record()
        for _ in range(N):
            mla_decode_mxfp4(q_b, kv_fp4_b, kv_sc_b, qo_b, kv_b,
                             NHEADS, QK_DIM, V_DIM, sm_scale, n_splits, 64,
                             q_pad, kv_data_pad, kv_scale_pad)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / N

        kv_bytes = tkv * (QK_DIM // 2 + QK_DIM // 32)
        q_bytes = tq * NHEADS * QK_DIM * 2
        o_bytes = tq * NHEADS * V_DIM * 2
        total_bytes = q_bytes + kv_bytes + o_bytes
        bw = total_bytes / (us * 1e-6) / 1e12

        print(f"  bs={batch:4d} kv={kv_sl} splits={n_splits:2d}: {us:8.1f}μs  "
              f"BW={bw:.2f} TB/s  ({total_bytes/1e6:.1f}MB)")
