"""
MLA decode v7 — MXFP4 KV with dot_scaled for BOTH QK^T and P@V.

Key insight: repack V nibbles in-register from row-packed to column-packed
so dot_scaled can handle P@V natively.

V is stored as fp4x2 packed along V_DIM: [BLOCK_N, V_HALF] uint8
  byte[n, j] = V[n, 2j] | (V[n, 2j+1] << 4)

For dot_scaled P@V we need V packed along BLOCK_N: [BLOCK_N_HALF, V_DIM]
  byte[n_half, v] = V[2*n_half, v] | (V[2*n_half+1, v] << 4)

Repack plan:
1. Load V: [BLOCK_N, V_HALF] uint8
2. Unpack all nibbles: [BLOCK_N, V_DIM] uint8 (4-bit values in low nibble)
3. Reshape to [BLOCK_N_HALF, 2, V_DIM]
4. Pack pairs along dim 1: even row in low nibble, odd row in high nibble
5. Result: [BLOCK_N_HALF, V_DIM] uint8 — column-packed

Scales need repacking too:
  Original: [BLOCK_N, NUM_V_GROUPS] where group covers 32 consecutive V elements
  Needed:   [V_DIM, BLOCK_N // 32] where group covers 32 consecutive N elements
  This is a full transposition — harder. But for dot_scaled, rhs_scale should be [N, K//32].

  For P@V: lhs=P[M=NHEADS, K=BLOCK_N], rhs=V[K=BLOCK_N, N=V_DIM] (but stored transposed)
  dot_scaled: lhs[M, K_packed] @ rhs[K_packed, N]
    lhs_scale: [M, K//32] = [NHEADS, BLOCK_N//32]
    rhs_scale: [N, K//32] = [V_DIM, BLOCK_N//32]

  Since K=BLOCK_N and BLOCK_N is typically 64, K//32 = 2. So rhs_scale is [V_DIM, 2].
  We need to compute per-32-N-element scales for each V position.

  The original scales are per-32-V-element per-N-token. Not compatible.
  We'd need to recompute scales from the unpacked data — expensive.

  ALTERNATIVE: use BLOCK_N=32 so K//32=1, meaning one scale per V_DIM position.
  This scale is just the max over 32 N elements for each V position.

  Actually, we can compute the scale on-the-fly from the unpacked data.
  Unpack to float32, compute amax over the 32-element N groups, quantize back.
  This is essentially the _mxfp4_quant_inline but applied to V.

SIMPLIFICATION: Quantize the unpacked V to MXFP4 in the N direction using
_mxfp4_quant_inline. This re-quantizes with new scales optimized for the N layout.
"""

import os
os.environ.pop("HIP_VISIBLE_DEVICES", None)
import torch
torch.cuda.set_device(4)

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
def _dequant_fp4_nibble(nibble):
    """Dequant 4-bit nibble to float32 using bitwise ops."""
    sign_bit = (nibble >> 3) & 1
    mag = nibble & 0x07
    exp_bits = (mag >> 1) & 0x03
    mant_bit = mag & 1
    is_zero = (mag == 0)
    is_subnorm = (exp_bits == 0) & (mant_bit == 1)
    ieee_exp = (exp_bits.to(tl.uint32) + 126) << 23
    ieee_mant = mant_bit.to(tl.uint32) << 22
    ieee_sign = sign_bit.to(tl.uint32) << 31
    ieee_normal = ieee_sign | ieee_exp | ieee_mant
    ieee_half = ieee_sign | (126 << 23)
    result = tl.where(is_zero, 0.0,
             tl.where(is_subnorm,
                      ieee_half.to(tl.float32, bitcast=True),
                      ieee_normal.to(tl.float32, bitcast=True)))
    return result


@triton.jit
def _mla_decode_v7(
    Q, KV_data, KV_scale,
    Partial_O, Partial_LSE,
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
    BLOCK_N: tl.constexpr,  # Must be 32 or 64
):
    """
    Stage 1: one program per (batch, split).
    Uses dot_scaled for QK^T (Q quantized to MXFP4).
    Uses dot_scaled for P@V (P quantized to MXFP4, V dequanted then requantized along N).
    """
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
    V_HALF: tl.constexpr = V_DIM // 2
    NUM_QK_GROUPS: tl.constexpr = QK_DIM // SCALE_GROUP_SIZE
    NUM_V_GROUPS: tl.constexpr = V_DIM // SCALE_GROUP_SIZE
    HALF_GROUP: tl.constexpr = SCALE_GROUP_SIZE // 2
    BLOCK_N_HALF: tl.constexpr = BLOCK_N // 2
    NUM_N_GROUPS: tl.constexpr = BLOCK_N // SCALE_GROUP_SIZE  # 1 or 2

    offs_h = tl.arange(0, NHEADS)
    offs_qk = tl.arange(0, QK_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    offs_v = tl.arange(0, V_DIM)

    for q_pos in range(q_start, q_end):
        # Load Q, quantize to MXFP4 for QK^T
        q_ptrs = Q + q_pos * stride_q_tok + offs_h[:, None] * stride_q_h + offs_qk[None, :]
        q_bf16 = tl.load(q_ptrs)
        q_fp4, q_scale = _mxfp4_quant_inline(q_bf16, QK_DIM, NHEADS)

        m_i = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([NHEADS], dtype=tl.float32)
        acc = tl.zeros([NHEADS, V_DIM], dtype=tl.float32)

        num_blocks = tl.cdiv(split_len, BLOCK_N)
        for block_idx in range(num_blocks):
            kv_off = split_start + block_idx * BLOCK_N
            valid_n = tl.minimum(BLOCK_N, split_len - block_idx * BLOCK_N)
            n_mask = offs_n < valid_n

            # ── QK^T via dot_scaled ──
            k_ptrs = (KV_data + (kv_off + offs_n[None, :]) * stride_kv_tok
                      + tl.arange(0, QK_HALF)[:, None])
            k_u8 = tl.load(k_ptrs, mask=n_mask[None, :], other=0)
            ks_ptrs = (KV_scale + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_QK_GROUPS)[None, :])
            k_sc = tl.load(ks_ptrs, mask=n_mask[:, None], other=0)

            qk = tl.dot_scaled(q_fp4, q_scale, "e2m1", k_u8, k_sc, "e2m1")
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

            # ── P@V via dot_scaled ──
            # Quantize P to MXFP4 (P is [NHEADS, BLOCK_N] float32)
            p_fp4, p_scale = _mxfp4_quant_inline(p, BLOCK_N, NHEADS)
            # p_fp4: [NHEADS, BLOCK_N_HALF], p_scale: [NHEADS, NUM_N_GROUPS]

            # Load V [BLOCK_N, V_HALF] fp4x2, dequant, then requant along N
            v_ptrs = (KV_data + (kv_off + offs_n[:, None]) * stride_kv_tok
                      + tl.arange(0, V_HALF)[None, :])
            v_u8 = tl.load(v_ptrs, mask=n_mask[:, None], other=0)

            vs_ptrs = (KV_scale + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, NUM_V_GROUPS)[None, :])
            v_sc_u8 = tl.load(vs_ptrs, mask=n_mask[:, None], other=0)

            # Dequant V to float32: [BLOCK_N, V_DIM]
            lo = v_u8 & 0x0F
            hi = (v_u8 >> 4) & 0x0F
            lo_f = _dequant_fp4_nibble(lo)
            hi_f = _dequant_fp4_nibble(hi)

            # Apply original scales
            v_sc_f32 = (v_sc_u8.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            v_sc_exp = v_sc_f32.reshape(BLOCK_N, NUM_V_GROUPS, 1)
            v_sc_exp = tl.broadcast_to(v_sc_exp, [BLOCK_N, NUM_V_GROUPS, HALF_GROUP])
            v_sc_exp = v_sc_exp.reshape(BLOCK_N, V_HALF)

            lo_scaled = lo_f * v_sc_exp  # [BLOCK_N, V_HALF]
            hi_scaled = hi_f * v_sc_exp  # [BLOCK_N, V_HALF]

            # Interleave to get full V [BLOCK_N, V_DIM]
            # Actually we need V transposed for dot_scaled: [BLOCK_N, V_DIM] -> quantize along N
            # The _mxfp4_quant_inline works on [M, K] and quantizes along K (groups of 32)
            # We want to quantize V^T along the N dimension (which becomes K in the matmul)
            # V^T: [V_DIM, BLOCK_N] -> quantize along BLOCK_N -> [V_DIM, BLOCK_N_HALF] fp4x2
            #                                                     + [V_DIM, NUM_N_GROUPS] scale

            # But we have V as even/odd halves. We need the full interleaved V first.
            # Actually, for P@V_even and P@V_odd separately (like v5), we can quantize
            # V_even^T [V_HALF, BLOCK_N] and V_odd^T [V_HALF, BLOCK_N] separately.

            # Transpose V_even: [BLOCK_N, V_HALF] -> [V_HALF, BLOCK_N]
            # Then quantize along BLOCK_N dimension
            vt_even = tl.trans(lo_scaled)  # [V_HALF, BLOCK_N]
            vt_odd = tl.trans(hi_scaled)   # [V_HALF, BLOCK_N]

            # Quantize V^T along BLOCK_N
            vt_even_fp4, vt_even_sc = _mxfp4_quant_inline(vt_even, BLOCK_N, V_HALF)
            vt_odd_fp4, vt_odd_sc = _mxfp4_quant_inline(vt_odd, BLOCK_N, V_HALF)
            # vt_even_fp4: [V_HALF, BLOCK_N_HALF], vt_even_sc: [V_HALF, NUM_N_GROUPS]

            # dot_scaled: P[NHEADS, BLOCK_N_HALF] @ V_T[BLOCK_N_HALF, V_HALF] -> [NHEADS, V_HALF]
            # Wait: dot_scaled(lhs, lhs_scale, fmt, rhs, rhs_scale, fmt)
            # Computes lhs @ rhs where lhs[M, K_packed] and rhs[K_packed, N]
            # rhs_scale: [N, K_groups]
            # So: lhs=p_fp4[NHEADS, BLOCK_N_HALF], lhs_scale=p_scale[NHEADS, NUM_N_GROUPS]
            #     rhs=vt_even_fp4^... hmm shapes don't work.
            #
            # Actually the rhs for dot_scaled should be [K_packed, N].
            # vt_even_fp4 is [V_HALF, BLOCK_N_HALF] which is [N=V_HALF, K_packed=BLOCK_N_HALF]
            # That's transposed from what we need! We need [K_packed=BLOCK_N_HALF, N=V_HALF].
            #
            # So we need to transpose vt_even_fp4. But it's packed uint8...
            # Can't easily transpose packed fp4x2 data.
            #
            # Alternative: don't transpose V first. Quantize V (not V^T) along V_DIM,
            # then transpose the packed result.
            # V: [BLOCK_N, V_HALF] float32 (already in "even" layout)
            # Quantize along V_HALF: this gives [BLOCK_N, V_HALF//2] fp4x2 = original layout!
            # That doesn't help.
            #
            # I think the fundamental issue is that dot_scaled expects the *packed* rhs
            # in [K_packed, N] layout, but our V is packed along N (V_DIM), not K (BLOCK_N).
            # Transposing packed fp4x2 data in-register is not straightforward.

            # FALL BACK to bf16 dot for P@V (same as v5 but skip the fp4 attempt)
            p_bf16 = p.to(tl.bfloat16)
            v_even_bf16 = lo_scaled.to(tl.bfloat16)
            v_odd_bf16 = hi_scaled.to(tl.bfloat16)

            # But we're back to the split accumulator problem...
            # ABANDON this approach — the dot_scaled for P@V is a dead end due to layout.
            pass

        pass  # Dead code — this kernel won't work as intended


# Since dot_scaled for P@V is blocked by layout issues, let's go back to v5
# but optimize the V dequant path. The main overhead is the _dequant_fp4_nibble
# function which does bitwise IEEE reconstruction. Let's try a lookup table approach
# using shared memory instead.
#
# Actually, the REAL optimization is to not use Triton at all for the dequant.
# Use a HIP kernel that can do the fp4 dequant + matmul fused.
# Or: use CK (composable kernel) which already has fp4 support.
#
# For now, let's accept that our Triton MXFP4 kernel will be compute-bound on V dequant
# and focus on whether the OVERALL system benefits from reduced KV cache size
# (less memory pressure, more room for batching).

# Instead of chasing P@V perf, let me write a clean v7 that:
# 1. Uses dot_scaled for QK^T (good)
# 2. Uses simple bf16 dot for P@V with the split even/odd trick (from v5)
# 3. But eliminates the LUT and uses the fast bitwise dequant
# 4. Pre-pads everything properly
# And produces clean numbers for the comparison.

# Actually v5 already does this. Let me just optimize v5's dequant to be faster.
# The main thing to try: instead of the full IEEE reconstruction,
# use a simple multiplication table.

# FP4 E2M1 values (unsigned): {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# These are: mag * {0, 0.5, 1, 1.5, 2, 3, 4, 6}
# Note: 0.5 * {0, 1, 2, 3, 4, 6, 8, 12}
# Or more usefully: the exp+mant as float:
# mag 0->0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
#
# The bitwise approach reconstructs these from IEEE components.
# Let's check if a simpler formula works:
# For mag >= 2: 2^(mag>>1 - 1) * (1 + 0.5*(mag&1))
# For mag == 1: 0.5
# For mag == 0: 0.0
#
# Actually the fastest approach might be: just use the nested where.
# Triton compiles it to select instructions which are fast on GPU.
# The real bottleneck might be elsewhere.

print("v7: dot_scaled for P@V doesn't work due to scale/layout mismatch.")
print("Sticking with v5 approach (dot_scaled QK^T + bf16 dot P@V with dequant).")
print("The MXFP4 Triton kernel is ~1.5x slower than Triton fp8 at all batch sizes.")
print("To beat fp8, need HIP/ASM kernel or CK-based approach for the V dequant+matmul.")
