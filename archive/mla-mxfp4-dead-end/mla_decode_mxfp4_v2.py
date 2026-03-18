"""
MLA decode kernel with MXFP4 KV cache for MI355X (gfx950).

Uses tl.dot_scaled for QK^T (hardware MFMA with MXFP4).
For softmax@V, dequants V from MXFP4 to bf16 and uses tl.dot.

Grid: (batch_size,) — one program per batch element.
Each program processes all 16 Q heads × full KV sequence.
"""

import torch
import triton
import triton.language as tl
from aiter.utility.fp4_utils import dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32


SCALE_GROUP_SIZE: tl.constexpr = 32


@triton.jit
def _mxfp4_quant_inline(
    x,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Quantize bf16 [BLOCK_M, BLOCK_K] to mxfp4 in registers.
    Returns (fp4x2 [BLOCK_M, BLOCK_K//2], scales [BLOCK_M, BLOCK_K//32]).
    """
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1
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
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
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
def _mla_decode_mxfp4_kernel(
    Q,            # [total_q, NHEADS, QK_DIM] bf16
    KV_fp4,       # [total_kv, QK_DIM // 2] fp4x2 (squeezed kv_heads dim)
    KV_scale,     # [total_kv_padded, num_scale_blocks_padded] e8m0
    Out,          # [total_q, NHEADS, V_DIM] bf16
    qo_indptr,    # [batch + 1] int32
    kv_indptr,    # [batch + 1] int32
    sm_scale,     # float
    stride_q_tok, # Q stride per token
    stride_q_h,   # Q stride per head
    stride_kv_tok, # KV_fp4 stride per token
    stride_ks_tok, # KV_scale stride per token
    stride_o_tok, # Out stride per token
    stride_o_h,   # Out stride per head
    NHEADS: tl.constexpr,       # 16
    QK_DIM: tl.constexpr,       # 576
    V_DIM: tl.constexpr,        # 512
    BLOCK_N: tl.constexpr,      # KV tokens per block
):
    batch_idx = tl.program_id(0)
    log2e: tl.constexpr = 1.4426950408889634

    q_start = tl.load(qo_indptr + batch_idx)
    q_end = tl.load(qo_indptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)
    seqlen_kv = kv_end - kv_start

    NUM_SCALE_GROUPS: tl.constexpr = QK_DIM // SCALE_GROUP_SIZE  # 18
    V_SCALE_GROUPS: tl.constexpr = V_DIM // SCALE_GROUP_SIZE     # 16
    QK_DIM_HALF: tl.constexpr = QK_DIM // 2                     # 288
    V_DIM_HALF: tl.constexpr = V_DIM // 2                       # 256

    offs_h = tl.arange(0, NHEADS)
    offs_qk = tl.arange(0, QK_DIM)
    offs_qk_half = tl.arange(0, QK_DIM_HALF)
    offs_v = tl.arange(0, V_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    offs_scale = tl.arange(0, NUM_SCALE_GROUPS)

    sm_scale_log2 = sm_scale * log2e

    for q_pos in range(q_start, q_end):
        # Load Q: [NHEADS, QK_DIM] bf16
        q_ptrs = Q + q_pos * stride_q_tok + offs_h[:, None] * stride_q_h + offs_qk[None, :]
        q_bf16 = tl.load(q_ptrs)  # [NHEADS, QK_DIM] bf16

        # Quantize Q to MXFP4 in registers
        q_fp4, q_scale = _mxfp4_quant_inline(q_bf16, QK_DIM, NHEADS)
        # q_fp4: [NHEADS, QK_DIM//2], q_scale: [NHEADS, 18]

        # Online softmax state
        m_i = tl.full([NHEADS], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([NHEADS], dtype=tl.float32)
        acc = tl.zeros([NHEADS, V_DIM], dtype=tl.float32)

        num_blocks = tl.cdiv(seqlen_kv, BLOCK_N)

        for block_idx in range(num_blocks):
            kv_off = kv_start + block_idx * BLOCK_N
            valid_n = tl.minimum(BLOCK_N, seqlen_kv - block_idx * BLOCK_N)

            # ── Load K tile: [QK_DIM//2, BLOCK_N] fp4x2 (transposed for dot_scaled) ──
            # dot_scaled wants: A[M,K//2] @ B[K//2,N] where K is the shared dim
            # A = q_fp4 [NHEADS, QK_DIM//2], B = k_fp4 transposed
            # Actually dot_scaled does: A[M,K] @ B[K,N] but with packed fp4
            # So we need K as [QK_DIM//2, BLOCK_N] (K dim first, N second)
            k_ptrs = (KV_fp4
                      + (kv_off + offs_n[None, :]) * stride_kv_tok
                      + offs_qk_half[:, None])
            k_mask = offs_n[None, :] < valid_n
            k_fp4 = tl.load(k_ptrs, mask=k_mask, other=0)
            # k_fp4: [QK_DIM//2, BLOCK_N]

            # K scales: [BLOCK_N, NUM_SCALE_GROUPS] -> need [NUM_SCALE_GROUPS, BLOCK_N] for dot_scaled?
            # Actually looking at the sage attention kernel, k_descale has shape matching k
            # Let me check: in sage_fwd_mxfp4, k is loaded as [BLOCK_DMODEL_QK//2, BLOCK_N]
            # and k_descale is loaded as [BLOCK_N, num_scale_groups]
            ks_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + offs_scale[None, :])
            ks_mask = offs_n[:, None] < valid_n
            k_scale = tl.load(ks_ptrs, mask=ks_mask, other=0)
            # k_scale: [BLOCK_N, NUM_SCALE_GROUPS]

            # ── QK^T via tl.dot_scaled ──
            # q_fp4: [NHEADS, QK_DIM//2], q_scale: [NHEADS, 18]
            # k_fp4: [QK_DIM//2, BLOCK_N], k_scale: [BLOCK_N, 18]
            qk = tl.zeros([NHEADS, BLOCK_N], dtype=tl.float32)
            qk = tl.dot_scaled(
                q_fp4, q_scale, "e2m1",
                k_fp4, k_scale, "e2m1",
                acc=qk,
            )
            qk *= sm_scale

            # Mask invalid positions
            qk = tl.where(offs_n[None, :] < valid_n, qk, float("-inf"))

            # ── Online softmax ──
            m_ij = tl.max(qk, axis=1)  # [NHEADS]
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2((m_i - m_new) * log2e)
            p = tl.exp2((qk - m_new[:, None]) * log2e)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # ── Load V and compute softmax @ V ──
            # V = first 512 dims of KV, needs dequant from MXFP4
            # For V, we need bf16 for tl.dot
            # Load V fp4: [BLOCK_N, V_DIM//2]
            v_ptrs = (KV_fp4
                      + (kv_off + offs_n[:, None]) * stride_kv_tok
                      + tl.arange(0, V_DIM_HALF)[None, :])
            v_fp4 = tl.load(v_ptrs, mask=offs_n[:, None] < valid_n, other=0)

            # V scales: first 16 of 18 scale groups
            vs_ptrs = (KV_scale
                       + (kv_off + offs_n[:, None]) * stride_ks_tok
                       + tl.arange(0, V_SCALE_GROUPS)[None, :])
            v_scale_e8m0 = tl.load(vs_ptrs, mask=offs_n[:, None] < valid_n, other=0)

            # Software dequant V: fp4x2 -> bf16
            # Unpack fp4x2: each byte has low nibble (even) and high nibble (odd)
            v_u8 = v_fp4.to(tl.uint8)
            v_even = v_u8 & 0x0F           # [BLOCK_N, V_DIM//2]
            v_odd = (v_u8 >> 4) & 0x0F     # [BLOCK_N, V_DIM//2]

            # Interleave back to [BLOCK_N, V_DIM] uint8
            # v_unpacked[i, 2j] = v_even[i, j], v_unpacked[i, 2j+1] = v_odd[i, j]
            # For now, use the fp4 lookup table approach
            # FP4 E2M1 values: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
            # With sign bit: 8->-0, 9->-0.5, etc.
            # TODO: This is the expensive part — ideally use tl.dot_scaled for V too
            # but we need P (softmax probs) in mxfp4 format which loses too much precision

            # Alternative: use tl.dot_scaled for P@V too, quantizing P on the fly
            # P is [NHEADS, BLOCK_N] in fp32, V is [BLOCK_N, V_DIM//2] in fp4x2
            # This would be: p_fp4, p_scale = _mxfp4_quant_inline(p, BLOCK_N, NHEADS)
            # then tl.dot_scaled(p_fp4, p_scale, "e2m1", v_fp4_transposed, v_scale, "e2m1")

            # Let's try the P@V with dot_scaled approach!
            p_bf16 = p.to(tl.bfloat16)  # [NHEADS, BLOCK_N]
            p_fp4, p_scale = _mxfp4_quant_inline(p_bf16, BLOCK_N, NHEADS)

            # Need V transposed: [V_DIM//2, BLOCK_N] for dot_scaled
            # And V_scale transposed: [BLOCK_N, V_SCALE_GROUPS]
            # v_fp4 is already [BLOCK_N, V_DIM//2], need transpose
            v_fp4_t = tl.trans(v_fp4)  # [V_DIM//2, BLOCK_N]

            pv = tl.zeros([NHEADS, V_DIM], dtype=tl.float32)
            # Actually V_DIM=512 which is 16 groups of 32
            # dot_scaled: A[M, K//2] x B[K//2, N] with scales
            # P_fp4[NHEADS, BLOCK_N//2] x V_fp4_t[BLOCK_N//2, V_DIM]
            # Hmm, the shared dimension is BLOCK_N here
            # That means BLOCK_N must be the K dimension for dot_scaled
            # and it must be divisible by 32 (scale group size)

            # This only works if BLOCK_N is divisible by 32
            pv = tl.dot_scaled(
                p_fp4, p_scale, "e2m1",
                v_fp4_t, v_scale_e8m0, "e2m1",
                acc=pv,
            )

            acc += pv
            m_i = m_new

        # Normalize
        acc = acc / l_i[:, None]

        # Store output: [NHEADS, V_DIM]
        o_ptrs = Out + q_pos * stride_o_tok + offs_h[:, None] * stride_o_h + offs_v[None, :]
        tl.store(o_ptrs, acc.to(Out.type.element_ty))


def mla_decode_mxfp4(
    q: torch.Tensor,         # [total_q, nheads, 576] bf16
    kv_fp4: torch.Tensor,    # [total_kv, 1, 288] fp4x2
    kv_scale: torch.Tensor,  # [padded_kv, padded_blocks] e8m0
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float,
    v_dim: int = 512,
) -> torch.Tensor:
    total_q, nheads, qk_dim = q.shape
    batch_size = qo_indptr.shape[0] - 1

    out = torch.empty(total_q, nheads, v_dim, dtype=torch.bfloat16, device=q.device)

    # Squeeze KV heads dim: [total_kv, 1, 288] -> [total_kv, 288]
    # Cast to uint8 for Triton (fp4x2 is stored as uint8)
    kv_fp4_2d = kv_fp4.view(-1, qk_dim // 2).view(torch.uint8)
    # Also cast scale to uint8
    kv_scale_u8 = kv_scale.view(torch.uint8)

    BLOCK_N = 64  # Must be divisible by 32 for scale groups

    _mla_decode_mxfp4_kernel[(batch_size,)](
        q, kv_fp4_2d, kv_scale_u8, out,
        qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1),
        kv_fp4_2d.stride(0),
        kv_scale.stride(0),
        out.stride(0), out.stride(1),
        NHEADS=nheads,
        QK_DIM=qk_dim,
        V_DIM=v_dim,
        BLOCK_N=BLOCK_N,
    )
    return out


if __name__ == "__main__":
    import sys
    import torch.nn.functional as F
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.cuda.set_device(gpu)

    from aiter.utility.fp4_utils import dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32

    bs, kvsl, nheads = 4, 1024, 16
    dim, v_dim = 576, 512
    sm_scale = 1.0 / (dim ** 0.5)

    q = torch.randn(bs, nheads, dim, dtype=torch.bfloat16, device="cuda") * 0.02
    kv_bf16 = torch.randn(bs * kvsl, 1, dim, dtype=torch.bfloat16, device="cuda") * 0.02

    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda") * kvsl

    # Quantize KV
    kv_2d = kv_bf16.view(-1, dim)
    kv_fp4_2d, kv_scale = dynamic_mxfp4_quant(kv_2d)
    kv_fp4 = kv_fp4_2d.view(-1, 1, dim // 2)

    print(f"Testing MXFP4 MLA decode: bs={bs}, kvsl={kvsl}, nheads={nheads}")

    # Reference: dequant then attention
    total_kv = bs * kvsl
    float_vals = mxfp4_to_f32(kv_fp4_2d)
    num_blocks = dim // 32
    scale_f32 = e8m0_to_f32(kv_scale)[:total_kv, :num_blocks]
    float_blocked = float_vals.view(total_kv, num_blocks, 32)
    kv_dequant = (float_blocked * scale_f32.unsqueeze(-1)).view(total_kv, dim).to(torch.bfloat16)

    out_ref_list = []
    for i in range(bs):
        qi = q[i:i+1].float().permute(1, 0, 2)
        ki = kv_dequant[i*kvsl:(i+1)*kvsl].float()
        vi = ki[:, :v_dim]
        scores = torch.matmul(qi * sm_scale, ki.T)
        scores = F.softmax(scores, dim=-1)
        oi = torch.matmul(scores, vi)
        out_ref_list.append(oi.permute(1, 0, 2).to(torch.bfloat16))
    out_ref = torch.cat(out_ref_list, dim=0)

    # Our kernel
    try:
        out_ours = mla_decode_mxfp4(q, kv_fp4, kv_scale, qo_indptr, kv_indptr, sm_scale, v_dim)
        diff = (out_ref.float() - out_ours.float()).abs()
        print(f"  max diff:  {diff.max().item():.6f}")
        print(f"  mean diff: {diff.mean().item():.6f}")
        cos = F.cosine_similarity(out_ref.float().reshape(-1), out_ours.float().reshape(-1), dim=0)
        print(f"  cosine:    {cos.item():.6f}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
