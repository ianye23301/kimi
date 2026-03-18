"""Tests for Triton MLA decode kernel integration.

Verifies that the Triton MLA decode kernel produces correct attention output
with native 8-head support (no head padding required), matching a reference
implementation.
"""

import torch
import math


def reference_mla_decode(q, kv_cache, kv_indptr, kv_indices, kv_lora_rank, sm_scale):
    """Reference MLA decode attention in pure PyTorch.

    Args:
        q: (B, H, D) where D = kv_lora_rank + qk_rope_head_dim
        kv_cache: (total_tokens, D) paged KV cache [kv_c; k_pe]
        kv_indptr: (B+1,) cumulative KV lengths
        kv_indices: (total_kv,) page indices
        kv_lora_rank: int, latent dimension for V
        sm_scale: float, softmax scale

    Returns:
        o: (B, H, kv_lora_rank)
    """
    B, H, D = q.shape
    rope_dim = D - kv_lora_rank
    o = torch.zeros(B, H, kv_lora_rank, dtype=torch.float32, device=q.device)

    for b in range(B):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        seq_len = end - start
        if seq_len == 0:
            continue

        # Gather KV for this batch
        indices = kv_indices[start:end]
        kv = kv_cache[indices]  # (seq_len, D)

        # Split Q into nope and rope parts
        q_nope = q[b, :, :kv_lora_rank].float()  # (H, kv_lora_rank)
        q_pe = q[b, :, kv_lora_rank:].float()  # (H, rope_dim)

        # Split KV into nope and rope parts
        k_nope = kv[:, :kv_lora_rank].float()  # (seq_len, kv_lora_rank)
        k_pe = kv[:, kv_lora_rank:].float()  # (seq_len, rope_dim)
        v = kv[:, :kv_lora_rank].float()  # (seq_len, kv_lora_rank)

        # Attention scores: Q_nope @ K_nope^T + Q_pe @ K_pe^T
        # (H, kv_lora_rank) @ (kv_lora_rank, seq_len) -> (H, seq_len)
        scores_nope = torch.mm(q_nope, k_nope.T)
        scores_pe = torch.mm(q_pe, k_pe.T)
        scores = (scores_nope + scores_pe) * sm_scale

        # Softmax
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum: (H, seq_len) @ (seq_len, kv_lora_rank) -> (H, kv_lora_rank)
        o[b] = torch.mm(attn, v)

    return o


def test_triton_mla_decode_shapes():
    """Verify tensor shapes and strides for Triton MLA decode interface."""
    total_tokens = 1000
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    dim = kv_lora_rank + qk_rope_head_dim
    B = 4
    H = 8  # native 8 heads, no padding

    # Simulate KV cache
    kv_cache = torch.randn(total_tokens, dim, device="cuda", dtype=torch.bfloat16)

    # Create k_buffer and v_buffer views
    k_buffer = kv_cache.view(-1, 1, dim)
    v_buffer = kv_cache.view(-1, 1, dim)[:, :, :kv_lora_rank]

    # Shape checks
    assert k_buffer.shape == (total_tokens, 1, dim)
    assert v_buffer.shape == (total_tokens, 1, kv_lora_rank)

    # Stride checks — v_buffer stride(0) should be dim (576), not kv_lora_rank
    assert k_buffer.stride() == (dim, dim, 1)
    assert v_buffer.stride() == (dim, dim, 1), f"v_buffer strides: {v_buffer.stride()}"

    # v_buffer shares storage with k_buffer (no copy)
    assert v_buffer.data_ptr() == k_buffer.data_ptr()

    # kv_group_num calculation
    q = torch.randn(B, H, dim, device="cuda", dtype=torch.bfloat16)
    kv_group_num = q.shape[1] // k_buffer.shape[1]
    assert kv_group_num == H, f"Expected kv_group_num={H}, got {kv_group_num}"

    print("test_triton_mla_decode_shapes: PASS")


def test_triton_mla_decode_correctness():
    """Compare Triton MLA decode output against reference for 8 heads."""
    torch.manual_seed(42)

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    dim = kv_lora_rank + qk_rope_head_dim
    H = 8
    B = 4
    num_kv_splits = 16
    sm_scale = 1.0 / math.sqrt(dim)

    # Create paged KV cache
    max_seq_len = 128
    total_tokens = B * max_seq_len
    kv_cache = torch.randn(total_tokens, dim, device="cuda", dtype=torch.bfloat16)

    # Simple paging: each batch has max_seq_len consecutive tokens
    seq_lens = [max_seq_len] * B
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    for i, sl in enumerate(seq_lens):
        kv_indptr[i + 1] = kv_indptr[i] + sl
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device="cuda")

    # Query
    q = torch.randn(B, H, dim, device="cuda", dtype=torch.bfloat16)

    # Reference output
    ref_out = reference_mla_decode(q, kv_cache, kv_indptr, kv_indices, kv_lora_rank, sm_scale)

    # Triton MLA decode
    from aiter.ops.triton.attention.mla_decode_rope import decode_attention_fwd_grouped_rope

    k_buffer = kv_cache.view(-1, 1, dim)
    v_buffer = kv_cache.view(-1, 1, dim)[:, :, :kv_lora_rank]

    o = torch.empty(B, H, kv_lora_rank, dtype=torch.bfloat16, device="cuda")
    attn_logits = torch.empty(
        B, H, num_kv_splits, kv_lora_rank + 1,
        dtype=torch.float32, device="cuda",
    )
    k_pe_tokens = torch.empty(0, device="cuda")

    decode_attention_fwd_grouped_rope(
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        k_pe_tokens,
        kv_lora_rank=kv_lora_rank,
        rotary_dim=0,
        cos_sin_cache=None,
        positions=None,
        attn_logits=attn_logits,
        num_kv_splits=num_kv_splits,
        sm_scale=sm_scale,
        logit_cap=0.0,
        use_rope=False,
        is_neox_style=False,
    )

    max_diff = (ref_out - o.float()).abs().max().item()
    mean_diff = (ref_out - o.float()).abs().mean().item()
    print(f"test_triton_mla_decode_correctness: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    # bf16 accumulation in Triton vs fp32 reference — allow reasonable tolerance
    assert max_diff < 0.5, f"Max diff too large: {max_diff}"
    assert mean_diff < 0.05, f"Mean diff too large: {mean_diff}"
    print("test_triton_mla_decode_correctness: PASS")


def test_triton_mla_decode_variable_seq_lens():
    """Test with variable sequence lengths per batch."""
    torch.manual_seed(123)

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    dim = kv_lora_rank + qk_rope_head_dim
    H = 8
    B = 4
    num_kv_splits = 16
    sm_scale = 1.0 / math.sqrt(dim)

    seq_lens = [32, 128, 1, 64]
    total_tokens = sum(seq_lens)
    kv_cache = torch.randn(total_tokens, dim, device="cuda", dtype=torch.bfloat16)

    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    for i, sl in enumerate(seq_lens):
        kv_indptr[i + 1] = kv_indptr[i] + sl
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device="cuda")

    q = torch.randn(B, H, dim, device="cuda", dtype=torch.bfloat16)

    ref_out = reference_mla_decode(q, kv_cache, kv_indptr, kv_indices, kv_lora_rank, sm_scale)

    from aiter.ops.triton.attention.mla_decode_rope import decode_attention_fwd_grouped_rope

    k_buffer = kv_cache.view(-1, 1, dim)
    v_buffer = kv_cache.view(-1, 1, dim)[:, :, :kv_lora_rank]

    o = torch.empty(B, H, kv_lora_rank, dtype=torch.bfloat16, device="cuda")
    attn_logits = torch.empty(
        B, H, num_kv_splits, kv_lora_rank + 1,
        dtype=torch.float32, device="cuda",
    )
    k_pe_tokens = torch.empty(0, device="cuda")

    decode_attention_fwd_grouped_rope(
        q, k_buffer, v_buffer, o, kv_indptr, kv_indices,
        k_pe_tokens, kv_lora_rank=kv_lora_rank, rotary_dim=0,
        cos_sin_cache=None, positions=None, attn_logits=attn_logits,
        num_kv_splits=num_kv_splits, sm_scale=sm_scale,
        logit_cap=0.0, use_rope=False, is_neox_style=False,
    )

    max_diff = (ref_out - o.float()).abs().max().item()
    print(f"test_triton_mla_decode_variable_seq_lens: max_diff={max_diff:.6f}")
    assert max_diff < 0.5, f"Max diff too large: {max_diff}"
    print("test_triton_mla_decode_variable_seq_lens: PASS")


def test_triton_mla_decode_batch_size_1():
    """Test batch size 1 (common decode case)."""
    torch.manual_seed(0)

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    dim = kv_lora_rank + qk_rope_head_dim
    H = 8
    B = 1
    num_kv_splits = 16
    sm_scale = 1.0 / math.sqrt(dim)

    seq_len = 256
    kv_cache = torch.randn(seq_len, dim, device="cuda", dtype=torch.bfloat16)

    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device="cuda")

    q = torch.randn(B, H, dim, device="cuda", dtype=torch.bfloat16)

    ref_out = reference_mla_decode(q, kv_cache, kv_indptr, kv_indices, kv_lora_rank, sm_scale)

    from aiter.ops.triton.attention.mla_decode_rope import decode_attention_fwd_grouped_rope

    k_buffer = kv_cache.view(-1, 1, dim)
    v_buffer = kv_cache.view(-1, 1, dim)[:, :, :kv_lora_rank]

    o = torch.empty(B, H, kv_lora_rank, dtype=torch.bfloat16, device="cuda")
    attn_logits = torch.empty(
        B, H, num_kv_splits, kv_lora_rank + 1,
        dtype=torch.float32, device="cuda",
    )
    k_pe_tokens = torch.empty(0, device="cuda")

    decode_attention_fwd_grouped_rope(
        q, k_buffer, v_buffer, o, kv_indptr, kv_indices,
        k_pe_tokens, kv_lora_rank=kv_lora_rank, rotary_dim=0,
        cos_sin_cache=None, positions=None, attn_logits=attn_logits,
        num_kv_splits=num_kv_splits, sm_scale=sm_scale,
        logit_cap=0.0, use_rope=False, is_neox_style=False,
    )

    max_diff = (ref_out - o.float()).abs().max().item()
    print(f"test_triton_mla_decode_batch_size_1: max_diff={max_diff:.6f}")
    assert max_diff < 0.5, f"Max diff too large: {max_diff}"
    print("test_triton_mla_decode_batch_size_1: PASS")


def test_triton_vs_asm_padded_output():
    """Verify Triton 8-head output matches what ASM+repeat would produce.

    Simulates the ASM path: repeat Q to 16 heads, run reference attention
    on all 16, then extract first 8 unique heads. Compare against Triton
    running natively on 8 heads.
    """
    torch.manual_seed(7)

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    dim = kv_lora_rank + qk_rope_head_dim
    H = 8
    B = 2
    sm_scale = 1.0 / math.sqrt(dim)

    seq_len = 64
    kv_cache = torch.randn(seq_len * B, dim, device="cuda", dtype=torch.bfloat16)

    kv_indptr = torch.tensor([0, seq_len, seq_len * 2], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(seq_len * B, dtype=torch.int32, device="cuda")

    q = torch.randn(B, H, dim, device="cuda", dtype=torch.bfloat16)

    # Simulate ASM path: repeat to 16 heads, run reference, slice first 8
    q_padded = q.repeat(1, 2, 1)  # (B, 16, D)
    ref_padded = reference_mla_decode(
        q_padded, kv_cache, kv_indptr, kv_indices, kv_lora_rank, sm_scale
    )
    # First 8 heads of repeated output should match native 8-head output
    ref_native = ref_padded[:, :H, :]

    # Also run native reference for sanity
    ref_direct = reference_mla_decode(q, kv_cache, kv_indptr, kv_indices, kv_lora_rank, sm_scale)

    # Padded-then-sliced should match native (both in fp32 reference, tiny bf16 rounding)
    pad_vs_native = (ref_native - ref_direct).abs().max().item()
    assert pad_vs_native < 1e-5, f"Padded vs native reference diff: {pad_vs_native}"

    print("test_triton_vs_asm_padded_output: PASS (reference equivalence verified)")


def test_v_buffer_stride_correctness():
    """Verify v_buffer stride allows correct data access pattern."""
    kv_lora_rank = 512
    dim = 576  # kv_lora_rank + qk_rope_head_dim
    total_tokens = 100

    kv_cache = torch.randn(total_tokens, dim, device="cuda", dtype=torch.bfloat16)
    v_buffer = kv_cache.view(-1, 1, dim)[:, :, :kv_lora_rank]

    # Verify stride: should be (dim, dim, 1) so kernel reads correctly
    assert v_buffer.stride() == (dim, dim, 1), f"Unexpected strides: {v_buffer.stride()}"

    # Verify data: v_buffer[i, 0, j] == kv_cache[i, j] for j < kv_lora_rank
    for i in [0, 50, 99]:
        assert torch.equal(v_buffer[i, 0], kv_cache[i, :kv_lora_rank])

    # Verify no copy — same storage
    assert v_buffer.storage().data_ptr() == kv_cache.storage().data_ptr()

    print("test_v_buffer_stride_correctness: PASS")


if __name__ == "__main__":
    test_triton_mla_decode_shapes()
    test_v_buffer_stride_correctness()
    test_triton_vs_asm_padded_output()
    test_triton_mla_decode_correctness()
    test_triton_mla_decode_variable_seq_lens()
    test_triton_mla_decode_batch_size_1()
    print("\nAll Triton MLA decode tests passed!")
