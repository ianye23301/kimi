"""Test that heads-first output layout with repeat (tile) produces correct results.

The optimization:
1. Use q.repeat(1, factor, 1) instead of repeat_interleave → [h0..h7, h0..h7]
2. Allocate output as (padded_N, B, L) contiguous, permute to (B, padded_N, L) for kernel
3. Slice o_heads_first[:num_heads] for contiguous (num_heads, B, L) → no .contiguous() copy
4. Pass heads_first=True to _v_up_proj_and_o_proj to skip view+transpose
"""

import torch


def test_repeat_vs_interleave():
    """Verify repeat gives first num_heads slots as unique heads."""
    B, num_heads, L = 4, 8, 512
    factor = 2
    padded_N = num_heads * factor  # 16

    q = torch.randn(B, num_heads, L, device="cuda")

    # repeat_interleave: [h0,h0,h1,h1,...,h7,h7]
    q_interleave = q.repeat_interleave(factor, dim=1)
    assert q_interleave.shape == (B, padded_N, L)
    assert torch.equal(q_interleave[:, 0, :], q[:, 0, :])
    assert torch.equal(q_interleave[:, 1, :], q[:, 0, :])  # duplicate
    assert torch.equal(q_interleave[:, 2, :], q[:, 1, :])

    # repeat (tile): [h0,h1,...,h7,h0,h1,...,h7]
    q_repeat = q.repeat(1, factor, 1)
    assert q_repeat.shape == (B, padded_N, L)
    assert torch.equal(q_repeat[:, 0, :], q[:, 0, :])
    assert torch.equal(q_repeat[:, 1, :], q[:, 1, :])  # next unique head
    assert torch.equal(q_repeat[:, 7, :], q[:, 7, :])
    assert torch.equal(q_repeat[:, 8, :], q[:, 0, :])  # second copy starts

    print("repeat vs interleave: PASS")


def test_heads_first_strides():
    """Verify stride trick gives kernel-compatible (B, N, L) view."""
    padded_N, B, L = 16, 4, 512

    o_heads_first = torch.randn(padded_N, B, L, device="cuda")
    o = o_heads_first.permute(1, 0, 2)

    assert o.shape == (B, padded_N, L)
    assert o.stride() == (L, B * L, 1)
    assert not o.is_contiguous()

    # Write through permuted view, read from storage
    o.fill_(0)
    o[2, 5, 100] = 42.0
    assert o_heads_first[5, 2, 100] == 42.0

    print("heads-first strides: PASS")


def test_heads_first_contiguous_slice():
    """Verify [:num_heads] from heads-first storage is contiguous and correct."""
    padded_N, B, L = 16, 4, 512
    num_heads = 8

    o_heads_first = torch.randn(padded_N, B, L, device="cuda")

    real_heads = o_heads_first[:num_heads]
    assert real_heads.shape == (num_heads, B, L)
    assert real_heads.is_contiguous()

    print("heads-first contiguous slice: PASS")


def test_mla_output_equivalence():
    """Simulate MLA decode with both approaches and verify identical results.

    MLA: each query head attends independently to shared KV.
    With repeat(factor=2): output[0..7] = unique heads, output[8..15] = duplicates.
    With repeat_interleave(factor=2): output[0,2,4,...,14] = unique heads.
    """
    B, num_heads, L = 4, 8, 512
    factor = 2
    padded_N = 16

    q = torch.randn(B, num_heads, L, device="cuda")
    # Shared KV (same for all heads in MLA)
    kv = torch.randn(B, 100, L, device="cuda")  # 100 KV tokens

    # Simulate attention per head (simplified dot-product attention)
    def fake_mla(q_padded):
        """q_padded: (B, padded_N, L) → output: (B, padded_N, L)"""
        # Each head independently: softmax(q @ kv.T) @ kv
        scores = torch.bmm(
            q_padded.reshape(B * padded_N, 1, L),
            kv.unsqueeze(1).expand(B, padded_N, 100, L).reshape(B * padded_N, 100, L).transpose(1, 2),
        )  # (B*padded_N, 1, 100)
        attn = torch.softmax(scores / (L ** 0.5), dim=-1)
        out = torch.bmm(
            attn,
            kv.unsqueeze(1).expand(B, padded_N, 100, L).reshape(B * padded_N, 100, L),
        )  # (B*padded_N, 1, L)
        return out.reshape(B, padded_N, L)

    # Standard approach: repeat_interleave + [::factor] slice
    q_interleave = q.repeat_interleave(factor, dim=1)
    o_interleave = fake_mla(q_interleave)
    result_standard = o_interleave[:, ::factor, :].contiguous()  # (B, 8, L)

    # Heads-first approach: repeat (tile) + [:num_heads] slice
    q_tile = q.repeat(1, factor, 1)
    o_tile = fake_mla(q_tile)
    # The kernel writes to (padded_N, B, L) via permuted view,
    # but for this test we can just check the logical output
    result_hf = o_tile[:, :num_heads, :].contiguous()  # (B, 8, L)

    # Both should give the same 8 unique head outputs
    assert torch.allclose(result_standard, result_hf, atol=1e-5), (
        f"Max diff: {(result_standard - result_hf).abs().max()}"
    )
    print("MLA output equivalence: PASS")


def test_v_up_proj_equivalence():
    """Test that heads_first=True in _v_up_proj_and_o_proj matches standard."""
    num_heads = 8
    kv_lora_rank = 512
    v_head_dim = 128
    B = 4

    W_V = torch.randn(num_heads, kv_lora_rank, v_head_dim, device="cuda", dtype=torch.bfloat16)

    # Standard path: (B, N, L)
    x_standard = torch.randn(B, num_heads, kv_lora_rank, device="cuda", dtype=torch.bfloat16)

    # Heads-first: (N, B, L) — same data
    x_hf = x_standard.transpose(0, 1).contiguous()

    # Standard: view → transpose → BMM
    x_s = x_standard.view(-1, num_heads, kv_lora_rank).transpose(0, 1)
    result_std = torch.bmm(x_s.float(), W_V.float()).transpose(0, 1)

    # Heads-first: skip view+transpose → BMM
    result_hf = torch.bmm(x_hf.float(), W_V.float()).transpose(0, 1)

    assert torch.allclose(result_std, result_hf, atol=1e-3), (
        f"Max diff: {(result_std - result_hf).abs().max()}"
    )
    print("v_up_proj equivalence: PASS")


def test_reduce_kernel_strides():
    """Verify reduce kernel stride parameters work with heads-first layout."""
    padded_N, B, L = 16, 4, 512

    # Standard: (B, padded_N, L)
    o_std = torch.empty(B, padded_N, L, device="cuda")
    assert o_std.stride(-3) == padded_N * L
    assert o_std.stride(-2) == L
    assert o_std.stride(-1) == 1

    # Heads-first: (padded_N, B, L) permuted to (B, padded_N, L)
    o_hf_storage = torch.empty(padded_N, B, L, device="cuda")
    o_hf = o_hf_storage.permute(1, 0, 2)
    assert o_hf.stride(-3) == L         # batch stride
    assert o_hf.stride(-2) == B * L     # head stride
    assert o_hf.stride(-1) == 1

    print("Reduce kernel stride compatibility: PASS")


def test_end_to_end_layout():
    """Full end-to-end: allocate heads-first, write through permuted view, slice."""
    padded_N, B, L = 16, 4, 512
    num_heads = 8

    # Known per-head data
    head_data = torch.randn(num_heads, B, L, device="cuda")

    # Allocate heads-first storage
    o_hf = torch.empty(padded_N, B, L, device="cuda")
    o_view = o_hf.permute(1, 0, 2)  # (B, padded_N, L) — kernel's view

    # Simulate kernel writing through the (B, padded_N, L) view
    # With repeat (tile), heads 0..7 get unique data, 8..15 are duplicates
    for h in range(num_heads):
        o_view[:, h, :] = head_data[h]
        o_view[:, h + num_heads, :] = head_data[h]  # duplicate

    # Slice from storage
    result = o_hf[:num_heads]  # (8, B, L) contiguous
    assert result.is_contiguous()
    assert torch.equal(result, head_data)

    print("End-to-end layout: PASS")


if __name__ == "__main__":
    test_repeat_vs_interleave()
    test_heads_first_strides()
    test_heads_first_contiguous_slice()
    test_mla_output_equivalence()
    test_v_up_proj_equivalence()
    test_reduce_kernel_strides()
    test_end_to_end_layout()
    print("\nAll heads-first layout tests passed!")
