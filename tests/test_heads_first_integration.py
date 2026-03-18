"""Integration test for heads-first MLA output layout.

Simulates the full _forward_decode flow:
1. q.repeat(1, factor, 1) to pad heads
2. Allocate output in heads-first (padded_N, B, L) layout
3. Permute to (B, padded_N, L) for kernel
4. Kernel writes attention output through permuted view
5. Slice o_heads_first[:num_heads] for contiguous (N, B, L)
6. Pass to _v_up_proj_and_o_proj with heads_first=True

Compares against the standard flow:
1. q.repeat_interleave(factor, dim=1) to pad heads
2. Allocate standard (B, padded_N, L) output
3. Kernel writes to contiguous output
4. o[:, ::factor, :].contiguous() to extract unique heads
5. Pass to _v_up_proj_and_o_proj with heads_first=False
"""

import torch
import torch.nn.functional as F


class FakeMLA:
    """Simulates MLA decode attention.

    Each query head independently attends to shared KV cache.
    Output[b, h, :] = softmax(q[b,h,:] @ K[b].T / sqrt(d)) @ V[b]
    """

    def __init__(self, num_heads, padded_num_heads, kv_lora_rank, dtype=torch.bfloat16):
        self.num_heads = num_heads
        self.padded_num_heads = padded_num_heads
        self.head_repeat_factor = padded_num_heads // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.dtype = dtype
        # Random W_V for v_up_proj BMM: (num_heads, kv_lora_rank, v_head_dim)
        self.v_head_dim = 128
        self.W_V = torch.randn(
            num_heads, kv_lora_rank, self.v_head_dim,
            device="cuda", dtype=dtype,
        )

    def fake_attention(self, q, kv_keys, kv_values, output):
        """Simulate attention kernel writing to output tensor.

        q: (B, padded_N, head_dim)
        kv_keys: (B, seq_len, head_dim) — shared across heads
        kv_values: (B, seq_len, kv_lora_rank) — shared across heads
        output: (B, padded_N, kv_lora_rank) — may be non-contiguous (heads-first)

        The kernel uses output strides, so handles non-contiguous output.
        """
        B = q.shape[0]
        for b in range(B):
            for h in range(self.padded_num_heads):
                # score = q[b,h] @ K[b].T
                scores = q[b, h].float() @ kv_keys[b].float().T
                attn = torch.softmax(scores / (q.shape[-1] ** 0.5), dim=-1)
                # output[b,h] = attn @ V[b]
                output[b, h] = (attn @ kv_values[b].float()).to(self.dtype)

    def v_up_proj(self, x_heads_first):
        """BMM: (N, B, L) @ (N, L, V) -> (N, B, V) -> (B, N*V)"""
        result = torch.bmm(x_heads_first.float(), self.W_V.float())
        return result.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim).to(self.dtype)

    def forward_standard(self, q, kv_keys, kv_values):
        """Standard flow: repeat_interleave + strided slice + contiguous."""
        B = q.shape[0]

        # Pad q: [h0,h0,h1,h1,...,h7,h7]
        q_padded = q.repeat_interleave(self.head_repeat_factor, dim=1)

        # Standard contiguous output
        o = torch.empty(B, self.padded_num_heads, self.kv_lora_rank,
                        dtype=self.dtype, device=q.device)

        self.fake_attention(q_padded, kv_keys, kv_values, o)

        # Extract unique heads: take every factor-th head
        o_unique = o[:, ::self.head_repeat_factor, :].contiguous()  # (B, N, L)

        # v_up_proj: (B, N, L) -> transpose -> (N, B, L) -> BMM
        x = o_unique.transpose(0, 1)  # (N, B, L)
        return self.v_up_proj(x)

    def forward_heads_first(self, q, kv_keys, kv_values):
        """Heads-first flow: repeat (tile) + heads-first layout + contiguous slice."""
        B = q.shape[0]

        # Pad q: [h0,h1,...,h7,h0,h1,...,h7]
        q_padded = q.repeat(1, self.head_repeat_factor, 1)

        # Heads-first output: (padded_N, B, L) contiguous
        o_heads_first = torch.empty(
            self.padded_num_heads, B, self.kv_lora_rank,
            dtype=self.dtype, device=q.device,
        )
        # Permute to (B, padded_N, L) for kernel
        o = o_heads_first.permute(1, 0, 2)

        self.fake_attention(q_padded, kv_keys, kv_values, o)

        # Slice real heads: o_heads_first[:num_heads] is (N, B, L) contiguous
        o_real = o_heads_first[:self.num_heads]
        assert o_real.is_contiguous()

        return self.v_up_proj(o_real)


def test_standard_vs_heads_first():
    """Verify both paths produce identical results."""
    torch.manual_seed(42)

    num_heads = 8
    padded_num_heads = 16
    kv_lora_rank = 512
    head_dim = 576  # kv_lora_rank + qk_rope_head_dim
    seq_len = 64
    B = 4

    mla = FakeMLA(num_heads, padded_num_heads, kv_lora_rank)

    q = torch.randn(B, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_keys = torch.randn(B, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_values = torch.randn(B, seq_len, kv_lora_rank, device="cuda", dtype=torch.bfloat16)

    out_standard = mla.forward_standard(q, kv_keys, kv_values)
    out_heads_first = mla.forward_heads_first(q, kv_keys, kv_values)

    assert out_standard.shape == out_heads_first.shape, (
        f"Shape mismatch: {out_standard.shape} vs {out_heads_first.shape}"
    )

    max_diff = (out_standard.float() - out_heads_first.float()).abs().max().item()
    assert max_diff < 1e-2, f"Max diff too large: {max_diff}"

    # Check relative error
    rel_err = (
        (out_standard.float() - out_heads_first.float()).abs()
        / (out_standard.float().abs() + 1e-8)
    ).mean().item()
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Mean rel err: {rel_err:.6f}")

    print("test_standard_vs_heads_first: PASS")


def test_various_batch_sizes():
    """Test with different batch sizes to catch edge cases."""
    torch.manual_seed(0)
    num_heads = 8
    padded_num_heads = 16
    kv_lora_rank = 512
    head_dim = 576
    seq_len = 32

    mla = FakeMLA(num_heads, padded_num_heads, kv_lora_rank)

    for B in [1, 2, 4, 8, 16, 32]:
        q = torch.randn(B, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        kv_keys = torch.randn(B, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        kv_values = torch.randn(B, seq_len, kv_lora_rank, device="cuda", dtype=torch.bfloat16)

        out_std = mla.forward_standard(q, kv_keys, kv_values)
        out_hf = mla.forward_heads_first(q, kv_keys, kv_values)

        max_diff = (out_std.float() - out_hf.float()).abs().max().item()
        assert max_diff < 1e-2, f"B={B}: max diff {max_diff}"

    print("test_various_batch_sizes: PASS")


def test_different_repeat_factors():
    """Test with head repeat factors other than 2."""
    torch.manual_seed(0)
    kv_lora_rank = 512
    head_dim = 576
    seq_len = 32
    B = 4

    configs = [
        (8, 16, 2),   # 8→16, factor=2 (Kimi K2.5 TP8)
        (4, 16, 4),   # 4→16, factor=4 (hypothetical)
        (2, 16, 8),   # 2→16, factor=8 (hypothetical)
        (16, 16, 1),  # No repeat needed
    ]

    for num_heads, padded, factor in configs:
        if factor == 1:
            continue  # heads-first not used when factor=1

        mla = FakeMLA(num_heads, padded, kv_lora_rank)

        q = torch.randn(B, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        kv_keys = torch.randn(B, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        kv_values = torch.randn(B, seq_len, kv_lora_rank, device="cuda", dtype=torch.bfloat16)

        out_std = mla.forward_standard(q, kv_keys, kv_values)
        out_hf = mla.forward_heads_first(q, kv_keys, kv_values)

        max_diff = (out_std.float() - out_hf.float()).abs().max().item()
        assert max_diff < 1e-2, (
            f"heads={num_heads}→{padded} (factor={factor}): max diff {max_diff}"
        )

    print("test_different_repeat_factors: PASS")


def test_output_strides_match_reduce_kernel():
    """Verify that the permuted output has strides compatible with reduce.cu.

    reduce.cu uses:
      params.stride_s_o = final_output.stride(-3)  // batch stride
      params.stride_h_o = final_output.stride(-2)  // head stride

    These must be valid positive integers for correct indexing.
    """
    B = 4
    padded_N = 16
    L = 512

    # Standard layout
    o_std = torch.empty(B, padded_N, L, device="cuda")
    s_s_std = o_std.stride(-3)
    s_h_std = o_std.stride(-2)
    assert s_s_std > 0 and s_h_std > 0

    # Heads-first layout (permuted)
    o_hf = torch.empty(padded_N, B, L, device="cuda").permute(1, 0, 2)
    s_s_hf = o_hf.stride(-3)
    s_h_hf = o_hf.stride(-2)
    assert s_s_hf > 0 and s_h_hf > 0

    # Verify the output write pattern is correct
    # For seq_idx=2, head_idx=5, dim=100:
    # Standard: offset = 2*s_s_std + 5*s_h_std + 100
    # Heads-first: offset = 2*s_s_hf + 5*s_h_hf + 100

    o_std.fill_(0)
    o_std[2, 5, 100] = 1.0
    flat_std = o_std.reshape(-1)
    offset_std = 2 * s_s_std + 5 * s_h_std + 100
    assert flat_std[offset_std] == 1.0

    o_hf_storage = torch.empty(padded_N, B, L, device="cuda")
    o_hf = o_hf_storage.permute(1, 0, 2)
    o_hf.fill_(0)
    o_hf[2, 5, 100] = 1.0
    # In storage: head=5, batch=2, dim=100
    assert o_hf_storage[5, 2, 100] == 1.0

    print("test_output_strides_match_reduce_kernel: PASS")


if __name__ == "__main__":
    test_standard_vs_heads_first()
    test_various_batch_sizes()
    test_different_repeat_factors()
    test_output_strides_match_reduce_kernel()
    print("\nAll heads-first integration tests passed!")
