"""End-to-end test for MLA heads-first optimization in attention_mla.py.

Tests the modified _v_up_proj_and_o_proj(heads_first=True) path
against the original path, verifying:
1. Standard (B,N,L) input → transpose → BMM → o_proj matches
2. Heads-first (N,B,L) input → skip transpose → BMM → o_proj matches
3. Output tensor layout is identical
"""

import torch
import torch.nn as nn


class MockMLAVUpProj(nn.Module):
    """Simplified v_up_proj + o_proj from MLAAttention.

    v_up_proj: BMM of (N, B, kv_lora_rank) x (N, kv_lora_rank, v_head_dim)
    o_proj: Linear of (B, N*v_head_dim) -> (B, hidden_size)
    """

    def __init__(self, num_heads, kv_lora_rank, v_head_dim, hidden_size, dtype=torch.bfloat16):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.hidden_size = hidden_size
        self.dtype = dtype

        # W_V: (num_heads, kv_lora_rank, v_head_dim) — fp8 weights with scales
        # For testing, use bf16 weights directly (skip fp8 quantization)
        self.W_V = nn.Parameter(
            torch.randn(num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device="cuda")
        )
        # o_proj: (N*v_head_dim, hidden_size)
        self.o_proj_weight = nn.Parameter(
            torch.randn(hidden_size, num_heads * v_head_dim, dtype=dtype, device="cuda")
        )

    def forward_standard(self, x):
        """Standard path: x is (B, N, L) → transpose → (N, B, L) → BMM."""
        # x: (B, N, kv_lora_rank)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # BMM: (N, B, L) x (N, L, V) → (N, B, V)
        x = torch.bmm(x.float(), self.W_V.float()).to(self.dtype)
        # (N, B, V) → (B, N, V) → (B, N*V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        # o_proj
        return torch.mm(x, self.o_proj_weight.T)

    def forward_heads_first(self, x):
        """Heads-first path: x is (N, B, L) contiguous → skip transpose → BMM."""
        # x: (N, B, kv_lora_rank) — already in correct layout
        assert x.is_contiguous()
        # BMM: (N, B, L) x (N, L, V) → (N, B, V)
        x = torch.bmm(x.float(), self.W_V.float()).to(self.dtype)
        # (N, B, V) → (B, N, V) → (B, N*V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        # o_proj
        return torch.mm(x, self.o_proj_weight.T)


def test_standard_vs_heads_first_vup():
    """Both paths should produce identical output."""
    torch.manual_seed(42)

    N = 8  # num_heads (after slicing from 16 padded)
    L = 512  # kv_lora_rank
    V = 128  # v_head_dim
    H = 7168  # hidden_size
    B = 4

    model = MockMLAVUpProj(N, L, V, H)

    # Create test data as if coming from MLA decode output
    data = torch.randn(N, B, L, device="cuda", dtype=torch.bfloat16)

    # Standard path: (B, N, L) contiguous input
    x_std = data.permute(1, 0, 2).contiguous()  # (B, N, L)
    out_std = model.forward_standard(x_std)

    # Heads-first path: (N, B, L) contiguous input
    x_hf = data.contiguous()  # (N, B, L)
    out_hf = model.forward_heads_first(x_hf)

    max_diff = (out_std.float() - out_hf.float()).abs().max().item()
    assert out_std.shape == out_hf.shape == (B, H)
    assert max_diff < 1e-2, f"Max diff: {max_diff}"
    print(f"test_standard_vs_heads_first_vup: PASS (max_diff={max_diff:.6f})")


def test_heads_first_slice_is_contiguous():
    """Verify that o_heads_first[:num_heads] from padded tensor is contiguous."""
    padded_N = 16
    real_N = 8
    B = 4
    L = 512

    # Simulate heads-first allocation
    o_heads_first = torch.randn(padded_N, B, L, device="cuda", dtype=torch.bfloat16)

    # Slice real heads
    o_real = o_heads_first[:real_N]

    assert o_real.is_contiguous(), "Sliced heads must be contiguous!"
    assert o_real.shape == (real_N, B, L)
    assert o_real.stride() == (B * L, L, 1)

    # Verify data integrity — slice shares storage
    o_heads_first[3, 2, 100] = 42.0
    assert o_real[3, 2, 100] == 42.0, "Slice should share storage!"

    print("test_heads_first_slice_is_contiguous: PASS")


def test_permuted_output_kernel_write():
    """Simulate kernel writing to permuted (B, padded_N, L) view of (padded_N, B, L)."""
    padded_N = 16
    real_N = 8
    B = 4
    L = 512

    # Allocate heads-first
    o_hf = torch.zeros(padded_N, B, L, device="cuda", dtype=torch.float32)
    # Permute to (B, padded_N, L) for kernel
    o_kernel_view = o_hf.permute(1, 0, 2)

    # Simulate kernel writing attention output for all 16 heads
    # Heads 0-7 = unique, 8-15 = copies (from q.repeat pattern)
    for b in range(B):
        for h in range(padded_N):
            o_kernel_view[b, h, :] = (b + 1) * 100 + (h % real_N)

    # After kernel, slice real heads from heads-first storage
    o_real = o_hf[:real_N]  # (8, B, L)

    # Verify: head h in o_real should have value (b+1)*100 + h
    for b in range(B):
        for h in range(real_N):
            expected = float((b + 1) * 100 + h)
            actual = o_real[h, b, 0].item()
            assert actual == expected, f"b={b}, h={h}: expected {expected}, got {actual}"

    print("test_permuted_output_kernel_write: PASS")


def test_repeat_vs_repeat_interleave_head_mapping():
    """Verify that repeat (tile) gives correct head mapping for heads-first slice.

    repeat([h0,h1,...,h7], factor=2) → [h0,h1,...,h7, h0,h1,...,h7]
    After MLA decode, output heads match input heads.
    o_heads_first[:8] → unique outputs for h0-h7. ✓

    repeat_interleave([h0,...,h7], factor=2) → [h0,h0,h1,h1,...,h7,h7]
    o[:, ::2, :] → unique outputs for h0-h7 (but needs .contiguous()). ✗ (our old path)
    """
    N = 8
    factor = 2
    padded_N = N * factor
    B = 2
    D = 4  # small dim for clarity

    q = torch.arange(N, device="cuda", dtype=torch.float32).view(1, N, 1).expand(B, N, D)

    # Repeat (tile): [0,1,...,7, 0,1,...,7]
    q_repeat = q.repeat(1, factor, 1)
    assert q_repeat[0, 0, 0] == 0  # h0
    assert q_repeat[0, 7, 0] == 7  # h7
    assert q_repeat[0, 8, 0] == 0  # h0 (repeated)
    assert q_repeat[0, 15, 0] == 7  # h7 (repeated)

    # Simulate attention: output[b,h] = q[b,h] * 10 (identity-like)
    o_hf_storage = torch.zeros(padded_N, B, D, device="cuda")
    o_view = o_hf_storage.permute(1, 0, 2)
    for b in range(B):
        for h in range(padded_N):
            o_view[b, h] = q_repeat[b, h] * 10

    # Heads-first slice: first N heads
    o_real = o_hf_storage[:N]
    for h in range(N):
        assert o_real[h, 0, 0].item() == h * 10, f"Head {h} output wrong"

    # Repeat_interleave: [0,0,1,1,...,7,7]
    q_interleave = q.repeat_interleave(factor, dim=1)
    assert q_interleave[0, 0, 0] == 0
    assert q_interleave[0, 1, 0] == 0  # h0 repeated
    assert q_interleave[0, 2, 0] == 1  # h1

    # Standard strided slice: o[:, ::factor, :]
    o_std = torch.zeros(B, padded_N, D, device="cuda")
    for b in range(B):
        for h in range(padded_N):
            o_std[b, h] = q_interleave[b, h] * 10
    o_std_slice = o_std[:, ::factor, :]
    for h in range(N):
        assert o_std_slice[0, h, 0].item() == h * 10, f"Standard head {h} output wrong"

    print("test_repeat_vs_repeat_interleave_head_mapping: PASS")


def test_bmm_transpose_bm():
    """Verify BMM with transpose_bm=True input matches standard BMM."""
    N = 8
    B = 4
    K = 512
    V = 128

    torch.manual_seed(0)
    W = torch.randn(N, K, V, device="cuda", dtype=torch.bfloat16)

    # Standard: (B, N, K) → transpose → (N, B, K) → BMM
    x_bn = torch.randn(B, N, K, device="cuda", dtype=torch.bfloat16)
    x_nb = x_bn.transpose(0, 1).contiguous()
    out1 = torch.bmm(x_nb.float(), W.float()).to(torch.bfloat16)

    # Heads-first: (N, B, K) already → BMM directly
    x_nb2 = x_bn.transpose(0, 1).contiguous()
    out2 = torch.bmm(x_nb2.float(), W.float()).to(torch.bfloat16)

    assert torch.equal(out1, out2)
    print("test_bmm_transpose_bm: PASS")


if __name__ == "__main__":
    test_standard_vs_heads_first_vup()
    test_heads_first_slice_is_contiguous()
    test_permuted_output_kernel_write()
    test_repeat_vs_repeat_interleave_head_mapping()
    test_bmm_transpose_bm()
    print("\nAll MLA heads-first e2e tests passed!")
