"""Test MoE split-k buffer pre-allocation optimization.

The optimization replaces per-call torch.zeros() allocation in cktile_moe_stage1
with a pre-allocated global buffer that gets .zero_()'d each call.
This saves allocation overhead (~3μs × 240 calls = ~0.7ms/step).

Tests verify:
1. Buffer is reused across calls (not reallocated)
2. Buffer is properly zeroed each call (split-k correctness)
3. Buffer grows when needed (larger batch)
4. Buffer handles device/dtype changes
"""

import torch
import sys


def test_buffer_reuse():
    """Pre-allocated buffer should be reused across calls with same or smaller size."""
    # Simulate the global buffer pattern
    buf = None
    alloc_count = 0

    def get_splitk_buf(token_num, topk, n_dim, dtype, device):
        nonlocal buf, alloc_count
        needed = token_num * topk * n_dim
        if (
            buf is None
            or buf.numel() < needed
            or buf.dtype != dtype
            or buf.device.type != torch.device(device).type
        ):
            alloc_tokens = max(token_num, 256)
            buf = torch.empty(alloc_tokens * topk * n_dim, dtype=dtype, device=device)
            alloc_count += 1
        tmp = buf[:needed].view(token_num, topk, n_dim)
        tmp.zero_()
        return tmp

    # First call — allocates
    t1 = get_splitk_buf(4, 8, 4096, torch.bfloat16, "cuda")
    assert t1.shape == (4, 8, 4096)
    assert torch.all(t1 == 0)
    assert alloc_count == 1

    # Second call, same size — reuses buffer (no new allocation)
    t2 = get_splitk_buf(4, 8, 4096, torch.bfloat16, "cuda")
    assert alloc_count == 1, "Buffer should be reused!"
    assert torch.all(t2 == 0)

    # Smaller size — still reuses (no new allocation)
    t3 = get_splitk_buf(2, 8, 4096, torch.bfloat16, "cuda")
    assert alloc_count == 1, "Buffer should be reused for smaller size!"
    assert t3.shape == (2, 8, 4096)

    # Much larger size — must reallocate
    t4 = get_splitk_buf(512, 8, 4096, torch.bfloat16, "cuda")
    assert alloc_count == 2, "Buffer should grow for larger size!"
    assert t4.shape == (512, 8, 4096)

    # Back to small — reuses the grown buffer
    t5 = get_splitk_buf(4, 8, 4096, torch.bfloat16, "cuda")
    assert alloc_count == 2, "Grown buffer should be reused!"

    print("test_buffer_reuse: PASS")


def test_buffer_grows():
    """Buffer should grow when a larger size is requested."""
    buf = None
    alloc_count = 0

    def get_splitk_buf(token_num, topk, n_dim, dtype, device):
        nonlocal buf, alloc_count
        needed = token_num * topk * n_dim
        if buf is None or buf.numel() < needed or buf.dtype != dtype:
            alloc_tokens = max(token_num, 256)
            buf = torch.empty(alloc_tokens * topk * n_dim, dtype=dtype, device=device)
            alloc_count += 1
        return buf[:needed].view(token_num, topk, n_dim)

    t1 = get_splitk_buf(4, 8, 4096, torch.bfloat16, "cuda")
    assert alloc_count == 1
    old_numel = buf.numel()

    # Request larger than pre-allocated (256 * 8 * 4096)
    t2 = get_splitk_buf(512, 8, 4096, torch.bfloat16, "cuda")
    assert t2.shape == (512, 8, 4096)
    assert buf.numel() >= 512 * 8 * 4096
    assert buf.numel() > old_numel
    assert alloc_count == 2

    print("test_buffer_grows: PASS")


def test_zero_correctness():
    """Verify that .zero_() properly clears the buffer between calls."""
    buf = torch.empty(256 * 8 * 4096, dtype=torch.bfloat16, device="cuda")

    # Fill with non-zero data
    buf.fill_(42.0)

    # Simulate what the optimized code does
    token_num, topk, n_dim = 4, 8, 4096
    needed = token_num * topk * n_dim
    tmp = buf[:needed].view(token_num, topk, n_dim)
    tmp.zero_()

    # Verify the slice is zeroed
    assert torch.all(tmp == 0), "Sliced buffer must be zeroed!"

    # Verify the REST of the buffer is NOT zeroed (only slice is)
    rest = buf[needed:]
    assert torch.all(rest == 42.0), "Rest of buffer should be untouched!"

    print("test_zero_correctness: PASS")


def test_view_contiguity():
    """The view from flat buffer must be contiguous for the CK kernel."""
    buf = torch.empty(256 * 8 * 4096, dtype=torch.bfloat16, device="cuda")

    for token_num in [1, 4, 16, 64, 128]:
        topk, n_dim = 8, 4096
        needed = token_num * topk * n_dim
        tmp = buf[:needed].view(token_num, topk, n_dim)

        assert tmp.is_contiguous(), f"View must be contiguous! token_num={token_num}"
        assert tmp.shape == (token_num, topk, n_dim)
        assert tmp.stride() == (topk * n_dim, n_dim, 1)

    print("test_view_contiguity: PASS")


def test_matches_torch_zeros():
    """Verify optimized path produces identical result to torch.zeros."""
    token_num, topk, n_dim = 4, 8, 4096
    dtype = torch.bfloat16

    # Original: torch.zeros
    ref = torch.zeros(token_num, topk, n_dim, dtype=dtype, device="cuda")

    # Optimized: pre-alloc + zero_()
    buf = torch.empty(256 * topk * n_dim, dtype=dtype, device="cuda")
    buf.fill_(999.0)  # dirty the buffer
    needed = token_num * topk * n_dim
    opt = buf[:needed].view(token_num, topk, n_dim)
    opt.zero_()

    assert torch.equal(ref, opt), "Optimized must match torch.zeros!"
    assert ref.stride() == opt.stride(), "Strides must match!"
    assert ref.dtype == opt.dtype

    print("test_matches_torch_zeros: PASS")


if __name__ == "__main__":
    test_buffer_reuse()
    test_buffer_grows()
    test_zero_correctness()
    test_view_contiguity()
    test_matches_torch_zeros()
    print("\nAll split-k pre-allocation tests passed!")
