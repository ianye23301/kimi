"""Benchmark: torch.zeros vs pre-allocated buffer for split-k accumulator.

Measures the allocation overhead saved by pre-allocating the buffer.
"""

import torch
import time


def bench_torch_zeros(token_num, topk, n_dim, dtype, n_iters=1000):
    """Original approach: torch.zeros every call."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        tmp = torch.zeros(token_num, topk, n_dim, dtype=dtype, device="cuda")
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1e6  # μs
    return elapsed


def bench_prealloc_zero(token_num, topk, n_dim, dtype, n_iters=1000):
    """Optimized: pre-allocated buffer + .zero_()."""
    alloc_tokens = max(token_num, 256)
    buf = torch.empty(alloc_tokens * topk * n_dim, dtype=dtype, device="cuda")
    needed = token_num * topk * n_dim

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        tmp = buf[:needed].view(token_num, topk, n_dim)
        tmp.zero_()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1e6  # μs
    return elapsed


if __name__ == "__main__":
    # Kimi K2.5 decode shapes: small batch, topk=8, intermediate=4096 (gate+up fused)
    configs = [
        (1, 8, 4096, "bs=1"),
        (4, 8, 4096, "bs=4"),
        (16, 8, 4096, "bs=16"),
        (32, 8, 4096, "bs=32"),
        (128, 8, 4096, "bs=128"),
    ]

    # Warmup
    for _ in range(100):
        torch.zeros(128, 8, 4096, dtype=torch.bfloat16, device="cuda")

    print(f"{'Config':<10} {'zeros (μs)':>12} {'prealloc (μs)':>14} {'speedup':>8}")
    print("-" * 50)

    for token_num, topk, n_dim, label in configs:
        t_zeros = bench_torch_zeros(token_num, topk, n_dim, torch.bfloat16)
        t_prealloc = bench_prealloc_zero(token_num, topk, n_dim, torch.bfloat16)
        speedup = t_zeros / t_prealloc if t_prealloc > 0 else float("inf")
        print(f"{label:<10} {t_zeros:>12.1f} {t_prealloc:>14.1f} {speedup:>7.2f}x")

    # Estimate savings for 240 MoE calls/step at bs=4
    t_z = bench_torch_zeros(4, 8, 4096, torch.bfloat16, n_iters=5000)
    t_p = bench_prealloc_zero(4, 8, 4096, torch.bfloat16, n_iters=5000)
    saved_per_call = t_z - t_p
    print(f"\nEstimated savings per step (240 calls × {saved_per_call:.1f}μs): {240 * saved_per_call / 1000:.2f} ms")
