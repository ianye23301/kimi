"""Benchmark: q.repeat vs q.repeat_interleave for MLA head padding.

Both copy data, but repeat (tile) enables the heads-first slice optimization
that eliminates the direct_copy kernel (~1.06ms/step).
"""

import torch
import time


def bench(label, fn, n_iters=5000):
    for _ in range(200):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1e6
    print(f"  {label:<45} {elapsed:>8.2f} μs")
    return elapsed


if __name__ == "__main__":
    # Kimi K2.5 TP8: 8 heads → 16 heads, head_dim = 576 (kv_lora_rank + rope)
    configs = [
        (1, 8, 576, 2, "bs=1, 8→16 heads"),
        (4, 8, 576, 2, "bs=4, 8→16 heads"),
        (32, 8, 576, 2, "bs=32, 8→16 heads"),
        (128, 8, 576, 2, "bs=128, 8→16 heads"),
    ]

    for B, N, D, factor, label in configs:
        print(f"\n{label}:")
        q = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)

        t_interleave = bench(
            "repeat_interleave(factor, dim=1)",
            lambda: q.repeat_interleave(factor, dim=1),
        )
        t_repeat = bench(
            "repeat(1, factor, 1)",
            lambda: q.repeat(1, factor, 1),
        )

        # Also measure the output extraction cost
        # Standard: o[:, ::factor, :].contiguous()
        o_std = torch.randn(B, N * factor, 512, device="cuda", dtype=torch.bfloat16)
        t_strided_copy = bench(
            "o[:, ::factor, :].contiguous() [standard]",
            lambda: o_std[:, ::factor, :].contiguous(),
        )

        # Heads-first: o_hf[:N] — already contiguous, no copy
        o_hf = torch.randn(N * factor, B, 512, device="cuda", dtype=torch.bfloat16)
        t_slice = bench(
            "o_heads_first[:N] [heads-first, no copy]",
            lambda: o_hf[:N],
        )

        # Net difference per step (244 layers)
        old_cost = t_interleave + t_strided_copy
        new_cost = t_repeat + t_slice
        saved = old_cost - new_cost
        print(f"  --- Per-step (×244): old={old_cost*244/1000:.2f}ms, new={new_cost*244/1000:.2f}ms, saved={saved*244/1000:.2f}ms")
