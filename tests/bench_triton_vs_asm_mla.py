"""Benchmark: Triton MLA decode (native 8 heads) vs ASM qh16 (padded from 8).

Full apples-to-apples comparison of the two decode paths for TP8 Kimi K2.5.
Measures the ENTIRE path for each approach including overhead (repeat, permute, etc.).

Run inside the ATOM container with GPU access:
    python kimi/tests/bench_triton_vs_asm_mla.py
"""

import torch
import time
import math
import sys


def bench(label, fn, n_iters=500, warmup=100):
    """Benchmark a function, return mean time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) / n_iters * 1e6
    print(f"  {label:<60} {elapsed_us:>8.2f} μs")
    return elapsed_us


def make_paging_metadata(B, seq_len, device="cuda"):
    """Create simple paging metadata (page_size=1, contiguous tokens)."""
    total_tokens = B * seq_len
    kv_indptr = torch.arange(0, (B + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=device)
    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device)
    return kv_indptr, kv_indices, kv_last_page_lens, qo_indptr, total_tokens


def bench_asm_path(B, H, seq_len, kv_lora_rank, qk_rope_head_dim, sm_scale):
    """Benchmark the full ASM path: repeat + ASM qh16 + heads_first slice."""
    from aiter.mla import mla_decode_fwd

    dim = kv_lora_rank + qk_rope_head_dim
    padded_H = 16
    factor = padded_H // H
    device = "cuda"

    kv_indptr, kv_indices, kv_last_page_lens, qo_indptr, total_tokens = \
        make_paging_metadata(B, seq_len, device)

    # ASM kernel needs fp8 KV cache (fp8,fp8 is the only qSeqLen=1 entry)
    kv_cache_bf16 = torch.randn(total_tokens, dim, device=device, dtype=torch.bfloat16)
    kv_cache_fp8 = kv_cache_bf16.to(torch.float8_e4m3fn)
    kv_buffer = kv_cache_fp8.view(-1, 1, 1, dim)

    # fp8 Q (as in actual ATOM path)
    q_fp8 = torch.randn(B, H, dim, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q_scale = torch.ones(1, device=device, dtype=torch.float32)
    kv_scale = torch.ones(1, device=device, dtype=torch.float32)

    # Pre-allocate heads-first output
    o_hf = torch.empty(padded_H, B, kv_lora_rank, dtype=torch.bfloat16, device=device)

    def run_asm_full():
        # Step 1: repeat Q to 16 heads (triggers float8_copy)
        q_padded = q_fp8.repeat(1, factor, 1)
        # Step 2: permute output buffer for kernel
        o_kernel = o_hf.permute(1, 0, 2)
        # Step 3: ASM MLA decode (non-persistent mode)
        mla_decode_fwd(
            q_padded,
            kv_buffer,
            o_kernel,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_q=1,
            page_size=1,
            nhead_kv=1,
            sm_scale=sm_scale,
            num_kv_splits=16,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        # Step 4: heads_first slice (free — just a view)
        return o_hf[:H]

    return bench("ASM qh16 full (repeat+kernel+slice)", run_asm_full)


def bench_triton_path(B, H, seq_len, kv_lora_rank, qk_rope_head_dim, sm_scale):
    """Benchmark the full Triton path: kernel + permute."""
    from aiter.ops.triton.attention.mla_decode_rope import decode_attention_fwd_grouped_rope

    dim = kv_lora_rank + qk_rope_head_dim
    device = "cuda"

    kv_indptr, kv_indices, _, _, total_tokens = \
        make_paging_metadata(B, seq_len, device)

    kv_cache = torch.randn(total_tokens, dim, device=device, dtype=torch.bfloat16)
    k_buffer = kv_cache.view(-1, 1, dim)
    v_buffer = kv_cache.view(-1, 1, dim)[:, :, :kv_lora_rank]

    # bf16 Q (Triton path skips fp8 quant)
    q_bf16 = torch.randn(B, H, dim, device=device, dtype=torch.bfloat16)

    o = torch.empty(B, H, kv_lora_rank, dtype=torch.bfloat16, device=device)
    attn_logits = torch.empty(B, H, 16, kv_lora_rank + 1, dtype=torch.float32, device=device)
    k_pe_tokens = torch.empty(0, device=device)

    def run_triton_full():
        # Triton kernel (native 8 heads) — output is (B,H,L), no permute needed
        decode_attention_fwd_grouped_rope(
            q_bf16, k_buffer, v_buffer, o,
            kv_indptr, kv_indices, k_pe_tokens,
            kv_lora_rank=kv_lora_rank, rotary_dim=0,
            cos_sin_cache=None, positions=None,
            attn_logits=attn_logits, num_kv_splits=16,
            sm_scale=sm_scale, logit_cap=0.0,
            use_rope=False, is_neox_style=False,
        )
        return o

    return bench("Triton qh8 full (kernel only)", run_triton_full)


def bench_individual_components(B, H, seq_len, kv_lora_rank, qk_rope_head_dim):
    """Benchmark individual overhead components for breakdown analysis."""
    dim = kv_lora_rank + qk_rope_head_dim
    device = "cuda"

    q_fp8 = torch.randn(B, H, dim, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q_bf16 = torch.randn(B, H, dim, device=device, dtype=torch.bfloat16)
    o_bf16 = torch.randn(B, H, kv_lora_rank, device=device, dtype=torch.bfloat16)

    bench("q.repeat(1,2,1) fp8 [float8_copy]", lambda: q_fp8.repeat(1, 2, 1))
    bench("q.repeat(1,2,1) bf16", lambda: q_bf16.repeat(1, 2, 1))
    bench("o.permute(1,0,2).contiguous()", lambda: o_bf16.permute(1, 0, 2).contiguous())
    bench("o.permute(1,0,2) [view only]", lambda: o_bf16.permute(1, 0, 2))


def main():
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    dim = kv_lora_rank + qk_rope_head_dim
    H = 8
    sm_scale = 1.0 / math.sqrt(dim)

    configs = [
        (1, 256),
        (1, 1024),
        (1, 4096),
        (4, 256),
        (4, 1024),
        (4, 4096),
        (32, 256),
        (32, 1024),
        (64, 256),
        (64, 1024),
        (128, 256),
        (128, 1024),
    ]

    print("=" * 80)
    print("Triton MLA qh8 vs ASM qh16+repeat — Full Path Benchmark")
    print(f"  H={H}, kv_lora_rank={kv_lora_rank}, rope_dim={qk_rope_head_dim}")
    print("=" * 80)

    results = []
    for B, seq_len in configs:
        print(f"\n--- bs={B}, seq_len={seq_len} ---")

        # Individual components
        bench_individual_components(B, H, seq_len, kv_lora_rank, qk_rope_head_dim)

        print()

        # Full paths
        try:
            t_asm = bench_asm_path(B, H, seq_len, kv_lora_rank, qk_rope_head_dim, sm_scale)
        except Exception as e:
            print(f"  ASM path failed: {e}")
            t_asm = float("inf")

        try:
            t_triton = bench_triton_path(B, H, seq_len, kv_lora_rank, qk_rope_head_dim, sm_scale)
        except Exception as e:
            print(f"  Triton path failed: {e}")
            t_triton = float("inf")

        diff = t_asm - t_triton
        winner = "Triton" if diff > 0 else "ASM"
        pct = abs(diff) / min(t_asm, t_triton) * 100

        print(f"  >>> Winner: {winner} by {abs(diff):.1f}μs ({pct:.1f}%)")
        print(f"  >>> Per-step (×244): ASM={t_asm*244/1000:.2f}ms, "
              f"Triton={t_triton*244/1000:.2f}ms, saved={diff*244/1000:.2f}ms")

        results.append((B, seq_len, t_asm, t_triton, diff))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'bs':>4} {'seq':>6} {'ASM (μs)':>10} {'Triton (μs)':>12} {'Δ (μs)':>10} {'Winner':>8} {'×244 saved':>12}")
    print("-" * 70)
    for B, seq_len, t_asm, t_triton, diff in results:
        winner = "Triton" if diff > 0 else "ASM"
        print(f"{B:>4} {seq_len:>6} {t_asm:>10.1f} {t_triton:>12.1f} {diff:>+10.1f} {winner:>8} {diff*244/1000:>+10.2f}ms")


if __name__ == "__main__":
    main()
