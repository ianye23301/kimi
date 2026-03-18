"""Test the DecoderLayer forward path change on single GPU.

On single GPU (tp_size=1), fused_allreduce is a no-op in RMSNorm,
so this tests that the split norm+quant path produces equivalent
results to the original fused norm+quant path.
"""
import os
os.environ.pop("HIP_VISIBLE_DEVICES", None)
import torch
torch.cuda.set_device(4)

# Test the kernel-level equivalence directly
from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant, mxfp4_quant_shuffled
from aiter import rmsnorm2d_fwd_with_add

torch.manual_seed(42)
device = "cuda"
N = 7168
eps = 1e-6

print("=" * 60)
print("Test: DecoderLayer input_layernorm path equivalence")
print("=" * 60)

for M in [1, 4, 32]:
    # Simulate layer inputs
    hidden_states = torch.randn(M, N, dtype=torch.bfloat16, device=device) * 0.1
    residual = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, dtype=torch.bfloat16, device=device).abs().clamp(min=0.01)

    # === Path A: Original fused path ===
    # _fuse_rmsnorm_quant(hidden_states, weight, eps, res1=residual)
    # This does: res_out = hidden_states + residual, normed = RMSNorm(res_out), quant(normed)
    (fp4_a, scale_a), _, _, res_a = fused_rms_mxfp4_quant(
        x1=hidden_states.clone(),
        x1_weight=weight,
        x1_epsilon=eps,
        res1=residual.clone(),
        shuffle=True,
        scale_shuffle_padding=True,
    )

    # === Path B: New split path ===
    # Step 1: input_layernorm(hidden_states, residual) -> normed, new_residual
    # On single GPU this is just rmsnorm_fwd_with_add (same as fused AR+RMSNorm minus AR)
    normed_b = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    res_b = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    rmsnorm2d_fwd_with_add(normed_b, hidden_states, residual, res_b, weight, eps, 0)

    # Step 2: mxfp4_quant_shuffled(normed) -> fp4, scale
    fp4_b, scale_b = mxfp4_quant_shuffled(normed_b, shuffle=True, scale_shuffle_padding=True)

    # === Compare ===
    # Residual should be bitwise identical (both compute x + residual)
    res_match = torch.equal(res_a, res_b)

    # Normed values should be identical (same RMSNorm implementation in aiter)
    # fp4 may differ slightly due to float32 vs bf16 intermediate precision
    fp4_diff_pct = 100 * (fp4_a != fp4_b).float().mean().item()
    scale_valid_a = scale_a[:M]
    scale_valid_b = scale_b[:M]
    scale_diff = (scale_valid_a != scale_valid_b).sum().item()

    print(f"\n  M={M}:")
    print(f"    residual exact match: {res_match}")
    print(f"    fp4 diff: {fp4_diff_pct:.2f}% (expected ~1% from bf16 intermediate)")
    print(f"    scale diffs: {scale_diff}/{scale_valid_a.numel()}")

    # Verify the fp4 diffs are within acceptable range
    assert res_match, "Residual mismatch!"
    assert fp4_diff_pct < 2.0, f"fp4 diff too large: {fp4_diff_pct}%"

print("\n" + "=" * 60)
print("PASS: All paths produce equivalent results")
print("=" * 60)

# Benchmark
print("\n" + "=" * 60)
print("Benchmark: overhead of split path (M=4)")
print("=" * 60)

M = 4
x = torch.randn(M, N, dtype=torch.bfloat16, device=device) * 0.1
res = torch.randn(M, N, dtype=torch.bfloat16, device=device)
w = torch.randn(N, dtype=torch.bfloat16, device=device).abs().clamp(min=0.01)
normed_buf = torch.empty(M, N, dtype=torch.bfloat16, device=device)
res_buf = torch.empty(M, N, dtype=torch.bfloat16, device=device)

NI = 500

# Warmup
for _ in range(20):
    fused_rms_mxfp4_quant(x, w, eps, res1=res, shuffle=True, scale_shuffle_padding=True)
    rmsnorm2d_fwd_with_add(normed_buf, x, res, res_buf, w, eps, 0)
    mxfp4_quant_shuffled(normed_buf)
torch.cuda.synchronize()

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# Path A: fused norm+quant (current)
s.record()
for _ in range(NI):
    fused_rms_mxfp4_quant(x, w, eps, res1=res, shuffle=True, scale_shuffle_padding=True)
e.record()
torch.cuda.synchronize()
us_fused = s.elapsed_time(e) * 1000 / NI

# Path B: split norm then quant (new)
s.record()
for _ in range(NI):
    rmsnorm2d_fwd_with_add(normed_buf, x, res, res_buf, w, eps, 0)
    mxfp4_quant_shuffled(normed_buf)
e.record()
torch.cuda.synchronize()
us_split = s.elapsed_time(e) * 1000 / NI

# Path C: quant only (measures pure quant cost)
s.record()
for _ in range(NI):
    mxfp4_quant_shuffled(normed_buf)
e.record()
torch.cuda.synchronize()
us_quant = s.elapsed_time(e) * 1000 / NI

print(f"\n  fused (res_add+norm+quant): {us_fused:.1f}μs  [1 Triton kernel]")
print(f"  split (norm + quant):       {us_split:.1f}μs  [1 HIP + 1 Triton kernel]")
print(f"  quant only:                 {us_quant:.1f}μs  [1 Triton kernel]")
print(f"  norm overhead:              {us_split - us_quant:.1f}μs")
print(f"  split vs fused overhead:    {us_split - us_fused:+.1f}μs")
print()
print(f"  In the real TP8 flow:")
print(f"    Current: standalone_AR(~27μs) + fused_norm_quant({us_fused:.0f}μs) = ~{27+us_fused:.0f}μs")
print(f"    New:     fused_AR+norm(~32μs) + quant_only({us_quant:.0f}μs) = ~{32+us_quant:.0f}μs")
print(f"    Per-layer savings: ~{(27+us_fused) - (32+us_quant):.0f}μs")
print(f"    Per-step savings (61 layers): ~{61*((27+us_fused) - (32+us_quant))/1000:.1f}ms")
