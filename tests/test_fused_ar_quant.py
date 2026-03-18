"""Test the full flow: fused AR+RMSNorm then MXFP4 quant-only.

Simulates single-GPU (no actual AR) to verify the norm+quant path
produces equivalent results to the current fused norm+quant path.
"""
import os
os.environ.pop("HIP_VISIBLE_DEVICES", None)
import torch
torch.cuda.set_device(4)

from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant, mxfp4_quant_shuffled
from aiter import rmsnorm2d_fwd_with_add

torch.manual_seed(42)
device = "cuda"

N = 7168  # hidden_size
eps = 1e-6

print("=== Correctness: fused(res_add+norm+quant) vs split(res_add+norm, quant) ===\n")

for M in [1, 4, 32, 128]:
    hidden_states = torch.randn(M, N, dtype=torch.bfloat16, device=device) * 0.1
    residual = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, dtype=torch.bfloat16, device=device).abs().clamp(min=0.01)

    # === Path A: Current fused path (res_add + RMSNorm + MXFP4 quant) ===
    (fp4_a, scale_a), _, _, res_a = fused_rms_mxfp4_quant(
        x1=hidden_states.clone(),
        x1_weight=weight,
        x1_epsilon=eps,
        res1=residual.clone(),
        shuffle=True,
        scale_shuffle_padding=True,
    )

    # === Path B: Split path (aiter RMSNorm with res_add, then quant-only) ===
    # Step 1: residual_add + RMSNorm (aiter HIP kernel, same as fused AR+RMSNorm minus the AR)
    normed_b = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    res_b = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    rmsnorm2d_fwd_with_add(
        normed_b,           # output: normed
        hidden_states,      # input: x
        residual,           # input: residual (added to x)
        res_b,              # output: x + residual (the new residual)
        weight,             # norm weight
        eps,                # epsilon
        0,                  # dim override (0 = use last dim)
    )

    # Step 2: MXFP4 quant-only
    fp4_b, scale_b = mxfp4_quant_shuffled(normed_b, shuffle=True, scale_shuffle_padding=True)

    # === Compare ===
    fp4_match = torch.equal(fp4_a, fp4_b)
    # Only compare valid scale region
    scale_valid_a = scale_a[:M]
    scale_valid_b = scale_b[:M]
    scale_match = torch.equal(scale_valid_a, scale_valid_b)
    res_match = torch.allclose(res_a, res_b, atol=0, rtol=0)

    status = "PASS" if (fp4_match and scale_match and res_match) else "FAIL"
    print(f"  M={M:4d}: {status}  fp4={fp4_match}  scale={scale_match}  residual={res_match}")

    if not fp4_match:
        diffs = (fp4_a != fp4_b).sum().item()
        print(f"         fp4 diffs: {diffs}/{fp4_a.numel()} ({100*diffs/fp4_a.numel():.2f}%)")
    if not scale_match:
        diffs = (scale_valid_a != scale_valid_b).sum().item()
        print(f"         scale diffs: {diffs}/{scale_valid_a.numel()}")
    if not res_match:
        max_diff = (res_a - res_b).abs().max().item()
        print(f"         residual max diff: {max_diff}")

print("\n=== Benchmark: fused vs split (M=4, N=7168) ===\n")
M = 4
hidden_states = torch.randn(M, N, dtype=torch.bfloat16, device=device) * 0.1
residual = torch.randn(M, N, dtype=torch.bfloat16, device=device)
weight = torch.randn(N, dtype=torch.bfloat16, device=device).abs().clamp(min=0.01)

normed_buf = torch.empty(M, N, dtype=torch.bfloat16, device=device)
res_buf = torch.empty(M, N, dtype=torch.bfloat16, device=device)

NI = 200

# Warmup
for _ in range(10):
    fused_rms_mxfp4_quant(hidden_states, weight, eps, res1=residual, shuffle=True, scale_shuffle_padding=True)
    rmsnorm2d_fwd_with_add(normed_buf, hidden_states, residual, res_buf, weight, eps, 0)
    mxfp4_quant_shuffled(normed_buf)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Bench fused (current path)
start.record()
for _ in range(NI):
    fused_rms_mxfp4_quant(hidden_states, weight, eps, res1=residual, shuffle=True, scale_shuffle_padding=True)
end.record()
torch.cuda.synchronize()
us_fused = start.elapsed_time(end) * 1000 / NI

# Bench split (new path: norm then quant)
start.record()
for _ in range(NI):
    rmsnorm2d_fwd_with_add(normed_buf, hidden_states, residual, res_buf, weight, eps, 0)
    mxfp4_quant_shuffled(normed_buf)
end.record()
torch.cuda.synchronize()
us_split = start.elapsed_time(end) * 1000 / NI

print(f"  fused (norm+quant):    {us_fused:.1f}μs")
print(f"  split (norm + quant):  {us_split:.1f}μs")
print(f"  overhead of split:     {us_split - us_fused:+.1f}μs")
print(f"\n  NOTE: In the real flow, the 'norm' part is fused with AR (no extra cost).")
print(f"  Net per-layer savings = standalone_AR({27:.0f}μs) - quant_only_overhead({us_split - us_fused:.0f}μs)")
print(f"  Estimated per-step savings (61 layers): {61 * (27 - (us_split - us_fused)) / 1000:.1f}ms")
