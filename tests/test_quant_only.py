"""Test that mxfp4_quant_shuffled matches fused_rms_mxfp4_quant for pre-normed input."""
import os
os.environ.pop("HIP_VISIBLE_DEVICES", None)
import torch
torch.cuda.set_device(4)

from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant, mxfp4_quant_shuffled

torch.manual_seed(42)
device = "cuda"

# Simulate the two paths:
# Path A (current): fused_rms_mxfp4_quant(x, weight, eps, res1=residual) -> does res_add + RMSNorm + quant
# Path B (new): RMSNorm(x + residual) externally, then mxfp4_quant_shuffled(normed) -> quant only

N = 7168  # hidden_size
for M in [1, 4, 32]:
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, dtype=torch.bfloat16, device=device).abs() + 0.1
    eps = 1e-6
    residual = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    # Path A: fused RMSNorm + quant
    (fp4_a, scale_a), _, _, res_out_a = fused_rms_mxfp4_quant(
        x1=x, x1_weight=weight, x1_epsilon=eps,
        res1=residual, shuffle=True, scale_shuffle_padding=True,
    )

    # Path B: manual residual_add + RMSNorm, then quant-only
    # Step 1: residual add
    added = x + residual
    # Step 2: RMSNorm
    rms = added.float().pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = (added.float() * rms * weight.float()).to(torch.bfloat16)
    # Step 3: quant only
    fp4_b, scale_b = mxfp4_quant_shuffled(normed, shuffle=True, scale_shuffle_padding=True)

    # Compare
    fp4_match = torch.equal(fp4_a, fp4_b)
    scale_match = torch.equal(scale_a, scale_b)

    # Also compare with fused but skip_rmsnorm on already-normed input
    (fp4_c, scale_c), _, _, _ = fused_rms_mxfp4_quant(
        x1=normed, x1_weight=weight, x1_epsilon=eps,
        res1=None, shuffle=True, scale_shuffle_padding=True,
        skip_rmsnorm=True,
    )
    fp4_c_match = torch.equal(fp4_b, fp4_c)
    scale_c_match = torch.equal(scale_b, scale_c)

    print(f"M={M}: fp4_match={fp4_match}, scale_match={scale_match}, "
          f"skip_vs_wrapper={fp4_c_match}/{scale_c_match}")

    if not fp4_match:
        diff = (fp4_a != fp4_b).sum().item()
        print(f"  fp4 diffs: {diff}/{fp4_a.numel()}")
    if not scale_match:
        diff = (scale_a != scale_b).sum().item()
        print(f"  scale diffs: {diff}/{scale_a.numel()}")

print("\nBenchmark mxfp4_quant_shuffled vs fused:")
M = 4
x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
weight = torch.randn(N, dtype=torch.bfloat16, device=device).abs() + 0.1

# Warmup
for _ in range(5):
    mxfp4_quant_shuffled(x, shuffle=True, scale_shuffle_padding=True)
    fused_rms_mxfp4_quant(x, weight, 1e-6, shuffle=True, scale_shuffle_padding=True)
torch.cuda.synchronize()

NI = 100
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(NI):
    mxfp4_quant_shuffled(x, shuffle=True, scale_shuffle_padding=True)
end.record()
torch.cuda.synchronize()
us_quant_only = start.elapsed_time(end) * 1000 / NI

start.record()
for _ in range(NI):
    fused_rms_mxfp4_quant(x, weight, 1e-6, shuffle=True, scale_shuffle_padding=True)
end.record()
torch.cuda.synchronize()
us_fused = start.elapsed_time(end) * 1000 / NI

print(f"  quant_only: {us_quant_only:.1f}μs")
print(f"  fused_norm+quant: {us_fused:.1f}μs")
print(f"  savings: {us_fused - us_quant_only:.1f}μs")
