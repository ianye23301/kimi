"""Debug scale mismatch - check if it's just padding."""
import os
os.environ.pop("HIP_VISIBLE_DEVICES", None)
import torch
torch.cuda.set_device(4)

from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant, mxfp4_quant_shuffled

torch.manual_seed(42)
device = "cuda"
N = 7168
M = 4

x = torch.randn(M, N, dtype=torch.bfloat16, device=device)

# Path 1: fused with skip_rmsnorm (should be equivalent)
(fp4_a, scale_a), _, _, _ = fused_rms_mxfp4_quant(
    x1=x, x1_weight=x[0], x1_epsilon=0.0,
    res1=None, shuffle=True, scale_shuffle_padding=True,
    skip_rmsnorm=True,
)

# Path 2: wrapper
fp4_b, scale_b = mxfp4_quant_shuffled(x, shuffle=True, scale_shuffle_padding=True)

print(f"fp4 match: {torch.equal(fp4_a, fp4_b)}")
print(f"scale match: {torch.equal(scale_a, scale_b)}")
print(f"scale_a shape: {scale_a.shape}, scale_b shape: {scale_b.shape}")

# Check valid region only (M=4, N=7168, BLOCK=32 -> 224 scale cols, padded to 224)
# Scale_N_valid = ceil(7168/32) = 224
# Scale_N_pad = ceil(224/8)*8 = 224 (already multiple of 8)
# Scale_M_pad = ceil(4/256)*256 = 256
# So valid region is [0:4, 0:224], padded to [256, 224]
valid_scale_a = scale_a[:M, :224]
valid_scale_b = scale_b[:M, :224]
print(f"valid scale match: {torch.equal(valid_scale_a, valid_scale_b)}")
if not torch.equal(valid_scale_a, valid_scale_b):
    diffs = (valid_scale_a != valid_scale_b).sum().item()
    print(f"  valid diffs: {diffs}/{valid_scale_a.numel()}")

# Now test: does the wrapper match calling fused with same input directly (no skip)?
# This checks quant_only produces same result as norm+quant when input is already normed
# (i.e., the quant part should be identical)
normed = x  # pretend x is already normed
fp4_skip, scale_skip = mxfp4_quant_shuffled(normed)

# Compare with fused_rms on same data (will apply RMSNorm which changes values)
(fp4_fused, scale_fused), _, _, _ = fused_rms_mxfp4_quant(
    normed, torch.ones(N, dtype=torch.bfloat16, device=device), 1e-6,
    shuffle=True, scale_shuffle_padding=True
)
print(f"\nskip vs fused+norm on same data: fp4={torch.equal(fp4_skip, fp4_fused)} (expected False - norm changes values)")

# The key test: two calls to fused_rms_mxfp4_quant with skip_rmsnorm should be deterministic
(fp4_d1, scale_d1), _, _, _ = fused_rms_mxfp4_quant(
    x, x[0], 0.0, shuffle=True, scale_shuffle_padding=True, skip_rmsnorm=True)
(fp4_d2, scale_d2), _, _, _ = fused_rms_mxfp4_quant(
    x, x[0], 0.0, shuffle=True, scale_shuffle_padding=True, skip_rmsnorm=True)
print(f"\ndeterministic: fp4={torch.equal(fp4_d1, fp4_d2)}, scale={torch.equal(scale_d1, scale_d2)}")
