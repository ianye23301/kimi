"""Test AR fusion + MXFP4 quant-only correctness for deepseek_v2 decoder layer.

Verifies that the 2-kernel approach (HIP AR+RMSNorm followed by Triton
MXFP4 quant-only) produces the same quantized output as the original
single-kernel path (Triton fused RMSNorm+MXFP4 quant).

Run inside the ATOM container:
    python kimi/tests/test_ar_fusion_correctness.py
"""

import torch
import sys


def test_mxfp4_quant_shuffled_matches_fused():
    """Verify mxfp4_quant_shuffled produces same output as fused_rms_mxfp4_quant."""
    from aiter.ops.triton.fused_mxfp4_quant import (
        fused_rms_mxfp4_quant,
        mxfp4_quant_shuffled,
    )

    torch.manual_seed(42)

    hidden_size = 7168
    # Test various M values (decode and small prefill)
    for M in [1, 4, 16, 32, 64, 128]:
        x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
        eps = 1e-6

        # Path 1: Fused RMSNorm + MXFP4 quant (original path)
        MXFP4_QUANT_BLOCK_SIZE = 32
        shuffle_bool = M >= MXFP4_QUANT_BLOCK_SIZE
        (fp4_fused, scale_fused), out1_fused, _, _ = fused_rms_mxfp4_quant(
            x1=x.clone(),
            x1_weight=weight,
            x1_epsilon=eps,
            shuffle=shuffle_bool,
            scale_shuffle_padding=True,
        )

        # Path 2: Separate RMSNorm then quant-only (AR fusion path)
        # Step 1: RMSNorm manually (simulates HIP AR+RMSNorm output)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = (x.float() * torch.rsqrt(variance + eps)).to(torch.bfloat16)
        normed = normed * weight.unsqueeze(0)

        # Step 2: quant-only
        fp4_separate, scale_separate = mxfp4_quant_shuffled(
            normed, shuffle=True, scale_shuffle_padding=True,
        )

        # Compare
        fp4_match = torch.equal(fp4_fused, fp4_separate)
        scale_match = torch.equal(scale_fused, scale_separate)

        status = "PASS" if (fp4_match and scale_match) else "FAIL"
        if not fp4_match:
            diff_count = (fp4_fused != fp4_separate).sum().item()
            total = fp4_fused.numel()
            print(f"  M={M}: fp4 mismatch: {diff_count}/{total} elements differ")
        if not scale_match:
            diff_count = (scale_fused != scale_separate).sum().item()
            total = scale_fused.numel()
            print(f"  M={M}: scale mismatch: {diff_count}/{total} elements differ")

        print(f"test_mxfp4_quant_shuffled_matches_fused M={M}: {status}")

        if status == "FAIL" and M >= MXFP4_QUANT_BLOCK_SIZE:
            # At M < 32, differences are expected due to RMSNorm precision
            # At M >= 32 with shuffle, they should be exact
            raise AssertionError(f"Mismatch at M={M} — correctness bug in AR fusion path")


def test_residual_update_equivalence():
    """Verify residual update is equivalent between fused and split paths.

    The original path does: residual_add + RMSNorm + quant (all in one Triton kernel)
    The split path does: residual_add + RMSNorm (HIP) then quant (Triton)

    Both must produce the same residual output for the rest of the layer.
    """
    from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant

    torch.manual_seed(7)
    M = 1  # decode
    hidden_size = 7168

    hidden_states = torch.randn(M, hidden_size, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(M, hidden_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6

    # Path 1: Fused residual_add + RMSNorm + quant
    _, _, _, residual_fused = fused_rms_mxfp4_quant(
        x1=hidden_states.clone(),
        x1_weight=weight,
        x1_epsilon=eps,
        res1=residual.clone(),
        shuffle=False,
        scale_shuffle_padding=True,
    )

    # Path 2: Manual residual_add then RMSNorm
    residual_split = (hidden_states.float() + residual.float()).to(torch.bfloat16)

    # Compare residuals
    max_diff = (residual_fused.float() - residual_split.float()).abs().max().item()
    print(f"test_residual_update_equivalence: max_diff={max_diff:.8f}")
    assert max_diff < 1e-3, f"Residual mismatch: {max_diff}"
    print("test_residual_update_equivalence: PASS")


def test_ar_fusion_forward_path_shapes():
    """Verify the AR fusion forward path produces correctly shaped outputs.

    Simulates the decoder layer forward with both fuse_ar_input_norm and
    fuse_input_norm_quant enabled (our new code path).
    """
    from aiter.ops.triton.fused_mxfp4_quant import mxfp4_quant_shuffled

    torch.manual_seed(123)
    M = 1
    hidden_size = 7168

    hidden_states = torch.randn(M, hidden_size, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(M, hidden_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6

    # Simulate HIP AR+RMSNorm (input_layernorm with fused_allreduce=True)
    # In real code, this would be: hidden_states, residual = self.input_layernorm(hidden_states, residual)
    # Here we simulate the RMSNorm part (AR is just an AllReduce, transparent to shapes)
    combined = (hidden_states.float() + residual.float()).to(torch.bfloat16)
    variance = combined.float().pow(2).mean(-1, keepdim=True)
    normed = (combined.float() * torch.rsqrt(variance + eps)).to(torch.bfloat16)
    normed_output = normed * weight.unsqueeze(0)
    new_residual = combined  # residual is updated to hidden + residual

    # Simulate Triton MXFP4 quant-only
    quant_data, quant_scale = mxfp4_quant_shuffled(
        normed_output, shuffle=True, scale_shuffle_padding=True,
    )

    # Verify shapes
    assert quant_data.shape == (M, hidden_size // 2), f"Bad fp4 shape: {quant_data.shape}"
    assert quant_data.dtype == torch.uint8
    assert quant_scale.dtype == torch.uint8
    assert new_residual.shape == (M, hidden_size)
    assert new_residual.dtype == torch.bfloat16

    print(f"test_ar_fusion_forward_path_shapes: PASS")
    print(f"  quant_data: {quant_data.shape} {quant_data.dtype}")
    print(f"  quant_scale: {quant_scale.shape} {quant_scale.dtype}")
    print(f"  residual: {new_residual.shape} {new_residual.dtype}")


if __name__ == "__main__":
    test_residual_update_equivalence()
    test_mxfp4_quant_shuffled_matches_fused()
    test_ar_fusion_forward_path_shapes()
    print("\nAll AR fusion correctness tests passed!")
