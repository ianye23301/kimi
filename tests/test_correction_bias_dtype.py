"""Test correction_bias dtype conversion in DeepseekV2MoE.

The AITER biased_grouped_topk_hip kernel requires:
  gating_output.dtype == correction_bias.dtype

The gate outputs bf16, but correction_bias is loaded as fp32 from checkpoint.
process_weights_after_loading converts it to bf16 to avoid per-step cast kernels.
"""

import torch
import torch.nn as nn


class FakeGate(nn.Module):
    """Minimal gate mock with e_score_correction_bias parameter."""

    def __init__(self, n_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, 7168, dtype=torch.bfloat16))
        # Loaded from checkpoint as float32
        self.e_score_correction_bias = nn.Parameter(
            torch.randn(n_experts, dtype=torch.float32)
        )


class FakeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_score_correction_bias = None


def process_weights_after_loading(gate, experts):
    """Mirror of DeepseekV2MoE.process_weights_after_loading logic."""
    bias = getattr(gate, "e_score_correction_bias", None)
    if bias is not None and bias.dtype != torch.bfloat16:
        gate.e_score_correction_bias = nn.Parameter(
            bias.data.to(torch.bfloat16), requires_grad=False
        )
        experts.e_score_correction_bias = gate.e_score_correction_bias


def test_bias_converted_to_bf16():
    """Correction bias should be converted from fp32 to bf16."""
    gate = FakeGate(384)
    experts = FakeExperts()

    assert gate.e_score_correction_bias.dtype == torch.float32

    process_weights_after_loading(gate, experts)

    assert gate.e_score_correction_bias.dtype == torch.bfloat16
    assert experts.e_score_correction_bias is gate.e_score_correction_bias
    assert experts.e_score_correction_bias.dtype == torch.bfloat16
    print("test_bias_converted_to_bf16: PASS")


def test_already_bf16_no_change():
    """If bias is already bf16, don't recreate the parameter."""
    gate = FakeGate(384)
    gate.e_score_correction_bias = nn.Parameter(
        torch.randn(384, dtype=torch.bfloat16)
    )
    experts = FakeExperts()

    original_ptr = gate.e_score_correction_bias.data_ptr()
    process_weights_after_loading(gate, experts)

    # Should not have been recreated
    assert gate.e_score_correction_bias.data_ptr() == original_ptr
    print("test_already_bf16_no_change: PASS")


def test_values_preserved():
    """Conversion should preserve values (within bf16 precision)."""
    gate = FakeGate(384)
    experts = FakeExperts()

    original_values = gate.e_score_correction_bias.data.clone()
    process_weights_after_loading(gate, experts)

    # bf16 has ~3 decimal digits of precision
    max_diff = (original_values.to(torch.bfloat16).float() - original_values).abs().max()
    actual_diff = (gate.e_score_correction_bias.float() - original_values).abs().max()
    assert actual_diff <= max_diff + 1e-8
    print("test_values_preserved: PASS")


def test_no_bias_does_not_crash():
    """If gate has no correction_bias, process_weights should be a no-op."""
    gate = FakeGate(384)
    delattr(gate, "e_score_correction_bias")
    experts = FakeExperts()

    # Should not raise
    bias = getattr(gate, "e_score_correction_bias", None)
    if bias is not None and bias.dtype != torch.bfloat16:
        gate.e_score_correction_bias = nn.Parameter(
            bias.data.to(torch.bfloat16), requires_grad=False
        )
    print("test_no_bias_does_not_crash: PASS")


def test_dtype_matches_gate_output():
    """Simulate gate forward and verify dtypes match for the topk kernel."""
    gate = FakeGate(384)
    experts = FakeExperts()
    experts.e_score_correction_bias = gate.e_score_correction_bias

    process_weights_after_loading(gate, experts)

    # Simulate gate forward: bf16 input → bf16 output
    hidden_states = torch.randn(4, 7168, dtype=torch.bfloat16, device="cpu")
    # Gate linear: hidden_states @ weight.T → bf16
    gating_output = hidden_states @ gate.weight.T

    assert gating_output.dtype == torch.bfloat16
    assert gate.e_score_correction_bias.dtype == torch.bfloat16
    assert gating_output.dtype == gate.e_score_correction_bias.dtype, (
        "Gate output and correction_bias must have same dtype for AITER kernel!"
    )
    print("test_dtype_matches_gate_output: PASS")


if __name__ == "__main__":
    test_bias_converted_to_bf16()
    test_already_bf16_no_change()
    test_values_preserved()
    test_no_bias_does_not_crash()
    test_dtype_matches_gate_output()
    print("\nAll correction_bias dtype tests passed!")
