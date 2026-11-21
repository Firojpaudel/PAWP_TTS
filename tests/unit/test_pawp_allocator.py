"""Unit tests for pure PAWP rank allocation, MAC breakeven, and regrowth math."""

from __future__ import annotations

from pawp_tts.domain.entities.models import LayerSpec
from pawp_tts.domain.pawp.accounting import low_rank_mac_breakeven, serialized_sizes
from pawp_tts.domain.pawp.allocator import (
    allocate_ranks_under_budget,
    rebalance_ranks_for_error_regrowth,
)


def test_mac_breakeven_and_serialized_sizes() -> None:
    assert low_rank_mac_breakeven(128, 256) == 85
    fp32, fp16, int8, int4 = serialized_sizes(1_000_000)
    assert fp32 == 4_000_000
    assert fp16 == 2_000_000
    assert int8 == 1_000_000
    assert int4 == 500_000


def test_knapsack_allocator_respects_budget_and_prioritizes_sensitive_layers() -> None:
    layers = (
        LayerSpec("layer_high", "pw", (64, 64), 64, 64, 4096),
        LayerSpec("layer_low", "pw", (64, 64), 64, 64, 4096),
    )
    sensitivities = {
        "layer_high": (10.0, 9.0, 8.0, 7.0, 6.0, 5.0),
        "layer_low": (0.5, 0.4, 0.3, 0.2, 0.1, 0.05),
    }
    budget = 6 * (64 + 64)  # exactly 6 total rank steps across both layers
    plan = allocate_ranks_under_budget(layers, sensitivities, target_budget=budget, min_rank=1)
    assert plan.allocated_params <= budget
    assert plan.layer_ranks["layer_high"] > plan.layer_ranks["layer_low"]
    assert plan.layer_ranks["layer_high"] == 5
    assert plan.layer_ranks["layer_low"] == 1


def test_error_driven_regrowth_strictly_conserves_budget() -> None:
    layers = (
        LayerSpec("enc_0", "pw", (64, 64), 64, 64, 4096),
        LayerSpec("ac_0", "pw", (64, 64), 64, 64, 4096),
    )
    init_sens = {
        "enc_0": (5.0, 4.0, 3.0, 2.0, 1.0),
        "ac_0": (5.0, 4.0, 3.0, 2.0, 1.0),
    }
    budget = 6 * 128
    initial_plan = allocate_ranks_under_budget(layers, init_sens, target_budget=budget, min_rank=1)

    # Diagnostic error feedback flags ac_0 as critical
    error_sens = {
        "enc_0": (0.5, 0.2, 0.1, 0.05, 0.01),
        "ac_0": (12.0, 10.0, 8.0, 6.0, 4.0),
    }
    regrown = rebalance_ranks_for_error_regrowth(
        initial_plan,
        layers,
        error_sens,
        regrowth_budget_delta=2 * 128,
        min_rank=1,
    )
    assert regrown.allocated_params <= budget
    assert regrown.layer_ranks["ac_0"] > initial_plan.layer_ranks["ac_0"]
    assert regrown.layer_ranks["enc_0"] < initial_plan.layer_ranks["enc_0"]
