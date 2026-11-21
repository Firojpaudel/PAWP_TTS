"""PAWP pure allocation and accounting math package."""

from pawp_tts.domain.pawp.accounting import (
    low_rank_mac_breakeven,
    low_rank_param_count,
    serialized_sizes,
)
from pawp_tts.domain.pawp.allocator import (
    allocate_ranks_under_budget,
    rebalance_ranks_for_error_regrowth,
)

__all__ = [
    "allocate_ranks_under_budget",
    "low_rank_mac_breakeven",
    "low_rank_param_count",
    "rebalance_ranks_for_error_regrowth",
    "serialized_sizes",
]
