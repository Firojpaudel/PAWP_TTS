"""Pure parameter and compute-breakeven accounting utilities."""

from __future__ import annotations


def low_rank_mac_breakeven(in_features: int, out_features: int) -> int:
    """Return max rank r where U*V uses strictly fewer MACs than dense W."""
    if in_features <= 0 or out_features <= 0:
        raise ValueError("Dimensions must be positive integers.")
    return (in_features * out_features) // (in_features + out_features)


def low_rank_param_count(in_features: int, out_features: int, rank: int) -> int:
    """Return exact number of parameters for an (out x rank) @ (rank x in) residual."""
    if rank <= 0:
        return 0
    return rank * (in_features + out_features)


def serialized_sizes(unique_params: int) -> tuple[int, int, int, int]:
    """Return raw tensor byte footprint across FP32, FP16/BF16, INT8, and INT4."""
    fp32 = unique_params * 4
    fp16 = unique_params * 2
    int8 = unique_params
    int4 = (unique_params + 1) // 2
    return fp32, fp16, int8, int4
