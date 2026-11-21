"""Pure budget-constrained rank allocation and phonetic error-driven regrowth solvers."""

from __future__ import annotations

from pawp_tts.domain.entities.models import LayerSpec, RankAllocationPlan


def allocate_ranks_under_budget(
    layers: tuple[LayerSpec, ...],
    singular_sensitivities: dict[str, tuple[float, ...]],
    target_budget: int,
    min_rank: int = 1,
    basis_counts: dict[str, int] | None = None,
    fixed_overhead_params: int = 0,
) -> RankAllocationPlan:
    """
    Solve discrete budget-constrained rank allocation greedily by marginal
    phonetic sensitivity gain per parameter (`delta_I / delta_P`).
    """
    if target_budget < fixed_overhead_params:
        raise ValueError("Target budget is smaller than fixed overhead parameters.")

    ranks: dict[str, int] = {}
    used_params = fixed_overhead_params
    total_score = 0.0

    for layer in layers:
        step_cost = layer.in_features + layer.out_features
        max_rank = min(layer.in_features, layer.out_features)
        s_vec = singular_sensitivities.get(layer.name, ())
        init_r = min(min_rank, max_rank, len(s_vec)) if s_vec else 0
        if used_params + init_r * step_cost > target_budget:
            init_r = max(0, (target_budget - used_params) // max(step_cost, 1))
        ranks[layer.name] = init_r
        used_params += init_r * step_cost
        total_score += sum(s_vec[:init_r])

    while True:
        best_layer: str | None = None
        best_efficiency = -1.0
        best_gain = 0.0
        best_cost = 0

        for layer in layers:
            cur_r = ranks[layer.name]
            s_vec = singular_sensitivities.get(layer.name, ())
            max_r = min(layer.in_features, layer.out_features, len(s_vec))
            if cur_r >= max_r:
                continue
            step_cost = layer.in_features + layer.out_features
            if used_params + step_cost > target_budget:
                continue
            gain = float(s_vec[cur_r])
            eff = gain / max(step_cost, 1)
            if eff > best_efficiency:
                best_efficiency = eff
                best_layer = layer.name
                best_gain = gain
                best_cost = step_cost

        if best_layer is None or best_gain <= 0.0:
            break

        ranks[best_layer] += 1
        used_params += best_cost
        total_score += best_gain

    return RankAllocationPlan(
        target_budget=target_budget,
        allocated_params=used_params,
        layer_ranks=ranks,
        pool_basis_counts=dict(basis_counts or {}),
        expected_score=total_score,
    )


def rebalance_ranks_for_error_regrowth(
    current_plan: RankAllocationPlan,
    layers: tuple[LayerSpec, ...],
    error_sensitivities: dict[str, tuple[float, ...]],
    regrowth_budget_delta: int,
    min_rank: int = 1,
) -> RankAllocationPlan:
    """
    Reallocate capacity toward error-linked layers while trimming equal parameter mass
    from low-error-sensitivity layers so total allocated params never exceed target_budget.
    """
    layer_by_name = {layer.name: layer for layer in layers}
    ranks = dict(current_plan.layer_ranks)
    used_params = current_plan.allocated_params

    # 1. Trim `regrowth_budget_delta` params from least error-sensitive active ranks
    freed = 0
    while freed < regrowth_budget_delta:
        worst_layer: str | None = None
        worst_eff = float("inf")
        worst_cost = 0

        for name, rank in ranks.items():
            if rank <= min_rank:
                continue
            layer = layer_by_name[name]
            step_cost = layer.in_features + layer.out_features
            s_vec = error_sensitivities.get(name, ())
            marginal = float(s_vec[rank - 1]) if rank - 1 < len(s_vec) else 0.0
            eff = marginal / max(step_cost, 1)
            if eff < worst_eff:
                worst_eff = eff
                worst_layer = name
                worst_cost = step_cost

        if worst_layer is None:
            break
        ranks[worst_layer] -= 1
        used_params -= worst_cost
        freed += worst_cost

    # 2. Grow ranks in highest error-sensitivity layers strictly within budget
    while True:
        best_layer: str | None = None
        best_eff = -1.0
        best_cost = 0

        for name, layer in layer_by_name.items():
            cur_r = ranks.get(name, 0)
            s_vec = error_sensitivities.get(name, ())
            max_r = min(layer.in_features, layer.out_features, len(s_vec))
            if cur_r >= max_r:
                continue
            step_cost = layer.in_features + layer.out_features
            if used_params + step_cost > current_plan.target_budget:
                continue
            marginal = float(s_vec[cur_r])
            eff = marginal / max(step_cost, 1)
            if eff > best_eff:
                best_eff = eff
                best_layer = name
                best_cost = step_cost

        if best_layer is None or best_eff <= 0.0:
            break
        ranks[best_layer] = ranks.get(best_layer, 0) + 1
        used_params += best_cost

    score = 0.0
    for name, rank in ranks.items():
        s_vec = error_sensitivities.get(name, ())
        score += sum(float(v) for v in s_vec[:rank])

    return RankAllocationPlan(
        target_budget=current_plan.target_budget,
        allocated_params=used_params,
        layer_ranks=ranks,
        pool_basis_counts=dict(current_plan.pool_basis_counts),
        expected_score=score,
    )
