"""Intervention experiments for the Coin task."""

from icl.coin.analysis.interventions.remove_task import (
    intervene_remove_task_subspace_coin,
    plot_intervention_remove_task_across_layers_coin,
)
from icl.coin.analysis.interventions.optimal_orth import (
    plot_optimal_orth_direction_across_layers_coin,
)
from icl.coin.analysis.interventions.inject_task import (
    intervene_inject_task_vector_coin,
    plot_inject_task_vector_across_layers_coin,
)
from icl.coin.analysis.interventions.inject_posterior import (
    intervene_inject_posterior_coin,
    intervene_direct_injection_coin,
    intervene_averaging_injection_coin,
    plot_inject_posterior_across_layers_coin,
)
from icl.utils.inject_common import (  # noqa: F401
    plot_inject_posterior_per_position,
)

__all__ = [
    "intervene_remove_task_subspace_coin",
    "plot_intervention_remove_task_across_layers_coin",
    "plot_optimal_orth_direction_across_layers_coin",
    "intervene_inject_task_vector_coin",
    "plot_inject_task_vector_across_layers_coin",
    "intervene_inject_posterior_coin",
    "intervene_direct_injection_coin",
    "intervene_averaging_injection_coin",
    "plot_inject_posterior_across_layers_coin",
]
