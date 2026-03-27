"""
Direction analysis, attention-head probes, and head ablation for latent Markov.

Re-exports all public symbols so that existing imports continue to work:
    from icl.latent_markov.analysis.directions import calculation_direction_analysis
"""

from icl.latent_markov.analysis.directions._subspace import (
    calculation_direction_analysis,
    plot_task_subspace_r2_over_positions,
)

from icl.latent_markov.analysis.directions._attention import (
    previous_token_head_score,
    induction_head_score,
    get_attention_scores_nonpadded,
    plot_head_scores,
)

from icl.latent_markov.analysis.directions._ablation import (
    head_ablation_experiment,
)

__all__ = [
    "calculation_direction_analysis",
    "plot_task_subspace_r2_over_positions",
    "previous_token_head_score",
    "induction_head_score",
    "get_attention_scores_nonpadded",
    "plot_head_scores",
    "head_ablation_experiment",
]
