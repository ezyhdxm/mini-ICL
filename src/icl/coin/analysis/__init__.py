"""Coin task analysis — public re-exports."""

from icl.coin.analysis._helpers import (
    compute_hiddens_multi_coin,
    compute_hiddens_token_conditioned_coin,
    _stable_rank_from_gram,
    compute_stable_rank_at_real_positions,
    get_token_conditioned_hiddens_coin,
)

from icl.coin.analysis.posterior import (
    plot_coin_task_posterior,
    plot_id_ood_loss_coin,
    plot_kl_lambda_vs_posterior_coin,
    plot_lambda_posterior_agreement_coin,
)

from icl.coin.analysis.probes import (
    train_linear_hidden_predictor_coin,
    plot_val_r2_across_layers_coin,
)

from icl.coin.legacy.coin_analysis_legacy import (
    plot_val_r2_across_layers_multi_k,
)

from icl.coin.legacy.softmax_probes import (
    train_linear_softmax_posterior_predictor_coin,
    train_linear_unigram_count_predictor_coin,
)

from icl.coin.analysis.variance import (
    get_task_variance_coin,
    plot_task_variance_coin,
    plot_p1_variance_coin,
    plot_task_vector_r2_coin,
    plot_probe_fit_r2_coin,
    plot_anova_separability_coin,
)

from icl.coin.analysis.trajectory import (
    traj_projection_plot_coin,
    traj_posterior_projection_plot_coin,
    traj_post_posterior_projection_plot_coin,
)

from icl.coin.analysis.directions import (
    decompose_hidden_posterior_vs_unigram_coin,
    plot_decomposition_across_layers_coin,
)

from icl.coin.legacy.orth_decomposition import (
    decompose_orth_projection_coin,
    plot_orth_decomposition_across_layers_coin,
)

from icl.coin.analysis.interventions import (
    _historical_injection_test_next_token_coin,
    historical_injection_coin,
    plot_historical_injection_coin,
    intervene_remove_task_subspace_coin,
    plot_intervention_remove_task_across_layers_coin,
    intervene_remove_unigram_orth_coin,
    plot_intervention_across_layers_coin,
    intervene_optimal_orth_direction_coin,
    plot_optimal_orth_direction_across_layers_coin,
    intervene_inject_task_vector_coin,
    plot_inject_task_vector_across_layers_coin,
)

__all__ = [
    # _helpers
    "compute_hiddens_multi_coin",
    "compute_hiddens_token_conditioned_coin",
    "_stable_rank_from_gram",
    "compute_stable_rank_at_real_positions",
    "get_token_conditioned_hiddens_coin",
    # posterior
    "plot_coin_task_posterior",
    "plot_id_ood_loss_coin",
    "plot_kl_lambda_vs_posterior_coin",
    "plot_lambda_posterior_agreement_coin",
    # probes
    "train_linear_softmax_posterior_predictor_coin",
    "train_linear_unigram_count_predictor_coin",
    "train_linear_hidden_predictor_coin",
    "plot_val_r2_across_layers_coin",
    "plot_val_r2_across_layers_multi_k",
    # variance
    "get_task_variance_coin",
    "plot_task_variance_coin",
    "plot_p1_variance_coin",
    "plot_task_vector_r2_coin",
    "plot_probe_fit_r2_coin",
    "plot_anova_separability_coin",
    # trajectory
    "traj_projection_plot_coin",
    "traj_posterior_projection_plot_coin",
    "traj_post_posterior_projection_plot_coin",
    # directions
    "decompose_hidden_posterior_vs_unigram_coin",
    "plot_decomposition_across_layers_coin",
    "decompose_orth_projection_coin",
    "plot_orth_decomposition_across_layers_coin",
    # interventions
    "_historical_injection_test_next_token_coin",
    "historical_injection_coin",
    "plot_historical_injection_coin",
    "intervene_remove_task_subspace_coin",
    "plot_intervention_remove_task_across_layers_coin",
    "intervene_remove_unigram_orth_coin",
    "plot_intervention_across_layers_coin",
    "intervene_optimal_orth_direction_coin",
    "plot_optimal_orth_direction_across_layers_coin",
    "intervene_inject_task_vector_coin",
    "plot_inject_task_vector_across_layers_coin",
]
