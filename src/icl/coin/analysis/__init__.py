"""Coin task analysis — public re-exports."""

from icl.coin.analysis._helpers import (
    compute_hiddens_token_conditioned_coin,
    get_token_conditioned_hiddens_coin,
)

from icl.coin.analysis.posterior import (
    plot_coin_task_posterior,
    plot_id_ood_loss_coin,
    plot_lambda_posterior_agreement_coin,
)

from icl.coin.analysis.probes import (
    train_linear_hidden_predictor_coin,
    plot_val_r2_across_layers_coin,
    plot_averaging_r2_coin,
)

from icl.coin.analysis.variance import (
    plot_task_vector_r2_coin,
    plot_anova_separability_coin,
)

from icl.coin.analysis.trajectory import (
    traj_posterior_projection_plot_coin,
    traj_post_posterior_projection_plot_coin,
    traj_cellmean_projection_plot_coin,
    traj_averaging_projection_plot_coin,
)

from icl.utils.ood_major_projection_r2 import (
    plot_maj_r2_ood_across_steps_coin,
)

from icl.coin.analysis.interventions import (
    intervene_remove_task_subspace_coin,
    plot_intervention_remove_task_across_layers_coin,
    plot_optimal_orth_direction_across_layers_coin,
    intervene_inject_task_vector_coin,
    plot_inject_task_vector_across_layers_coin,
    intervene_inject_posterior_coin,
    intervene_direct_injection_coin,
    intervene_averaging_injection_coin,
    plot_inject_posterior_across_layers_coin,
)

__all__ = [
    # _helpers
    "compute_hiddens_token_conditioned_coin",
    "get_token_conditioned_hiddens_coin",
    # posterior
    "plot_coin_task_posterior",
    "plot_id_ood_loss_coin",
    "plot_lambda_posterior_agreement_coin",
    # probes
    "train_linear_hidden_predictor_coin",
    "plot_val_r2_across_layers_coin",
    "plot_averaging_r2_coin",
    # variance
    "plot_task_vector_r2_coin",
    "plot_anova_separability_coin",
    # trajectory
    "traj_posterior_projection_plot_coin",
    "traj_post_posterior_projection_plot_coin",
    "traj_cellmean_projection_plot_coin",
    "traj_averaging_projection_plot_coin",
    # ood
    "plot_maj_r2_ood_across_steps_coin",
    # interventions
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
