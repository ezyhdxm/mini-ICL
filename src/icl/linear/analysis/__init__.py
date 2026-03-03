"""Linear regression post-training analysis (probes, interventions, plotting).

All public symbols are re-exported here so users can do::

    from icl.linear.analysis import plot_task_posterior, intervene_remove_task_subspace

Old ``_linear_nonpadded``-suffixed names are available as aliases for backward
compatibility.
"""

# -- probes ------------------------------------------------------------------
from icl.linear.analysis.probes import (  # noqa: F401
    train_linear_hidden_predictor,
    get_task_variance,
    probe_gaussian_posterior,
)

# -- token-conditioned hiddens (P1 variance utility) ------------------------
from icl.linear.p1_variance import get_token_conditioned_hiddens  # noqa: F401

# -- P1 variance plotting ---------------------------------------------------
from icl.linear.p1_variance import plot_p1_variance  # noqa: F401

# -- posterior computation & plotting ----------------------------------------
from icl.linear.analysis.posterior import (  # noqa: F401
    task_posterior_linear_regression,
    task_posterior_over_time_linear_regression,
    task_posterior_with_gaussian_linear_regression,
    plot_kl_model_vs_two_bayes_linear,
    plot_kl_model_vs_two_bayes_linear_across_k,
    plot_task_posterior,
    plot_task_variance,
    plot_val_r2_across_layers,
    plot_id_ood_loss,
)
from icl.linear.legacy.plot_posterior_predictor_loss_vs_k import (  # noqa: F401
    plot_posterior_predictor_loss_vs_k,
)

# -- trajectory projections -------------------------------------------------
from icl.linear.analysis.trajectory import (  # noqa: F401
    traj_posterior_projection_plot,
    traj_post_posterior_projection_plot,
    plot_lambda_posterior_agreement,
)

# -- causal interventions ----------------------------------------------------
from icl.linear.analysis.interventions import (  # noqa: F401
    intervene_optimal_orth_direction,
    plot_optimal_orth_direction_across_layers,
    intervene_remove_task_subspace,
    plot_intervention_remove_task_across_layers,
    intervene_inject_task_vector,
    plot_inject_task_vector_across_layers,
    intervene_remove_suffstats_orth,
    plot_remove_suffstats_orth_across_layers,
    plot_gaussian_posterior_probe,
    intervene_remove_gaussian_direction,
    plot_remove_gaussian_direction_across_layers,
    intervene_remove_ood_deltah_subspace,
    analyze_ood_deltah_direction,
)

# -- direction analysis ------------------------------------------------------
from icl.linear.analysis.directions import (  # noqa: F401
    calculation_direction_analysis,
    plot_task_subspace_r2_over_positions,
)

# ---------------------------------------------------------------------------
# Backward-compatible aliases (old ``_linear_nonpadded`` suffixed names)
# ---------------------------------------------------------------------------
train_linear_hidden_predictor_linear_nonpadded = train_linear_hidden_predictor
get_token_conditioned_hiddens_linear_nonpadded = get_token_conditioned_hiddens
probe_gaussian_posterior_linear_nonpadded = probe_gaussian_posterior

plot_p1_variance_linear_nonpadded = plot_p1_variance
plot_task_posterior_linear_nonpadded = plot_task_posterior
plot_val_r2_across_layers_linear_nonpadded = plot_val_r2_across_layers
plot_posterior_predictor_loss_vs_k_linear_nonpadded = plot_posterior_predictor_loss_vs_k
plot_id_ood_loss_linear_nonpadded = plot_id_ood_loss

traj_posterior_projection_plot_linear_nonpadded = traj_posterior_projection_plot
traj_post_posterior_projection_plot_linear_nonpadded = traj_post_posterior_projection_plot
plot_lambda_posterior_agreement_linear_nonpadded = plot_lambda_posterior_agreement

intervene_optimal_orth_direction_linear_nonpadded = intervene_optimal_orth_direction
plot_optimal_orth_direction_across_layers_linear_nonpadded = plot_optimal_orth_direction_across_layers
intervene_remove_task_subspace_linear_nonpadded = intervene_remove_task_subspace
plot_intervention_remove_task_across_layers_linear_nonpadded = plot_intervention_remove_task_across_layers
intervene_inject_task_vector_linear_nonpadded = intervene_inject_task_vector
plot_inject_task_vector_across_layers_linear_nonpadded = plot_inject_task_vector_across_layers
intervene_remove_suffstats_orth_linear_nonpadded = intervene_remove_suffstats_orth
plot_remove_suffstats_orth_across_layers_linear_nonpadded = plot_remove_suffstats_orth_across_layers
plot_gaussian_posterior_probe_linear_nonpadded = plot_gaussian_posterior_probe
intervene_remove_gaussian_direction_linear_nonpadded = intervene_remove_gaussian_direction
plot_remove_gaussian_direction_across_layers_linear_nonpadded = plot_remove_gaussian_direction_across_layers
intervene_remove_ood_deltah_subspace_linear_nonpadded = intervene_remove_ood_deltah_subspace

calculation_direction_analysis_linear_nonpadded = calculation_direction_analysis
analyze_ood_deltah_direction_linear_nonpadded = analyze_ood_deltah_direction
plot_task_subspace_r2_over_positions_linear_nonpadded = plot_task_subspace_r2_over_positions
