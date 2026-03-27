"""Linear regression post-training analysis (probes, interventions, plotting).

All public symbols are re-exported here so users can do::

    from icl.linear.analysis import plot_task_posterior, intervene_remove_task_subspace
"""

# -- probes ------------------------------------------------------------------
from icl.linear.analysis.probes import (  # noqa: F401
    train_linear_hidden_predictor,
    plot_task_vector_r2_linear,
    plot_averaging_r2_linear,
    plot_ancova_separability_linear,
)

# -- posterior computation & plotting ----------------------------------------
from icl.linear.analysis.posterior import (  # noqa: F401
    task_posterior_linear_regression,
    task_posterior_over_time_linear_regression,
    task_posterior_with_gaussian_linear_regression,
    plot_kl_model_vs_two_bayes_linear_over_steps,
    plot_kl_model_vs_two_bayes_linear_transition_across_k,
    plot_task_posterior,
    plot_val_r2_across_layers,
    plot_id_ood_loss,
)

# -- trajectory projections -------------------------------------------------
from icl.linear.analysis.trajectory import (  # noqa: F401
    traj_posterior_projection_plot,
    traj_averaging_projection_plot,
)

# -- causal interventions ----------------------------------------------------
from icl.linear.analysis.interventions import (  # noqa: F401
    intervene_optimal_orth_direction,
    plot_optimal_orth_direction_across_layers,
    intervene_remove_task_subspace,
    plot_intervention_remove_task_across_layers,
    intervene_inject_task_vector,
    plot_inject_task_vector_across_layers,
    intervene_remove_ood_deltah_subspace,
    intervene_averaging_injection,
)
