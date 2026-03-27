"""Posterior computation and plotting for linear regression analysis."""

from icl.linear.analysis.posterior._compute import (  # noqa: F401
    task_posterior_linear_regression,
    task_posterior_over_time_linear_regression,
    task_posterior_with_gaussian_linear_regression,
)
from icl.linear.analysis.posterior._kl_plots import (  # noqa: F401
    plot_kl_model_vs_two_bayes_linear_over_steps,
    plot_kl_model_vs_two_bayes_linear_transition_across_k,
)
from icl.linear.analysis.posterior._analysis_plots import (  # noqa: F401
    plot_task_posterior,
    plot_val_r2_across_layers,
    plot_id_ood_loss,
)
