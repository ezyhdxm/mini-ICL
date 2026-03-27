"""
Analysis tools for latent Markov models.

Re-exports all public analysis functions for convenient access via:
    from icl.latent_markov.analysis import <function_name>
"""

from icl.latent_markov.analysis.bayes import (
    GroupUniformKnownBayes,
    ThreeKnownPlusNewDirichletBayes,
    ThreeKnownUniformBayes,
    task_posterior_over_time,
)

from icl.latent_markov.analysis.variance import (
    get_token_conditioned_hiddens,
    plot_p1_variance,
    plot_task_vector_r2_latent,
    plot_task_vector_r2_blocked_latent,
    plot_probe_fit_r2_latent,
    plot_anova_separability_latent,
)

from icl.latent_markov.analysis.probes import (
    train_linear_hidden_predictor,
    plot_val_r2_across_layers,
    plot_averaging_r2_latent,
)

from icl.latent_markov.analysis.trajectory import (
    traj_posterior_projection_plot,
    plot_weight_row_cosine_heatmap,
    compute_weight_row_orthogonality_metrics,
    traj_post_posterior_projection_plot,
    traj_cellmean_projection_plot,
    traj_averaging_projection_plot,
)

from icl.latent_markov.analysis.posterior import (
    plot_lambda_posterior_agreement,
    plot_id_ood_loss,
    plot_latent_task_posterior,
)

from icl.latent_markov.analysis.directions import (
    calculation_direction_analysis,
    plot_task_subspace_r2_over_positions,
)

from icl.latent_markov.analysis.kl_divergence import (
    plot_kl_model_vs_two_bayes_latent,
    plot_kl_model_vs_two_bayes_latent_across_k,
    plot_kl_model_vs_two_bayes_latent_over_steps,
    plot_kl_model_vs_two_bayes_latent_transition_across_k,
)

from icl.latent_markov.analysis.ood import (
    compute_latent_ood_metrics,
    get_latent_sampler,
    get_all_samples,
)

from icl.latent_markov.analysis.interventions import (
    intervene_remove_task_subspace,
    plot_intervention_remove_task_across_layers,
    intervene_inject_task_vector,
    plot_inject_task_vector_across_layers,
    intervene_optimal_orth_direction,
    plot_optimal_orth_direction_across_layers,
    intervene_remove_ood_deltah_subspace,
    intervene_remove_bigram_subspace,
    plot_remove_bigram_subspace_across_layers,
)
