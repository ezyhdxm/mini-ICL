"""Regenerate the linear OLS probe R^2 figure (Plot 3 of
``plot_optimal_orth_direction_across_layers``) and the matching filtered
intervention figure (Plot 4).

Mirrors the first invocation of the function in
``notebooks/Linear.ipynb`` and saves ``fig_r2`` and ``fig_filt`` to the
paper's Figs folder.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

from icl.linear.analysis import plot_optimal_orth_direction_across_layers

# Fix the optimisation seed so the V_opt initialisation is reproducible.
# The layer-by-layer V_opt is found by stochastic Adam from a random init,
# so without this both ``fig_r2`` and ``fig_filt`` shift slightly between
# runs (especially around the layer-4 R^2 transition dip).
torch.manual_seed(0)


# Existing checkpoint trained with the canonical E2 hyperparameters
# (n_tasks=3, n_minor_tasks=1024, p_minor=0.1, noise_scale=0.5, n_layer=16,
# total_steps=30_000, warmup_steps=15_000, batch_size=256, max_grad_norm=1.0).
# Hard-coded because the legacy wandb.project name in lr_config differs from
# the value baked into the saved config.json, so get_exp_name() no longer
# round-trips to this hash.
exp_name = "train_0826e18eead7b4d4df074d56cae39dd1"

_figs_dir = os.path.join(
    os.path.dirname(__file__), '..', '..',
    'representation-geometry-two-modes', 'Figs',
)
save_path = os.path.join(_figs_dir, 'linear_ols_probe_r2.png')
save_path_filt = os.path.join(_figs_dir, 'linear_filtered_intervention.png')

fig, fig_loss, fig_r2, fig_filt, fig_per_task, all_results = (
    plot_optimal_orth_direction_across_layers(
        exp_name=exp_name,
        layers=range(16),
        n_directions=2,
        n_opt_steps=25,
        opt_lr=0.01,
        opt_B=256,
        patience=50,
        grad_clip_norm=1.0,
        n_samples_eval=2**10,
        n_samples_probe=2**10,
        n_ood=2**10,
        scale=2,
        opt_source="minor",
        extraction_point="post_mlp",
        probe_method="averaging",
        # Lowered from the default 0.1 to 0.05: keeps the marginally explained
        # second V_opt direction at layer 10 (joint-probe R^2 ~0.10), so the
        # filtered bar there matches V_opt instead of partially collapsing.
        # Layer 4 is left as the single visible exception where one V_opt
        # direction has joint-probe R^2 < 0.05 and is dropped; that outlier
        # is discussed in the appendix narrative.
        r2_filter_threshold=0.05,
        figsize=(6, 4),
        show=False,
    )
)

fig_r2.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"\nSaved fig_r2 to: {save_path}")

fig_filt.savefig(save_path_filt, dpi=300, bbox_inches="tight")
print(f"Saved fig_filt to: {save_path_filt}")
