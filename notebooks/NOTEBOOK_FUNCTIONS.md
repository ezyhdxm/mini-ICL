# Functions used in analysis notebooks

This reference lists **imported callables** and **other library/API calls** that appear in the code cells of:

- `notebooks/Dyck.ipynb`
- `notebooks/coins.ipynb`
- `notebooks/Latent.ipynb`
- `notebooks/Linear.ipynb`

**How it was generated:** Python `ast` parsing of each notebook’s code cells (imports plus call names). Names are the symbols as used in the notebook (aliases shown in parentheses). Brief descriptions are taken from docstrings or from the implementation role in this codebase.

**Companion:** `NOTEBOOK_UNUSED_FUNCTIONS.md` is produced by `scripts/find_unused_notebook_functions.py` and resolves imports to defining modules (including `from icl.coin.coin_analysis import …`, which re-exports `icl.coin.analysis` via `import *` with a static `__all__`). A function listed as used here should not appear under `icl.coin` in that report unless the notebook calls it under a different import path than the analyzer sees.

---

## Standard / third-party modules

| Symbol | Typical import | Role |
|--------|------------------|------|
| **NumPy** | `import numpy as np` | Arrays, `np.linspace`, `np.concatenate`, `np.mean`, etc. |
| **Matplotlib** | `import matplotlib.pyplot as plt` | Figures; notebooks call `plt.show()` to display. |
| **PyTorch** | `import torch` | Tensors and ops (e.g. `torch.cdist` in Linear); underlying models require `torch`. |
| **tqdm** | `from tqdm.notebook import trange` | Notebook-friendly progress bar over a range (`trange`). |
| **notebook_utils** | `import icl.utils.notebook_utils as nu` | Loading checkpoints, configs, samplers (`nu.load_everything`, etc.). |

---

## `icl.utils.notebook_utils` (coins notebook)

| Function | Description |
|----------|-------------|
| **`load_everything(task_name, train_folder, get_log=False)`** | Loads config, trained model, and sampler from under `results/<task_name>/…`; optional training log. Adjusts paths when the cwd is `notebooks/`. |

---

## Notebook: `Dyck.ipynb`

| Function | Import (as used) | Description |
|----------|-------------------|-------------|
| `get_exp_name` | `icl.utils.unified_interface` | Builds the canonical experiment folder name from task name, `k`, and hyperparameters (matches training). |
| `unified_train` | `icl.utils.unified_interface` | Single entry point to train Dyck (or other) transformer experiments with a unified config. |
| `plot_dyck_task_posterior` | `icl.utils.unified_interface` (re-export) | Plots Dyck task / posterior visualization (see `icl.dyck.analysis.posterior_plots`). |
| `plot_id_ood_loss_dyck` | `icl.utils.unified_interface` (re-export) | ID vs OOD loss curves for Dyck. |
| `plot_ood_loss_vs_steps_dyck` | `icl.utils.unified_ood_analysis` | OOD loss vs training step for Dyck. |
| `plot_kl_model_vs_two_bayes_dyck` | `icl.dyck.dyck` | KL between model predictions and a two-component Bayes reference (Dyck). |
| `plot_kl_model_vs_two_bayes_dyck_across_k` | `icl.dyck.dyck` | Same KL diagnostic, compared across `k` (minor-task count / regime). |
| `plot_task_vector_r2_dyck` | `icl.dyck.analysis` | Task-vector \(R^2\) style diagnostic for Dyck representations. |
| `plot_2d_scatter` | `icl.dyck.dyck_prefix_probe` | 2D scatter of prefix probes, hierarchical colouring by prefix length. |
| `plot_3d_scatter` | `icl.dyck.dyck_prefix_probe` | 3D variant of the prefix-probe scatter. |
| `plot_accuracy_bar` | `icl.dyck.dyck_prefix_probe` | Bar chart of validation accuracy vs prefix length. |
| `plot_training_loss` | `icl.dyck.dyck_prefix_probe` | Simple training-loss curve for the prefix probe. |
| `run_prefix_probe` | `icl.dyck.dyck_prefix_probe` | End-to-end: load model, collect prefix data, train probe, return results for plots. |

**Other calls in cells:** `plt.show`, `print`, `range`, `list`, `results.get` (on loaded result dicts).

---

## Notebook: `coins.ipynb`

| Function | Import (as used) | Description |
|----------|-------------------|-------------|
| `get_exp_name` | `icl.utils.unified_interface` | Standard experiment name for coin runs. |
| `unified_train` | `icl.utils.unified_interface` | Trains coin (or other) models via unified config. |
| `plot_coin_task_posterior` | `icl.coin.coin_analysis` (shim → `icl.coin.analysis`) | Visualizes coin task posterior / belief state. |
| `plot_id_ood_loss_coin` | `icl.coin.coin_analysis` | ID vs OOD loss for the coin task. |
| `plot_kl_model_vs_two_bayes_coin_transition_across_k` | `icl.coin.coin` | KL vs two-Bayes baseline across transitions and `k`. |
| `plot_anova_separability_coin` | `icl.coin.analysis` | ANOVA-style separability plot for coin hidden states. |
| `plot_averaging_r2_coin` | `icl.coin.analysis` | \(R^2\) for averaging / task-vector geometry (coin). |
| `plot_task_vector_r2_coin` | `icl.coin.analysis` | Task-vector \(R^2\) across layers/settings (coin). |
| `plot_val_r2_across_layers_coin` | `icl.coin.analysis.probes` | Sweep linear probe \(h \sim\) [posterior, one-hot] across layers; bar plot of \(R^2\). |
| `train_linear_hidden_predictor_coin` | `icl.coin.analysis.probes` | Joint linear map from hidden state to task + token (OLS-style probe). |
| `plot_inject_task_vector_across_layers_coin` | `icl.coin.coin_analysis` | Intervention: inject task vector; effect across layers. |
| `plot_intervention_remove_task_across_layers_coin` | `icl.coin.coin_analysis` | Remove task subspace intervention across layers. |
| `plot_optimal_orth_direction_across_layers_coin` | `icl.coin.coin_analysis` | Optimal orthogonal direction intervention plot (coin). |
| `traj_posterior_projection_plot_coin` | `icl.coin.coin_analysis` | Trajectory of hidden states projected along posterior-related directions. |
| `traj_averaging_projection_plot_coin` | `icl.coin.coin_analysis` | Trajectory projection for averaging-based task signal. |
| `plot_maj_r2_ood_across_steps_coin` | `icl.utils.ood_major_projection_r2` | “Major” \(R^2\) on OOD vs training step for coin. |
| `plot_task_vector_geometry` | `icl.utils.unified_plot` | Compares task vectors estimated at different positions. |
| `intervene_averaging_injection_coin` | `icl.coin.analysis.interventions` | Posterior averaging–style injection intervention (coin). |
| `intervene_residual_removal` | `icl.utils.unified_plot` | Ablates residual component in an additive decomposition; measures output change. |
| `intervene_scale_task_component` | `icl.utils.unified_plot` | Scales task-subspace component; measures sensitivity. |
| `trange` | `tqdm.notebook` | `tqdm` wrapper over `range` for progress display. |

**Other calls in cells:** `nu.load_everything`, `np.linspace`, `np.concatenate`, `max`, `sorted`, `list`, `print`, `range`, dict keys on `anova_results` / `pos_dict`, etc.

---

## Notebook: `Latent.ipynb`

| Function | Import (as used) | Description |
|----------|-------------------|-------------|
| `get_exp_name` | `icl.utils.unified_interface` | Experiment naming for latent Markov runs. |
| `unified_train` | `icl.utils.unified_interface` | Trains latent Markov experiments. |
| `get_latent_sampler` | `icl.latent_markov.analysis.ood` | Builds sampler for OOD/minor evaluation on latent task. |
| `plot_latent_task_posterior` | `icl.latent_markov.analysis.posterior` | Samples latent Markov sequences and plots Bayesian task posterior. |
| `plot_id_ood_loss` | `icl.latent_markov.analysis` | ID vs OOD loss (latent task). |
| `plot_kl_model_vs_two_bayes_latent_transition_across_k` | `icl.latent_markov.analysis` | KL vs two-Bayes across transitions and `k`. |
| `plot_anova_separability_latent` | `icl.latent_markov.analysis` | ANOVA-style separability (latent). |
| `plot_averaging_r2_latent` | `icl.latent_markov.analysis` | Averaging \(R^2\) diagnostic. |
| `plot_task_vector_r2_latent` | `icl.latent_markov.analysis` | Task-vector \(R^2\). |
| `plot_probe_fit_r2_latent` | `icl.latent_markov.analysis` | Probe fit \(R^2\) (latent). |
| `plot_val_r2_across_layers` | `icl.latent_markov.analysis` | Validation \(R^2\) across layers. |
| `plot_inject_task_vector_across_layers` | `icl.latent_markov.analysis` | Task-vector injection across layers. |
| `plot_intervention_remove_task_across_layers` | `icl.latent_markov.analysis` | Remove-task intervention across layers. |
| `plot_optimal_orth_direction_across_layers` | `icl.latent_markov.analysis` | Optimal orthogonal direction across layers. |
| `traj_posterior_projection_plot` | `icl.latent_markov.analysis` | Posterior trajectory projection. |
| `traj_averaging_projection_plot` | `icl.latent_markov.analysis` | Averaging trajectory projection. |
| `plot_maj_r2_ood_across_steps_latent` | `icl.utils.ood_major_projection_r2` | Major \(R^2\) on OOD vs step (latent). |
| `plot_task_vector_geometry` | `icl.utils.unified_plot` | Task vector geometry across positions. |
| `get_attention_scores_nonpadded` | `icl.latent_markov.analysis.directions` | PTH and IH attention scores per layer/head (non-padded). |
| `plot_head_scores` | `icl.latent_markov.analysis.directions` | Heatmaps of PTH vs IH scores. |
| `head_ablation_experiment` | `icl.latent_markov.analysis.directions` | Ablation by zeroing attention heads; measures effect. |
| `intervene_averaging_injection` | `icl.latent_markov.analysis.interventions` | Averaging-style posterior injection (latent). |
| `intervene_residual_removal` | `icl.utils.unified_plot` | Residual removal intervention. |
| `intervene_scale_task_component` | `icl.utils.unified_plot` | Scale task-component intervention. |

**Other calls in cells:** `np.linspace`, `np.concatenate`, `np.mean`, `max`, `sorted`, `list`, `print`, `range`, dict `.keys` / `.items` on `anova_results` / `pos_dict`.

---

## Notebook: `Linear.ipynb`

| Function | Import (as used) | Description |
|----------|-------------------|-------------|
| `get_exp_name` | `icl.utils.unified_interface` | Experiment name for linear regression–ICL task. |
| `unified_train` | `icl.utils.unified_interface` | Trains linear transformer / ICL linear model. |
| `unified_train_parallel` | `icl.utils.unified_interface` | Runs several `k` values in parallel on multiple GPUs. |
| `load_model_task_config` | `icl.linear.linear_path_utils` | Loads trained model, task object, and config from an experiment. |
| `plot_task_posterior` | `icl.linear.analysis` | Task posterior visualization (linear). |
| `plot_id_ood_loss` | `icl.linear.analysis` | ID vs OOD loss (linear). |
| `plot_kl_model_vs_two_bayes_linear_transition_across_k` | `icl.linear.lr_task` | KL vs two-Bayes across transitions and `k` (linear). |
| `plot_ancova_separability_linear` | `icl.linear.analysis` | ANCOVA-style separability plot. |
| `plot_averaging_r2_linear` | `icl.linear.analysis` | Averaging \(R^2\) (linear). |
| `plot_task_vector_r2_linear` | `icl.linear.analysis` | Task-vector \(R^2\). |
| `plot_val_r2_across_layers` | `icl.linear.analysis` | Validation \(R^2\) across layers. |
| `train_linear_hidden_predictor` | `icl.linear.analysis` | Linear probe on hidden states (linear task). |
| `plot_inject_task_vector_across_layers` | `icl.linear.analysis` | Task-vector injection across layers. |
| `plot_intervention_remove_task_across_layers` | `icl.linear.analysis` | Remove-task intervention across layers. |
| `plot_optimal_orth_direction_across_layers` | `icl.linear.analysis` | Optimal orthogonal direction across layers. |
| `traj_posterior_projection_plot` | `icl.linear.analysis` | Posterior trajectory projection. |
| `traj_averaging_projection_plot` | `icl.linear.analysis` | Averaging trajectory projection. |
| `plot_maj_r2_ood_across_steps_linear` | `icl.utils.ood_major_projection_r2` | Major \(R^2\) on OOD vs step (linear). |
| `plot_task_vector_geometry` | `icl.utils.unified_plot` | Task vector geometry. |
| `intervene_averaging_injection` | `icl.linear.analysis.interventions` | Averaging injection (linear). |
| `intervene_remove_ood_deltah_subspace` | `icl.linear.analysis` | Removes OOD \(\Delta h\) subspace component (intervention). |
| `intervene_residual_removal` | `icl.utils.unified_plot` | Residual removal. |
| `intervene_scale_task_component` | `icl.utils.unified_plot` | Task-component scaling. |

**Other calls in cells:** `torch.cdist`, `np.linspace`, `np.concatenate`, `getattr`, `dict`, `W.norm` (tensor norm), `train_task.task_pool.squeeze`, `ancova_results.keys`, `pos_dict.keys`, `print`, `range`, `list`, `max`, `sorted`.

---

## Built-ins and object methods (all notebooks)

These appear in AST call lists but are not project-specific: **`print`**, **`range`**, **`list`**, **`max`**, **`sorted`**, **`dict`**, **`getattr`**, **`str`/`int`** (if any), and method calls like **`results.get`**, **`anova_results.keys`**, **`pos_dict.keys`**, **`pos_dict.items`**, **`train_task.task_pool.squeeze`**, tensor **`.norm`**, **`plt.show`**.

---

## Quick counts (unique imported symbols per notebook)

| Notebook | Approx. distinct `icl.*` / analysis imports |
|----------|---------------------------------------------|
| Dyck | 14 |
| coins | 24 |
| Latent | 23 |
| Linear | 25 |

*Third-party modules (`numpy`, `matplotlib`, `torch`, `tqdm`) are counted separately.*
