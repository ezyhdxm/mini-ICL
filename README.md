# Task-Vector Geometry of In-Context Learning — Code

A research framework for investigating **in-context learning (ICL)** in
Transformers through controlled synthetic tasks. Small Transformers are trained
from scratch on sequence distributions with known Bayesian-optimal predictors,
then analysed with a mechanistic-interpretability toolkit
— probes, causal interventions, task-vector geometry, and posterior tracking
— to understand *how* the models learn in context.

This repository accompanies the NeurIPS 2026 submission
*"Task Vector Geometry Underlies Dual Modes of Task Inference in Transformers"*
and contains all code needed to reproduce the figures of the paper.

> Author identifying information has been removed for double-blind review.

---

## Tasks

The framework includes four synthetic task families. Each has its own sampler,
Bayesian baselines, and analysis suite.

| Task                      | Module              | Description                                                                                       |
|---------------------------|---------------------|---------------------------------------------------------------------------------------------------|
| **Latent Markov** (E3)    | `icl.latent_markov` | High-order Markov chains with latent transition matrices drawn from a Dirichlet prior             |
| **Coin** (E1)             | `icl.coin`          | Categorical (biased coin) sequences with known and novel coin pools                                |
| **Noisy Linear Regression** (E2) | `icl.linear` | In-context linear regression with Gaussian noise                                                  |
| **Dyck Paths** (E4)       | `icl.dyck`          | Dyck (balanced parenthesis) path sequences with trie-based Bayesian predictor                      |

Each task supports **in-distribution (ID)** and **out-of-distribution (OOD)**
evaluation, with configurable major/minor task pools. We use the labels
**E1 = Coin**, **E2 = Linear**, **E3 = Latent Markov**, **E4 = Dyck**
throughout the paper and the code.

A separate "real LLM" analysis (Qwen2.5-7B) lives in `icl.real_llm` and is
driven by `notebooks/RealLLM.ipynb`.

---

## Installation

### Using Conda

```bash
conda create -n icl python=3.10 -y
conda activate icl

# PyTorch first (allows platform-specific builds)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Editable install of this package
pip install -e .
```

### Using `uv` (faster, no conda)

```bash
uv sync
```

### Optional: HuggingFace token (only for `notebooks/RealLLM.ipynb`)

Copy `.env.example` to `.env` and set `HF_TOKEN` to a read-only HuggingFace
access token. Required to download Qwen2.5-7B; not needed for the synthetic
experiments (E1–E4).

---

## Project Structure

```
code/
├── src/icl/
│   ├── models/              # Transformer architecture (RoPE, flash attention, KV cache)
│   ├── latent_markov/       # E3 task, config, and analysis
│   ├── coin/                # E1 task, config, and analysis
│   ├── linear/              # E2 task, config, and analysis
│   ├── dyck/                # E4 task, config, and analysis
│   ├── real_llm/            # Qwen2.5-7B ICL analysis utilities
│   ├── utils/               # Training loop, plotting, separability metrics
│   └── figures/             # Attention and task-vector visualisations
├── notebooks/               # One notebook per task (Coins/Linear/Latent/Dyck/RealLLM)
├── scripts/                 # CLI training pipelines and paper-figure scripts
├── tests/                   # Unit tests
├── docs/                    # Internal documentation (Markdown, LaTeX)
├── results/                 # Experiment outputs (gitignored)
├── paper_figs/              # Output directory for paper figures
├── Makefile                 # Convenience targets for training and figures
└── pyproject.toml
```

---

## Reproducing the paper figures

All figures are written to `paper_figs/`. Most figures rely on trained
checkpoints in `results/`; you must therefore train the models **before** you
can render the figures. We outline the pipeline below.

### Step 1 — Train the synthetic models (E1–E3)

```bash
# Train E1, E2, and E3 over k-list = 0..9 (default).
# k indexes the major-task seed; the paper uses k = 0..9 and aggregates.
make train

# Or piecewise:
make train-coin     # E1
make train-linear   # E2
make train-latent   # E3

# Quick smoke test (k = 0, 1 only):
make train-test
```

Training runs in parallel across two GPUs by default (see
`scripts/run_pipeline.py`). Each task takes ~30k steps. The full pipeline
populates `results/{coin,linear,latent}/...`.

For E4 (Dyck, used only in the motivating example):
```bash
make train-dyck
```

### Step 2 — Render the figures

The `Makefile` defines a `figs` target that calls each figure script
individually. Pass `RECOMPUTE=1` to force regeneration when the PNG already
exists.

```bash
make figs                # all combined paper figures (skip already-rendered)
make figs RECOMPUTE=1    # regenerate everything
```

### Figure-by-figure mapping

The table below maps every figure included in the paper (main text + appendix)
to the script or notebook that produces it. Filenames refer to entries inside
`paper_figs/`.

#### Main text

| Paper section | Figure file | How to reproduce |
|---|---|---|
| Schematic / overview (Sec. 1) | `main-Figs/main-drawio-v5.pdf` | Hand-drawn diagram in `draw.io`; not generated by code |
| Task-vector R² and averaging (Sec. *Task vectors*) | `averaging_r2_combined.png` | `make fig-averaging-r2` |
| Posterior alignment of β/α (Sec. *Task vectors*) | `beta_alpha_traj_c3_l9_t3_simplex.png` | `make fig-beta-alpha-traj` |
| Simplex-injection causal intervention (Sec. *Task vectors*) | `injection_simplex_combined.png` | `make fig-injection-simplex` |
| KL phase-transition between modes (Sec. *Two modes*) | `kl_transition_combined_logx.png` | `make fig-kl-transition` |
| OOD R² projection curves (Sec. *Coexistence & orthogonality*) | `ood_r2_c4_l10_t4_logx.png` | `make fig-ood-r2` |
| Aligned simplex trajectories on E3 (Sec. *Coexistence & orthogonality*) | `latent_traj_aligned.png` | `python scripts/fig_latent_traj_aligned.py` |
| Real-LLM trajectory at layer 20 (Sec. *Coexistence & orthogonality*) | `real_llm_traj_l20.png` | Run `notebooks/RealLLM.ipynb` end-to-end (writes one PNG per layer to `paper_figs/real_llm_traj_l{LL}.png`) |
| Dyck motivating example (Sec. *Motivating example*) | `dyck_combined_color.png` | `make fig-dyck` (the script also writes `dyck_projection.png` for the appendix) |

#### Appendix

| Paper subsection | Figure file | How to reproduce |
|---|---|---|
| **App.** Long-context residual variance | `task_vector_r2_combined.png` | `make fig-task-vector-r2` |
| **App.** Two-way ANOVA interaction | `anova_interaction_combined_notitle.png` | `python scripts/fig_anova_interaction_combined.py` |
| **App.** OLS probe decomposition for E3 | `probe_e3.png` | `python scripts/fig_probe_e3_no_position.py` |
| **App.** Linear-interpolation, larger E2 | `beta_alpha_traj_c5_l15_t5_simplex.png` | `python scripts/fig_beta_alpha_traj_combined.py --coin-layer 5 --linear-layer 15 --latent-layer 5` |
| **App.** Linear-interpolation, no simplex projection | `beta_alpha_traj_c3_l9_t3_nosimplex.png` | `python scripts/fig_beta_alpha_traj_combined.py --no-simplex` |
| **App.** Mode-task vs. injection comparison | `mode_task_comparison_combined.png` | `python scripts/fig_injection_simplex_combined.py` (writes both `injection_simplex_combined.png` and `mode_task_comparison_combined.png`) |
| **App.** ID/OOD loss across diversities (Sec. *Two modes*) | `id_ood_loss_combined_c6.png` | `python scripts/fig_id_ood_loss_combined.py` |
| **App.** Real-LLM lambda projection (layer 20) | `real_llm_lambda_id_l20.png` | `notebooks/RealLLM.ipynb` (Section *Per-layer lambda*) |
| **App.** Coin/Linear/Latent simplex trajectories at training step 0 / 1024 | `coin_traj_0.png`, `coin_traj_1024.png`, `linear_traj_0.png`, `linear_traj_1024.png`, `latent_traj_0.png`, `latent_traj_1024.png` | Run the corresponding notebook (`notebooks/Coins.ipynb`, `notebooks/Linear.ipynb`, `notebooks/Latent.ipynb`) and execute the *"OOD trajectory simplex"* section, which iterates over checkpoint steps `[0, 1024]` |
| **App.** Causal intervention on orthogonal subspace | `coin_ortho_intervene.png`, `latent_ortho_intervene.png`, `linear_ortho_intervene.png` | Run the *"Orthogonal-subspace intervention"* section in the corresponding notebook (`Coins.ipynb` / `Latent.ipynb` / `Linear.ipynb`); driven by `icl.{coin,latent_markov,linear}.analysis.interventions.optimal_orth` |
| **App.** Task-subspace loss bars | `coin_task_subspace.png`, `latent_task_subspace.png`, `linear_task_subspace.png` | Same notebooks, *"Remove task subspace"* section (`icl.{coin,latent_markov,linear}.analysis.interventions.remove_task`) |
| **App.** OLS probe R² for the optimal orthogonal direction | `coin_ols_probe_r2.png`, `latent_ols_probe_r2.png`, `linear_ols_probe_r2.png` | Coin/Latent: notebook *"Probe orthogonal direction"* section. Linear: `python scripts/fig_linear_ols_probe_r2.py` |
| **App.** Filtered (feature-removed) intervention | `coin_filtered_intervention.png`, `latent_filtered_intervention.png`, `linear_filtered_intervention.png` | Notebook *"Filtered intervention"* section in each task notebook |
| **App.** Dyck prefix-memorization scatter | `dyck_projection.png` | `make fig-dyck` (same script as `dyck_combined_color.png`) |

#### Optional / supporting figures

| Figure file | Producer | Notes |
|---|---|---|
| `anova_major_vs_minor_{coin,latent,linear}.png`, `anova_major_vs_minor_table.tex` | `python scripts/fig_major_vs_minor_anova.py` | Used in an alternative appendix subsection comparing major / minor variance rates |
| `id_ood_loss_major_only_combined.png` | `python scripts/fig_id_ood_loss_major_only_combined.py` | Major-only ablation of the ID/OOD loss figure |
| `kl_transition_major_only_combined.png` | `python scripts/fig_kl_transition_major_only_combined.py` | Major-only ablation of the KL phase-transition figure |
| `task_vector_r2_{coin,latent,linear}_k10_minor.png`, `task_vector_r2_combined_c16.png`, etc. | `scripts/fig_task_vector_r2_*` | Variants over different vocabulary sizes / minor-task pools |

### Notes on training cost and seeds

* Each E1/E2/E3 model is ~6M parameters. A full sweep over k = 0..9 trains
  30 models (10 per task), ~30k steps each. On 2× A100 GPUs the full
  pipeline finishes in ~6 hours.
* Seeds are set inside the per-task config (`get_config_*` functions in
  `src/icl/{coin,linear,latent_markov,dyck}/`). Re-running with identical
  configs reproduces the figures bit-for-bit (modulo CUDA non-determinism
  in flash attention).
* Some figure scripts (e.g. `fig_beta_alpha_e3_per_task.py`,
  `fig_linear_traj_aligned.py`) cache intermediate `.npz` arrays inside
  `paper_figs/`. Pass `--recompute` (or delete the cache) to re-run.

### A note on `notebooks/RealLLM.ipynb`

`RealLLM.ipynb` downloads `Qwen/Qwen2.5-7B` (~15 GB) from HuggingFace and
runs forward passes over the function-vector evaluation prompts in
`src/icl/real_llm/data/function_vectors/`. A read-only HF token in `.env`
is required (see `.env.example`). On a single A100 (80 GB) the notebook
runs in ~25 minutes and writes per-layer figures
`real_llm_traj_l{LL}.png`, `real_llm_lambda_{id,ood}_l{LL}.png`, and
`real_llm_r2_vs_layer.png` into `paper_figs/`.

---

## Programmatic API

### Quick start

```python
from icl.models import Transformer
from icl.latent_markov import LatentMarkov, get_config_base
from icl.utils import train_model, load_everything
```

### Training a single model

```python
from icl.latent_markov import get_config_base
from icl.utils import train_model

config = get_config_base()
train_model(config)
```

Each task ships its own config function with sensible defaults:

```python
from icl.latent_markov import get_config_base    # E3
from icl.coin import get_config_coin             # E1
from icl.linear import get_config                # E2
from icl.dyck import get_config_dyck             # E4
```

### Loading a trained model

```python
from icl.utils import load_everything
model, sampler, config = load_everything("results/your_experiment")
```

---

## Analysis toolkit

Each task includes a structured `analysis/` subpackage:

* **Probes** — linear probes on hidden representations to predict task
  identity, posterior beliefs, or sufficient statistics.
* **Interventions** — causal edits on hidden states:
    - *Remove task subspace*: zero out the task-vector direction and
      measure loss increase.
    - *Inject task vector*: swap in another task's direction and check
      whether predictions follow.
    - *Optimal orthogonal direction*: find and remove the single
      direction most predictive of OOD loss.
    - *Task-specific*: bigram (latent), unigram (coin), sufficient-stats
      (linear).
* **Directions** — R² regression of hidden states against task subspaces,
  posterior projections, and OOD δh directions.
* **Posteriors** — KL divergence between model output and known-pool /
  Dirichlet Bayes; per-position posterior tracking.
* **Variance analysis** — token-conditioned residual variance (P1) and
  task-level variance decomposition across layers.
* **Trajectory plots** — evolution of hidden-state projections across
  sequence positions and training time.

---

## Model architecture

The default Transformer (`icl.models.Transformer`) uses

* Rotary positional embeddings (RoPE)
* Flash attention with causal masking
* SiLU-activated MLP blocks
* Pre-norm layer normalisation
* Optional KV cache for incremental decoding

The linear-regression task uses a separate GPT-2-style architecture
(`icl.linear.lr_transformer`) adapted from the public `icl-task-diversity`
codebase.

---

## Experiment browser

The repo includes a small SQLite/HTML browser for navigating the
experiments produced by the training pipeline.

```bash
make browse            # index then start a local server at :8000
# or piecewise
make index             # just (re-)build the index
make serve             # just serve the existing index
```

Programmatic search:

```python
from icl.utils.experiment_index import ExperimentIndex
index = ExperimentIndex()
results = index.search_experiments(task_name="latent", vocab_size=10, emb_dim=128)
for exp in results:
    print(f"{exp['exp_name']}: {exp['exp_path']}")
```

---

## License & attribution

Several modules adapt code from publicly available repositories; their
sources are credited inline in the file headers:

* GPT-2 ICL backbone for E2 — adapted from
  `https://github.com/mansheej/icl-task-diversity`
* Function-vector dataset for `RealLLM.ipynb` — Todd et al., ICLR 2024
  (`https://github.com/ericwtodd/function_vectors`)
* RoPE/ALiBi positional encoders — adapted from public reference
  implementations cited at the top of `src/icl/models/pos_encoder.py`

Trained checkpoints, the wandb run history, and intermediate `.npz`
artefacts are too large to include in the supplementary; they will be
released alongside the camera-ready version.

---

## Anonymisation note for reviewers

This codebase has been scrubbed of author-identifying information. Before
running the supplementary archive, please be aware that:

* The `.git/` directory and any `wandb/` run logs (which embed personal
  user identifiers) are **not** included in the supplementary zip.
* Run `wandb offline` or unset `WANDB_API_KEY` before training if you
  wish to avoid creating identifying wandb runs.
* The default wandb projects are named generically (`icl-coin`,
  `icl-linear`, `icl-latent`, `icl-dyck`).
