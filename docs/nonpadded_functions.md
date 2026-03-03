# Non-Padded Functions Reference

This document describes all non-padded functions created for extracting and analysing hidden representations from transformer models trained on discrete-token in-context learning tasks (coin, latent Markov, linear regression).

## Background: Padded vs Non-Padded Sequences

**Padded** sequences interleave real tokens with separator/padding tokens:

$$
\text{padded} = [x_0,\ \texttt{pad},\ x_1,\ \texttt{pad},\ x_2,\ \texttt{pad},\ \ldots,\ x_{n-1}]
$$

- Real tokens at even positions: $0, 2, 4, \ldots$
- Padding tokens at odd positions: $1, 3, 5, \ldots$
- Hidden extraction happens at **padding positions** $2i + 1$

**Non-padded** sequences contain only real tokens:

$$
\text{non-padded} = [x_0,\ x_1,\ x_2,\ \ldots,\ x_{n-1}]
$$

- All positions are real tokens
- Hidden extraction happens at **real-token positions** $i$

---

## File: `src/icl/utils/unified_interface.py`

### 1. `_compute_hiddens_at_real_tokens`

**Purpose:** Core hook-based hidden extraction at real-token positions for discrete-token tasks (coin, latent, dyck).

**How it works:**
- Registers forward hooks on `model.layers[l].attn_block` for each layer $l$.
- Runs a forward pass on non-padded input `(B, seq_len)`.
- Each hook captures the output at positions $[0, 1, \ldots, \text{seq\_len} - 2]$ via `index_select`.

**What the hook captures:**
The attention block computes:

$$
h_l^{(t)} = x_l^{(t)} + \text{Attn}_l\bigl(\text{LN}(x_l^{(t)})\bigr)
$$

where $x_l^{(t)}$ is the residual stream at layer $l$, position $t$. The hook captures $h_l^{(t)}$, which is the **residual stream + attention output combined**.

**Output shape:** `(n_layers, n_tasks, seq_len - 1, B, n_embd)`

---

### 2. `_get_hiddens_at_real_positions`

**Purpose:** Unified dispatcher that extracts non-padded hidden representations for any task type.

**How it works:**
- For **coin/dyck**: forces `sampler.pad = False`, calls `_compute_hiddens_at_real_tokens`.
- For **latent**: forces `sampler.pad = False`, calls `_compute_hiddens_at_real_tokens` via a cloned sampler with OOD tasks.
- For **linear**: determines `task_pos` based on the model's padding format:
  - `pad="mapsto"`: $\text{task\_pos} = [0, 3, 6, \ldots]$ (data tokens in mapsto layout)
  - `pad="none"`: $\text{task\_pos} = [0, 2, 4, \ldots]$ (data tokens in none layout)
  - `pad="bos"`: $\text{task\_pos} = [1, 3, 5, \ldots]$ (data tokens in bos layout)

---

## File: `src/icl/utils/coin_nonpadded.py`

### 3. `compute_hiddens_multi_coin_nonpadded`

**Purpose:** Extract hidden representations across multiple layers for all tasks in the coin/latent sampler.

**How it works:**
For each task $k \in \{0, \ldots, K-1\}$:
1. Generate $B$ non-padded sequences from task $k$: $(B, \text{seq\_len})$
2. Forward pass through model
3. Extract hiddens at positions of interest via hooks

**Position mapping:**
$$
\text{task\_pos}[i] = \text{positions\_of\_interest}[i] \quad \text{(direct, no } 2i+1 \text{ mapping)}
$$

**Output shape:** `(L, n_tasks, n_positions, B, n_embd)`

---

### 4. `compute_hiddens_token_conditioned_coin_nonpadded`

**Purpose:** Extract hidden representations conditioned on fixing a specific token value at a given position.

**How it works:**
For each position $t$ and each unique token value $v$:
1. Generate sequences for each task
2. Replace the token at position $t$ with $v$: $x_t \leftarrow v$
3. Extract hidden at position $t$ (the modified token itself)

**Key difference from padded version:**
- Padded: fix at $2t$ (real token), extract at $2t + 1$ (padding token after it)
- Non-padded: fix at $t$, extract at $t$ (the real token itself)

**Output shape:** `(L, n_positions, n_unique_tokens, n_tasks, B, n_embd)`

---

### 5. `plot_coin_task_posterior_nonpadded`

**Purpose:** Generate coin samples and plot the Bayesian task posterior over time.

**The posterior computation:**
Given a sequence of tokens $x_1, x_2, \ldots, x_T$ and $K$ candidate coin probability vectors $p_1, \ldots, p_K$, the posterior after observing $t$ tokens is:

$$
P(Z = k \mid x_{1:t}) = \frac{\pi_k \prod_{\tau=1}^{t} p_k(x_\tau)}{\sum_{j=1}^{K} \pi_j \prod_{\tau=1}^{t} p_j(x_\tau)}
$$

where $\pi_k$ is the prior probability of task $k$.

Equivalently, using cumulative token counts $n_v^{(t)} = \sum_{\tau=1}^{t} \mathbf{1}[x_\tau = v]$:

$$
\log P(Z = k \mid x_{1:t}) \propto \log \pi_k + \sum_{v=0}^{V-1} n_v^{(t)} \log p_k(v)
$$

**Key difference from padded:** Uses `x = samples.long()` (all tokens are real) instead of `x = samples[..., ::2].long()` (strip padding).

---

### 6. `compute_stable_rank_at_real_positions`

**Purpose:** Compute the stable rank of hidden representations at every (layer, position) pair.

**Stable rank definition:**
For a matrix $A \in \mathbb{R}^{m \times d}$ with singular values $\sigma_1 \geq \sigma_2 \geq \ldots$:

$$
\text{srank}(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_{\max}^2} = \frac{\text{tr}(A^\top A)}{\lambda_{\max}(A^\top A)}
$$

**Memory-efficient streaming implementation:**
Instead of materialising the full matrix $A$ of shape $(K \cdot B, D)$ (which can be tens of GB), we accumulate the Gram matrix:

$$
G = A^\top A = \sum_{k=1}^{K} H_k^\top H_k
$$

where $H_k \in \mathbb{R}^{B \times D}$ is the hidden matrix for task $k$. Each $H_k$ is computed on GPU and discarded after accumulating into $G$.

Then:

$$
\text{srank}(A) = \frac{\text{tr}(G)}{\lambda_{\max}(G)}
$$

where $\lambda_{\max}$ is computed via `torch.linalg.eigvalsh` on the $(D, D)$ matrix.

**Memory:** $O(L \cdot P \cdot D^2)$ instead of $O(L \cdot K \cdot P \cdot B \cdot D)$.

For the default config ($L=6, P=127, D=128$): **~100 MB** instead of ~51 GB for $k=11$.

---

### 7. `plot_max_stable_rank_vs_k_nonpadded`

**Purpose:** For each $k$ value (where $n_\text{minor} = 2^k$), compute the stable rank across all layers and positions, take the maximum, and plot against $k$.

$$
\text{max\_srank}(k) = \max_l \max_t \ \text{srank}\bigl(H_{l,t}\bigr)
$$

where $H_{l,t}$ is the $(K \cdot B, D)$ matrix of hidden representations at layer $l$, position $t$.

---

### 8. `get_task_variance_coin_nonpadded`

**Purpose:** Compute task variance — the variance of batch-averaged hidden representations across tasks.

**The computation:**
1. For each task $k$ and position $t$, compute the batch mean:

$$
\bar{h}_{k,t} = \frac{1}{B} \sum_{b=1}^{B} h_{k,t,b}
$$

2. Task variance at position $t$:

$$
\text{Var}_\text{task}(t) = \frac{1}{K} \sum_{k=1}^{K} \|\bar{h}_{k,t} - \bar{h}_{\cdot,t}\|^2
$$

where $\bar{h}_{\cdot,t}$ is the global mean across tasks.

3. Normalised task variance:

$$
\text{NormVar}_\text{task}(t) = \frac{\text{Var}_\text{task}(t)}{\frac{1}{K} \sum_k \|\bar{h}_{k,t}\|^2}
$$

---

### 9. `plot_task_variance_coin_nonpadded`

**Purpose:** Convenience wrapper — computes task variance and plots normalised task variance vs position for each layer.

---

### 10. `get_token_conditioned_hiddens_coin_nonpadded`

**Purpose:** Wrapper that loads model/sampler and calls `compute_hiddens_token_conditioned_coin_nonpadded`.

---

### 11. `plot_p1_variance_coin_nonpadded`

**Purpose:** Compute and plot the P1 variance (conditional residual variance given fixed token and task).

**P1 variance computation:**
For each position $t$, token value $v$, and task $k$:

1. Conditional mean:

$$
\mu_{t,v,k} = \mathbb{E}[h \mid x_t = v,\ Z = k]
$$

2. Conditional variance (residual after conditioning on both token and task):

$$
\text{Var}_{t,v,k} = \mathbb{E}\bigl[\|h - \mu_{t,v,k}\|^2 \mid x_t = v,\ Z = k\bigr]
$$

3. Average across tasks:

$$
\text{Var}_{t,v} = \frac{1}{K} \sum_k \text{Var}_{t,v,k}
$$

4. Average across tokens:

$$
\text{Var}_\text{P1}(t) = \frac{1}{|V_t|} \sum_v \text{Var}_{t,v}
$$

5. Normalised:

$$
\text{NormVar}_\text{P1}(t) = \frac{\text{Var}_\text{P1}(t)}{\frac{1}{|V_t|} \sum_v \|\mu_{t,v}\|^2}
$$

where $\mu_{t,v} = \frac{1}{K} \sum_k \mu_{t,v,k}$.

**Interpretation:** P1 variance measures how much hidden-state variation remains **after** controlling for both the current token and the task identity. Low P1 variance means the hidden state is almost fully determined by (task, token).

---

### 12. `train_linear_softmax_posterior_predictor_coin_nonpadded`

**Purpose:** Train a linear softmax model to predict task posteriors from hidden representations.

**Model:**

$$
\hat{P}(Z = k \mid h) = \text{softmax}(W h + b)_k
$$

where $W \in \mathbb{R}^{T \times D}$, $b \in \mathbb{R}^T$.

**Loss:** KL divergence between predicted and true posteriors:

$$
\mathcal{L} = \text{KL}\bigl(\hat{P} \| P^*\bigr) = \sum_k P^*(k) \log \frac{P^*(k)}{\hat{P}(k)}
$$

where $P^*$ is the Bayesian posterior computed from `task_posterior_coins`.

**Baselines:**
- **Permutation:** shuffled posteriors (break hidden↔posterior pairing)
- **Logits:** model's own output logits → posteriors (what the output layer captures)

---

### 13. `plot_posterior_predictor_loss_vs_k_coin_nonpadded`

**Purpose:** For each $k$ value, train the posterior predictor and plot validation KL loss vs $k$. Supports multiple layers on the same plot.

---

### 14. `traj_projection_plot_nonpadded`

**Purpose:** 2D trajectory projection of task vectors onto the plane defined by the 3 major tasks. Supports coin, latent, and linear.

**The computation:**

1. Extract hiddens at a specific layer: $H \in \mathbb{R}^{K \times T \times B \times D}$

2. Compute batch-averaged task vectors:

$$
\tau_k(t) = \frac{1}{B} \sum_b h_{k,t,b} - \bar{\mu}(t)
$$

where $\bar{\mu}(t) = \frac{1}{3B} \sum_{k=0}^{2} \sum_b h_{k,t,b}$ is the mean over the 3 major tasks.

3. Final task vectors (anchors): $\tau_0(T), \tau_1(T), \tau_2(T)$

4. Fit coefficients $\lambda$ via regression:

$$
\tau_k(t) \approx \sum_{j=0}^{2} \lambda_{k,t,j} \cdot \tau_j(T)
$$

with $R^2$ score measuring fit quality.

5. Project onto 2D plane via SVD of the 3 anchor vectors.

---

### 15. `historical_injection_coin_nonpadded` and `_historical_injection_test_next_token_nonpadded`

**Purpose:** Measure how much the model's prediction depends on historical hidden representations by randomly perturbing them.

**The perturbation procedure:**
At position $p$ in a non-padded sequence $[x_0, x_1, \ldots, x_{n-1}]$:

1. **Baseline:** run model normally, record $\log P_\text{base}(x_{p+1} \mid x_{0:p})$

2. **Perturbed:** hook into layer $l$'s attention block output. For each historical position $j \in \{0, 1, \ldots, p-1\}$:
   - With probability $p_\text{perturb}$: replace $h_l^{(j)} \leftarrow r$ where $r$ is random noise
   - With probability $1 - p_\text{perturb}$: leave unchanged
   - Position $p$ is **never perturbed**

3. **Metric:**

$$
\Delta \log P = \log P_\text{perturbed}(x_{p+1}) - \log P_\text{base}(x_{p+1})
$$

averaged over batch and tasks.

**Noise modes:**
- `"standard_normal"`: $r \sim \mathcal{N}(0, I)$
- `"match_std"`: $r \sim \mathcal{N}(0, \sigma^2 I)$ where $\sigma = \text{std}(h_l^{(j)})$ across batch
- `"match_mean_std"`: $r \sim \mathcal{N}(\mu, \sigma^2 I)$ where $\mu, \sigma$ from batch statistics

**What it measures:** $\Delta \log P \approx 0$ means the model doesn't rely on historical hidden states at layer $l$ for prediction at position $p$. $\Delta \log P \ll 0$ means historical information at layer $l$ is critical.

---

### 16. `plot_historical_injection_coin_nonpadded`

**Purpose:** Convenience wrapper — runs the perturbation analysis and plots $\Delta \log P$ vs position. Supports multiple layers (each perturbed independently, one at a time) on the same plot.

**Parameters:** `stride` (default 10) controls which positions are evaluated; `start`/`end` control the range.

---

### 17. `train_linear_hidden_predictor_coin_nonpadded`

**Purpose:** Train a linear regression model in the **reverse direction**: predict hidden representations from task posteriors.

**Models fitted:**

1. **Main:** $\hat{h} = W_\text{post} \cdot \log P^*(Z \mid x_{1:T}) + b_\text{post}$

2. **Logits baseline:** $\hat{h} = W_\text{logit} \cdot \ell + b_\text{logit}$ where $\ell$ is the model's output logits

3. **One-hot baseline:** $\hat{h} = W_\text{oh} \cdot \text{onehot}(x_t) + b_\text{oh}$

4. **Combined:** $\hat{h} = W_\text{comb} \cdot [\log P^*;\ \text{onehot}(x_t);\ \ell] + b_\text{comb}$

5. **Permutation baseline:** same as main but with shuffled targets (measures chance-level $R^2$)

All fitted via closed-form pseudoinverse: $W = (X^\top X)^{-1} X^\top Y$.

**Metrics:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \|\hat{h}_i - h_i\|^2, \qquad R^2 = 1 - \frac{\sum_i \|\hat{h}_i - h_i\|^2}{\sum_i \|h_i - \bar{h}\|^2}
$$

**Orthogonality analysis:**
Measures whether the three predictors (posterior, logits, one-hot) capture **independent** components of the hidden state:

- Cosine similarity between centered predictions: $\cos(\hat{h}_A, \hat{h}_B)$
- Cross $R^2$: how much of predictor $A$'s output can be linearly recovered from predictor $B$'s output

If individual $R^2$ values are roughly additive ($R^2_\text{combined} \approx R^2_\text{post} + R^2_\text{logit} + R^2_\text{oh}$) and cross $R^2$ values are near zero, the three components are orthogonal in the hidden space.

**Sampling modes:**
- `sample_mode="train"`: mixture of major + minor tasks
- `sample_mode="major"`: only major tasks
- `sample_mode="minor"`: only minor tasks

**Modifications from initial implementation:**

- **Posterior probabilities instead of logits:** The main fit now uses raw posterior probabilities $P^*(Z \mid x_{1:T})$ instead of $\log P^*(Z \mid x_{1:T})$. This preserves simplex structure: if the model perfectly encodes the posterior, the predicted hidden $\hat{h} = W \cdot P^* + b$ is a convex combination of the rows of $W$ (plus bias).

- **Major-only posterior in major mode:** When `sample_mode="major"`, the posterior is computed with `include_minor=False`, giving a $(n, 3)$ design matrix instead of $(n, 3 + n_\text{minor})$. This avoids fitting near-zero minor posterior columns when all samples come from major tasks.

---

### 18. `plot_val_r2_across_layers_coin_nonpadded`

**Purpose:** Sweep `train_linear_hidden_predictor_coin_nonpadded` across all layers and plot val $R^2$ for each baseline as a grouped bar chart (matplotlib).

**Baselines shown:** Posterior, Logits, Token (one-hot of current token).

**Output:** Grouped bar chart with layer on x-axis, val $R^2$ on y-axis, one bar per baseline per layer. Returns `(fig, all_results)` where `all_results = {layer: results_dict}`.

---

### 19. `plot_val_r2_across_layers_multi_k`

**Purpose:** Compare val $R^2$ across layers for multiple $k$ values (experiments). Produces a figure with one subplot per baseline, where each subplot shows layer on the x-axis and one line per $k$ value.

**How it works:** For each $k$, calls `plot_val_r2_across_layers_coin_nonpadded` to get per-layer results, then assembles a 3-panel figure (Posterior, Logits, Token).

---

### 20. `traj_posterior_projection_plot_coin_nonpadded`

**Purpose:** 2D trajectory projection using **posterior-derived** task vectors (W-based) instead of raw hidden means.

**How it differs from `traj_projection_plot_nonpadded` (function 14):**

1. **Final task vectors** come from a linear fit $P^*(Z) \cdot W + b \approx h$ in major mode at late positions (default: positions 100 to `seq_len`):

$$
\tau_j^{\text{final}} = W_j - \bar{W}, \qquad \bar{W} = \frac{1}{3}\sum_{j=0}^{2} W_j
$$

This isolates the posterior-encoding component of the hidden representation. The bias $b$ cancels after centering.

2. **Rescaling:** The centered $W$ rows have a different scale from raw hidden deviations. We rescale by the Frobenius norm ratio:

$$
\tau^{\text{final}} \leftarrow \tau^{\text{final}} \cdot \frac{\|\tau^{\text{actual}}_{\text{endpoints}}\|_F}{\|\tau^{\text{final}}\|_F}
$$

where $\tau^{\text{actual}}_{\text{endpoints}}$ are the batch-averaged major task vectors at the final time step.

3. **Trajectories** remain raw batch-averaged (or per-sample) hidden deviations, unchanged from function 14. $R^2$ measures how much of the actual hidden trajectory lies in the posterior plane.

4. **Point selection:** Uses farthest-point sampling on the projected 2D coordinates (median across all trajectories) to select informative time indices, avoiding clustering.

**Visual parameters:** Blue shades for major tasks, red for OOD, with low OOD colour jitter.

---

### 21. `traj_post_posterior_projection_plot_coin_nonpadded`

**Purpose:** Compare projected coefficients $\lambda$ with the true Bayesian posterior $P(Z \mid x_{1:t})$ for individual sequences.

**How it works:**

1. Fits $W$ and $b$ as in function 20.
2. Extracts hiddens and raw token data for a chosen task (`task_id`: 0-2 for major, $\geq 3$ for OOD).
3. For each of `n_rows` random batch samples:
   - Computes hidden trajectory and projects onto W-plane to get $\lambda(t) \in \mathbb{R}^3$
   - Computes Bayesian posterior $P(Z \mid x_{1:t})$ from the token sequence
   - Projects $\lambda$ onto the probability simplex using the Euclidean projection algorithm (Duchi et al., 2008)

4. Plots an `n_rows` $\times$ 3 grid: one row per sample, one column per task component. Each cell shows $\lambda_k(t)$ (solid) vs $P(Z=k \mid x_{1:t})$ (dashed).

**Simplex projection (Duchi et al., 2008):**

Given unconstrained $\lambda \in \mathbb{R}^n$, find the closest point in the simplex $\Delta^n = \{w \geq 0, \sum w_i = 1\}$:

1. Sort $\lambda$ in decreasing order: $\lambda_{(1)} \geq \ldots \geq \lambda_{(n)}$
2. Find $\rho = \max\{j : \lambda_{(j)} - \frac{1}{j}(\sum_{i=1}^{j} \lambda_{(i)} - 1) > 0\}$
3. Set $\theta = \frac{1}{\rho}(\sum_{i=1}^{\rho} \lambda_{(i)} - 1)$
4. Return $\max(\lambda_i - \theta, 0)$

---

### 22. `process_ood_minor_metric_coin_nonpadded`

**Purpose:** Compute OOD and minor task metrics across training steps and layers for coin task on non-padded sequences.

**How it differs from `process_ood_minor_metric` (in `unified_ood_analysis.py`):**

1. **Major task vectors:** Uses W-based approach (fit $P^* \cdot W + b \approx h$ in major mode, centered $W$ rows, norm-rescaled) instead of raw hidden means at the last time step. The fit is re-done at every training step since $W$ changes as the model trains.

2. **Minor task vectors:** Kept as SVD-based (top-3 singular vectors of minor tasks' final-time-step task vectors), unchanged from the original.

3. **Hiddens:** Extracted via `_get_hiddens_at_real_positions` (real-token positions, non-padded sequences).

**Metrics computed (per step, per layer):**

For each projection basis (major W-based, minor SVD-based):

- $R^2_{\text{OOD}}$: mean $R^2$ at the final time step for OOD tasks
- $R^2_{\text{minor}}$: mean $R^2$ at the final time step for minor tasks
- $\text{Var}_{\text{OOD}}$: dispersion of OOD tasks' $\lambda$ vectors
- $\text{Var}_{\text{minor}}$: dispersion of minor tasks' $\lambda$ vectors

**Caching:** Results are cached per (step, layer) as pickle files with `"coin_nonpadded"` in the filename to avoid collisions with the padded version.

---

### 23. `plot_training_curves_all_experiments_coin_nonpadded`

**Purpose:** Plot OOD/minor metrics and OOD loss across training steps for multiple coin experiments (different $k$ values).

**How it works:**

1. For each $k$ in `k_list`, calls `process_ood_minor_metric_coin_nonpadded` to collect metrics.
2. Loads OOD loss from `results/coin/<exp_name>/log.json`.
3. Produces two matplotlib figures:
   - **Figure 1:** Chosen metric vs training step, with raw (faint) + EMA-smoothed (bold) overlay. One line per $k$ value.
   - **Figure 2:** OOD loss vs training step. Same colour per $k$.

**Consistent colours** across both figures via a fixed `tab10`/`tab20` colour map keyed by $k$.

---

### 21. `plot_id_ood_loss_coin_nonpadded`

**Purpose:** Plot ID loss and OOD loss side by side for multiple coin experiments.

**How it works:**
1. For each $k$ in `k_list`, loads `log.json` from the experiment directory.
2. Extracts `eval/step`, `eval/IDLoss`, `eval/OODLoss`.
3. Produces a single figure with two panels: ID loss (left) and OOD loss (right) vs training step.

**Example:**
```python
from icl.coin.coin_nonpadded import plot_id_ood_loss_coin_nonpadded

out = plot_id_ood_loss_coin_nonpadded(
    k_list=[0, 4, 6, 8],
    logx=True,
)
```

---

### 23b. `plot_stable_rank_vs_maj_r2_min_coin_nonpadded`

**Purpose:** Plot stable rank of minor-task hidden representations and R² (minor → major basis) vs training step for multiple coin experiments.

**Two panels:**
- **Left:** Stable rank of minor-task vectors (batch-averaged deviations from major mean) at the specified position(s).
- **Right:** R² of projecting minor tasks onto the major basis (same as `maj_r2_min` in `plot_training_curves_all_experiments_coin_nonpadded`).

**Minor tasks:** Uses `min(2**k, n_minor_max)` per experiment (default `n_minor_max=256`), so larger-k experiments use more minor tasks.

**Position control:** Use `metric_positions` to evaluate at earlier positions:
- `None` (default): final position
- `10`: single position 10
- `(10, 16)`: average over positions [10, 16)

**Caching:** Stable rank values are cached per (exp_name, step, layer, metric_positions, n_minor, n_ood, B) as pickle files in the experiment directory. Use `force_recompute=True` to bypass.

**Example:**
```python
from icl.coin.coin_nonpadded import plot_stable_rank_vs_maj_r2_min_coin_nonpadded

# Final position (default)
out = plot_stable_rank_vs_maj_r2_min_coin_nonpadded(
    steps=range(0, 30_001, 5000),
    k_list=[0, 4, 6, 8],
)

# Earlier position (e.g. 10)
out = plot_stable_rank_vs_maj_r2_min_coin_nonpadded(
    steps=range(0, 30_001, 5000),
    k_list=[0, 4, 6, 8],
    metric_positions=10,
)

# Average over positions 10-15
out = plot_stable_rank_vs_maj_r2_min_coin_nonpadded(
    steps=range(0, 30_001, 5000),
    k_list=[0, 4, 6, 8],
    metric_positions=(10, 16),
)
```
