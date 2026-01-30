# Projection Removal Intervention - Implementation Notes

## Overview

This intervention removes the projection of hidden states onto the majority task subspace (defined by `final_task_vecs`) at padded token positions in the coin task.

## Tensor Shapes

### Hidden States
- **Input hidden states**: Shape `(B, L, d)` or `(L, B, d)` where:
  - `B` = batch size
  - `L` = sequence length (with padding: `2 * seq_len - 1`)
  - `d` = embedding dimension (`config.model.emb_dim`)

### final_task_vecs
- **Shape**: `(3, d)` where:
  - First dimension (3) = number of major tasks
  - Second dimension (d) = embedding dimension
- **Computation**: Extracted from hidden states at the final padded position (`p_idx = seq_len - 2`, `token_pos = 2 * seq_len - 3`) for the first 3 major tasks, then centered by subtracting the global mean.

### Orthonormal Basis Q
- **Shape**: `(d, rank)` where:
  - `d` = embedding dimension
  - `rank` = rank of `final_task_vecs` (at most 3)
- **Computation**: Computed via QR decomposition of `final_task_vecs.T` (shape `(d, 3)`)

## Where the Hook is Applied

The hook is registered on `model.layers[layer_idx].attn_block` (or the module specified by `hook_attr`). The intervention modifies the hidden state **output** of the attention block at the specified layer, specifically at the padded token position.

## How Padded Positions are Detected

For coin task with padding enabled (`sampler.pad == True`):
- **Real tokens** are at even positions: `0, 2, 4, 6, ...`
- **Padded tokens** are at odd positions: `1, 3, 5, 7, ...`
- The mapping from `p_idx` (padded position index) to `token_pos` (actual token position in sequence) is:
  ```
  token_pos = 2 * p_idx + 1
  ```
- For example:
  - `p_idx = 0` → `token_pos = 1` (first padded position)
  - `p_idx = 1` → `token_pos = 3` (second padded position)
  - `p_idx = seq_len - 2` → `token_pos = 2 * seq_len - 3` (final padded position)

The intervention verifies that the token at `token_pos` is indeed a PAD token by checking `batch_data[:, token_pos] == pad_id`.

## Projection Removal Algorithm

For each hidden vector `h` at a padded position:

1. **Compute orthonormal basis Q**: 
   - QR decomposition of `final_task_vecs.T` → `Q` (shape `(d, rank)`)
   - Only columns corresponding to non-zero diagonal elements of R are kept

2. **Compute projection**:
   ```
   proj = Q @ (Q.T @ h)
   ```
   - `Q.T @ h`: `(rank, B)` (transpose h first if needed)
   - `Q @ (Q.T @ h)`: `(d, B)` → transpose to `(B, d)`

3. **Remove projection**:
   ```
   h_new = h - proj
   ```

4. **Sanity check**: Verify `Q.T @ h_new ≈ 0` (should be approximately zero)

## How to Run the Intervention

### Basic Usage

```python
from icl.utils.projection_removal_intervention import (
    projection_removal_coin_padded_only,
    compute_final_task_vecs_coin
)

# Run intervention for all padded positions
out, info = projection_removal_coin_padded_only(
    exp_name="your_experiment_name",
    n_minor=64,
    n_ood=30,
    B=64,
    layer_idx=5,
    verbose=True
)

# out: list[float] - aggregated delta_target_logprob_mean per padded position
# info: dict with task_ids, per_task values, and final_task_vecs_shape
```

### With Pre-computed final_task_vecs

```python
import torch
from icl.utils.projection_removal_intervention import (
    projection_removal_coin_padded_only,
    compute_final_task_vecs_coin
)
import icl.utils.notebook_utils as nu

# Load model and sampler
_, sampler, config = nu.load_everything("coin", "your_experiment_name")
model, _ = nu.load_checkpoint(config, step=60000, exp_name="your_experiment_name")

# Compute final_task_vecs once
final_task_vecs = compute_final_task_vecs_coin(
    config=config,
    model=model,
    sampler=sampler,
    layer_idx=5,
    B=64
)

# Use pre-computed final_task_vecs
out, info = projection_removal_coin_padded_only(
    exp_name="your_experiment_name",
    final_task_vecs=final_task_vecs,
    compute_final_task_vecs=False,  # Don't recompute
    layer_idx=5,
    verbose=True
)
```

### Single Position Evaluation

```python
from icl.utils.projection_removal_intervention import (
    projection_removal_test_next_token_padded_only,
    compute_final_task_vecs_coin
)
import icl.utils.notebook_utils as nu

# Load model and sampler
_, sampler, config = nu.load_everything("coin", "your_experiment_name")
model, _ = nu.load_checkpoint(config, step=60000, exp_name="your_experiment_name")

# Compute final_task_vecs
final_task_vecs = compute_final_task_vecs_coin(
    config=config,
    model=model,
    sampler=sampler,
    layer_idx=5,
    B=64
)

# Evaluate at a specific padded position
results = projection_removal_test_next_token_padded_only(
    config=config,
    model=model,
    sampler=sampler,
    final_task_vecs=final_task_vecs,
    layer_idx=5,
    p_idx=10,  # padded position index
    task=0,     # task ID
    B=64,
    verbose=True,
    return_logits=True
)

# Check sanity check metrics
print(f"Q.T @ h_new norm (mean): {results['QTh_new_norm_mean']:.6f}")
print(f"Q.T @ h_new norm (max): {results['QTh_new_norm_max']:.6f}")
print(f"Delta target logprob: {results['delta_target_logprob_mean']:.4f}")
```

## Return Format

The intervention returns the same metrics format as `injection_test_next_token_padded_only`:

- `delta_target_logprob_mean`: Mean change in target token log-probability
- `delta_target_logprob_std`: Std of change in target token log-probability
- `base_acc` / `int_acc`: Accuracy before/after intervention
- `top1_flip_rate`: Fraction of predictions that changed
- `kl_mean` / `kl_std`: KL divergence between distributions
- `QTh_new_norm_mean` / `QTh_new_norm_max`: Sanity check metrics (should be near zero)
- `n_padded_modified`: Number of padded tokens modified (should equal batch size B)

## Assumptions

1. **Coin task with padding**: Assumes `sampler.pad == True` and the even/odd padding pattern
2. **At least 3 major tasks**: Requires `sampler.n_major_tasks >= 3` to compute `final_task_vecs`
3. **Model structure**: Assumes `model.layers[layer_idx].attn_block` exists and outputs hidden states
4. **Device/dtype matching**: `final_task_vecs` are automatically moved to the correct device and dtype

## Numerical Stability

- Uses QR decomposition (via `torch.linalg.qr`) for numerically stable orthonormal basis computation
- Handles rank-deficient `final_task_vecs` by only keeping columns corresponding to non-zero diagonal elements
- Uses `eps=1e-8` threshold for rank detection (can be adjusted in `_compute_orthonormal_basis`)

## Sanity Check Explanation

### How the Sanity Check Works

The sanity check verifies that the projection removal is working correctly by computing:

```
Q.T @ h_new
```

where:
- `Q` is the orthonormal basis for the span of `final_task_vecs` (shape `(d, rank)`)
- `h_new` is the modified hidden state after removing the projection (shape `(B, d)`)
- The result `Q.T @ h_new` should be approximately zero if the projection was correctly removed

**Computation details**:
1. For each hidden vector `h` in the batch (B samples), we compute `h_new = h - Q @ (Q.T @ h)`
2. We then compute `Q.T @ h_new.T` which gives shape `(rank, B)` - the projection of each `h_new` onto the Q subspace
3. We take the norm along the rank dimension: `norm = ||Q.T @ h_new||` for each sample → shape `(B,)`
4. We report the mean and max of these norms across the batch

**Why the values vary slightly**:
- Each task generates different sequences, leading to different hidden states `h`
- After projection removal, `h_new` is different for each task/batch
- The norm `||Q.T @ h_new||` measures how much of `h_new` still lies in the Q subspace
- Small non-zero values (e.g., 0.000001-0.000004) are expected due to:
  - Numerical precision limits (floating point arithmetic)
  - The fact that `h` may have had components slightly outside the exact span of Q
  - Batch-to-batch variation in the hidden states

**What good values look like**:
- Values in the range `1e-6` to `1e-5` indicate excellent projection removal
- Values much larger (e.g., `> 1e-3`) might indicate a problem with the computation
- The consistency of small values across tasks confirms the intervention is working correctly

### Why Multiple Print Statements Appear

When you set `n_minor=64` (and assuming `n_major=3`), the total number of tasks is `n_tasks = 3 + 64 = 67`.

For `p_idx=0` with `verbose=True`, the wrapper function:
1. Loops over all 67 tasks
2. For each task, calls `projection_removal_test_next_token_padded_only` 
3. Each call processes a different batch (different sequences for that task)
4. Each call computes the sanity check on its batch

**Previous behavior**: Each task would print its own sanity check, resulting in 67 print statements.

**Improved behavior** (after update): The wrapper now aggregates sanity check metrics across all tasks and prints a single summary with:
- Mean norm across all tasks
- Range of mean norms (min-max)
- Maximum norm across all tasks

This provides the same information in a more concise format.

## Interpreting Results

### Example Results and Interpretation

Here's an example output from running the intervention:

```
Computing final_task_vecs from model...
Computed final_task_vecs: shape torch.Size([3, 128])
  [p_idx=0] Q.T @ h_new norm: mean=0.000001, max=0.000001
  [p_idx=0] Q.T @ h_new norm: mean=0.000002, max=0.000002
  [p_idx=0] Q.T @ h_new norm: mean=0.000002, max=0.000003
[coin] p_idx=00/219 (padded token_pos=001) Δlogp(mean over 3 tasks)=-1.0029
[coin] p_idx=05/219 (padded token_pos=011) Δlogp(mean over 3 tasks)=-1.0728
...
[coin] p_idx=219/219 (padded token_pos=439) Δlogp(mean over 3 tasks)=-1.0887
```

#### Sanity Check Verification

The `Q.T @ h_new norm` values (0.000001 to 0.000003) are extremely small, confirming that:
- The projection removal is working correctly
- The modified hidden states (`h_new`) are indeed orthogonal to the majority task subspace
- The numerical implementation is stable

#### Delta Log-Probability Interpretation

**Key Observation**: All `Δlogp` values are **negative** and consistently around **-1.0** (ranging from approximately -0.95 to -1.09).

**What this means**:

1. **Removing the projection hurts performance**: The negative values indicate that removing the projection onto the majority task subspace **decreases** the log-probability assigned to the correct next token. This means the model's predictions become **worse** after the intervention.

2. **The majority task subspace is important**: The consistent negative impact (around -1.0 nats) suggests that the information encoded in the projection onto the majority task subspace is **crucial** for accurate next-token prediction. The model relies on this subspace representation to make correct predictions.

3. **Consistency across positions**: The values are relatively stable across all padded positions (standard deviation ~0.04), suggesting that:
   - The majority task subspace information is important throughout the sequence
   - The effect is not position-dependent in a dramatic way
   - The model consistently uses this subspace representation across different context lengths

4. **Magnitude of effect**: A change of approximately -1.0 nats corresponds to roughly a **2.7x reduction** in probability (since exp(-1.0) ≈ 0.37). This is a substantial degradation, indicating that the majority task subspace contains significant predictive information.

#### Implications

These results suggest that:

- **The model encodes task-specific information in the majority task subspace**: The hidden states at padded positions contain projections onto this subspace that are informative for prediction.

- **The majority task subspace is functionally important**: Removing this information consistently degrades performance, indicating it's not just present but actively used by the model.

- **The intervention successfully isolates this component**: The small `Q.T @ h_new` norms confirm we've cleanly removed the projection, allowing us to measure its contribution.

#### Comparison with Other Interventions

If you compare these results with random injection interventions (which typically show smaller or more variable effects), the consistent -1.0 nats effect suggests that:
- The majority task subspace projection is a **specific, structured** component of the representation
- Random perturbations may affect many dimensions, while projection removal targets a specific, functionally important subspace
- The structured nature of this subspace makes it more critical for performance than random noise

#### Next Steps for Analysis

To further understand these results, consider:

1. **Compare with baseline**: Check what the baseline log-probabilities are to understand the relative impact (is -1.0 nats a 10% or 50% reduction?).

2. **Task-specific analysis**: Look at `info['per_task']` to see if the effect varies across different tasks (major vs minor).

3. **Layer comparison**: Run the intervention at different layers to see if the effect is layer-dependent.

4. **Position analysis**: Examine if there are any positions with notably different effects (e.g., early vs late in sequence).

5. **Visualization**: Plot the `Δlogp` values across positions to identify any patterns or trends.

