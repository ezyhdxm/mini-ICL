"""Averaging-based task and token vector estimation."""

from typing import Optional, Sequence

import torch


def estimate_task_vectors_by_averaging(
    all_hiddens_layer: torch.Tensor,
    estimation_positions: Sequence[int],
) -> tuple:
    """Estimate task vectors by averaging hidden states per task.

    Pools hidden states across ``estimation_positions`` and batch
    samples for each task, then centers.

    Parameters
    ----------
    all_hiddens_layer : (n_tasks, n_points, B, D)
        Hidden states for one layer.
    estimation_positions : list of int
        Late position indices to average over.

    Returns
    -------
    (task_vecs, grand_mean)
        task_vecs : (K, D) centred task vectors (sum to zero)
        grand_mean : (D,) grand mean across tasks
    """
    K, n_points, B, D = all_hiddens_layer.shape

    task_means = torch.zeros(K, D, dtype=torch.float32)
    for k in range(K):
        parts = [all_hiddens_layer[k, t, :, :] for t in estimation_positions]
        task_means[k] = torch.cat(parts, dim=0).float().mean(dim=0)

    grand_mean = task_means.mean(dim=0)
    task_vecs = task_means - grand_mean.unsqueeze(0)

    return task_vecs, grand_mean


def per_position_task_vectors(
    all_hiddens_layer: torch.Tensor,
    per_position_mean: bool = True,
) -> tuple:
    """Compute task vectors at every position independently.

    Parameters
    ----------
    all_hiddens_layer : (K, T, B, D)
        Hidden states for one layer, grouped by task.
    per_position_mean : bool
        If True, centre with a position-specific grand mean mu_t.
        If False, use a single grand mean pooled across all positions.

    Returns
    -------
    (task_vecs_by_pos, grand_means)
        task_vecs_by_pos : (K, T, D) centred task vectors at each position
        grand_means : (T, D) grand mean at each position (or broadcast)
    """
    K, T, B, D = all_hiddens_layer.shape

    task_means = all_hiddens_layer.float().mean(dim=2)  # (K, T, D)

    if per_position_mean:
        grand_means = task_means.mean(dim=0)            # (T, D)
    else:
        grand_means = task_means.mean(dim=(0, 1)).unsqueeze(0).expand(T, D)  # (T, D)

    task_vecs_by_pos = task_means - grand_means.unsqueeze(0)  # (K, T, D)

    return task_vecs_by_pos, grand_means


# ---- Balanced (orthogonal-design) variants --------------------------------

def per_position_task_vectors_balanced(
    cell_hiddens: torch.Tensor,
    per_position_mean: bool = True,
) -> tuple:
    """Compute task vectors at every position from a balanced design.

    Uses cell means from token-conditioned (interventional) data and
    uniformly averages over tokens, eliminating token leakage.  This is
    the per-position analogue of the task effect in a two-way ANOVA.

    Parameters
    ----------
    cell_hiddens : (T, V, K, B, D)
        Token-conditioned hidden states for one layer.
        ``T`` positions, ``V`` unique tokens, ``K`` tasks, ``B`` samples.
    per_position_mean : bool
        If True, centre with a position-specific grand mean.

    Returns
    -------
    (task_vecs_by_pos, grand_means)
        task_vecs_by_pos : (K, T, D)
        grand_means      : (T, D)
    """
    T, V, K, B, D = cell_hiddens.shape
    cell_means = cell_hiddens.float().mean(dim=3)       # (T, V, K, D)
    task_means = cell_means.mean(dim=1)                 # (T, K, D)

    if per_position_mean:
        grand_means = task_means.mean(dim=1)            # (T, D)
    else:
        grand_means = task_means.mean(dim=(0, 1)).unsqueeze(0).expand(T, D)

    task_vecs_by_pos = (
        task_means - grand_means.unsqueeze(1)
    ).permute(1, 0, 2)                                  # (K, T, D)

    return task_vecs_by_pos, grand_means


def estimate_task_vectors_by_averaging_balanced(
    cell_hiddens: torch.Tensor,
    estimation_positions: Sequence[int],
) -> tuple:
    """Estimate reference task vectors from a balanced design.

    Pools cell means across ``estimation_positions`` and uniformly
    averages over tokens, then centres.

    Parameters
    ----------
    cell_hiddens : (T, V, K, B, D)
        Token-conditioned hidden states for one layer.
    estimation_positions : list of int
        Position indices (into the T axis) to pool over.

    Returns
    -------
    (task_vecs, grand_mean)
        task_vecs  : (K, D) centred, sum-to-zero
        grand_mean : (D,)
    """
    h = cell_hiddens.float()
    est = h[estimation_positions]                       # (T_est, V, K, B, D)
    est_cell_means = est.mean(dim=3)                    # (T_est, V, K, D)
    task_means = est_cell_means.mean(dim=(0, 1))        # (K, D)
    grand_mean = task_means.mean(dim=0)                 # (D,)
    task_vecs = task_means - grand_mean.unsqueeze(0)    # (K, D)
    return task_vecs, grand_mean


def per_position_token_vectors_balanced(
    cell_hiddens: torch.Tensor,
    per_position_mean: bool = True,
) -> torch.Tensor:
    """Compute token (main-effect) vectors at every position from a balanced design.

    Parameters
    ----------
    cell_hiddens : (T, V, K, B, D)
        Token-conditioned hidden states for one layer.
    per_position_mean : bool
        If True, centre with a position-specific grand mean.

    Returns
    -------
    token_vecs_by_pos : (V, T, D)
    """
    T, V, K, B, D = cell_hiddens.shape
    cell_means = cell_hiddens.float().mean(dim=3)       # (T, V, K, D)
    token_means = cell_means.mean(dim=2)                # (T, V, D)

    if per_position_mean:
        grand_means = cell_means.mean(dim=(1, 2))       # (T, D)
    else:
        grand_means = cell_means.mean(dim=(0, 1, 2)).unsqueeze(0).expand(T, D)

    token_vecs_by_pos = (
        token_means - grand_means.unsqueeze(1)
    ).permute(1, 0, 2)                                  # (V, T, D)

    return token_vecs_by_pos


def estimate_task_and_token_vectors_jointly(
    all_hiddens_layer: torch.Tensor,
    token_ids: torch.Tensor,
    estimation_positions: Sequence[int],
) -> tuple:
    """Jointly estimate task and token vectors via two-way additive ANOVA.

    Solves the OLS problem

        h̃_i = Σ_{k} z_{ik} θ_k + Σ_{s} w_{is} ν_s + ε_i

    on position-demeaned hidden states pooled across estimation
    positions, using drop-one-column coding for identifiability and
    reconstructing sum-to-zero vectors afterwards.

    This avoids the bias that arises when task and token effects are
    estimated by independent averaging under an unbalanced design
    (e.g. latent Markov chains where the token distribution depends
    on the task).

    Parameters
    ----------
    all_hiddens_layer : (K, n_points, B, D)
    token_ids : (K, B, seq_len)
    estimation_positions : list of int

    Returns
    -------
    task_vecs : (K, D)  centred, sum-to-zero
    token_vecs : (V, D) centred, sum-to-zero
    grand_mean : (D,)   grand mean of *raw* hidden states
    """
    K, n_points, B, D = all_hiddens_layer.shape

    h_parts, task_parts, tok_parts = [], [], []
    task_labels = torch.arange(K).unsqueeze(1).expand(K, B).reshape(K * B)

    for t in estimation_positions:
        h_t = all_hiddens_layer[:, t, :, :].reshape(K * B, D).float()
        mu_t = h_t.mean(dim=0)
        h_parts.append(h_t - mu_t)
        task_parts.append(task_labels)
        tok_parts.append(token_ids[:, :, t].reshape(K * B))

    h_pooled = torch.cat(h_parts, dim=0)           # (M, D)
    z_task = torch.cat(task_parts, dim=0).long()    # (M,)
    z_tok = torch.cat(tok_parts, dim=0).long()      # (M,)

    M = h_pooled.shape[0]
    V = int(z_tok.max().item()) + 1

    n_cols = (K - 1) + (V - 1)
    X = torch.zeros(M, n_cols, dtype=h_pooled.dtype, device=h_pooled.device)
    for k in range(K - 1):
        X[:, k] = (z_task == k).float()
    for s in range(V - 1):
        X[:, (K - 1) + s] = (z_tok == s).float()

    W = torch.linalg.lstsq(X, h_pooled).solution   # (n_cols, D)

    task_coefs = torch.cat([W[:K - 1], torch.zeros(1, D, dtype=W.dtype, device=W.device)], dim=0)
    task_vecs = task_coefs - task_coefs.mean(dim=0, keepdim=True)

    tok_coefs = torch.cat([W[K - 1 : K - 1 + V - 1], torch.zeros(1, D, dtype=W.dtype, device=W.device)], dim=0)
    token_vecs = tok_coefs - tok_coefs.mean(dim=0, keepdim=True)

    raw_parts = [
        all_hiddens_layer[:, t, :, :].reshape(K * B, D).float()
        for t in estimation_positions
    ]
    grand_mean = torch.cat(raw_parts, dim=0).mean(dim=0)

    return task_vecs, token_vecs, grand_mean


def estimate_token_vectors_by_averaging(
    all_hiddens_layer: torch.Tensor,
    token_ids: torch.Tensor,
    estimation_positions: Sequence[int],
    task_vecs: "Optional[torch.Tensor]" = None,
) -> tuple:
    """Estimate token (vocab) vectors by averaging hidden states per token.

    At each estimation position the per-position mean is subtracted to
    remove position-encoding effects.  When ``task_vecs`` is provided
    the task effect is also subtracted before averaging by token
    identity, preventing task--token confounding from contaminating
    the token vectors (critical when the token distribution depends
    on the task, e.g. latent Markov chains).

    Parameters
    ----------
    all_hiddens_layer : (K, n_points, B, D)
        Hidden states for one layer.
    token_ids : (K, B, seq_len)
        Integer token ids for each (task, sample, position).
    estimation_positions : list of int
        Position indices to average over (same as for task vectors).
    task_vecs : (K, D), optional
        Centred task vectors (sum-to-zero).  When provided, each
        sample's task effect is subtracted before averaging by token,
        so the resulting token vectors are not confounded with the
        task effect.

    Returns
    -------
    token_vecs : (V, D)
        Centred token vectors.
    grand_mean : (D,)
        Grand mean across all pooled (raw) samples.
    """
    K, n_points, B, D = all_hiddens_layer.shape

    task_labels = (
        torch.arange(K).unsqueeze(1).expand(K, B).reshape(K * B)
    )

    h_parts = []
    s_parts = []
    for t in estimation_positions:
        h_t = all_hiddens_layer[:, t, :, :].reshape(K * B, D).float()
        mu_t = h_t.mean(dim=0)
        h_demeaned = h_t - mu_t

        if task_vecs is not None:
            h_demeaned = h_demeaned - task_vecs.to(h_demeaned.device)[task_labels].float()

        h_parts.append(h_demeaned)
        s_parts.append(token_ids[:, :, t].reshape(K * B))

    h_pooled = torch.cat(h_parts, dim=0)   # (M, D)
    s_pooled = torch.cat(s_parts, dim=0).long()

    raw_parts = [
        all_hiddens_layer[:, t, :, :].reshape(K * B, D).float()
        for t in estimation_positions
    ]
    grand_mean = torch.cat(raw_parts, dim=0).mean(dim=0)

    V = int(s_pooled.max().item()) + 1
    token_means = torch.zeros(V, D, dtype=torch.float32)
    for s in range(V):
        mask = (s_pooled == s)
        if mask.any():
            token_means[s] = h_pooled[mask].mean(dim=0)
        else:
            token_means[s] = torch.zeros(D, dtype=torch.float32)

    return token_means, grand_mean
