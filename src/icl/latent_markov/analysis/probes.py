"""
Linear probes for Latent Markov task on **non-padded** sequences.

- Joint OLS hidden predictor (posterior + token + logit → hiddens)
- Val R² across layers sweep

Legacy softmax predictor functions are re-exported from
``icl.latent_markov.legacy.softmax_predictor``.
"""

import gc
import numpy as np
import torch
from torch import nn
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger
from icl.latent_markov.analysis.bayes import task_posterior_over_time
from icl.latent_markov.analysis.ood import get_latent_sampler

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _collect_multi_layer_data(
    exp_name: str,
    layers: list,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    positions: Optional[list] = None,
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    device: Optional[str] = None,
    verbose: bool = False,
    anchor_minor_samples: Optional[int] = None,
    extraction_point: str = "post_attn",
    use_task_identity: bool = False,
) -> dict:
    """Load model once, generate data once, hook all *layers* in one pass.

    Returns a dict with per-layer hiddens and shared logits / posteriors /
    tokens so that ``_fit_probe`` can be called cheaply per layer.

    Parameters
    ----------
    extraction_point : ``"post_attn"`` | ``"post_mlp"``
        Where to hook each transformer layer.
        ``"post_attn"`` (default) — after the attention block, before MLP.
        ``"post_mlp"`` — after the full block (attention + MLP), i.e. the
        standard residual-stream representation used in most probing work.
    use_task_identity : bool
        If True, collect the ground-truth task label (latent index) for each
        sequence by forcing ``mode="major"`` generation, which returns the
        latent alongside the samples.  The labels are returned under the
        ``"task_ids"`` key and can be used by ``_fit_probe`` to replace the
        continuous Bayesian posterior with a one-hot task-identity encoding.
    """
    from icl.latent_markov.analysis.ood import get_latent_sampler

    _, orig_sampler, config = nu.load_everything("latent", exp_name)
    if device is not None:
        config.device = device
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True
    )
    model.eval().to(config.device)

    if n_minor is None:
        n_minor = 1_000_000
    elif n_minor == -1:
        n_minor = 0

    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood=0)
    if device is not None and hasattr(sampler, "to"):
        sampler = sampler.to(device)
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks

    _original_p_minor = float(sampler.p_minor)
    if uniform_sampling and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (
            sampler.n_major_tasks + sampler.n_minor_tasks
        )

    if device is not None and hasattr(orig_sampler, "to"):
        orig_sampler = orig_sampler.to(device)

    seq_len = sampler.seq_len
    if positions is None:
        positions = list(range(min(10, seq_len)))
    else:
        positions = list(positions)

    dev = config.device
    position_indices = torch.tensor(positions, device=dev, dtype=torch.long)

    if verbose:
        logger.info(
            f"[collect] layers={layers}, B={B}, n_samples={n_samples}, "
            f"mode={sample_mode!r}, positions={positions[:5]}{'...' if len(positions) > 5 else ''}"
            + (" [task_identity]" if use_task_identity else "")
        )

    # When use_task_identity=True we force "major" mode so that the sampler
    # returns the latent index as gen_out[2].
    _gen_mode = "major" if use_task_identity else sample_mode

    acc_hiddens = {l: [] for l in layers}
    acc_logits: list = []
    acc_posteriors: list = []
    acc_tokens: list = []
    acc_task_ids: list = []
    n_batches = (n_samples + B - 1) // B

    for batch_idx in range(n_batches):
        gen_out = sampler.generate(
            mode=_gen_mode, task=None, num_samples=B, epochs=1,
        )
        samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples = samples.to(dev)

        # For "major" mode with task=None the sampler returns (samples, probs, latent).
        if use_task_identity and isinstance(gen_out, (tuple, list)) and len(gen_out) >= 3:
            acc_task_ids.append(gen_out[2].cpu())

        posteriors = task_posterior_over_time(
            orig_sampler, samples, include_minor=True,
        )
        posteriors_at_pos = posteriors[:, position_indices, :]
        real_tokens_batch = samples[:, position_indices]

        caches: dict = {}
        handles = []
        for l in layers:
            layer_mod = (
                model.layers[l] if extraction_point == "post_mlp"
                else model.layers[l].attn_block
            )

            def _make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    if torch.is_tensor(out):
                        caches[layer_idx] = out.index_select(
                            dim=1, index=position_indices,
                        ).detach()
                    elif (isinstance(out, tuple) and len(out) > 0
                          and torch.is_tensor(out[0])):
                        caches[layer_idx] = out[0].index_select(
                            dim=1, index=position_indices,
                        ).detach()
                return hook_fn

            handles.append(layer_mod.register_forward_hook(_make_hook(l)))

        try:
            with torch.no_grad():
                logits_full = model(samples)
                logits_batch = logits_full.index_select(
                    dim=1, index=position_indices,
                )
        finally:
            for h in handles:
                h.remove()

        for l in layers:
            acc_hiddens[l].append(caches[l].cpu())
        acc_logits.append(logits_batch.cpu())
        acc_posteriors.append(posteriors_at_pos.cpu())
        acc_tokens.append(real_tokens_batch.cpu())

        del samples, posteriors, logits_full, logits_batch, caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Anchor with minor-task sequences to break the dummy-variable trap.
    # When sample_mode="major" and the sampler has minor tasks, generate
    # extra sequences via mode="train" (which mixes major + minor) so that
    # the truncated posterior doesn't always sum to 1.
    # Only anchor if the minor tasks are genuine (original p_minor is
    # non-negligible).  When k=-1 the config sets n_minor=1, p_minor=1e-12
    # as a ghost — anchoring with it would be meaningless.
    _do_anchor = (
        sample_mode == "major"
        and sampler.n_minor_tasks > 0
        and _original_p_minor > 1e-6
        and (anchor_minor_samples is None or anchor_minor_samples > 0)
    )
    if _do_anchor:
        n_anchor = anchor_minor_samples if anchor_minor_samples is not None else max(B, n_samples // 5)
        n_anchor_batches = (n_anchor + B - 1) // B
        if verbose:
            logger.info(f"[collect] anchoring with {n_anchor} train-mode samples ({n_anchor_batches} batches)")
        for _ in range(n_anchor_batches):
            gen_out = sampler.generate(
                mode="train", task=None, num_samples=B, epochs=1,
            )
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(dev)

            posteriors = task_posterior_over_time(
                orig_sampler, samples, include_minor=True,
            )
            posteriors_at_pos = posteriors[:, position_indices, :]
            real_tokens_batch = samples[:, position_indices]

            caches_a: dict = {}
            handles_a = []
            for l in layers:
                layer_mod_a = (
                    model.layers[l] if extraction_point == "post_mlp"
                    else model.layers[l].attn_block
                )

                def _make_hook_a(layer_idx):
                    def hook_fn(module, inp, out):
                        if torch.is_tensor(out):
                            caches_a[layer_idx] = out.index_select(
                                dim=1, index=position_indices,
                            ).detach()
                        elif (isinstance(out, tuple) and len(out) > 0
                              and torch.is_tensor(out[0])):
                            caches_a[layer_idx] = out[0].index_select(
                                dim=1, index=position_indices,
                            ).detach()
                    return hook_fn

                handles_a.append(layer_mod_a.register_forward_hook(_make_hook_a(l)))

            try:
                with torch.no_grad():
                    logits_full = model(samples)
                    logits_batch = logits_full.index_select(
                        dim=1, index=position_indices,
                    )
            finally:
                for h in handles_a:
                    h.remove()

            for l in layers:
                acc_hiddens[l].append(caches_a[l].cpu())
            acc_logits.append(logits_batch.cpu())
            acc_posteriors.append(posteriors_at_pos.cpu())
            acc_tokens.append(real_tokens_batch.cpu())

            del samples, posteriors, logits_full, logits_batch, caches_a
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "hiddens_by_layer": {
            l: torch.cat(acc_hiddens[l], dim=0) for l in layers
        },
        "logits": torch.cat(acc_logits, dim=0),
        "posteriors": torch.cat(acc_posteriors, dim=0),
        "real_tokens": torch.cat(acc_tokens, dim=0),
        "n_major": sampler.n_major_tasks,
        "n_tasks": n_tasks,
        "positions": positions,
        # Ground-truth task labels (shape N_seq,) when use_task_identity=True.
        "task_ids": torch.cat(acc_task_ids, dim=0) if acc_task_ids else None,
    }


# ---- module-level OLS / MLP helpers (shared by _fit_probe) ----

def _fit_ols(Xtr, Ytr, Xva, Yva):
    ones_tr = torch.ones(Xtr.shape[0], 1, dtype=Xtr.dtype, device=Xtr.device)
    Xtr_aug = torch.cat([Xtr, ones_tr], dim=1)
    W_aug = torch.linalg.pinv(Xtr_aug) @ Ytr
    W = W_aug[:-1, :]
    b = W_aug[-1, :]

    pred_tr = Xtr @ W + b
    pred_va = Xva @ W + b
    tr_res = Ytr - pred_tr
    va_res = Yva - pred_va

    tr_ss_res = (tr_res ** 2).sum().item()
    va_ss_res = (va_res ** 2).sum().item()
    tr_ss_tot = ((Ytr - Ytr.mean(dim=0)) ** 2).sum().item()
    va_ss_tot = ((Yva - Yva.mean(dim=0)) ** 2).sum().item()
    n_dim = Ytr.shape[1]
    return W, b, {
        "tr_mse": tr_ss_res / (Ytr.shape[0] * n_dim),
        "va_mse": va_ss_res / (Yva.shape[0] * n_dim),
        "tr_r2": 1.0 - tr_ss_res / tr_ss_tot if tr_ss_tot > 0 else float("nan"),
        "va_r2": 1.0 - va_ss_res / va_ss_tot if va_ss_tot > 0 else float("nan"),
        "tr_ss_res": tr_ss_res,
        "va_ss_res": va_ss_res,
        "n_features": Xtr.shape[1],
    }


def _fit_mlp_r2(Xtr, Ytr, Xva, Yva,
                hidden_dim=128, epochs=200, lr=1e-3, batch_size=4096):
    from torch import nn
    d_in, d_out = Xtr.shape[1], Ytr.shape[1]
    mlp = nn.Sequential(
        nn.Linear(d_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_out),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], batch_size):
            batch = perm[i:i + batch_size]
            loss = ((mlp(Xtr[batch]) - Ytr[batch]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    mlp.eval()
    with torch.no_grad():
        pred = mlp(Xva)
        ss_res = ((Yva - pred) ** 2).sum().item()
        ss_tot = ((Yva - Yva.mean(dim=0)) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _fit_probe(
    hiddens_all,      # (N, P, D) — this layer's hiddens
    posteriors_all,    # (N, P, T) — full posteriors (will be cropped)
    logits_all,        # (N, P, V)
    real_tokens_all,   # (N, P)
    n_major: int,
    n_tasks: int,
    layer: int,
    positions: list,
    validation_split: float = 0.2,
    include_position_bias: bool = True,
    include_logit: bool = True,
    use_log_posterior: bool = False,
    skip_baselines: bool = False,
    sample_mode: str = "train",
    seq_perm: Optional[torch.Tensor] = None,
    per_position_mean: bool = False,
    task_ids_all: Optional[torch.Tensor] = None,
) -> dict:
    """All OLS fitting for a single layer on pre-collected data.

    If *seq_perm* is given the same permutation is reused across layers so
    the train/val split is consistent.

    If *per_position_mean* is True, the mean hidden state at each position
    (computed from training sequences only) is subtracted before fitting,
    removing position-specific DC offsets so the probe captures only
    task-driven variation.

    If *task_ids_all* is provided (shape ``(N,)`` of integer task indices),
    the "posterior" feature block is replaced with a one-hot task-identity
    encoding.  One column is dropped to avoid the dummy-variable trap.
    This gives a cleaner, binary design matrix with no within-group
    collinearity, making partial-R² comparisons unambiguous.
    """
    posteriors_maj = posteriors_all[:, :, :n_major]

    vocab_size_actual = logits_all.shape[-1]
    rt_flat = real_tokens_all.reshape(-1)
    n_vocab = max(int(rt_flat.max().item()) + 1, vocab_size_actual)

    N_seq = hiddens_all.shape[0]
    n_seq_train = int(N_seq * (1 - validation_split))
    if seq_perm is None:
        seq_perm = torch.randperm(N_seq)
    seq_tr, seq_va = seq_perm[:n_seq_train], seq_perm[n_seq_train:]

    def _flatten(tensor, indices):
        return tensor[indices].reshape(-1, tensor.shape[-1]).float()

    if per_position_mean:
        # Estimate per-position mean from training sequences only to avoid
        # leakage; subtract from both splits.  Shape: (P, D)
        pos_mean = hiddens_all[seq_tr].float().mean(dim=0)
        Ytr = (hiddens_all[seq_tr].float() - pos_mean).reshape(-1, hiddens_all.shape[-1])
        Yva = (hiddens_all[seq_va].float() - pos_mean).reshape(-1, hiddens_all.shape[-1])
    else:
        Ytr = _flatten(hiddens_all, seq_tr)
        Yva = _flatten(hiddens_all, seq_va)
    post_tr = _flatten(posteriors_maj, seq_tr)
    post_va = _flatten(posteriors_maj, seq_va)
    logit_tr = _flatten(logits_all, seq_tr)
    logit_va = _flatten(logits_all, seq_va)

    rt_tr = real_tokens_all[seq_tr].reshape(-1).long()
    rt_va = real_tokens_all[seq_va].reshape(-1).long()
    oh_tr = torch.zeros(rt_tr.shape[0], n_vocab, dtype=torch.float32)
    oh_tr.scatter_(1, rt_tr.unsqueeze(1), 1.0)
    oh_va = torch.zeros(rt_va.shape[0], n_vocab, dtype=torch.float32)
    oh_va.scatter_(1, rt_va.unsqueeze(1), 1.0)

    # Position nuisance features: one-hot(position index), drop last column
    # to avoid yet another dummy-variable trap with the intercept.
    n_pos = hiddens_all.shape[1]
    use_pos_bias = include_position_bias and n_pos > 1
    if use_pos_bias:
        pos_tr = torch.arange(n_pos).unsqueeze(0).expand(seq_tr.shape[0], n_pos).reshape(-1)
        pos_va = torch.arange(n_pos).unsqueeze(0).expand(seq_va.shape[0], n_pos).reshape(-1)
        X_pos_full_tr = torch.zeros(pos_tr.shape[0], n_pos, dtype=torch.float32)
        X_pos_full_tr.scatter_(1, pos_tr.unsqueeze(1), 1.0)
        X_pos_full_va = torch.zeros(pos_va.shape[0], n_pos, dtype=torch.float32)
        X_pos_full_va.scatter_(1, pos_va.unsqueeze(1), 1.0)
        X_pos_tr = X_pos_full_tr[:, :-1]
        X_pos_va = X_pos_full_va[:, :-1]
    else:
        X_pos_tr = None
        X_pos_va = None

    if task_ids_all is not None:
        # One-hot task-identity encoding at the sequence level, expanded to
        # positions.  Last column dropped to avoid the dummy-variable trap.
        # Shape after drop: (N_seq * P, n_major - 1).
        _n_posterior_orig = n_major  # K columns before drop
        _post_redundant = False       # we handle the drop explicitly below

        def _task_onehot(seq_indices):
            ids = task_ids_all[seq_indices].clamp(0, n_major - 1)  # (N_split,)
            oh = torch.zeros(len(seq_indices), n_major, dtype=torch.float32)
            oh.scatter_(1, ids.unsqueeze(1), 1.0)
            # Expand across positions: (N_split, n_major) → (N_split * P, n_major)
            return oh.unsqueeze(1).expand(-1, n_pos, -1).reshape(-1, n_major)

        oh_task_tr = _task_onehot(seq_tr)
        oh_task_va = _task_onehot(seq_va)
        # Drop last column (K → K-1 features, no dummy-variable trap).
        X_main_tr = oh_task_tr[:, :-1]
        X_main_va = oh_task_va[:, :-1]
    elif use_log_posterior:
        X_main_tr = torch.log(post_tr + 1e-10)
        X_main_va = torch.log(post_va + 1e-10)
        _post_sum_tr = X_main_tr.sum(dim=1)
        _post_redundant = False
        _n_posterior_orig = X_main_tr.shape[1]
    else:
        X_main_tr, X_main_va = post_tr, post_va
        # Detect dummy-variable trap: if posterior columns sum to ~1 (no minor
        # tasks), the intercept is redundant.  Drop the last posterior column.
        _post_sum_tr = X_main_tr.sum(dim=1)
        _post_redundant = (
            X_main_tr.shape[1] > 1
            and (_post_sum_tr - 1.0).abs().max().item() < 1e-4
        )
        _n_posterior_orig = X_main_tr.shape[1]
        if _post_redundant:
            X_main_tr = X_main_tr[:, :-1]
            X_main_va = X_main_va[:, :-1]

    # Drop last column from token one-hot to avoid a second dummy-variable
    # trap (the V one-hot columns also sum to 1, collinear with intercept).
    _n_tok_orig = oh_tr.shape[1]
    X_tok_tr = oh_tr[:, :-1]
    X_tok_va = oh_va[:, :-1]

    if include_logit:
        X_logit_tr, X_logit_va = logit_tr, logit_va
    else:
        X_logit_tr = torch.zeros(X_tok_tr.shape[0], 0, dtype=torch.float32)
        X_logit_va = torch.zeros(X_tok_va.shape[0], 0, dtype=torch.float32)

    D = hiddens_all.shape[-1]
    n_total = hiddens_all.shape[0] * hiddens_all.shape[1]
    n_train = Ytr.shape[0]

    # ---- joint fit ----
    X_joint_tr_parts = [X_main_tr, X_tok_tr, X_logit_tr]
    X_joint_va_parts = [X_main_va, X_tok_va, X_logit_va]
    if X_pos_tr is not None:
        X_joint_tr_parts.append(X_pos_tr)
        X_joint_va_parts.append(X_pos_va)
    X_joint_tr = torch.cat(X_joint_tr_parts, dim=1)
    X_joint_va = torch.cat(X_joint_va_parts, dim=1)
    W_joint, b_joint, joint_s = _fit_ols(X_joint_tr, Ytr, X_joint_va, Yva)

    d_main = X_main_tr.shape[1]
    d_tok = X_tok_tr.shape[1]
    d_logit = X_logit_tr.shape[1]
    d_pos = X_pos_tr.shape[1] if X_pos_tr is not None else 0
    W_task_raw = W_joint[:d_main, :]
    W_tok_raw = W_joint[d_main:d_main + d_tok, :]
    W_logit_block = W_joint[d_main + d_tok:d_main + d_tok + d_logit, :]
    W_pos_raw = W_joint[d_main + d_tok + d_logit:, :] if d_pos > 0 else None

    # Reconstruct full-sized weight matrices for each group where a column
    # was dropped.  The dropped category's weight is set to 0 (gauge choice);
    # the centered vectors W - w_bar are gauge-invariant.
    if _post_redundant:
        W_task = torch.zeros((_n_posterior_orig, W_task_raw.shape[1]),
                             dtype=W_task_raw.dtype)
        W_task[:_n_posterior_orig - 1, :] = W_task_raw
    else:
        W_task = W_task_raw

    W_tok_block = torch.zeros((_n_tok_orig, W_tok_raw.shape[1]),
                              dtype=W_tok_raw.dtype)
    W_tok_block[:_n_tok_orig - 1, :] = W_tok_raw

    if W_pos_raw is not None:
        W_pos_block = torch.zeros((n_pos, W_pos_raw.shape[1]),
                                  dtype=W_pos_raw.dtype)
        W_pos_block[:n_pos - 1, :] = W_pos_raw
    else:
        W_pos_block = None

    # ---- marginal / pairwise fits ----
    if X_pos_tr is not None:
        X_main_marg_tr = torch.cat([X_main_tr, X_pos_tr], dim=1)
        X_main_marg_va = torch.cat([X_main_va, X_pos_va], dim=1)
        X_tok_marg_tr = torch.cat([X_tok_tr, X_pos_tr], dim=1)
        X_tok_marg_va = torch.cat([X_tok_va, X_pos_va], dim=1)
        X_logit_marg_tr = torch.cat([X_logit_tr, X_pos_tr], dim=1)
        X_logit_marg_va = torch.cat([X_logit_va, X_pos_va], dim=1)
    else:
        X_main_marg_tr, X_main_marg_va = X_main_tr, X_main_va
        X_tok_marg_tr, X_tok_marg_va = X_tok_tr, X_tok_va
        X_logit_marg_tr, X_logit_marg_va = X_logit_tr, X_logit_va

    _, _, pi_s = _fit_ols(X_main_marg_tr, Ytr, X_main_marg_va, Yva)
    _, _, tok_s = _fit_ols(X_tok_marg_tr, Ytr, X_tok_marg_va, Yva)
    _, _, logit_s = _fit_ols(X_logit_marg_tr, Ytr, X_logit_marg_va, Yva)

    X_pt_parts_tr = [X_main_tr, X_tok_tr]
    X_pt_parts_va = [X_main_va, X_tok_va]
    if X_pos_tr is not None:
        X_pt_parts_tr.append(X_pos_tr)
        X_pt_parts_va.append(X_pos_va)
    X_pt_tr = torch.cat(X_pt_parts_tr, dim=1)
    X_pt_va = torch.cat(X_pt_parts_va, dim=1)
    _, _, post_tok_s = _fit_ols(X_pt_tr, Ytr, X_pt_va, Yva)

    X_pl_parts_tr = [X_main_tr, X_logit_tr]
    X_pl_parts_va = [X_main_va, X_logit_va]
    if X_pos_tr is not None:
        X_pl_parts_tr.append(X_pos_tr)
        X_pl_parts_va.append(X_pos_va)
    X_pl_tr = torch.cat(X_pl_parts_tr, dim=1)
    X_pl_va = torch.cat(X_pl_parts_va, dim=1)
    _, _, post_logit_s = _fit_ols(X_pl_tr, Ytr, X_pl_va, Yva)

    X_tl_parts_tr = [X_tok_tr, X_logit_tr]
    X_tl_parts_va = [X_tok_va, X_logit_va]
    if X_pos_tr is not None:
        X_tl_parts_tr.append(X_pos_tr)
        X_tl_parts_va.append(X_pos_va)
    X_tl_tr = torch.cat(X_tl_parts_tr, dim=1)
    X_tl_va = torch.cat(X_tl_parts_va, dim=1)
    _, _, tok_logit_s = _fit_ols(X_tl_tr, Ytr, X_tl_va, Yva)

    # ---- partial R^2 ----
    _eps = 1e-10
    partial_r2_post = (
        (joint_s["va_r2"] - tok_logit_s["va_r2"])
        / max(1.0 - tok_logit_s["va_r2"], _eps)
    )
    partial_r2_tok = (
        (joint_s["va_r2"] - post_logit_s["va_r2"])
        / max(1.0 - post_logit_s["va_r2"], _eps)
    )
    partial_r2_logit = (
        (joint_s["va_r2"] - post_tok_s["va_r2"])
        / max(1.0 - post_tok_s["va_r2"], _eps)
    )

    # ---- incremental F-test ----
    n_tr = Ytr.shape[0]
    p_full = d_main + d_tok + d_logit + d_pos
    df_den = n_tr - p_full - 1

    def _f_test(ss_reduced, ss_full, q, df_d):
        if q <= 0 or df_d <= 0 or ss_full <= 0:
            return {"F": float("nan"), "p": float("nan"),
                    "df_num": q, "df_den": df_d}
        f_val = ((ss_reduced - ss_full) / q) / (ss_full / df_d)
        try:
            from scipy.stats import f as f_dist
            p_val = float(f_dist.sf(max(f_val, 0.0), q, df_d))
        except ImportError:
            p_val = float("nan")
        return {"F": f_val, "p": p_val, "df_num": q, "df_den": df_d}

    f_test_post = _f_test(
        tok_logit_s["tr_ss_res"], joint_s["tr_ss_res"], d_main, df_den,
    )
    f_test_tok = _f_test(
        post_logit_s["tr_ss_res"], joint_s["tr_ss_res"], d_tok, df_den,
    )
    f_test_logit = _f_test(
        post_tok_s["tr_ss_res"], joint_s["tr_ss_res"], d_logit, df_den,
    )

    # ---- collinearity diagnostics ----
    cond_num = float(torch.linalg.cond(
        torch.cat([X_joint_tr,
                    torch.ones(n_tr, 1, dtype=X_joint_tr.dtype)], dim=1),
    ).item())

    def _group_vif(X_group, X_rest):
        ones = torch.ones(X_rest.shape[0], 1, dtype=X_rest.dtype)
        X_aug = torch.cat([X_rest, ones], dim=1)
        W_ = torch.linalg.pinv(X_aug) @ X_group
        pred = X_aug @ W_
        ss_res = ((X_group - pred) ** 2).sum().item()
        ss_tot = ((X_group - X_group.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return 1.0 / max(1.0 - r2, 1e-10), r2

    if X_pos_tr is not None and d_logit > 0:
        vif_post, r2_post_rest = _group_vif(
            X_main_tr, torch.cat([X_tok_tr, X_logit_tr, X_pos_tr], dim=1),
        )
        vif_tok, r2_tok_rest = _group_vif(
            X_tok_tr, torch.cat([X_main_tr, X_logit_tr, X_pos_tr], dim=1),
        )
        vif_logit, r2_logit_rest = _group_vif(
            X_logit_tr, torch.cat([X_main_tr, X_tok_tr, X_pos_tr], dim=1),
        )
    elif d_logit > 0:
        vif_post, r2_post_rest = _group_vif(
            X_main_tr, torch.cat([X_tok_tr, X_logit_tr], dim=1),
        )
        vif_tok, r2_tok_rest = _group_vif(
            X_tok_tr, torch.cat([X_main_tr, X_logit_tr], dim=1),
        )
        vif_logit, r2_logit_rest = _group_vif(
            X_logit_tr, torch.cat([X_main_tr, X_tok_tr], dim=1),
        )
    else:
        if X_pos_tr is not None:
            vif_post, r2_post_rest = _group_vif(
                X_main_tr, torch.cat([X_tok_tr, X_pos_tr], dim=1),
            )
            vif_tok, r2_tok_rest = _group_vif(
                X_tok_tr, torch.cat([X_main_tr, X_pos_tr], dim=1),
            )
        else:
            vif_post, r2_post_rest = _group_vif(X_main_tr, X_tok_tr)
            vif_tok, r2_tok_rest = _group_vif(X_tok_tr, X_main_tr)
        vif_logit, r2_logit_rest = float("nan"), float("nan")

    gvif_post = vif_post ** (1.0 / (2 * d_main)) if d_main > 0 else float("nan")
    gvif_tok = vif_tok ** (1.0 / (2 * d_tok)) if d_tok > 0 else float("nan")
    gvif_logit = vif_logit ** (1.0 / (2 * d_logit)) if d_logit > 0 else float("nan")

    def _pairwise_r2(Xa, Xb):
        ones = torch.ones(Xb.shape[0], 1, dtype=Xb.dtype)
        X_aug = torch.cat([Xb, ones], dim=1)
        W_ = torch.linalg.pinv(X_aug) @ Xa
        pred = X_aug @ W_
        ss_res = ((Xa - pred) ** 2).sum().item()
        ss_tot = ((Xa - Xa.mean(0)) ** 2).sum().item()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pw_pt = _pairwise_r2(X_main_tr, X_tok_tr)
    pw_pl = _pairwise_r2(X_main_tr, X_logit_tr)
    pw_tl = _pairwise_r2(X_tok_tr, X_logit_tr)

    design_diagnostics = {
        "condition_number": cond_num,
        "n_features": {"posterior": d_main, "token": d_tok, "logit": d_logit,
                        "position": d_pos,
                        "total": d_main + d_tok + d_logit + d_pos},
        "vif": {"posterior": vif_post, "token": vif_tok, "logit": vif_logit},
        "gvif_adj": {"posterior": gvif_post, "token": gvif_tok, "logit": gvif_logit},
        "r2_from_rest": {"posterior": r2_post_rest, "token": r2_tok_rest,
                         "logit": r2_logit_rest},
        "pairwise_r2": {"post_tok": pw_pt, "post_logit": pw_pl,
                         "tok_logit": pw_tl},
    }

    diagnostics = {
        "r2_posterior_only": pi_s["va_r2"],
        "r2_token_only": tok_s["va_r2"],
        "r2_logit_only": logit_s["va_r2"],
        "r2_post_tok": post_tok_s["va_r2"],
        "r2_post_logit": post_logit_s["va_r2"],
        "r2_tok_logit": tok_logit_s["va_r2"],
        "r2_joint": joint_s["va_r2"],
        "partial_r2_posterior": partial_r2_post,
        "partial_r2_token": partial_r2_tok,
        "partial_r2_logit": partial_r2_logit,
        "f_test_posterior": f_test_post,
        "f_test_token": f_test_tok,
        "f_test_logit": f_test_logit,
        "condition_number": cond_num,
        "design_diagnostics": design_diagnostics,
        "mlp_val_r2": None,
        "position_bias_included": bool(use_pos_bias),
        "posterior_column_dropped": bool(_post_redundant),
    }

    # ---- optional heavier diagnostics ----
    geometry = None

    if not skip_baselines:
        diagnostics["mlp_val_r2"] = _fit_mlp_r2(
            X_joint_tr, Ytr, X_joint_va, Yva,
        )

        eps = 1e-10
        rank_tol = 1e-5
        Wt_f = W_task_raw.T.float()
        Wx_f = W_tok_raw.T.float()

        def _rank_basis(M: torch.Tensor) -> torch.Tensor:
            U, S, _ = torch.linalg.svd(M, full_matrices=False)
            r = (S > S[0] * rank_tol).sum().item()
            return U[:, :r]

        Qt = _rank_basis(Wt_f)
        Qx = _rank_basis(Wx_f)

        cos_angles = torch.linalg.svdvals(Qt.T @ Qx).clamp(0.0, 1.0)
        angles_deg = torch.rad2deg(torch.acos(cos_angles))

        subspace_angles = {
            "principal_angles_deg": angles_deg.tolist(),
            "mean_angle_deg": angles_deg.mean().item(),
            "min_angle_deg": angles_deg.min().item(),
            "max_cos": cos_angles.max().item(),
            "mean_cos2": cos_angles.pow(2).mean().item(),
            "rank_task": Qt.shape[1],
            "rank_token": Qx.shape[1],
        }

        c_task_va = (X_main_va @ W_task_raw).float()
        c_tok_va = (X_tok_va @ W_tok_raw).float()
        dot = (c_task_va * c_tok_va).sum(dim=1)
        norms = c_task_va.norm(dim=1) * c_tok_va.norm(dim=1) + eps
        per_sample_cos = dot / norms

        component_cosine = {
            "mean": per_sample_cos.mean().item(),
            "std": per_sample_cos.std().item(),
            "median": per_sample_cos.median().item(),
            "abs_mean": per_sample_cos.abs().mean().item(),
        }

        Wt_rows = W_task_raw.float()
        Wx_rows = W_tok_raw.float()
        Wt_norm = Wt_rows / (Wt_rows.norm(dim=1, keepdim=True) + eps)
        Wx_norm = Wx_rows / (Wx_rows.norm(dim=1, keepdim=True) + eps)
        row_cos_matrix = Wt_norm @ Wx_norm.T
        row_cosine = {
            "matrix": row_cos_matrix.cpu(),
            "max_abs": row_cos_matrix.abs().max().item(),
            "mean_abs": row_cos_matrix.abs().mean().item(),
        }

        geometry = {
            "joint_train_mse": joint_s["tr_mse"],
            "joint_val_mse": joint_s["va_mse"],
            "joint_train_r2": joint_s["tr_r2"],
            "joint_val_r2": joint_s["va_r2"],
            "subspace_angles": subspace_angles,
            "component_cosine": component_cosine,
            "row_cosine": row_cosine,
            "component_weight": {
                "main": W_task.cpu(),
                "token": W_tok_block.cpu(),
                "logit": W_logit_block.cpu(),
                "bias": b_joint.cpu(),
            },
        }

    return {
        "train_mse": joint_s["tr_mse"],
        "val_mse": joint_s["va_mse"],
        "train_r2": joint_s["tr_r2"],
        "val_r2": joint_s["va_r2"],
        "model_weight": W_task.cpu(),
        "model_bias": b_joint.cpu(),
        "token_weight": W_tok_block.cpu(),
        "logit_weight": W_logit_block.cpu(),
        "position_weight": W_pos_block.cpu() if W_pos_block is not None else None,
        "diagnostics": diagnostics,
        "geometry": geometry,
        "layer": layer,
        "n_tasks": n_tasks,
        "hidden_dim": D,
        "vocab_size": vocab_size_actual,
        "n_samples": n_total,
        "n_train": n_train,
        "n_val": Yva.shape[0],
        "positions": positions,
        "_internals": {
            "joint_s": joint_s, "pi_s": pi_s, "tok_s": tok_s,
            "logit_s": logit_s, "post_tok_s": post_tok_s,
        },
    }


def _print_probe_summary(results: dict, sample_mode: str = "train"):
    """Print the fit summary table for one layer."""
    layer = results["layer"]
    diag = results["diagnostics"]
    geometry = results.get("geometry")
    internals = results.get("_internals", {})
    joint_s = internals.get("joint_s", {})
    pi_s = internals.get("pi_s", {})
    tok_s = internals.get("tok_s", {})
    logit_s = internals.get("logit_s", {})
    post_tok_s = internals.get("post_tok_s", {})

    has_logit = diag.get("design_diagnostics", {}).get("n_features", {}).get("logit", 0) > 0
    has_pos = diag.get("position_bias_included", False)
    pos_tag = "+pos" if has_pos else ""

    _r2 = "\u00b2"
    dropped = diag.get("posterior_column_dropped", False)
    print(f"\n=== Fit Summary (Latent non-padded, layer {layer}, mode={sample_mode!r}) ===")
    if dropped:
        print("  [posterior col dropped: sum(pi)=1 detected, last column removed to avoid dummy-variable trap]")
    print("  [token & position one-hot: last column dropped from each to avoid dummy-variable traps]")
    print(f"{'Model':<30} {'Train R' + _r2:>10} {'Val R' + _r2:>10} {'Val MSE':>12}")
    print("-" * 64)

    if has_logit:
        rows = [
            (f"Joint (task+tok+logit{pos_tag})", joint_s),
            (f"Post+tok{pos_tag} (no logit)", post_tok_s),
            (f"Posterior{pos_tag}", pi_s),
            (f"Token{pos_tag}", tok_s),
            (f"Logit{pos_tag}", logit_s),
        ]
    else:
        rows = [
            (f"Joint (task+tok{pos_tag})", joint_s),
            (f"Posterior{pos_tag}", pi_s),
            (f"Token{pos_tag}", tok_s),
        ]

    for label, s in rows:
        if s:
            print(f"{label:<30} {s['tr_r2']:>10.4f} {s['va_r2']:>10.4f} {s['va_mse']:>12.6f}")
    print()
    _pr2 = "Partial R" + _r2
    if has_logit:
        print(f"{_pr2}  posterior|rest = {diag['partial_r2_posterior']:.4f}"
              f"    token|rest = {diag['partial_r2_token']:.4f}"
              f"    logit|rest = {diag['partial_r2_logit']:.4f}")
    else:
        print(f"{_pr2}  posterior|rest = {diag['partial_r2_posterior']:.4f}"
              f"    token|rest = {diag['partial_r2_token']:.4f}")
    fp = diag["f_test_posterior"]
    ft = diag["f_test_token"]
    fl = diag["f_test_logit"]
    if has_logit:
        print(f"F-test  posterior: F={fp['F']:.1f} p={fp['p']:.2e}"
              f"   token: F={ft['F']:.1f} p={ft['p']:.2e}"
              f"   logit: F={fl['F']:.1f} p={fl['p']:.2e}")
    else:
        print(f"F-test  posterior: F={fp['F']:.1f} p={fp['p']:.2e}"
              f"   token: F={ft['F']:.1f} p={ft['p']:.2e}")
    print(f"Condition number: {diag['condition_number']:.1f}")

    dd = diag["design_diagnostics"]
    _arrow = "\u2194"
    _r2_rest_hdr = "R" + _r2 + " from rest"
    _pw_hdr = "Pairwise R" + _r2 + " between groups:"
    print(f"\n  Design matrix collinearity:")
    print(f"    {'Group':<12} {'dims':>5} {'VIF':>10} {'GVIF^(1/2p)':>12} {_r2_rest_hdr:>14}")
    print(f"    {'-' * 55}")
    groups = ["posterior", "token"] + (["logit"] if has_logit else [])
    for grp in groups:
        ndim = dd["n_features"][grp]
        vif_val = dd["vif"][grp]
        gvif_val = dd["gvif_adj"][grp]
        r2_rest = dd["r2_from_rest"][grp]
        print(f"    {grp:<12} {ndim:>5d} {vif_val:>10.2f} {gvif_val:>12.4f} {r2_rest:>14.4f}")
    if has_pos:
        d_pos = dd["n_features"].get("position", 0)
        print(f"    {'position':<12} {d_pos:>5d}       (nuisance — VIF not computed)")
    print(f"\n    {_pw_hdr}")
    print(f"      post{_arrow}tok  = {dd['pairwise_r2']['post_tok']:.4f}")
    if has_logit:
        print(f"      post{_arrow}logit= {dd['pairwise_r2']['post_logit']:.4f}")
        print(f"      tok{_arrow}logit = {dd['pairwise_r2']['tok_logit']:.4f}")

    if diag.get("mlp_val_r2") is not None:
        gap = diag["mlp_val_r2"] - results["val_r2"]
        _mlp = "MLP val R" + _r2
        print(f"{_mlp}: {diag['mlp_val_r2']:.4f}  (linear gap = {gap:.4f})")
    if geometry is not None:
        sa = geometry["subspace_angles"]
        cc = geometry["component_cosine"]
        _deg = "\u00b0"
        print(f"Subspace angles (task vs token): "
              f"mean={sa['mean_angle_deg']:.1f}{_deg}  "
              f"min={sa['min_angle_deg']:.1f}{_deg}  "
              f"max_cos={sa['max_cos']:.4f}  "
              f"(rank {sa['rank_task']} vs {sa['rank_token']})")
        print(f"Component cos(task, token):      "
              f"mean={cc['mean']:.4f}  "
              f"std={cc['std']:.4f}  "
              f"|mean|={cc['abs_mean']:.4f}")
        rc = geometry["row_cosine"]
        print(f"Row cos(W_task, W_tok):          "
              f"max|cos|={rc['max_abs']:.4f}  "
              f"mean|cos|={rc['mean_abs']:.4f}  "
              f"({rc['matrix'].shape[0]}x{rc['matrix'].shape[1]})")


# ---------------------------------------------------------------------------
# 1.  train_linear_hidden_predictor
# ---------------------------------------------------------------------------

def train_linear_hidden_predictor(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    include_position_bias: bool = True,
    include_logit: bool = True,
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    skip_baselines: bool = False,
    print_summary: bool = True,
    device: Optional[str] = None,
    use_log_posterior: bool = False,
    anchor_minor_samples: Optional[int] = None,
    extraction_point: str = "post_attn",
    per_position_mean: bool = False,
    use_task_identity: bool = False,
) -> dict:
    """Joint OLS: h = [phi(pi), onehot_t, (optional) logit_t] @ W + b.

    phi is log or identity (controlled by *use_log_posterior*).
    Joint fitting ensures W_task directions are orthogonal to token/logit
    confounds (Frisch-Waugh-Lovell).

    Returns dict with fitted weights, R^2, partial R^2, F-tests, and
    design-matrix collinearity diagnostics (VIF, condition number).
    When multiple positions are fit jointly, includes one-hot position
    nuisance features (enabled by ``include_position_bias``) so the
    intercept can vary with position.

    If *use_task_identity* is True, the posterior feature is replaced with
    a one-hot task-identity encoding (ground-truth discrete label) to
    eliminate the dummy-variable collinearity.
    """
    data = _collect_multi_layer_data(
        exp_name=exp_name, layers=[layer], B=B, n_samples=n_samples,
        step=step, n_minor=n_minor, positions=positions,
        uniform_sampling=uniform_sampling, sample_mode=sample_mode,
        device=device, verbose=verbose,
        anchor_minor_samples=anchor_minor_samples,
        extraction_point=extraction_point,
        use_task_identity=use_task_identity,
    )
    results = _fit_probe(
        hiddens_all=data["hiddens_by_layer"][layer],
        posteriors_all=data["posteriors"],
        logits_all=data["logits"],
        real_tokens_all=data["real_tokens"],
        n_major=data["n_major"],
        n_tasks=data["n_tasks"],
        layer=layer,
        positions=data["positions"],
        validation_split=validation_split,
        include_position_bias=include_position_bias,
        include_logit=include_logit,
        use_log_posterior=use_log_posterior,
        skip_baselines=skip_baselines,
        sample_mode=sample_mode,
        per_position_mean=per_position_mean,
        task_ids_all=data.get("task_ids"),
    )
    if print_summary:
        _print_probe_summary(results, sample_mode=sample_mode)
    return results


# ---------------------------------------------------------------------------
# 2.  plot_val_r2_across_layers
# ---------------------------------------------------------------------------

def plot_val_r2_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    extraction_point: str = "post_attn",
    **kwargs,
):
    """Sweep OLS probe across layers; two-panel R² / partial-R² plot.

    Collects data for **all layers in a single forward pass** (1 model load,
    1 data generation), then fits the OLS probe per layer.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        Layer indices to evaluate. ``None`` → all layers (auto-detected).
    title : str, optional
    show : bool
    save_path : str, optional
    extraction_point : ``"post_attn"`` | ``"post_mlp"`` | ``"both"``
        Where to hook each transformer layer.
        ``"post_attn"`` (default) — after the attention block, before MLP.
        ``"post_mlp"`` — after the full block (attention + MLP).
        ``"both"`` — sweeps both extraction points and displays them
        interleaved on the x-axis (post_attn then post_mlp per layer).
    **kwargs
        Forwarded to ``_collect_multi_layer_data`` and ``_fit_probe``
        (e.g. ``B``, ``n_samples``, ``skip_baselines``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    all_results : dict
        When ``extraction_point`` is a single point: ``{layer_index: results_dict}``.
        When ``extraction_point="both"``: ``{(layer_index, ep): results_dict}``.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))

    # Separate kwargs into collection vs fitting parameters
    # extraction_point is handled explicitly and excluded from collect_kw
    collect_keys = {
        "B", "n_samples", "step", "n_minor", "positions",
        "uniform_sampling", "sample_mode", "device", "verbose",
        "anchor_minor_samples", "use_task_identity",
    }
    fit_keys = {
        "validation_split", "include_position_bias", "include_logit", "use_log_posterior", "skip_baselines",
        "sample_mode", "per_position_mean",
    }
    collect_kw = {k: v for k, v in kwargs.items() if k in collect_keys}
    fit_kw = {k: v for k, v in kwargs.items() if k in fit_keys}

    eps = ["post_attn", "post_mlp"] if extraction_point == "both" else [extraction_point]

    # Collect hidden states (one forward pass per extraction point)
    all_data = {}
    for ep in eps:
        logger.info(f"[sweep] collecting data for {len(layers)} layers (ep={ep!r}) ...")
        all_data[ep] = _collect_multi_layer_data(
            exp_name=exp_name, layers=layers, extraction_point=ep, **collect_kw,
        )

    # Fit a probe per (layer, extraction_point); share a seq_perm per ep pass
    all_results = {}
    for ep in eps:
        data = all_data[ep]
        N_seq = data["hiddens_by_layer"][layers[0]].shape[0]
        seq_perm = torch.randperm(N_seq)
        for layer in layers:
            logger.info(f"[sweep] fitting layer {layer} (ep={ep!r}) ...")
            result = _fit_probe(
                hiddens_all=data["hiddens_by_layer"][layer],
                posteriors_all=data["posteriors"],
                logits_all=data["logits"],
                real_tokens_all=data["real_tokens"],
                n_major=data["n_major"],
                n_tasks=data["n_tasks"],
                layer=layer,
                positions=data["positions"],
                seq_perm=seq_perm,
                task_ids_all=data.get("task_ids"),
                **fit_kw,
            )
            key = (layer, ep) if extraction_point == "both" else layer
            all_results[key] = result

    def _diag(r, key, default=float("nan")):
        d = r.get("diagnostics")
        return d[key] if d is not None else default

    # ---- print design-matrix collinearity (from first layer, first ep) ----
    first_key = (layers[0], eps[0]) if extraction_point == "both" else layers[0]
    first_res = all_results[first_key]
    dd = first_res.get("diagnostics", {}).get("design_diagnostics")
    if dd is not None:
        _r2 = "\u00b2"
        print(f"\n{'=' * 65}")
        print(f"  Design Matrix Collinearity Summary (layer-independent)")
        print(f"{'=' * 65}")
        _use_tid = kwargs.get("use_task_identity", False)
        _post_label = "task id" if _use_tid else "posterior"
        _post_abbr  = "task"   if _use_tid else "post"
        print(f"  Condition number: {dd['condition_number']:.2e}")
        print(f"  Features: {_post_label}={dd['n_features']['posterior']}  "
              f"token={dd['n_features']['token']}  "
              f"logit={dd['n_features']['logit']}  "
              f"(total={dd['n_features']['total']})")
        print()
        print(f"  {'Group':<12} {'dims':>5} {'VIF':>10} "
              f"{'GVIF^(1/2p)':>12} {'R' + _r2 + ' from rest':>14}")
        print(f"  {'-' * 55}")
        _grp_labels = {"posterior": _post_label, "token": "token", "logit": "logit"}
        for grp in ("posterior", "token", "logit"):
            ndim = dd["n_features"][grp]
            vif_val = dd["vif"][grp]
            gvif_val = dd["gvif_adj"][grp]
            r2_rest = dd["r2_from_rest"][grp]
            print(f"  {_grp_labels[grp]:<12} {ndim:>5d} {vif_val:>10.2f} "
                  f"{gvif_val:>12.4f} {r2_rest:>14.4f}")
        print()
        pw = dd["pairwise_r2"]
        _arrow = "\u2194"
        print(f"  Pairwise R{_r2} between feature groups:")
        print(f"    {_post_abbr}{_arrow}tok   = {pw['post_tok']:.4f}")
        print(f"    {_post_abbr}{_arrow}logit = {pw['post_logit']:.4f}")
        print(f"    tok{_arrow}logit  = {pw['tok_logit']:.4f}")
        print(f"{'=' * 65}\n")

    # ---- build ordered key list and x-axis labels ----
    _ep_short = {"post_attn": "attn", "post_mlp": "mlp"}
    if extraction_point == "both":
        ordered_keys = [(l, ep) for l in layers for ep in eps]
        layer_labels = [f"{l}\n{_ep_short.get(ep, ep)}" for l in layers for ep in eps]
        xlabel = "Layer / Extraction point"
    else:
        ordered_keys = [l for l in layers]
        layer_labels = [str(l) for l in layers]
        xlabel = "Layer"

    # ---- build two-panel figure ----
    x = np.arange(len(ordered_keys))

    _use_tid = kwargs.get("use_task_identity", False)
    _p_label = "Task id" if _use_tid else "Posterior"
    marginal_metrics = {
        "Joint": lambda r: r["val_r2"],
        f"{_p_label} only": lambda r: _diag(r, "r2_posterior_only"),
        "Token only": lambda r: _diag(r, "r2_token_only"),
        "Logit only": lambda r: _diag(r, "r2_logit_only"),
    }
    partial_metrics = {
        f"{_p_label} | rest": lambda r: _diag(r, "partial_r2_posterior"),
        "Token | rest": lambda r: _diag(r, "partial_r2_token"),
        "Logit | rest": lambda r: _diag(r, "partial_r2_logit"),
    }

    panels = [marginal_metrics, partial_metrics]
    panel_titles = ["Val R\u00b2 (marginal)", "Partial R\u00b2 (unique contribution)"]
    panel_ylabels = ["R\u00b2", "Partial R\u00b2"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(5 * len(ordered_keys) / 4, 12), 6),
        dpi=150,
    )

    for ax, metrics, ptitle, ylabel in zip(
        axes, panels, panel_titles, panel_ylabels,
    ):
        n_m = len(metrics)
        bw = 0.8 / n_m
        colors = plt.cm.Set2(np.linspace(0, 0.8, n_m))
        for i, (name, ext) in enumerate(metrics.items()):
            vals = [ext(all_results[k]) for k in ordered_keys]
            offset = (i - (n_m - 1) / 2) * bw
            bars = ax.bar(x + offset, vals, bw, label=name, color=colors[i])
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    txt = f"{v:.2f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), txt,
                            ha="center", va="bottom", fontsize=9)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=ptitle)
        ax.set_xticks(x, layer_labels)
        ax.tick_params(labelsize=12)
        ax.legend(
            fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=n_m,
            framealpha=0.9,
        )
        ax.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, all_results


# Legacy re-exports (moved to icl.latent_markov.legacy.softmax_predictor)
from icl.latent_markov.legacy.softmax_predictor import (  # noqa: F401, E402
    train_linear_softmax_posterior_predictor as _legacy_train_linear_softmax_posterior_predictor,
    plot_posterior_predictor_loss_vs_k,
)


# Legacy padded-sequence probes (moved to icl.latent_markov.legacy.padded_probes)
from icl.latent_markov.legacy.padded_probes import (  # noqa: F401, E402
    train_linear_softmax_posterior_predictor,
    train_linear_hidden_predictor_padded,
    train_mlp_hidden_predictor,
)


# ---------------------------------------------------------------------------
# Averaging-based R² for latent Markov
# ---------------------------------------------------------------------------

def plot_averaging_r2_latent(
    exp_name: str,
    layers: Optional[list] = None,
    estimation_positions: Optional[list] = None,
    evaluation_positions: Optional[list] = None,
    batch_size: int = 64,
    step: Optional[int] = None,
    n_minor: int = 0,
    fit_token: str = "anova",
    per_position_mean: bool = True,
    per_position_token_vecs: bool = False,
    eval_subset: str = "major",
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    simplex: bool = True,
    verbose: bool = False,
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    task_batch_size: int = 8,
) -> dict:
    """Task-subspace R² using interventional (token-conditioned) data.

    Collects hidden states with the token at each position fixed
    (interventional), breaking task-token confounding.  Task and token
    vectors are estimated from cell-mean ANOVA marginals at estimation
    positions, then R² is evaluated at all positions by projecting onto
    the task subspace (task-only) and the combined task + token subspace
    (additive).

    Parameters
    ----------
    exp_name : str
    layers : list, optional — ``None`` -> all layers
    estimation_positions : list, optional
        Positions used to estimate task and token vectors (default: last 10).
    evaluation_positions : list, optional
        Positions at which R² is computed (default: all).
    batch_size : int
        Samples per (task, token) cell.
    step : int, optional
    n_minor : int
    fit_token : "none" | "anova"
    per_position_mean : bool
    per_position_token_vecs : bool
        If True and ``fit_token="anova"``, estimate token vectors from
        the cell data at each evaluation position independently,
        instead of using a single set from the estimation positions.
        Useful when token effects are position-dependent (e.g. Markov
        chains that haven't reached stationarity at early positions).
    eval_subset : "all" | "major" | "minor"
    verbose : bool
    figsize, log_x, show, show_ylabel : plot options
    task_batch_size : int
        Tasks to batch per forward pass during data collection.

    Returns
    -------
    dict with 'task_vecs', 'token_vecs', 'results',
    'fig_task', 'fig_additive'.
    """
    if fit_token not in ("none", "anova"):
        raise ValueError(
            f"fit_token must be 'none' or 'anova' for discrete-token tasks, "
            f"got {fit_token!r}"
        )

    import matplotlib.pyplot as plt
    from icl.latent_markov.analysis.variance import get_token_conditioned_hiddens
    from icl.utils.separability import AveragingR2Result

    # ---- Determine positions ----
    _, sampler_orig, config = nu.load_everything("latent", exp_name)
    n_major = sampler_orig.n_major_tasks
    T = sampler_orig.seq_len - 1

    if layers is None:
        layers = list(range(config.model.num_layers))
    layers_idx = list(layers)

    if estimation_positions is None:
        estimation_positions = list(range(max(0, T - 10), T))
    if evaluation_positions is None:
        evaluation_positions = list(range(T))

    all_positions = sorted(set(estimation_positions) | set(evaluation_positions))

    if verbose:
        logger.info(
            f"[averaging R² latent] collecting token-conditioned data at "
            f"{len(all_positions)} positions, batch_size={batch_size}, "
            f"n_minor={n_minor}, task_batch_size={task_batch_size}"
        )

    # ---- Collect interventional data ----
    all_hiddens, token_info = get_token_conditioned_hiddens(
        exp_name=exp_name,
        layers=layers_idx,
        batch_size=batch_size,
        positions_of_interest=all_positions,
        n_minor=n_minor,
        step=step,
        verbose=verbose,
        task_batch_size=task_batch_size,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )
    # all_hiddens: (L, n_positions, n_tokens_max, n_tasks, B, D)
    L, n_pos, n_tok_max, n_tasks_total, B, D = all_hiddens.shape
    pos_list_data = token_info["positions"]
    pos_to_idx = {p: i for i, p in enumerate(pos_list_data)}
    n_unique = token_info["n_unique_tokens"]

    # ---- Eval subset ----
    _subset_ranges = {
        "all":   (0, n_tasks_total),
        "major": (0, n_major),
        "minor": (n_major, n_tasks_total),
    }
    if eval_subset not in _subset_ranges:
        raise ValueError(
            f"eval_subset must be one of {list(_subset_ranges)}, "
            f"got {eval_subset!r}"
        )
    eval_start, eval_end = _subset_ranges[eval_subset]
    if eval_end <= eval_start:
        raise ValueError(f"No tasks in subset '{eval_subset}'")

    if verbose:
        logger.info(
            f"  hiddens shape: {all_hiddens.shape}, "
            f"estimation_pos={estimation_positions}, "
            f"eval on '{eval_subset}' [{eval_start}:{eval_end}]"
        )

    # ---- Per-layer estimation and evaluation ----
    results: dict = {}
    task_vecs_by_layer: dict = {}
    token_vecs_by_layer: dict = {}

    for li, l_num in enumerate(layers_idx):
        # --- Estimate task/token vectors from cell-mean marginals ---
        est_parts = []
        for p in estimation_positions:
            pi = pos_to_idx[p]
            V_p = n_unique.get(p, n_tok_max)
            est_parts.append(
                all_hiddens[li, pi, :V_p, :n_major].float()
            )  # (V_p, K_major, B, D)

        # Cell means per estimation position
        cell_means_list = [part.mean(dim=2) for part in est_parts]
        # (each entry: (V_p, K, D))

        # Raw grand mean (before demeaning) — needed for per_position_mean=False
        raw_pooled = torch.stack(cell_means_list, dim=0).mean(dim=0)  # (V, K, D)
        grand_mean = raw_pooled.mean(dim=(0, 1))  # (D,)

        # Remove per-position mean μ_t before pooling across positions
        demeaned_list = []
        for cm in cell_means_list:
            mu_t = cm.mean(dim=(0, 1))  # mean over tokens and tasks -> (D,)
            demeaned_list.append(cm - mu_t)
        pooled = torch.stack(demeaned_list, dim=0).mean(dim=0)  # (V, K, D)

        task_vecs = pooled.mean(dim=0) - pooled.mean(dim=(0, 1))   # (K, D)
        token_vecs = pooled.mean(dim=1) - pooled.mean(dim=(0, 1))  # (V, D)

        task_vecs_by_layer[l_num] = task_vecs
        token_vecs_by_layer[l_num] = token_vecs

        if verbose:
            logger.info(
                f"  Layer {l_num}: task norms = "
                + ", ".join(f"{n:.3f}" for n in task_vecs.norm(dim=1).tolist())
                + " | token norms = "
                + ", ".join(f"{n:.3f}" for n in token_vecs.norm(dim=1).tolist())
            )

        # --- Build task projector ---
        tv = task_vecs.float()
        if not simplex:
            V_basis = tv[:-1]  # (K-1, D), drop-one for identifiability
            P_task = V_basis.T @ torch.linalg.solve(
                V_basis @ V_basis.T, V_basis
            )  # (D, D)

        fixed_mean = None if per_position_mean else grand_mean

        # --- Evaluate R² at each position ---
        from icl.utils.separability import _simplex_project_coeffs
        layer_results: dict = {}
        for pos in evaluation_positions:
            if pos not in pos_to_idx:
                continue
            pi = pos_to_idx[pos]
            V_p = n_unique.get(pos, n_tok_max)
            cell_data = all_hiddens[
                li, pi, :V_p, eval_start:eval_end
            ].float()  # (V_p, K_eval, B, D)

            h = cell_data.reshape(-1, D)
            N = h.shape[0]
            mu = fixed_mean if fixed_mean is not None else h.mean(dim=0)
            h_c = h - mu

            ss_total = (h_c ** 2).sum().item()
            eps = 1e-10

            if simplex:
                h_task_hat = _simplex_project_coeffs(tv, h_c)
                ss_task_res = ((h_c - h_task_hat) ** 2).sum().item()
                ss_task = ss_total - ss_task_res

                if fit_token == "anova":
                    if per_position_token_vecs:
                        pos_tv = cell_data.mean(dim=(1, 2))
                        pos_tv = pos_tv - pos_tv.mean(dim=0)
                        h_no_tok = cell_data - pos_tv[:, None, None, :]
                    else:
                        h_no_tok = cell_data - token_vecs[:V_p, None, None, :]
                    h_no_tok = h_no_tok.reshape(-1, D) - mu
                    h_task_hat_nt = _simplex_project_coeffs(tv, h_no_tok)
                    residual = h_no_tok - h_task_hat_nt
                    ss_residual = (residual ** 2).sum().item()
                    ss_additive = ss_total - ss_residual
                else:
                    ss_additive = ss_task
            else:
                h_task = h_c @ P_task
                ss_task = (h_task ** 2).sum().item()

                if fit_token == "anova":
                    if per_position_token_vecs:
                        pos_tv = cell_data.mean(dim=(1, 2))
                        pos_tv = pos_tv - pos_tv.mean(dim=0)
                        h_no_tok = cell_data - pos_tv[:, None, None, :]
                    else:
                        h_no_tok = cell_data - token_vecs[:V_p, None, None, :]
                    h_no_tok = h_no_tok.reshape(-1, D) - mu
                    h_no_tok_task = h_no_tok @ P_task
                    residual = h_no_tok - h_no_tok_task
                    ss_residual = (residual ** 2).sum().item()
                    ss_additive = ss_total - ss_residual
                else:
                    ss_additive = ss_task

            r = AveragingR2Result(
                r2_task=ss_task / (ss_total + eps),
                r2_additive=ss_additive / (ss_total + eps),
                ss_total=ss_total,
                ss_task=ss_task,
                ss_token=ss_additive - ss_task,
                n_tasks=eval_end - eval_start,
                n_samples=N,
            )
            r.layer_num = l_num
            r.position = pos
            layer_results[pos] = r

        results[l_num] = layer_results

    del all_hiddens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---- Plotting ----
    from icl.utils.separability import _layer_style

    sorted_layers = sorted(results.keys())

    fig_task, ax1 = plt.subplots(figsize=figsize)
    pos_list = []
    for i, l_num in enumerate(sorted_layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        vals = [pos_results[p].r2_task for p in pos_list]
        ax1.plot(pos_list, vals, label=str(l_num),
                 **_layer_style(l_num, len(pos_list)))

    ax1.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax1.set_ylabel("Task subspace $R^2$", fontsize=14)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax1.set_xscale("symlog", linthresh=1)
    ax1.set_ylim(-0.02, 1.02)
    ax1.tick_params(labelsize=12)
    n_layers = len(sorted_layers)
    _ncol = 2 if n_layers > 6 else 1
    ax1.legend(title="Layer", fontsize=12, title_fontsize=12,
               framealpha=0.9, loc="best", ncol=_ncol,
               borderaxespad=0.3, handlelength=2.2)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    fig_task.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_task)

    fig_add, ax2 = plt.subplots(figsize=figsize)
    for i, l_num in enumerate(sorted_layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        vals = [pos_results[p].r2_additive for p in pos_list]
        ax2.plot(pos_list, vals, label=str(l_num),
                 **_layer_style(l_num, len(pos_list)))

    ax2.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax2.set_ylabel(r"$R^2$: $\mu_t + \theta_z + \nu_a$", fontsize=14)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax2.set_xscale("symlog", linthresh=1)
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(labelsize=12)
    ax2.legend(title="Layer", fontsize=12, title_fontsize=12,
               framealpha=0.9, loc="best", ncol=_ncol,
               borderaxespad=0.3, handlelength=2.2)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    fig_add.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_add)

    return {
        "all_hiddens_shape": (L, n_pos, n_tok_max, n_tasks_total, B, D),
        "task_vecs": task_vecs_by_layer,
        "token_vecs": token_vecs_by_layer,
        "results": results,
        "fig_task": fig_task,
        "fig_additive": fig_add,
    }
