"""Internal data collection and OLS fitting for Coin probes."""

import gc

import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def _collect_coin_probe_data(
    exp_name: str,
    layers: list,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    positions: Optional[list] = None,
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    verbose: bool = False,
    anchor_minor_samples: Optional[int] = None,
    extraction_point: str = "post_attn",
    use_task_identity: bool = False,
) -> dict:
    """Collect hidden states for multiple layers in a single forward pass.

    Loads model once, generates data once, hooks all requested layers
    simultaneously per batch.  Returns a dict with shared posteriors,
    tokens, and per-layer hiddens ready for OLS fitting.

    Parameters
    ----------
    extraction_point : ``"post_attn"`` | ``"post_mlp"``
        Where to hook each transformer layer.
        ``"post_attn"`` (default) — after the attention block, before MLP.
        ``"post_mlp"`` — after the full block (attention + MLP).
    use_task_identity : bool
        If True, collect ground-truth task labels via ``mode="major"``
        generation (which returns the latent as ``gen_out[2]``).
    """
    from icl.coin.coin import task_posterior_coins

    _, sampler_orig, config = nu.load_everything("coin", exp_name)
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

    sampler, k_minor = get_new_sampler(exp_name, n_minor, n_ood=0)
    n_tasks = sampler_orig.n_major_tasks + sampler_orig.n_minor_tasks

    original_p_minor = sampler.p_minor
    if uniform_sampling and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (
            sampler.n_major_tasks + sampler.n_minor_tasks
        )

    seq_len = sampler.seq_len
    if positions is None:
        positions = list(range(min(10, seq_len)))
    else:
        positions = list(positions)

    device = config.device
    position_indices = torch.tensor(positions, device=device, dtype=torch.long)

    if verbose:
        logger.info(
            f"[coin] collecting probe data: layers={layers}, "
            f"n_tasks={n_tasks}, B={B}, n_samples={n_samples}, "
            f"sample_mode={sample_mode!r}"
            + (" [task_identity]" if use_task_identity else "")
        )

    # When use_task_identity=True force "major" mode so the sampler returns
    # the latent index as gen_out[2].
    _gen_mode = "major" if use_task_identity else sample_mode

    per_layer_hiddens = {l: [] for l in layers}
    all_posteriors, all_real_tokens, all_task_ids = [], [], []
    n_batches = (n_samples + B - 1) // B

    for batch_idx in range(n_batches):
        gen_out = sampler.generate(mode=_gen_mode, task=None, num_samples=B, epochs=1)
        samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples = samples.to(device)

        # For "major" mode with task=None the sampler returns (samples, probs, latent).
        if use_task_identity and isinstance(gen_out, (tuple, list)) and len(gen_out) >= 3:
            all_task_ids.append(gen_out[2].cpu())

        posteriors = task_posterior_coins(
            sampler_orig, samples, include_minor=True,
        )
        posteriors_expanded = posteriors.unsqueeze(1).expand(-1, len(positions), -1)
        real_tokens_batch = samples[:, position_indices]

        cache = {}
        handles = []
        for layer in layers:
            layer_mod = (
                model.layers[layer] if extraction_point == "post_mlp"
                else model.layers[layer].attn_block
            )

            def _make_hook(layer_id):
                def hook_fn(module, inp, out):
                    h = out if torch.is_tensor(out) else out[0]
                    cache[layer_id] = h.index_select(dim=1, index=position_indices).detach()
                return hook_fn

            handles.append(layer_mod.register_forward_hook(_make_hook(layer)))

        try:
            with torch.no_grad():
                model(samples)
        finally:
            for h in handles:
                h.remove()

        for layer in layers:
            per_layer_hiddens[layer].append(cache[layer].cpu())

        all_posteriors.append(posteriors_expanded.cpu())
        all_real_tokens.append(real_tokens_batch.cpu())

        del samples, posteriors, cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Anchor with minor-task sequences when fitting on major-task data only.
    _do_anchor = (
        sample_mode == "major"
        and sampler.n_minor_tasks > 0
        and original_p_minor > 1e-6
        and (anchor_minor_samples is None or anchor_minor_samples > 0)
    )
    if _do_anchor:
        n_anchor = anchor_minor_samples if anchor_minor_samples is not None else max(B, n_samples // 5)
        n_anchor_batches = (n_anchor + B - 1) // B
        if verbose:
            logger.info(f"[coin] anchoring with {n_anchor} train-mode samples ({n_anchor_batches} batches)")
        for _ in range(n_anchor_batches):
            gen_out = sampler.generate(mode="train", task=None, num_samples=B, epochs=1)
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)

            posteriors = task_posterior_coins(
                sampler_orig, samples, include_minor=True,
            )
            posteriors_expanded = posteriors.unsqueeze(1).expand(-1, len(positions), -1)
            real_tokens_batch = samples[:, position_indices]

            cache = {}
            handles = []
            for layer in layers:
                layer_mod_a = (
                    model.layers[layer] if extraction_point == "post_mlp"
                    else model.layers[layer].attn_block
                )

                def _make_hook(layer_id):
                    def hook_fn(module, inp, out):
                        h = out if torch.is_tensor(out) else out[0]
                        cache[layer_id] = h.index_select(dim=1, index=position_indices).detach()
                    return hook_fn

                handles.append(layer_mod_a.register_forward_hook(_make_hook(layer)))

            try:
                with torch.no_grad():
                    model(samples)
            finally:
                for h in handles:
                    h.remove()

            for layer in layers:
                per_layer_hiddens[layer].append(cache[layer].cpu())
            all_posteriors.append(posteriors_expanded.cpu())
            all_real_tokens.append(real_tokens_batch.cpu())

            del samples, posteriors, cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sampler.p_minor = original_p_minor
    model.cpu(); del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    hiddens_by_layer = {
        l: torch.cat(per_layer_hiddens[l], dim=0) for l in layers
    }
    return {
        "hiddens_by_layer": hiddens_by_layer,
        "posteriors_all": torch.cat(all_posteriors, dim=0),
        "real_tokens_all": torch.cat(all_real_tokens, dim=0),
        "n_major": sampler_orig.n_major_tasks,
        "n_tasks": n_tasks,
        "positions": positions,
        # Ground-truth task labels (shape N_seq,) when use_task_identity=True.
        "task_ids": torch.cat(all_task_ids, dim=0) if all_task_ids else None,
    }


def _fit_coin_probe(
    hiddens_all: torch.Tensor,
    posteriors_all: torch.Tensor,
    real_tokens_all: torch.Tensor,
    layer: int,
    n_tasks: int,
    positions: list,
    validation_split: float = 0.2,
    include_position_bias: bool = True,
    skip_baselines: bool = False,
    print_summary: bool = True,
    sample_mode: str = "train",
    n_major: Optional[int] = None,
    per_position_mean: bool = False,
    task_ids_all: Optional[torch.Tensor] = None,
) -> dict:
    """OLS fitting logic for a single layer given pre-collected data.

    If *per_position_mean* is True, the mean hidden state at each position
    (computed from training sequences only) is subtracted before fitting,
    removing position-specific DC offsets so the probe captures only
    task-driven variation.

    If *task_ids_all* is provided (shape ``(N,)``), the posterior feature
    block is replaced with a one-hot task-identity encoding (last column
    dropped to avoid the dummy-variable trap).
    """

    # Crop to major-task posterior columns (minor columns are in the full
    # posterior but not used as regressors).
    _n_major = n_major if n_major is not None else posteriors_all.shape[-1]
    if n_major is not None and posteriors_all.shape[-1] > n_major:
        posteriors_all = posteriors_all[:, :, :n_major]

    D = hiddens_all.shape[-1]
    n_vocab = int(real_tokens_all.max().item()) + 1

    N_seq = hiddens_all.shape[0]
    n_seq_train = int(N_seq * (1 - validation_split))
    seq_perm = torch.randperm(N_seq)
    seq_tr, seq_va = seq_perm[:n_seq_train], seq_perm[n_seq_train:]

    def _flatten(tensor, indices):
        return tensor[indices].reshape(-1, tensor.shape[-1]).float()

    if per_position_mean:
        pos_mean = hiddens_all[seq_tr].float().mean(dim=0)  # (P, D)
        Ytr = (hiddens_all[seq_tr].float() - pos_mean).reshape(-1, D)
        Yva = (hiddens_all[seq_va].float() - pos_mean).reshape(-1, D)
    else:
        Ytr = _flatten(hiddens_all, seq_tr)
        Yva = _flatten(hiddens_all, seq_va)

    if task_ids_all is not None:
        # One-hot task-identity encoding at the sequence level, expanded to
        # positions.  Last column dropped to avoid the dummy-variable trap.
        n_pos = hiddens_all.shape[1]
        _n_posterior_orig = _n_major
        _post_redundant = False

        def _task_onehot(seq_indices):
            ids = task_ids_all[seq_indices].clamp(0, _n_major - 1)
            oh = torch.zeros(len(seq_indices), _n_major, dtype=torch.float32)
            oh.scatter_(1, ids.unsqueeze(1), 1.0)
            return oh.unsqueeze(1).expand(-1, n_pos, -1).reshape(-1, _n_major)

        X_main_tr = _task_onehot(seq_tr)[:, :-1]
        X_main_va = _task_onehot(seq_va)[:, :-1]
    else:
        X_main_tr = _flatten(posteriors_all, seq_tr)
        X_main_va = _flatten(posteriors_all, seq_va)

        _post_sum_tr = X_main_tr.sum(dim=1)
        _post_redundant = (
            X_main_tr.shape[1] > 1
            and (_post_sum_tr - 1.0).abs().max().item() < 1e-4
        )
        _n_posterior_orig = X_main_tr.shape[1]
        if _post_redundant:
            X_main_tr = X_main_tr[:, :-1]
            X_main_va = X_main_va[:, :-1]

    rt_tr = real_tokens_all[seq_tr].reshape(-1).long()
    rt_va = real_tokens_all[seq_va].reshape(-1).long()
    oh_tr = torch.zeros(rt_tr.shape[0], n_vocab, dtype=torch.float32)
    oh_tr.scatter_(1, rt_tr.unsqueeze(1), 1.0)
    oh_va = torch.zeros(rt_va.shape[0], n_vocab, dtype=torch.float32)
    oh_va.scatter_(1, rt_va.unsqueeze(1), 1.0)
    _n_tok_orig = n_vocab
    X_tok_tr = oh_tr[:, :-1]
    X_tok_va = oh_va[:, :-1]

    # Position nuisance features: one-hot(position index), drop last column.
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

    n_total = hiddens_all.shape[0] * hiddens_all.shape[1]
    n_train = Ytr.shape[0]

    def _fit_ols(Xtr, Ytr_, Xva, Yva_):
        ones_tr = torch.ones(Xtr.shape[0], 1, dtype=Xtr.dtype, device=Xtr.device)
        Xtr_aug = torch.cat([Xtr, ones_tr], dim=1)
        W_aug = torch.linalg.pinv(Xtr_aug) @ Ytr_
        W = W_aug[:-1, :]
        b = W_aug[-1, :]

        pred_tr = Xtr @ W + b
        pred_va = Xva @ W + b
        tr_res = Ytr_ - pred_tr
        va_res = Yva_ - pred_va

        tr_ss_res = (tr_res ** 2).sum().item()
        va_ss_res = (va_res ** 2).sum().item()
        tr_ss_tot = ((Ytr_ - Ytr_.mean(dim=0)) ** 2).sum().item()
        va_ss_tot = ((Yva_ - Yva_.mean(dim=0)) ** 2).sum().item()
        n_dim = Ytr_.shape[1]
        return W, b, {
            "tr_mse": tr_ss_res / (Ytr_.shape[0] * n_dim),
            "va_mse": va_ss_res / (Yva_.shape[0] * n_dim),
            "tr_r2": 1.0 - tr_ss_res / tr_ss_tot if tr_ss_tot > 0 else float("nan"),
            "va_r2": 1.0 - va_ss_res / va_ss_tot if va_ss_tot > 0 else float("nan"),
            "tr_ss_res": tr_ss_res,
            "va_ss_res": va_ss_res,
            "n_features": Xtr.shape[1],
        }

    # ---- Primary fit: joint [posterior, one_hot] -> hidden ----
    X_joint_tr_parts = [X_main_tr, X_tok_tr]
    X_joint_va_parts = [X_main_va, X_tok_va]
    if X_pos_tr is not None:
        X_joint_tr_parts.append(X_pos_tr)
        X_joint_va_parts.append(X_pos_va)
    X_joint_tr = torch.cat(X_joint_tr_parts, dim=1)
    X_joint_va = torch.cat(X_joint_va_parts, dim=1)
    W_joint, b_joint, joint_s = _fit_ols(X_joint_tr, Ytr, X_joint_va, Yva)

    d_main = X_main_tr.shape[1]
    d_tok = X_tok_tr.shape[1]
    d_pos = X_pos_tr.shape[1] if X_pos_tr is not None else 0
    W_task_raw = W_joint[:d_main, :]
    W_tok_raw = W_joint[d_main:d_main + d_tok, :]
    W_pos_raw = W_joint[d_main + d_tok:, :] if d_pos > 0 else None

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

    # ---- Marginal fits ----
    if X_pos_tr is not None:
        X_main_marg_tr = torch.cat([X_main_tr, X_pos_tr], dim=1)
        X_main_marg_va = torch.cat([X_main_va, X_pos_va], dim=1)
        X_tok_marg_tr = torch.cat([X_tok_tr, X_pos_tr], dim=1)
        X_tok_marg_va = torch.cat([X_tok_va, X_pos_va], dim=1)
    else:
        X_main_marg_tr, X_main_marg_va = X_main_tr, X_main_va
        X_tok_marg_tr, X_tok_marg_va = X_tok_tr, X_tok_va

    _, _, pi_s = _fit_ols(X_main_marg_tr, Ytr, X_main_marg_va, Yva)
    _, _, tok_s = _fit_ols(X_tok_marg_tr, Ytr, X_tok_marg_va, Yva)

    _eps = 1e-10
    partial_r2_post = (
        (joint_s["va_r2"] - tok_s["va_r2"])
        / max(1.0 - tok_s["va_r2"], _eps)
    )
    partial_r2_tok = (
        (joint_s["va_r2"] - pi_s["va_r2"])
        / max(1.0 - pi_s["va_r2"], _eps)
    )

    n_tr = Ytr.shape[0]
    p_full = d_main + d_tok + d_pos
    df_den = n_tr - p_full - 1

    def _f_test(ss_reduced, ss_full, q, df_d):
        if df_d <= 0 or ss_full <= 0:
            return {"F": float("nan"), "p": float("nan"),
                    "df_num": q, "df_den": df_d}
        f_val = ((ss_reduced - ss_full) / q) / (ss_full / df_d)
        try:
            from scipy.stats import f as f_dist
            p_val = float(f_dist.sf(max(f_val, 0.0), q, df_d))
        except ImportError:
            p_val = float("nan")
        return {"F": f_val, "p": p_val, "df_num": q, "df_den": df_d}

    f_test_post = _f_test(tok_s["tr_ss_res"], joint_s["tr_ss_res"], d_main, df_den)
    f_test_tok = _f_test(pi_s["tr_ss_res"], joint_s["tr_ss_res"], d_tok, df_den)

    cond_num = float(torch.linalg.cond(
        torch.cat([X_joint_tr, torch.ones(n_tr, 1, dtype=X_joint_tr.dtype)], dim=1)
    ).item())

    def _group_vif(X_group, X_rest):
        ones = torch.ones(X_rest.shape[0], 1, dtype=X_rest.dtype)
        X_aug = torch.cat([X_rest, ones], dim=1)
        W_v = torch.linalg.pinv(X_aug) @ X_group
        pred = X_aug @ W_v
        ss_res = ((X_group - pred) ** 2).sum().item()
        ss_tot = ((X_group - X_group.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return 1.0 / max(1.0 - r2, 1e-10), r2

    if X_pos_tr is not None:
        vif_post, r2_post_from_rest = _group_vif(X_main_tr, torch.cat([X_tok_tr, X_pos_tr], dim=1))
        vif_tok, r2_tok_from_rest = _group_vif(X_tok_tr, torch.cat([X_main_tr, X_pos_tr], dim=1))
    else:
        vif_post, r2_post_from_rest = _group_vif(X_main_tr, X_tok_tr)
        vif_tok, r2_tok_from_rest = _group_vif(X_tok_tr, X_main_tr)

    gvif_post = vif_post ** (1.0 / (2 * d_main)) if d_main > 0 else float("nan")
    gvif_tok = vif_tok ** (1.0 / (2 * d_tok)) if d_tok > 0 else float("nan")

    def _pairwise_r2(Xa, Xb):
        ones = torch.ones(Xb.shape[0], 1, dtype=Xb.dtype)
        X_aug = torch.cat([Xb, ones], dim=1)
        W_v = torch.linalg.pinv(X_aug) @ Xa
        pred = X_aug @ W_v
        ss_res = ((Xa - pred) ** 2).sum().item()
        ss_tot = ((Xa - Xa.mean(0)) ** 2).sum().item()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pairwise_r2_post_tok = _pairwise_r2(X_main_tr, X_tok_tr)

    design_diagnostics = {
        "condition_number": cond_num,
        "n_features": {"posterior": d_main, "token": d_tok,
                        "position": d_pos,
                        "total": d_main + d_tok + d_pos},
        "vif": {"posterior": vif_post, "token": vif_tok},
        "gvif_adj": {"posterior": gvif_post, "token": gvif_tok},
        "r2_from_rest": {"posterior": r2_post_from_rest, "token": r2_tok_from_rest},
        "pairwise_r2": {"post_tok": pairwise_r2_post_tok},
    }

    diagnostics = {
        "r2_posterior_only": pi_s["va_r2"],
        "r2_token_only": tok_s["va_r2"],
        "r2_joint": joint_s["va_r2"],
        "partial_r2_posterior": partial_r2_post,
        "partial_r2_token": partial_r2_tok,
        "f_test_posterior": f_test_post,
        "f_test_token": f_test_tok,
        "condition_number": cond_num,
        "design_diagnostics": design_diagnostics,
        "mlp_val_r2": None,
        "position_bias_included": bool(use_pos_bias),
        "posterior_column_dropped": bool(_post_redundant),
    }

    geometry = None
    if not skip_baselines:
        def _fit_mlp_r2(Xtr, Ytr_, Xva, Yva_,
                        hidden_dim=128, epochs=200, lr=1e-3, batch_size=4096):
            from torch import nn
            d_in, d_out = Xtr.shape[1], Ytr_.shape[1]
            mlp = nn.Sequential(
                nn.Linear(d_in, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_out),
            )
            opt = torch.optim.Adam(mlp.parameters(), lr=lr)
            for _ in range(epochs):
                perm = torch.randperm(Xtr.shape[0])
                for i in range(0, Xtr.shape[0], batch_size):
                    batch = perm[i:i + batch_size]
                    loss = ((mlp(Xtr[batch]) - Ytr_[batch]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            mlp.eval()
            with torch.no_grad():
                pred = mlp(Xva)
                ss_res = ((Yva_ - pred) ** 2).sum().item()
                ss_tot = ((Yva_ - Yva_.mean(dim=0)) ** 2).sum().item()
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        diagnostics["mlp_val_r2"] = _fit_mlp_r2(X_joint_tr, Ytr, X_joint_va, Yva)

        eps = 1e-10
        rank_tol = 1e-5
        Wt_f = W_task_raw.T.float()
        Wx_f = W_tok_raw.T.float()

        def _rank_basis(M):
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
            "component_cosine": {
                "mean": per_sample_cos.mean().item(),
                "std": per_sample_cos.std().item(),
                "median": per_sample_cos.median().item(),
                "abs_mean": per_sample_cos.abs().mean().item(),
            },
            "row_cosine": row_cosine,
            "component_weight": {
                "main": W_task.cpu(), "token": W_tok_block.cpu(),
                "bias": b_joint.cpu(),
            },
        }

    results = {
        "train_mse": joint_s["tr_mse"],
        "val_mse": joint_s["va_mse"],
        "train_r2": joint_s["tr_r2"],
        "val_r2": joint_s["va_r2"],
        "model_weight": W_task.cpu(),
        "model_bias": b_joint.cpu(),
        "token_weight": W_tok_block.cpu(),
        "position_weight": W_pos_block.cpu() if W_pos_block is not None else None,
        "diagnostics": diagnostics,
        "geometry": geometry,
        "layer": layer,
        "n_tasks": n_tasks,
        "hidden_dim": D,
        "vocab_size": n_vocab,
        "n_samples": n_total,
        "n_train": n_train,
        "n_val": Yva.shape[0],
        "positions": positions,
    }

    if print_summary:
        diag = diagnostics
        has_pos = diag.get("position_bias_included", False)
        pos_tag = "+pos" if has_pos else ""
        _r2 = "\u00b2"
        dropped = diag.get("posterior_column_dropped", False)
        print(f"\n=== Fit Summary (Coin, layer {layer}, mode={sample_mode!r}) ===")
        if dropped:
            print("  [posterior col dropped: sum(pi)=1 detected, last column removed to avoid dummy-variable trap]")
        print("  [token & position one-hot: last column dropped from each to avoid dummy-variable traps]")
        print(f"{'Model':<30} {'Train R' + _r2:>10} {'Val R' + _r2:>10} {'Val MSE':>12}")
        print("-" * 64)
        print(f"{f'Joint (post+tok{pos_tag})':<30} {joint_s['tr_r2']:>10.4f} {joint_s['va_r2']:>10.4f} {joint_s['va_mse']:>12.6f}")
        print(f"{f'Posterior{pos_tag}':<30} {pi_s['tr_r2']:>10.4f} {pi_s['va_r2']:>10.4f} {pi_s['va_mse']:>12.6f}")
        print(f"{f'Token{pos_tag}':<30} {tok_s['tr_r2']:>10.4f} {tok_s['va_r2']:>10.4f} {tok_s['va_mse']:>12.6f}")
        print()
        _pr2 = "Partial R" + _r2
        print(f"{_pr2}  posterior|token = {diag['partial_r2_posterior']:.4f}"
              f"    token|posterior = {diag['partial_r2_token']:.4f}")
        fp, ft = diag["f_test_posterior"], diag["f_test_token"]
        print(f"F-test  posterior: F={fp['F']:.1f} p={fp['p']:.2e}"
              f"   token: F={ft['F']:.1f} p={ft['p']:.2e}")
        print(f"Condition number: {diag['condition_number']:.1f}")

        dd = diag["design_diagnostics"]
        _r2_rest_hdr = "R" + _r2 + " from rest"
        print(f"\n  Design matrix collinearity:")
        print(f"    {'Group':<12} {'dims':>5} {'VIF':>10} {'GVIF^(1/2p)':>12} {_r2_rest_hdr:>14}")
        print(f"    {'-' * 55}")
        for grp in ("posterior", "token"):
            ndim = dd["n_features"][grp]
            vif_val = dd["vif"][grp]
            gvif_val = dd["gvif_adj"][grp]
            r2_rest = dd["r2_from_rest"][grp]
            print(f"    {grp:<12} {ndim:>5d} {vif_val:>10.2f} {gvif_val:>12.4f} {r2_rest:>14.4f}")
        if has_pos:
            d_pos_val = dd["n_features"].get("position", 0)
            print(f"    {'position':<12} {d_pos_val:>5d}       (nuisance — VIF not computed)")
        print(f"\n    Pairwise R{_r2}: post\u2194tok = {dd['pairwise_r2']['post_tok']:.4f}")

        if diag["mlp_val_r2"] is not None:
            gap = diag["mlp_val_r2"] - joint_s["va_r2"]
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

    return results
