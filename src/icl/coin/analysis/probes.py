"""Linear probes and R² sweep analysis for the Coin task."""

import gc

import numpy as np
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
) -> dict:
    """Collect hidden states for multiple layers in a single forward pass.

    Loads model once, generates data once, hooks all requested layers
    simultaneously per batch.  Returns a dict with shared posteriors,
    tokens, and per-layer hiddens ready for OLS fitting.
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
        )

    per_layer_hiddens = {l: [] for l in layers}
    all_posteriors, all_real_tokens = [], []
    n_batches = (n_samples + B - 1) // B

    for batch_idx in range(n_batches):
        gen_out = sampler.generate(mode=sample_mode, task=None, num_samples=B, epochs=1)
        samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples = samples.to(device)

        posteriors = task_posterior_coins(
            sampler_orig, samples, include_minor=(sample_mode != "major"),
        )
        posteriors_expanded = posteriors.unsqueeze(1).expand(-1, len(positions), -1)
        real_tokens_batch = samples[:, position_indices]

        cache = {}
        handles = []
        for layer in layers:
            layer_mod = model.layers[layer].attn_block

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
        "n_tasks": n_tasks,
        "positions": positions,
    }


def _fit_coin_probe(
    hiddens_all: torch.Tensor,
    posteriors_all: torch.Tensor,
    real_tokens_all: torch.Tensor,
    layer: int,
    n_tasks: int,
    positions: list,
    validation_split: float = 0.2,
    skip_baselines: bool = False,
    print_summary: bool = True,
    sample_mode: str = "train",
) -> dict:
    """OLS fitting logic for a single layer given pre-collected data."""

    D = hiddens_all.shape[-1]
    n_vocab = int(real_tokens_all.max().item()) + 1

    N_seq = hiddens_all.shape[0]
    n_seq_train = int(N_seq * (1 - validation_split))
    seq_perm = torch.randperm(N_seq)
    seq_tr, seq_va = seq_perm[:n_seq_train], seq_perm[n_seq_train:]

    def _flatten(tensor, indices):
        return tensor[indices].reshape(-1, tensor.shape[-1]).float()

    Ytr = _flatten(hiddens_all, seq_tr)
    Yva = _flatten(hiddens_all, seq_va)

    X_main_tr = _flatten(posteriors_all, seq_tr)
    X_main_va = _flatten(posteriors_all, seq_va)

    rt_tr = real_tokens_all[seq_tr].reshape(-1).long()
    rt_va = real_tokens_all[seq_va].reshape(-1).long()
    X_tok_tr = torch.zeros(rt_tr.shape[0], n_vocab, dtype=torch.float32)
    X_tok_tr.scatter_(1, rt_tr.unsqueeze(1), 1.0)
    X_tok_va = torch.zeros(rt_va.shape[0], n_vocab, dtype=torch.float32)
    X_tok_va.scatter_(1, rt_va.unsqueeze(1), 1.0)

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
    X_joint_tr = torch.cat([X_main_tr, X_tok_tr], dim=1)
    X_joint_va = torch.cat([X_main_va, X_tok_va], dim=1)
    W_joint, b_joint, joint_s = _fit_ols(X_joint_tr, Ytr, X_joint_va, Yva)

    d_main = X_main_tr.shape[1]
    d_tok = X_tok_tr.shape[1]
    W_task = W_joint[:d_main, :]
    W_tok_block = W_joint[d_main:, :]

    # ---- Marginal fits ----
    _, _, pi_s = _fit_ols(X_main_tr, Ytr, X_main_va, Yva)
    _, _, tok_s = _fit_ols(X_tok_tr, Ytr, X_tok_va, Yva)

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
    p_full = d_main + d_tok
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
                        "total": d_main + d_tok},
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
        Wt_f = W_task.T.float()
        Wx_f = W_tok_block.T.float()

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

        c_task_va = (X_main_va @ W_task).float()
        c_tok_va = (X_tok_va @ W_tok_block).float()
        dot = (c_task_va * c_tok_va).sum(dim=1)
        norms = c_task_va.norm(dim=1) * c_tok_va.norm(dim=1) + eps
        per_sample_cos = dot / norms

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
        _r2 = "\u00b2"
        print(f"\n=== Fit Summary (layer {layer}, mode={sample_mode!r}) ===")
        print(f"{'Model':<25} {'Train R' + _r2:>10} {'Val R' + _r2:>10} {'Val MSE':>12}")
        print("-" * 59)
        print(f"{'Joint (post+tok)':<25} {joint_s['tr_r2']:>10.4f} {joint_s['va_r2']:>10.4f} {joint_s['va_mse']:>12.6f}")
        print(f"{'Posterior only':<25} {pi_s['tr_r2']:>10.4f} {pi_s['va_r2']:>10.4f} {pi_s['va_mse']:>12.6f}")
        print(f"{'Token only':<25} {tok_s['tr_r2']:>10.4f} {tok_s['va_r2']:>10.4f} {tok_s['va_mse']:>12.6f}")
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

    return results


def train_linear_hidden_predictor_coin(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    skip_baselines: bool = False,
    print_summary: bool = True,
) -> dict:
    """Joint OLS: h = [posterior, one_hot(x_t)] @ [W_task; W_tok] + b.

    Joint fitting ensures W_task directions are orthogonal to token
    confounds (Frisch-Waugh-Lovell), eliminating the omitted-variable bias
    that arises when posterior and token features are fit separately.

    Logits are omitted because in the coin task the Bayes-optimal prediction
    is a linear function of the posterior alone (tokens are i.i.d. given
    the task), making logits nearly collinear with the posterior.

    Returns dict with fitted weights, R², partial R², F-tests, and
    design-matrix collinearity diagnostics (VIF, condition number).

    Always uses ``n_ood=0``.

    Parameters
    ----------
    exp_name : str
    layer : int
    B : int
    n_samples : int
    step : int, optional
    n_minor : int, optional
        Capped at ``sampler.n_minor_tasks``.
    verbose : bool
    positions : list, optional
        ``None`` → first 10 positions.
    validation_split : float
    uniform_sampling : bool
    sample_mode : str, default ``"train"``
        Sampling mode passed to ``sampler.generate(mode=...)``.

        - ``"train"`` — mixture of major + minor tasks (default).
        - ``"major"`` — sample **only** from major tasks.
        - ``"minor"`` — sample only from minor tasks.
    skip_baselines : bool
        If True, skips heavier diagnostics (MLP, subspace geometry).
    print_summary : bool, default True
        Print a formatted results table after fitting.

    Returns
    -------
    dict
        Keys aligned with ``train_linear_hidden_predictor`` (linear).
    """
    data = _collect_coin_probe_data(
        exp_name=exp_name, layers=[layer], B=B, n_samples=n_samples,
        step=step, n_minor=n_minor, positions=positions,
        uniform_sampling=uniform_sampling, sample_mode=sample_mode,
        verbose=verbose,
    )
    return _fit_coin_probe(
        hiddens_all=data["hiddens_by_layer"][layer],
        posteriors_all=data["posteriors_all"],
        real_tokens_all=data["real_tokens_all"],
        layer=layer, n_tasks=data["n_tasks"], positions=data["positions"],
        validation_split=validation_split, skip_baselines=skip_baselines,
        print_summary=print_summary, sample_mode=sample_mode,
    )


def plot_val_r2_across_layers_coin(
    exp_name: str,
    layers: Optional[list] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Sweep OLS probe h ~ [posterior, one_hot] across layers; plot R² and partial-R² bars.

    Also prints design-matrix collinearity diagnostics (condition number,
    VIF, GVIF, pairwise R² between feature groups).

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        Layer indices to evaluate.  ``None`` → all layers (auto-detected).
    title : str, optional
    show : bool
    save_path : str, optional
    **kwargs
        Forwarded to ``train_linear_hidden_predictor_coin``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    all_results : dict
        ``{layer_index: results_dict}`` from each per-layer call.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
        layers = list(range(config.model.num_layers))

    def _diag(r, key, default=float("nan")):
        d = r.get("diagnostics")
        return d[key] if d is not None else default

    collect_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ("B", "n_samples", "step", "n_minor", "positions",
                 "uniform_sampling", "sample_mode", "verbose")
    }
    fit_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ("validation_split", "skip_baselines")
    }
    sample_mode = kwargs.get("sample_mode", "train")

    logger.info(f"[sweep] collecting data for {len(layers)} layers in one pass ...")
    data = _collect_coin_probe_data(
        exp_name=exp_name, layers=layers, **collect_kwargs,
    )

    all_results = {}
    for layer in layers:
        logger.info(f"[sweep] fitting layer {layer} ...")
        all_results[layer] = _fit_coin_probe(
            hiddens_all=data["hiddens_by_layer"][layer],
            posteriors_all=data["posteriors_all"],
            real_tokens_all=data["real_tokens_all"],
            layer=layer, n_tasks=data["n_tasks"], positions=data["positions"],
            print_summary=False, sample_mode=sample_mode, **fit_kwargs,
        )

    # ── Print design matrix diagnostics (layer-independent, first layer) ──
    first_res = all_results[layers[0]]
    dd = first_res.get("diagnostics", {}).get("design_diagnostics")
    if dd is not None:
        _r2 = "\u00b2"
        print(f"\n{'=' * 60}")
        print(f"  Design Matrix Collinearity Summary (layer-independent)")
        print(f"{'=' * 60}")
        print(f"  Condition number: {dd['condition_number']:.2e}")
        print(f"  Features: posterior={dd['n_features']['posterior']}  "
              f"token={dd['n_features']['token']}  "
              f"(total={dd['n_features']['total']})")
        print()
        print(f"  {'Group':<12} {'dims':>5} {'VIF':>10} "
              f"{'GVIF^(1/2p)':>12} {'R' + _r2 + ' from rest':>14}")
        print(f"  {'-' * 55}")
        for grp in ("posterior", "token"):
            ndim = dd["n_features"][grp]
            vif_val = dd["vif"][grp]
            gvif_val = dd["gvif_adj"][grp]
            r2_rest = dd["r2_from_rest"][grp]
            print(f"  {grp:<12} {ndim:>5d} {vif_val:>10.2f} "
                  f"{gvif_val:>12.4f} {r2_rest:>14.4f}")
        print()
        pw = dd["pairwise_r2"]
        _arrow = "\u2194"
        print(f"  Pairwise R{_r2} between feature groups:")
        print(f"    post{_arrow}tok = {pw['post_tok']:.4f}")
        print(f"{'=' * 60}\n")

    # ── Two-panel figure: marginal R² and partial R² ──
    x = np.arange(len(layers))
    layer_labels = [str(l) for l in layers]

    marginal_metrics = {
        "Joint": lambda r: r["val_r2"],
        "Posterior only": lambda r: _diag(r, "r2_posterior_only"),
        "Token only": lambda r: _diag(r, "r2_token_only"),
    }
    partial_metrics = {
        "Posterior | token": lambda r: _diag(r, "partial_r2_posterior"),
        "Token | posterior": lambda r: _diag(r, "partial_r2_token"),
    }

    panels = [marginal_metrics, partial_metrics]
    panel_titles = ["Val R\u00b2 (marginal)", "Partial R\u00b2 (unique contribution)"]
    panel_ylabels = ["R\u00b2", "Partial R\u00b2"]

    fig, axes = plt.subplots(1, 2, figsize=(max(5 * len(layers) / 4, 12), 5))

    for ax, metrics, ptitle, ylabel in zip(
        axes, panels, panel_titles, panel_ylabels,
    ):
        n_m = len(metrics)
        bw = 0.8 / n_m
        colors = plt.cm.Set2(np.linspace(0, 0.8, n_m))
        for i, (name, ext) in enumerate(metrics.items()):
            vals = [ext(all_results[l]) for l in layers]
            offset = (i - (n_m - 1) / 2) * bw
            bars = ax.bar(x + offset, vals, bw, label=name, color=colors[i])
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    txt = f"{v:.2f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), txt,
                            ha="center", va="bottom", fontsize=9)
        ax.set(xlabel="Layer", ylabel=ylabel, title=ptitle)
        ax.set_xticks(x, layer_labels)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, all_results
