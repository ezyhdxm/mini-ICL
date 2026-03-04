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
) -> dict:
    """Load model once, generate data once, hook all *layers* in one pass.

    Returns a dict with per-layer hiddens and shared logits / posteriors /
    tokens so that ``_fit_probe`` can be called cheaply per layer.
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
        )

    acc_hiddens = {l: [] for l in layers}
    acc_logits: list = []
    acc_posteriors: list = []
    acc_tokens: list = []
    n_batches = (n_samples + B - 1) // B

    for batch_idx in range(n_batches):
        gen_out = sampler.generate(
            mode=sample_mode, task=None, num_samples=B, epochs=1,
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

        caches: dict = {}
        handles = []
        for l in layers:
            layer_mod = model.layers[l].attn_block

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
    use_log_posterior: bool = False,
    skip_baselines: bool = False,
    sample_mode: str = "train",
    seq_perm: Optional[torch.Tensor] = None,
) -> dict:
    """All OLS fitting for a single layer on pre-collected data.

    If *seq_perm* is given the same permutation is reused across layers so
    the train/val split is consistent.
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

    if use_log_posterior:
        X_main_tr = torch.log(post_tr + 1e-10)
        X_main_va = torch.log(post_va + 1e-10)
    else:
        X_main_tr, X_main_va = post_tr, post_va
    X_tok_tr, X_tok_va = oh_tr, oh_va
    X_logit_tr, X_logit_va = logit_tr, logit_va

    D = hiddens_all.shape[-1]
    n_total = hiddens_all.shape[0] * hiddens_all.shape[1]
    n_train = Ytr.shape[0]

    # ---- joint fit ----
    X_joint_tr = torch.cat([X_main_tr, X_tok_tr, X_logit_tr], dim=1)
    X_joint_va = torch.cat([X_main_va, X_tok_va, X_logit_va], dim=1)
    W_joint, b_joint, joint_s = _fit_ols(X_joint_tr, Ytr, X_joint_va, Yva)

    d_main = X_main_tr.shape[1]
    d_tok = X_tok_tr.shape[1]
    d_logit = X_logit_tr.shape[1]
    W_task = W_joint[:d_main, :]
    W_tok_block = W_joint[d_main:d_main + d_tok, :]
    W_logit_block = W_joint[d_main + d_tok:, :]

    # ---- marginal / pairwise fits ----
    _, _, pi_s = _fit_ols(X_main_tr, Ytr, X_main_va, Yva)
    _, _, tok_s = _fit_ols(X_tok_tr, Ytr, X_tok_va, Yva)
    _, _, logit_s = _fit_ols(X_logit_tr, Ytr, X_logit_va, Yva)

    X_pt_tr = torch.cat([X_main_tr, X_tok_tr], dim=1)
    X_pt_va = torch.cat([X_main_va, X_tok_va], dim=1)
    _, _, post_tok_s = _fit_ols(X_pt_tr, Ytr, X_pt_va, Yva)

    X_pl_tr = torch.cat([X_main_tr, X_logit_tr], dim=1)
    X_pl_va = torch.cat([X_main_va, X_logit_va], dim=1)
    _, _, post_logit_s = _fit_ols(X_pl_tr, Ytr, X_pl_va, Yva)

    X_tl_tr = torch.cat([X_tok_tr, X_logit_tr], dim=1)
    X_tl_va = torch.cat([X_tok_va, X_logit_va], dim=1)
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
    p_full = d_main + d_tok + d_logit
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

    vif_post, r2_post_rest = _group_vif(
        X_main_tr, torch.cat([X_tok_tr, X_logit_tr], dim=1),
    )
    vif_tok, r2_tok_rest = _group_vif(
        X_tok_tr, torch.cat([X_main_tr, X_logit_tr], dim=1),
    )
    vif_logit, r2_logit_rest = _group_vif(
        X_logit_tr, torch.cat([X_main_tr, X_tok_tr], dim=1),
    )

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
                        "total": d_main + d_tok + d_logit},
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
    }

    # ---- optional heavier diagnostics ----
    geometry = None

    if not skip_baselines:
        diagnostics["mlp_val_r2"] = _fit_mlp_r2(
            X_joint_tr, Ytr, X_joint_va, Yva,
        )

        eps = 1e-10
        rank_tol = 1e-5
        Wt_f = W_task.T.float()
        Wx_f = W_tok_block.T.float()

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

        c_task_va = (X_main_va @ W_task).float()
        c_tok_va = (X_tok_va @ W_tok_block).float()
        dot = (c_task_va * c_tok_va).sum(dim=1)
        norms = c_task_va.norm(dim=1) * c_tok_va.norm(dim=1) + eps
        per_sample_cos = dot / norms

        component_cosine = {
            "mean": per_sample_cos.mean().item(),
            "std": per_sample_cos.std().item(),
            "median": per_sample_cos.median().item(),
            "abs_mean": per_sample_cos.abs().mean().item(),
        }

        geometry = {
            "joint_train_mse": joint_s["tr_mse"],
            "joint_val_mse": joint_s["va_mse"],
            "joint_train_r2": joint_s["tr_r2"],
            "joint_val_r2": joint_s["va_r2"],
            "subspace_angles": subspace_angles,
            "component_cosine": component_cosine,
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

    _r2 = "\u00b2"
    print(f"\n=== Fit Summary (Latent non-padded, layer {layer}, mode={sample_mode!r}) ===")
    print(f"{'Model':<25} {'Train R' + _r2:>10} {'Val R' + _r2:>10} {'Val MSE':>12}")
    print("-" * 59)
    for label, s in [
        ("Joint (task+tok+logit)", joint_s),
        ("Post+tok (no logit)", post_tok_s),
        ("Posterior only", pi_s),
        ("Token only", tok_s),
        ("Logit only", logit_s),
    ]:
        if s:
            print(f"{label:<25} {s['tr_r2']:>10.4f} {s['va_r2']:>10.4f} {s['va_mse']:>12.6f}")
    print()
    _pr2 = "Partial R" + _r2
    print(f"{_pr2}  posterior|rest = {diag['partial_r2_posterior']:.4f}"
          f"    token|rest = {diag['partial_r2_token']:.4f}"
          f"    logit|rest = {diag['partial_r2_logit']:.4f}")
    fp = diag["f_test_posterior"]
    ft = diag["f_test_token"]
    fl = diag["f_test_logit"]
    print(f"F-test  posterior: F={fp['F']:.1f} p={fp['p']:.2e}"
          f"   token: F={ft['F']:.1f} p={ft['p']:.2e}"
          f"   logit: F={fl['F']:.1f} p={fl['p']:.2e}")
    print(f"Condition number: {diag['condition_number']:.1f}")

    dd = diag["design_diagnostics"]
    _arrow = "\u2194"
    _r2_rest_hdr = "R" + _r2 + " from rest"
    _pw_hdr = "Pairwise R" + _r2 + " between groups:"
    print(f"\n  Design matrix collinearity:")
    print(f"    {'Group':<12} {'dims':>5} {'VIF':>10} {'GVIF^(1/2p)':>12} {_r2_rest_hdr:>14}")
    print(f"    {'-' * 55}")
    for grp in ("posterior", "token", "logit"):
        ndim = dd["n_features"][grp]
        vif_val = dd["vif"][grp]
        gvif_val = dd["gvif_adj"][grp]
        r2_rest = dd["r2_from_rest"][grp]
        print(f"    {grp:<12} {ndim:>5d} {vif_val:>10.2f} {gvif_val:>12.4f} {r2_rest:>14.4f}")
    print(f"\n    {_pw_hdr}")
    print(f"      post{_arrow}tok  = {dd['pairwise_r2']['post_tok']:.4f}")
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
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    skip_baselines: bool = False,
    print_summary: bool = True,
    device: Optional[str] = None,
    use_log_posterior: bool = False,
) -> dict:
    """Joint OLS: h = [phi(pi), onehot_t, logit_t] @ [W_task; W_tok; W_logit] + b.

    phi is log or identity (controlled by *use_log_posterior*).
    Joint fitting ensures W_task directions are orthogonal to token/logit
    confounds (Frisch-Waugh-Lovell).

    Returns dict with fitted weights, R^2, partial R^2, F-tests, and
    design-matrix collinearity diagnostics (VIF, condition number).
    """
    data = _collect_multi_layer_data(
        exp_name=exp_name, layers=[layer], B=B, n_samples=n_samples,
        step=step, n_minor=n_minor, positions=positions,
        uniform_sampling=uniform_sampling, sample_mode=sample_mode,
        device=device, verbose=verbose,
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
        use_log_posterior=use_log_posterior,
        skip_baselines=skip_baselines,
        sample_mode=sample_mode,
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
    **kwargs
        Forwarded to ``_collect_multi_layer_data`` and ``_fit_probe``
        (e.g. ``B``, ``n_samples``, ``skip_baselines``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    all_results : dict
        ``{layer_index: results_dict}`` from each per-layer fit.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))

    # Separate kwargs into collection vs fitting parameters
    collect_keys = {
        "B", "n_samples", "step", "n_minor", "positions",
        "uniform_sampling", "sample_mode", "device", "verbose",
    }
    fit_keys = {
        "validation_split", "use_log_posterior", "skip_baselines",
        "sample_mode",
    }
    collect_kw = {k: v for k, v in kwargs.items() if k in collect_keys}
    fit_kw = {k: v for k, v in kwargs.items() if k in fit_keys}

    logger.info(f"[sweep] collecting data for {len(layers)} layers in one pass ...")
    data = _collect_multi_layer_data(exp_name=exp_name, layers=layers, **collect_kw)

    # Use a single sequence permutation for all layers (fair comparison)
    N_seq = data["hiddens_by_layer"][layers[0]].shape[0]
    seq_perm = torch.randperm(N_seq)

    all_results = {}
    for layer in layers:
        logger.info(f"[sweep] fitting layer {layer} ...")
        all_results[layer] = _fit_probe(
            hiddens_all=data["hiddens_by_layer"][layer],
            posteriors_all=data["posteriors"],
            logits_all=data["logits"],
            real_tokens_all=data["real_tokens"],
            n_major=data["n_major"],
            n_tasks=data["n_tasks"],
            layer=layer,
            positions=data["positions"],
            seq_perm=seq_perm,
            **fit_kw,
        )

    def _diag(r, key, default=float("nan")):
        d = r.get("diagnostics")
        return d[key] if d is not None else default

    # ---- print design-matrix collinearity (from first layer) ----
    first_res = all_results[layers[0]]
    dd = first_res.get("diagnostics", {}).get("design_diagnostics")
    if dd is not None:
        _r2 = "\u00b2"
        print(f"\n{'=' * 65}")
        print(f"  Design Matrix Collinearity Summary (layer-independent)")
        print(f"{'=' * 65}")
        print(f"  Condition number: {dd['condition_number']:.2e}")
        print(f"  Features: posterior={dd['n_features']['posterior']}  "
              f"token={dd['n_features']['token']}  "
              f"logit={dd['n_features']['logit']}  "
              f"(total={dd['n_features']['total']})")
        print()
        print(f"  {'Group':<12} {'dims':>5} {'VIF':>10} "
              f"{'GVIF^(1/2p)':>12} {'R' + _r2 + ' from rest':>14}")
        print(f"  {'-' * 55}")
        for grp in ("posterior", "token", "logit"):
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
        print(f"    post{_arrow}tok   = {pw['post_tok']:.4f}")
        print(f"    post{_arrow}logit = {pw['post_logit']:.4f}")
        print(f"    tok{_arrow}logit  = {pw['tok_logit']:.4f}")
        print(f"{'=' * 65}\n")

    # ---- build two-panel figure ----
    x = np.arange(len(layers))
    layer_labels = [str(l) for l in layers]

    marginal_metrics = {
        "Joint": lambda r: r["val_r2"],
        "Posterior only": lambda r: _diag(r, "r2_posterior_only"),
        "Token only": lambda r: _diag(r, "r2_token_only"),
        "Logit only": lambda r: _diag(r, "r2_logit_only"),
    }
    partial_metrics = {
        "Posterior | rest": lambda r: _diag(r, "partial_r2_posterior"),
        "Token | rest": lambda r: _diag(r, "partial_r2_token"),
        "Logit | rest": lambda r: _diag(r, "partial_r2_logit"),
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
        fig.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()

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
