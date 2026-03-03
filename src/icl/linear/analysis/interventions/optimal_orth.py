"""Causal intervention: find optimal orthogonal directions to remove."""
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger
from icl.linear.analysis._helpers import _show_or_close
from icl.linear.analysis.interventions._helpers import (
    _cleanup_model,
    _create_ood_task,
    _joint_fit_task_token,
    _build_protected_subspace,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Pure utilities (no model / task dependencies)
# ---------------------------------------------------------------------------

def _ols_r2(X, Y):
    """Held-out R² (80/20 split) for OLS with intercept: Y = [X, 1] W."""
    N = X.shape[0]
    n_tr = int(0.8 * N)
    perm = torch.randperm(N)
    Xtr, Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
    Xte, Yte = X[perm[n_tr:]], Y[perm[n_tr:]]
    Xa = torch.cat([Xtr, torch.ones(n_tr, 1)], dim=1)
    W = torch.linalg.pinv(Xa) @ Ytr
    pred = torch.cat([Xte, torch.ones(N - n_tr, 1)], dim=1) @ W
    ss_r = ((Yte - pred) ** 2).sum().item()
    ss_t = ((Yte - Yte.mean(0)) ** 2).sum().item()
    return 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")


def _fit_r2_mlp(X, Y, hidden_dim=64, n_epochs=200, lr=1e-3):
    """Train/test R² (80/20 split) for a two-layer MLP probe X → Y."""
    N = X.shape[0]
    n_train = int(0.8 * N)
    perm = torch.randperm(N)
    X_tr, Y_tr = X[perm[:n_train]], Y[perm[:n_train]]
    X_te, Y_te = X[perm[n_train:]], Y[perm[n_train:]]
    in_dim, out_dim = X_tr.shape[1], Y_tr.shape[1]
    mlp = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, out_dim),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    for _ in range(n_epochs):
        pred = mlp(X_tr)
        loss = ((pred - Y_tr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_te = mlp(X_te)
    ss_res = ((Y_te - pred_te) ** 2).sum().item()
    ss_tot = ((Y_te - Y_te.mean(0)) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _compute_running_beta_hat(demo_data, demo_target):
    """Running ridge regression estimate at each position t.

    ŵₜ = (Σ_{s≤t} xₛxₛᵀ + λI)⁻¹ (Σ_{s≤t} xₛyₛ)

    Returns (B, T, D).
    """
    B, T, D = demo_data.shape
    XtY = torch.zeros(B, D, device=demo_data.device, dtype=demo_data.dtype)
    XtX = torch.zeros(B, D, D, device=demo_data.device, dtype=demo_data.dtype)
    reg = 1e-4 * torch.eye(D, device=demo_data.device, dtype=demo_data.dtype)
    out = torch.zeros(B, T, D, device=demo_data.device, dtype=demo_data.dtype)
    for t in range(T):
        x_t, y_t = demo_data[:, t, :], demo_target[:, t]
        XtY = XtY + torch.einsum("bd,b->bd", x_t, y_t)      # XᵀY += xₜyₜ
        XtX = XtX + torch.einsum("bi,bj->bij", x_t, x_t)    # XᵀX += xₜxₜᵀ
        out[:, t, :] = torch.linalg.solve(XtX + reg, XtY)    # ŵₜ = (XᵀX + λI)⁻¹ XᵀY
    return out


def _compute_running_xty(demo_data, demo_target):
    """Running average of XᵀY:  out[t] = (Σ_{s≤t} xₛyₛ) / (t+1).

    Returns (B, T, D).
    """
    B, T, D = demo_data.shape
    XtY = torch.zeros(B, D, device=demo_data.device, dtype=demo_data.dtype)
    out = torch.zeros(B, T, D, device=demo_data.device, dtype=demo_data.dtype)
    for t in range(T):
        XtY = XtY + torch.einsum("bd,b->bd", demo_data[:, t, :], demo_target[:, t])
        out[:, t, :] = XtY / float(t + 1)
    return out


# ---------------------------------------------------------------------------
# V_opt optimisation via gradient ascent
# ---------------------------------------------------------------------------

def _optimize_v_directions(
    model, layer, U_orth, orth_dim,
    *, device, opt_task, eval_positions,
    n_directions, n_opt_steps, opt_lr, opt_B, patience, grad_clip_norm,
    scale,
    opt_source,
    max_mse_oracle, min_opt_steps,
    verbose,
):
    """Gradient ascent in the orthogonal complement to maximise intervened MSE.

    Parameterises search directions as  V = U_⊥ W  where  W ∈ ℝ^{m×r}.
    At each step QR-orthonormalises V, forms P_V = V Vᵀ, and intervenes:
        h′ = h − s · P_V h
    which removes the V-component from the hidden state.

    Objective (maximised):  L_ood(V)  = avg MSE on OOD sequences after
    removing the V-subspace.

    Uses Adam on −L_ood with EMA parameter averaging (bias-corrected)
    for the final V_opt.

    Returns (V_opt, loss_history).
    """
    opt_minor = (opt_source == "minor")

    saved_opt_bs = int(opt_task.batch_size)
    opt_task.batch_size = opt_B

    # W_param in R^{orth_dim x n_directions} parameterises the directions
    # V = U_orth @ W_param  that live in the orthogonal complement.
    W_param = torch.randn(orth_dim, n_directions, device=device)
    W_param = (W_param / W_param.norm(dim=0, keepdim=True)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([W_param], lr=opt_lr)

    loss_history = []
    steps_no_improve = 0
    smoothed_loss = None
    smooth_alpha = 0.1
    ema_decay = 0.99
    W_ema = torch.zeros_like(W_param)
    best_smoothed = -float("inf")

    for step_i in range(n_opt_steps):
        # ---- Sample optimisation data ----
        if opt_minor:
            demo_data, demo_tasks, demo_target = opt_task.sample_batch(
                step=step_i + 9999, minor_only=True,
            )
        else:
            demo_data, demo_tasks, demo_target = opt_task.sample_batch(
                step=step_i + 9999, is_eval=True,
            )
        demo_data, demo_target = demo_data.to(device), demo_target.to(device)
        demo_tasks = demo_tasks.to(device)

        # ---- Intervened forward pass ----
        # Hook removes the V_opt component from hidden states:
        #   V = U_orth @ W_param            (map params to full space)
        #   V_q = QR(V)                     (orthonormalise)
        #   h' = h - scale * V_q V_q^T h    (subtract projection)
        def _hook(mod, inp, out):
            h = out if torch.is_tensor(out) else out[0]
            V = U_orth @ W_param
            V_q, _ = torch.linalg.qr(V)
            h_mod = h - scale * (h @ V_q @ V_q.T)
            return h_mod if torch.is_tensor(out) else (h_mod,) + out[1:]

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(_hook)
        try:
            preds = model(demo_data, demo_target)
        finally:
            handle.remove()

        loss_acc = torch.tensor(0.0, device=device)
        cnt = 0
        for p in eval_positions:
            if p >= preds.shape[1]:
                continue
            loss_acc = loss_acc + ((preds[:, p] - demo_target[:, p]) ** 2).mean()
            cnt += 1

        if cnt == 0:
            del demo_data, demo_tasks, demo_target, preds
            continue

        avg_loss = loss_acc / cnt
        loss_history.append(avg_loss.item())

        del demo_data, demo_tasks, demo_target, preds

        if max_mse_oracle is not None and step_i + 1 >= min_opt_steps:
            if opt_minor:
                val_data, val_tasks, val_target = opt_task.sample_batch(
                    step=step_i + 77777, minor_only=True,
                )
            else:
                val_data, val_tasks, val_target = opt_task.sample_batch(
                    step=step_i + 77777, is_eval=True,
                )
            val_data, val_target = val_data.to(device), val_target.to(device)
            val_tasks = val_tasks.to(device)

            with torch.no_grad():
                V_val = U_orth @ W_param.detach()
                V_q_val, _ = torch.linalg.qr(V_val)

                def _val_hook(mod, inp, out, _Vq=V_q_val):
                    h = out if torch.is_tensor(out) else out[0]
                    h_mod = h - scale * (h @ _Vq @ _Vq.T)
                    return h_mod if torch.is_tensor(out) else (h_mod,) + out[1:]

                handle_val = model.transformer.blocks[layer].attn_block.register_forward_hook(_val_hook)
                try:
                    val_preds = model(val_data, val_target)
                finally:
                    handle_val.remove()

                val_oracle = (val_data @ val_tasks).squeeze(-1)
                oracle_acc = sum(
                    ((val_preds[:, p] - val_oracle[:, p]) ** 2).mean()
                    for p in eval_positions if p < val_preds.shape[1]
                )
                cur_oracle = (oracle_acc / cnt).item()

            del val_data, val_tasks, val_target, val_preds, val_oracle

            if cur_oracle >= max_mse_oracle:
                if verbose:
                    logger.info(
                        f"[opt-dir] Stopping at step {step_i + 1}: "
                        f"val MSE-to-oracle {cur_oracle:.4f} >= {max_mse_oracle}"
                    )
                break

        cur_obj = avg_loss.item()
        if smoothed_loss is None:
            smoothed_loss = cur_obj
        else:
            smoothed_loss = smooth_alpha * cur_obj + (1 - smooth_alpha) * smoothed_loss

        if smoothed_loss > best_smoothed:
            best_smoothed = smoothed_loss
            steps_no_improve = 0
        else:
            steps_no_improve += 1

        optimizer.zero_grad()
        (-avg_loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([W_param], max_norm=grad_clip_norm)
        optimizer.step()

        W_ema = ema_decay * W_ema + (1 - ema_decay) * W_param.detach()

        if verbose and (step_i + 1) % 50 == 0:
            src = "Minor" if opt_minor else "OOD"
            logger.info(
                f"[opt-dir] step {step_i + 1}/{n_opt_steps}, "
                f"{src} MSE: {loss_history[-1]:.4f}, "
                f"smoothed: {smoothed_loss:.4f}"
            )

        if patience > 0 and steps_no_improve >= patience:
            if verbose:
                logger.info(
                    f"[opt-dir] Early stopping at step {step_i + 1} "
                    f"(no improvement for {patience} steps)"
                )
            break

    opt_task.batch_size = saved_opt_bs

    # EMA with bias correction: W_corrected = W_ema / (1 - decay^T).
    # This removes the bias toward zero from initialising W_ema = 0.
    n_steps_done = len(loss_history)
    bias_correction = 1.0 - ema_decay ** max(n_steps_done, 1)
    W_avg = W_ema / bias_correction
    with torch.no_grad():
        V_final = U_orth @ W_avg
        V_opt, _ = torch.linalg.qr(V_final)

    return V_opt, loss_history


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

def _collect_features(
    model, layer, task_obj, train_task,
    *, device, B, eval_positions, n_dims, n_samples,
):
    """Collect hidden states and diagnostic feature vectors at *eval_positions*.

    For each sequence position t in eval_positions, we extract:

    - h      : hidden state h_l(xₜ) at layer l after the attention block
    - beta   : running ridge estimate  ŵₜ = (XᵀX + λI)⁻¹ XᵀY
    - xty    : running average  XᵀY / t
    - xt     : current input xₜ
    - pgauss : P(z = K+1 | X, Y) — posterior probability that the
               sequence comes from a Gaussian/OOD task (not one of the
               K known major tasks)

    Returns dict with 2-D float tensors (N_total, ...).
    """
    from icl.linear.analysis.posterior import (
        task_posterior_with_gaussian_linear_regression,
    )

    eval_seq_pos = torch.tensor(
        [2 * p for p in eval_positions], device=device, dtype=torch.long,
    )
    point_pos = torch.tensor(eval_positions, device=device, dtype=torch.long)
    P_ev = len(eval_positions)

    lists = {k: [] for k in ("h", "beta", "xty", "xt", "pgauss")}
    n_batches = (n_samples + B - 1) // B
    orig_bs = int(task_obj.batch_size)
    task_obj.batch_size = B

    for bi in range(n_batches):
        data, _, target = task_obj.sample_batch(
            step=bi + 77777, is_eval=True,
        )
        data, target = data.to(device), target.to(device)

        cache = {}
        def _hook(mod, inp, out, _c=cache):
            h = out if torch.is_tensor(out) else out[0]
            _c["h"] = h.index_select(1, eval_seq_pos).detach()

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(data, target)
        finally:
            handle.remove()

        beta = _compute_running_beta_hat(data, target).index_select(1, point_pos)
        xty = _compute_running_xty(data, target).index_select(1, point_pos)
        xt = data.index_select(1, point_pos)

        with torch.no_grad():
            post = task_posterior_with_gaussian_linear_regression(
                train_task, data, target, include_minor=False,
            )
        pgauss = post[:, -1:].unsqueeze(1).expand(-1, P_ev, -1)

        lists["h"].append(cache["h"].cpu())
        lists["beta"].append(beta.cpu())
        lists["xty"].append(xty.cpu())
        lists["xt"].append(xt.cpu())
        lists["pgauss"].append(pgauss.cpu())

    task_obj.batch_size = orig_bs

    D = lists["h"][0].shape[-1]
    return {
        "h": torch.cat(lists["h"], 0).reshape(-1, D).float(),
        "beta": torch.cat(lists["beta"], 0).reshape(-1, n_dims).float(),
        "xty": torch.cat(lists["xty"], 0).reshape(-1, n_dims).float(),
        "xt": torch.cat(lists["xt"], 0).reshape(-1, n_dims).float(),
        "pgauss": torch.cat(lists["pgauss"], 0).reshape(-1, 1).float(),
    }


# ---------------------------------------------------------------------------
# Unified intervention evaluation
# ---------------------------------------------------------------------------

def _run_intervention_eval(
    model, layer, proj_matrix, task_obj,
    *, device, B, eval_positions, scale, n_samples,
    track_oracle=True,
):
    """Remove ``proj_matrix`` component from hidden states and measure MSE.

    The intervention replaces h with:

        h' = h - scale * P h

    where P = ``proj_matrix`` (typically V V^T for some direction set V).

    We compare three quantities at each eval position t:

    - MSE_baseline(t)   = E[(f(x,y)_t  - y_t)^2]     (unmodified model)
    - MSE_intervened(t) = E[(f'(x,y)_t - y_t)^2]     (with intervention)
    - MSE_oracle(t)     = E[(f'(x,y)_t - x_t w*)^2]  (vs. noiseless truth)

    delta = mean(MSE_intervened - MSE_baseline) across positions.

    Parameters
    ----------
    track_oracle : bool
        If True, also compute MSE relative to oracle predictions and
        per-position breakdowns.
    """
    bl_by_pos = {p: [] for p in eval_positions}
    it_by_pos = {p: [] for p in eval_positions}
    bl_oracle = {p: [] for p in eval_positions} if track_oracle else None
    it_oracle = {p: [] for p in eval_positions} if track_oracle else None
    bl_pos0 = [] if track_oracle else None
    bl_pos0_oracle = [] if track_oracle else None
    pred_delta_sum, pred_delta_count = 0.0, 0

    n_batches = (n_samples + B - 1) // B
    orig_bs = int(task_obj.batch_size)
    task_obj.batch_size = B

    for bi in range(n_batches):
        data, tasks, target = task_obj.sample_batch(
            step=bi + 55555, is_eval=True,
        )
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            preds_base = model(data, target)

        # Intervention hook: h' = h - scale * h P  (= h - scale * P^T h)
        def _hook(mod, inp, out, _P=proj_matrix):
            h = out if torch.is_tensor(out) else out[0]
            h_mod = h - scale * (h @ _P)
            return h_mod if torch.is_tensor(out) else (h_mod,) + out[1:]

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                preds_int = model(data, target)
        finally:
            handle.remove()

        if track_oracle:
            tasks = tasks.to(device)
            oracle_preds = (data @ tasks).squeeze(-1)

        for p in eval_positions:
            if p >= preds_base.shape[1]:
                continue
            bl_by_pos[p].append(
                ((preds_base[:, p] - target[:, p]) ** 2).mean().item()
            )
            it_by_pos[p].append(
                ((preds_int[:, p] - target[:, p]) ** 2).mean().item()
            )
            if track_oracle:
                bl_oracle[p].append(
                    ((preds_base[:, p] - oracle_preds[:, p]) ** 2).mean().item()
                )
                it_oracle[p].append(
                    ((preds_int[:, p] - oracle_preds[:, p]) ** 2).mean().item()
                )
                d_pred = (preds_int[:, p] - preds_base[:, p]).cpu()
                pred_delta_sum += d_pred.abs().sum().item()
                pred_delta_count += d_pred.shape[0]

        if track_oracle and preds_base.shape[1] > 0:
            bl_pos0.append(
                ((preds_base[:, 0] - target[:, 0]) ** 2).mean().item()
            )
            bl_pos0_oracle.append(
                ((preds_base[:, 0] - oracle_preds[:, 0]) ** 2).mean().item()
            )

        del data, target, preds_base, preds_int

    task_obj.batch_size = orig_bs

    def _avg(by_pos):
        vals = [np.mean(by_pos[p]) for p in eval_positions if by_pos[p]]
        return float(np.mean(vals)) if vals else float("nan")

    valid = [p for p in eval_positions if bl_by_pos[p]]
    bl_avg, it_avg = _avg(bl_by_pos), _avg(it_by_pos)
    result = {
        "baseline": bl_avg,
        "intervened": it_avg,
        "delta": it_avg - bl_avg,
    }
    if track_oracle:
        bl_o, it_o = _avg(bl_oracle), _avg(it_oracle)
        result.update({
            "baseline_to_oracle": bl_o,
            "intervened_to_oracle": it_o,
            "delta_to_oracle": it_o - bl_o,
            "baseline_per_pos": [np.mean(bl_by_pos[p]) for p in valid],
            "intervened_per_pos": [np.mean(it_by_pos[p]) for p in valid],
            "positions": valid,
            "mean_abs_pred_delta": pred_delta_sum / max(pred_delta_count, 1),
            "baseline_pos0": float(np.mean(bl_pos0)) if bl_pos0 else float("nan"),
            "baseline_pos0_to_oracle": float(np.mean(bl_pos0_oracle)) if bl_pos0_oracle else float("nan"),
        })
    return result


def _eval_per_major_task(
    model, layer, proj_matrix, train_task,
    *, device, B, eval_positions, scale, n_samples,
):
    """Evaluate the intervention separately for each major task.

    Samples batches from ``train_task`` (which mixes all tasks), identifies
    each sequence's major task by comparing its weight vector to
    ``task_pool``, and aggregates MSE per task.

    Returns a list of dicts (one per major task), each with keys
    ``task_idx``, ``baseline``, ``intervened``, ``delta``, ``pct``.
    """
    task_pool = train_task.task_pool.to(device)
    n_major = task_pool.shape[0]

    bl = [{p: [] for p in eval_positions} for _ in range(n_major)]
    it = [{p: [] for p in eval_positions} for _ in range(n_major)]

    n_batches = (n_samples + B - 1) // B
    orig_bs = int(train_task.batch_size)
    train_task.batch_size = B

    for bi in range(n_batches):
        data, tasks, target = train_task.sample_batch(
            step=bi + 88888, is_eval=True,
        )
        data, target = data.to(device), target.to(device)
        tasks = tasks.to(device)

        dists = (tasks.unsqueeze(1) - task_pool.unsqueeze(0)).norm(dim=2).squeeze(-1)
        task_idx = dists.argmin(dim=1)
        is_major = dists.min(dim=1).values < 1e-4

        with torch.no_grad():
            preds_base = model(data, target)

        def _hook(mod, inp, out, _P=proj_matrix):
            h = out if torch.is_tensor(out) else out[0]
            h_mod = h - scale * (h @ _P)
            return h_mod if torch.is_tensor(out) else (h_mod,) + out[1:]

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                preds_int = model(data, target)
        finally:
            handle.remove()

        for p in eval_positions:
            if p >= preds_base.shape[1]:
                continue
            bl_mse = (preds_base[:, p] - target[:, p]) ** 2
            it_mse = (preds_int[:, p] - target[:, p]) ** 2
            for k in range(n_major):
                mask = is_major & (task_idx == k)
                if mask.any():
                    bl[k][p].append(bl_mse[mask].mean().item())
                    it[k][p].append(it_mse[mask].mean().item())

        del data, target, preds_base, preds_int

    train_task.batch_size = orig_bs

    results = []
    for k in range(n_major):
        valid = [p for p in eval_positions if bl[k][p]]
        if valid:
            bl_avg = float(np.mean([np.mean(bl[k][p]) for p in valid]))
            it_avg = float(np.mean([np.mean(it[k][p]) for p in valid]))
        else:
            bl_avg, it_avg = float("nan"), float("nan")
        delta = it_avg - bl_avg
        pct = 100.0 * delta / bl_avg if bl_avg > 0 else float("nan")
        results.append({
            "task_idx": k,
            "baseline": bl_avg,
            "intervened": it_avg,
            "delta": delta,
            "pct": pct,
        })
    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_intervention_summary(results):
    """Print formatted summary of intervention results."""
    r = results
    n_directions = r["n_directions"]
    loss_history = r["loss_history"]
    vs = r["vopt_summary"]
    enriched_r2 = r["enriched_r2"]
    n_dims = r["n_dims"]
    input_loadings = r["input_loadings"]
    opt_src_label = "Minor" if r["opt_source"] == "minor" else "OOD"

    print(f"\n{'=' * 65}")
    print(f"Causal Intervention: Optimal Rank-{n_directions} Orth Subspace  "
          f"(layer {r['layer']}, scale={r['scale']}, opt={r['opt_source']})")
    print(f"{'=' * 65}")
    print(
        f"  Joint fit R\u00b2: {r.get('joint_r2', float('nan')):.4f}  |  "
        f"Task rank: {r['task_subspace_rank']}  |  "
        f"Protected rank: {r['protected_rank']}  |  "
        f"Token: {r['token_protection_mode']} ({r['input_protection_features']}), "
        f"rank {r['token_protection_rank']}/{r['token_protection_max_rank']}  |  "
        f"Opt steps: {len(loss_history)}/{r['n_opt_steps_total']}  |  "
        f"{opt_src_label} MSE: {loss_history[0]:.4f} \u2192 {loss_history[-1]:.4f}"
    )

    # V_opt variance summary
    print(f"\n  V_opt summary (rank {vs['n_directions']}, orth dim {vs['orth_dim']}):")
    print(f"    {'':30s} {'Major':>8} {'OOD':>8}")
    print(f"    {'-'*46}")
    print(f"    {'Var explained (% of total)':<30s} "
          f"{100*vs['frac_var_major']:>7.2f}% {100*vs['frac_var_ood']:>7.2f}%")
    print(f"    {'Var explained (% of orth)':<30s} "
          f"{100*vs['frac_orth_var_major']:>7.2f}% {100*vs['frac_orth_var_ood']:>7.2f}%")
    print(f"    Stable rank (OOD proj): {vs['stable_rank_proj_ood']:.2f}")

    # Joint probe diagnostics
    jd = results.get("joint_diagnostics")
    if jd is not None:
        _r2 = "\u00b2"
        print(f"\n  Joint probe \u2192 V_opt (all features):")
        print(f"    Joint R{_r2} (OOD): {jd['joint_r2_ood']:.4f}  |  "
              f"Joint R{_r2} (Major): {jd['joint_r2_major']:.4f}  |  "
              f"Cond #: {jd['condition_number']:.1f}")
        print(f"    {'Feature':<22} {'dim':>4} {'Partial R'+_r2:>12} {'Marginal R'+_r2:>12}")
        print(f"    {'-'*52}")
        for i, name in enumerate(jd["group_names"]):
            pr2 = jd["partial_r2"][name]
            mr2 = enriched_r2[name]["fwd_ood"]
            dim = jd["group_dims"][i]
            print(f"    {name:<22} {dim:>4} {pr2:>12.4f} {mr2:>12.4f}")

    # Individual probe R² table (marginal)
    _probe_hdr = 'Probe \u2192 V_opt  R\u00b2'
    print(f"\n  {_probe_hdr:<28} {'Major':>8} {'OOD':>8}")
    print(f"  {'-'*44}")
    for fn, rv in enriched_r2.items():
        print(f"  {fn:<28} {rv['fwd_major']:>8.4f} {rv['fwd_ood']:>8.4f}")

    # Input-space loadings
    dim_labels = ["y_ch"] + [f"x{i}" for i in range(n_dims)]
    print(f"\n  Input-space loadings (V_opt^T @ W_input_proj):")
    for d_i in range(input_loadings.shape[0]):
        vals = input_loadings[d_i]
        top_idx = vals.abs().argsort(descending=True)[:3]
        top_str = ", ".join(
            f"{dim_labels[int(j)]}={vals[j]:.3f}" for j in top_idx
        )
        print(f"    dir {d_i}: {top_str}")

    # Intervention effect
    print(f"\n  Mean |\u0394 pred| (intervention effect):")
    print(f"  Major: {results['mean_abs_pred_delta_major']:.4f}")
    print(f"  OOD:   {results['mean_abs_pred_delta_ood']:.4f}")

    print(f"\n  Eval positions: {len(results['eval_positions'])} positions\n")

    # MSE table
    def _row(label, key_m, key_o, fmt=".4f"):
        print(f"{label:<30} {results[key_m]:>12{fmt}} {results[key_o]:>12{fmt}}")

    def _pct_row(label, key_m, key_o):
        print(f"{label:<30} {results[key_m]:>11.1f}% {results[key_o]:>11.1f}%")

    def _safe_pct(delta, baseline):
        return 100.0 * delta / baseline if baseline > 0 else float("nan")

    print(f"{'Metric':<30} {'Major':>12} {'OOD':>12}")
    print("-" * 54)
    _row("Baseline MSE (\u2192 target)",  "baseline_loss_major",    "baseline_loss_ood")
    _row("Intervened MSE (\u2192 target)", "intervened_loss_major",  "intervened_loss_ood")
    _row("\u0394 MSE (\u2192 target)",     "delta_loss_major",       "delta_loss_ood")
    _pct_row("% increase (\u2192 target)", "pct_increase_major",     "pct_increase_ood")
    print()
    _row("Baseline MSE (\u2192 oracle)",   "baseline_to_oracle_major", "baseline_to_oracle_ood")
    _row("Intervened MSE (\u2192 oracle)",  "intervened_to_oracle_major", "intervened_to_oracle_ood")
    _row("\u0394 MSE (\u2192 oracle)",      "delta_to_oracle_major",  "delta_to_oracle_ood")
    _row("Rand orth 3x \u0394 MSE",        "rand3x_int_delta_major", "rand3x_int_delta_ood")
    _pct_row("Rand orth 3x % increase",    "rand3x_int_pct_major",   "rand3x_int_pct_ood")
    print()
    _row("MSE at pos 0 (\u2192 target)",   "baseline_pos0_major",    "baseline_pos0_ood")
    ctx_gain_m = results["baseline_pos0_major"] - results["baseline_loss_major"]
    ctx_gain_o = results["baseline_pos0_ood"]   - results["baseline_loss_ood"]
    print(f"{'Context gain (l_0 - l_t)':<30} {ctx_gain_m:>12.4f} {ctx_gain_o:>12.4f}")
    delta_m = results["delta_loss_major"]
    delta_o = results["delta_loss_ood"]
    frac_m = 100.0 * delta_m / ctx_gain_m if ctx_gain_m > 0 else float("nan")
    frac_o = 100.0 * delta_o / ctx_gain_o if ctx_gain_o > 0 else float("nan")
    _frac_label = "\u0394 / context gain"
    print(f"{_frac_label:<30} {frac_m:>11.1f}% {frac_o:>11.1f}%")

    # Per-task breakdown
    per_task = results.get("per_task")
    if per_task:
        print(f"\n  Per-task intervention effect:")
        _delta_hdr = "\u0394 MSE"
        print(f"    {'Task':<8} {'Baseline':>10} {'Intervened':>11} "
              f"{_delta_hdr:>10} {'% increase':>11}")
        print(f"    {'-'*51}")
        for pt in per_task:
            print(f"    Task {pt['task_idx']:<3d} {pt['baseline']:>10.4f} "
                  f"{pt['intervened']:>11.4f} {pt['delta']:>10.4f} "
                  f"{pt['pct']:>10.1f}%")

    # Filtered V_opt section
    r2_thr = results.get("r2_filter_threshold", 0.1)
    kept_r2s = results["filtered_per_dir_r2"]
    print(f"\n  Filtered V_opt (R\u00b2 \u2265 {r2_thr}):")
    print(f"    Directions kept: {results['filtered_n_dirs']} / {n_directions}")
    print(f"    Joint R\u00b2 (OOD): {results['filtered_joint_r2']:.4f}")
    if kept_r2s:
        print(f"    Per-direction R\u00b2: [{', '.join(f'{r:.3f}' for r in kept_r2s)}]")
    filt_pct_m = _safe_pct(results["filtered_delta_loss_major"], results.get("filtered_baseline_loss_major", 0.0))
    filt_pct_o = _safe_pct(results["filtered_delta_loss_ood"],   results.get("filtered_baseline_loss_ood", 0.0))
    print(f"    {'Metric':<30} {'Major':>12} {'OOD':>12}")
    print(f"    {'-'*54}")
    print(f"    {'Filt. MSE (oracle)':<30} "
          f"{results['filtered_intervened_to_oracle_major']:>12.4f} "
          f"{results['filtered_intervened_to_oracle_ood']:>12.4f}")
    _filt_delta = "Filt. \u0394 MSE (\u2192 target)"
    print(f"    {_filt_delta:<30} "
          f"{results['filtered_delta_loss_major']:>12.4f} "
          f"{results['filtered_delta_loss_ood']:>12.4f}")
    print(f"    {'Filt. % increase':<30} {filt_pct_m:>11.1f}% {filt_pct_o:>11.1f}%")


# ---------------------------------------------------------------------------
# V_opt probing, filtering, and summary statistics
# ---------------------------------------------------------------------------

def _joint_probe_r2(feature_groups, Y, *, exclude_idx=None):
    """OLS R² from concatenated feature groups; drop group *exclude_idx* for partial R²."""
    parts = [g for i, g in enumerate(feature_groups) if i != exclude_idx]
    return _ols_r2(torch.cat(parts, dim=1), Y)


def _probe_and_filter_vopt(
    model, layer, V_opt_cpu,
    train_task, ood_task_eval,
    *, device, B, eval_positions, n_dims, n_samples_probe,
    r2_filter_threshold=0.1,
):
    """Probe V_opt with known features and filter to interpretable directions.

    Fits a **joint** linear probe from all feature groups simultaneously
    to ``proj(h, V_opt)`` and reports:

    - Joint R² (all features together, OOD held-out)
    - Partial R² for each feature group (unique contribution)
    - Condition number of the joint design matrix (collinearity check)
    - Individual R² per feature (marginal, for comparison)
    - MLP R² for X^TY/t (linearity check)

    Then filters V_opt to keep only interpretable directions (SVD of
    joint prediction, per-direction R² threshold).

    Returns a dict with ``enriched_r2``, ``joint_diagnostics``,
    ``V_opt_filtered``, and associated metadata.
    """
    n_directions = V_opt_cpu.shape[1]

    feat_maj = _collect_features(
        model, layer, train_task, train_task,
        device=device, B=B, eval_positions=eval_positions,
        n_dims=n_dims, n_samples=n_samples_probe,
    )
    feat_ood = _collect_features(
        model, layer, ood_task_eval, train_task,
        device=device, B=B, eval_positions=eval_positions,
        n_dims=n_dims, n_samples=n_samples_probe,
    )

    proj_maj = feat_maj["h"] @ V_opt_cpu
    proj_ood = feat_ood["h"] @ V_opt_cpu

    xt_beta_maj = (feat_maj["xt"] * feat_maj["beta"]).sum(-1, keepdim=True)
    xt_beta_ood = (feat_ood["xt"] * feat_ood["beta"]).sum(-1, keepdim=True)

    probe_groups_ood = [
        feat_ood["xty"],
        feat_ood["beta"],
        feat_ood["xt"],
        xt_beta_ood,
        feat_ood["pgauss"],
    ]
    probe_groups_maj = [
        feat_maj["xty"],
        feat_maj["beta"],
        feat_maj["xt"],
        xt_beta_maj,
        feat_maj["pgauss"],
    ]
    group_names = ["X^TY/t", "beta_hat", "x_t", "x_t @ beta_hat", "P(Z=K+1)"]

    # ---- 1. Joint probe (all features simultaneously) ----
    joint_r2_ood = _joint_probe_r2(probe_groups_ood, proj_ood)
    joint_r2_maj = _joint_probe_r2(probe_groups_maj, proj_maj)

    # ── Partial R² ─────────────────────────────────────────────────
    # Unique contribution of feature group i after controlling for
    # all other groups:
    #   ΔR²(i | rest) = (R²_full − R²_{without i}) / (1 − R²_{without i})
    # ≈ 0 means group i is redundant given the rest; ≈ 1 means essential.
    partial_r2 = {}
    for i, name in enumerate(group_names):
        r2_without = _joint_probe_r2(probe_groups_ood, proj_ood, exclude_idx=i)
        denom = max(1.0 - r2_without, 1e-12)
        partial_r2[name] = 1.0 - (1.0 - joint_r2_ood) / denom

    # Condition number of the joint design matrix (collinearity diagnostic)
    X_all = torch.cat(probe_groups_ood, dim=1)
    X_all_aug = torch.cat([X_all, torch.ones(X_all.shape[0], 1)], dim=1)
    cond_num = torch.linalg.cond(X_all_aug).item()

    joint_diagnostics = {
        "joint_r2_ood": joint_r2_ood,
        "joint_r2_major": joint_r2_maj,
        "partial_r2": partial_r2,
        "condition_number": cond_num,
        "group_names": group_names,
        "group_dims": [g.shape[1] for g in probe_groups_ood],
    }

    # ---- 2. Individual probes (marginal R², for comparison) ----
    enriched_r2 = {}
    for i, name in enumerate(group_names):
        enriched_r2[name] = {
            "fwd_major": _ols_r2(probe_groups_maj[i], proj_maj),
            "fwd_ood":   _ols_r2(probe_groups_ood[i], proj_ood),
        }
    enriched_r2["X^TY/t (MLP)"] = {
        "fwd_major": _fit_r2_mlp(probe_groups_maj[0], proj_maj),
        "fwd_ood":   _fit_r2_mlp(probe_groups_ood[0], proj_ood),
    }

    # ── Joint probe filtering of V_opt ──────────────────────────────
    # Fit all feature groups jointly to predict  proj(h, V_opt) ∈ ℝ^r.
    # Then SVD on the predicted values identifies the directions of
    # V_opt that the features can explain.  Directions with R² below
    # the threshold are discarded — they are not interpretable from
    # the known covariates.
    F_aug = X_all_aug
    W_j = torch.linalg.pinv(F_aug) @ proj_ood   # OLS: features → proj
    pred = F_aug @ W_j

    ss_r = ((proj_ood - pred) ** 2).sum().item()
    ss_t = ((proj_ood - proj_ood.mean(0)) ** 2).sum().item()
    filt_joint_r2 = 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")

    # SVD of centred predictions → principal interpretable directions
    pred_c = pred - pred.mean(0)
    proj_c = proj_ood - proj_ood.mean(0)
    _, _, Vh = torch.linalg.svd(pred_c, full_matrices=False)

    # Keep direction k only if R²(proj onto Vₕ[k]) ≥ threshold
    kept_dirs, kept_r2s = [], []
    for k in range(min(Vh.shape[0], n_directions)):
        p_k = proj_c @ Vh[k]
        pr_k = pred_c @ Vh[k]
        ss_rk = ((p_k - pr_k) ** 2).sum().item()
        ss_tk = (p_k ** 2).sum().item()
        r2_k = 1.0 - ss_rk / ss_tk if ss_tk > 1e-12 else 0.0
        if r2_k >= r2_filter_threshold:
            kept_dirs.append(k)
            kept_r2s.append(r2_k)

    if kept_dirs:
        V_filt, _ = torch.linalg.qr(V_opt_cpu @ Vh[kept_dirs].T)
    else:
        V_filt = V_opt_cpu.clone()
        kept_r2s = []

    return {
        "enriched_r2": enriched_r2,
        "joint_diagnostics": joint_diagnostics,
        "V_opt_filtered": V_filt,
        "filtered_n_dirs": V_filt.shape[1],
        "good_feat_names": group_names,
        "filtered_per_dir_r2": kept_r2s,
        "filtered_joint_r2": filt_joint_r2,
        "r2_filter_threshold": r2_filter_threshold,
        "feat_maj": feat_maj,
        "feat_ood": feat_ood,
    }


def _compute_vopt_summary(V_opt_cpu, P_orth_cpu, feat_maj, feat_ood, model,
                           n_directions, orth_dim):
    """Variance-decomposition summary and input-space loadings for V_opt.

    Returns (vopt_summary_dict, input_loadings_tensor).
    """
    h_maj, h_ood = feat_maj["h"], feat_ood["h"]
    h_maj_c = h_maj - h_maj.mean(0, keepdim=True)   # centred hidden states
    h_ood_c = h_ood - h_ood.mean(0, keepdim=True)

    def _frac(proj, total):
        """‖proj‖² / ‖total‖²  — fraction of energy in the projection."""
        return (proj ** 2).sum().item() / max((total ** 2).sum().item(), 1e-30)

    # Fraction of total variance captured by V_opt:
    #   ‖h_c V_opt‖² / ‖h_c‖²
    frac_var_maj = _frac(h_maj_c @ V_opt_cpu, h_maj_c)
    frac_var_ood = _frac(h_ood_c @ V_opt_cpu, h_ood_c)
    # Fraction of orthogonal-complement variance captured by V_opt:
    #   ‖h_c V_opt‖² / ‖h_c P_⊥‖²
    frac_orth_maj = _frac(h_maj_c @ V_opt_cpu, h_maj_c @ P_orth_cpu)
    frac_orth_ood = _frac(h_ood_c @ V_opt_cpu, h_ood_c @ P_orth_cpu)

    # Stable rank of h projected onto V_opt:
    #   stable_rank = (Σᵢ σᵢ²) / σ₁²
    # Equals 1 when one direction dominates; equals r when all equal.
    svd_vals = torch.linalg.svdvals(h_ood_c @ V_opt_cpu)
    stable_rank = (
        (svd_vals ** 2).sum().item() / (svd_vals[0] ** 2).item()
        if svd_vals[0] > 0 else float("nan")
    )

    # Input-space loadings: V_optᵀ W_input  shows which input dimensions
    # (y-channel and x-coordinates) each V_opt direction is sensitive to.
    W_input = model.input_proj.weight.detach().cpu().float()
    loadings = V_opt_cpu.T @ W_input

    summary = {
        "frac_var_major": frac_var_maj,
        "frac_var_ood": frac_var_ood,
        "frac_orth_var_major": frac_orth_maj,
        "frac_orth_var_ood": frac_orth_ood,
        "stable_rank_proj_ood": stable_rank,
        "n_directions": n_directions,
        "orth_dim": orth_dim,
    }
    return summary, loadings


def _rand_orth_trials(
    model, layer, U_orth_cpu, orth_dim, n_directions,
    train_task, ood_task_eval,
    *, device, B, eval_positions, scale, n_samples_eval,
    n_trials=3,
):
    """Average delta-MSE from random orthogonal directions of given rank."""
    actual_rank = min(n_directions, orth_dim)
    d_maj, d_ood, p_maj, p_ood = [], [], [], []
    for _ in range(n_trials):
        W_r = torch.randn(orth_dim, actual_rank)
        V_r, _ = torch.linalg.qr(U_orth_cpu @ W_r)
        P_r = (V_r @ V_r.T).to(device)
        rm = _run_intervention_eval(
            model, layer, P_r, train_task,
            device=device, B=B, eval_positions=eval_positions,
            scale=scale, n_samples=n_samples_eval,
            track_oracle=False,
        )
        ro = _run_intervention_eval(
            model, layer, P_r, ood_task_eval,
            device=device, B=B, eval_positions=eval_positions,
            scale=scale, n_samples=n_samples_eval,
            track_oracle=False,
        )
        d_maj.append(rm["delta"])
        d_ood.append(ro["delta"])
        p_maj.append(100.0 * rm["delta"] / rm["baseline"] if rm["baseline"] > 0 else float("nan"))
        p_ood.append(100.0 * ro["delta"] / ro["baseline"] if ro["baseline"] > 0 else float("nan"))
    return (
        float(np.mean(d_maj)), float(np.mean(d_ood)),
        float(np.mean(p_maj)), float(np.mean(p_ood)),
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def intervene_optimal_orth_direction(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_opt_steps: int = 200,
    opt_lr: float = 0.01,
    opt_B: int = 32,
    patience: int = 30,
    grad_clip_norm: float = 1.0,
    n_samples_eval: int = 500,
    n_samples_probe: int = 1000,
    n_ood: int = 30,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    eval_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    n_directions: int = 1,
    scale: float = 1.0,
    token_protection_mode: str = "residual_pca",
    token_protection_rank: Optional[int] = None,
    token_protection_var_explained: float = 0.9,
    input_protection_features: str = "x_y",
    opt_source: str = "ood",
    max_mse_oracle: Optional[float] = None,
    min_opt_steps: int = 0,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Find a rank-r subspace in the orthogonal complement of the task
    subspace that maximally increases OOD loss when removed.

    High-level algorithm
    --------------------
    1. **Joint fit**: ``h = [pi_major, tok_feat] @ [W_task; W_tok] + b``.
       Gives unbiased task directions ``W_task`` (by FWL) and token
       directions ``W_tok``.  SVD of centred ``W_task`` gives the task
       subspace basis B (D x rank).

    2. **Protected subspace**: augment B with ``W_tok`` directions,
       SVD to get S_prot.  ``P_orth = I - S_prot S_prot^T``.

    3. **Optimise V_opt**: parameterise V = U_orth @ W (W is learnable)
       so V always lies in the complement.  Gradient ascent maximises:

           L_ood(V) - penalty * L_major(V)

       where L is MSE after intervention  h' = h - scale * V V^T h.

    4. **Probe V_opt**: fit linear probes from known features (X'Y/t,
       beta_hat, x_t, P(Z=K+1), ...) to proj(h, V_opt) and report R^2.
       Optionally filter V_opt to keep only interpretable directions.

    5. **Evaluate**: run the actual intervention on held-out data and
       measure delta MSE for both major tasks and OOD tasks.  Compare
       against random-direction baselines of the same rank.

    Parameters
    ----------
    input_protection_features : str
        Which token features to use in the joint fit for input protection:
        - ``"x"``       : only current x_t  (n_dims features)
        - ``"x_y"``     : [x_t, y_{t-1}]   (n_dims + 1 features)  **default**
        - ``"x_y_suff"`` : [x_t, y_{t-1}, suff_stats_t]
    opt_source : str
        Data source for the optimisation objective.
        ``"ood"`` (default) — maximise intervened MSE on fresh OOD tasks.
        ``"minor"`` — maximise intervened MSE on the minor task pool.
    """
    from icl.linear.linear_path_utils import load_model_task_config

    # ---- 1. Setup ----
    _, train_task, config = load_model_task_config(exp_name)
    if step is None:
        step = config.training.total_steps
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    n_points = int(config.task.n_points)
    n_dims = int(config.task.n_dims)
    n_embd = int(config.model.n_embd)
    noise_scale = float(train_task.noise_scale)

    if fit_positions is None:
        fit_positions = list(range(min(10, n_points), n_points))
    if eval_positions is None:
        eval_positions = list(range(n_points))

    if opt_source not in {"ood", "minor"}:
        raise ValueError(f"opt_source={opt_source!r} not in {{'ood', 'minor'}}")
    if opt_source == "minor":
        if not (int(getattr(train_task, "n_minor_tasks", 0)) > 0
                and getattr(train_task, "minor_pool", None) is not None):
            raise ValueError(
                "opt_source='minor' requires a train_task with a non-empty minor_pool "
                f"(n_minor_tasks={getattr(train_task, 'n_minor_tasks', 0)})"
            )

    ood_task = _create_ood_task(train_task, config, B, n_ood, device)
    ood_task_eval = _create_ood_task(train_task, config, B, n_ood, device)

    # ---- 2. Joint fit + protected subspace ----
    fit = _joint_fit_task_token(
        model, layer, train_task,
        device=device, fit_positions=fit_positions,
        fit_n_samples=fit_n_samples, B=B,
        n_dims=n_dims, n_embd=n_embd,
        input_protection_features=input_protection_features,
        center_task_vecs=center_task_vecs,
    )
    joint_r2 = fit["joint_r2"]
    rank = fit["rank"]

    prot = _build_protected_subspace(
        fit["basis"], fit["W_tok"], n_embd,
        device=device,
        token_protection_mode=token_protection_mode,
        token_protection_rank=token_protection_rank,
        token_protection_var_explained=token_protection_var_explained,
    )
    orth_dim = prot["orth_dim"]

    if verbose:
        logger.info(
            f"[opt-dir linear] Joint fit R\u00b2={joint_r2:.4f}, "
            f"task rank={rank}, protected rank={prot['protected_rank']}, "
            f"token-mode={token_protection_mode}, "
            f"token-rank={prot['token_basis_rank']}/{prot['token_basis_max_rank']}, "
            f"input-features={input_protection_features}, "
            f"orth dim={orth_dim}"
        )

    # ---- 3. Optimise V_opt ----
    opt_task = train_task if opt_source == "minor" else ood_task
    V_opt, loss_history = _optimize_v_directions(
        model, layer, prot["U_orth"], orth_dim,
        device=device, opt_task=opt_task,
        eval_positions=eval_positions, n_directions=n_directions,
        n_opt_steps=n_opt_steps, opt_lr=opt_lr, opt_B=opt_B,
        patience=patience, grad_clip_norm=grad_clip_norm,
        scale=scale,
        opt_source=opt_source,
        max_mse_oracle=max_mse_oracle, min_opt_steps=min_opt_steps,
        verbose=verbose,
    )

    if verbose:
        logger.info(
            f"[opt-dir linear] Optimisation done. "
            f"OOD MSE: {loss_history[0]:.4f} \u2192 {loss_history[-1]:.4f}"
        )

    # ---- 4. Probe + filter V_opt ----
    V_opt_cpu = V_opt.cpu().float()
    probe = _probe_and_filter_vopt(
        model, layer, V_opt_cpu, train_task, ood_task_eval,
        device=device, B=B, eval_positions=eval_positions,
        n_dims=n_dims, n_samples_probe=n_samples_probe,
    )

    # ---- 5. V_opt summary stats ----
    P_orth_cpu = prot["P_orth"].cpu().float()
    vopt_summary, input_loadings = _compute_vopt_summary(
        V_opt_cpu, P_orth_cpu,
        probe["feat_maj"], probe["feat_ood"], model,
        n_directions, orth_dim,
    )

    del probe["feat_maj"], probe["feat_ood"]

    # ---- 6. Run causal interventions ----
    P_v = (V_opt @ V_opt.T).to(device)
    eval_kw = dict(device=device, B=B, eval_positions=eval_positions,
                   scale=scale, n_samples=n_samples_eval)

    res_major = _run_intervention_eval(model, layer, P_v, train_task, **eval_kw)
    res_ood   = _run_intervention_eval(model, layer, P_v, ood_task_eval, **eval_kw)

    rand3x_rank = 3 * n_directions
    rand3x_d_m, rand3x_d_o, rand3x_p_m, rand3x_p_o = _rand_orth_trials(
        model, layer, prot["U_orth"].cpu().float(), orth_dim, rand3x_rank,
        train_task, ood_task_eval,
        device=device, B=B, eval_positions=eval_positions,
        scale=scale, n_samples_eval=n_samples_eval,
    )

    V_filt = probe["V_opt_filtered"]
    P_v_filt = (V_filt @ V_filt.T).to(device)
    res_filt_major = _run_intervention_eval(model, layer, P_v_filt, train_task, **eval_kw)
    res_filt_ood   = _run_intervention_eval(model, layer, P_v_filt, ood_task_eval, **eval_kw)

    per_task = _eval_per_major_task(
        model, layer, P_v, train_task,
        device=device, B=B, eval_positions=eval_positions,
        scale=scale, n_samples=n_samples_eval,
    )

    _cleanup_model(model)

    # ---- 7. Assemble results ----
    def _pct(res):
        return 100.0 * res["delta"] / res["baseline"] if res["baseline"] > 0 else float("nan")

    def _unpack(res, suffix):
        return {
            f"baseline_loss_{suffix}": res["baseline"],
            f"intervened_loss_{suffix}": res["intervened"],
            f"delta_loss_{suffix}": res["delta"],
            f"pct_increase_{suffix}": _pct(res),
            f"baseline_to_oracle_{suffix}": res["baseline_to_oracle"],
            f"intervened_to_oracle_{suffix}": res["intervened_to_oracle"],
            f"delta_to_oracle_{suffix}": res["delta_to_oracle"],
            f"baseline_per_pos_{suffix}": res["baseline_per_pos"],
            f"intervened_per_pos_{suffix}": res["intervened_per_pos"],
            f"mean_abs_pred_delta_{suffix}": res["mean_abs_pred_delta"],
            f"baseline_pos0_{suffix}": res.get("baseline_pos0", float("nan")),
            f"baseline_pos0_to_oracle_{suffix}": res.get("baseline_pos0_to_oracle", float("nan")),
        }

    results = {
        **_unpack(res_major, "major"),
        **_unpack(res_ood, "ood"),
        "eval_positions": res_major["positions"],
        "layer": layer,
        "scale": scale,
        "task_subspace_rank": rank,
        "protected_rank": prot["protected_rank"],
        "token_protection_mode": token_protection_mode,
        "token_protection_rank": prot["token_basis_rank"],
        "token_protection_max_rank": prot["token_basis_max_rank"],
        "token_protection_var_explained": token_protection_var_explained,
        "input_protection_features": input_protection_features,
        "joint_r2": joint_r2,
        "opt_source": opt_source,
        "n_directions": n_directions,
        "n_dims": n_dims,
        "n_opt_steps_total": n_opt_steps,
        "directions": V_opt.cpu(),
        "directions_filtered": V_filt,
        "loss_history": loss_history,
        "rand3x_int_delta_major": rand3x_d_m,
        "rand3x_int_delta_ood": rand3x_d_o,
        "rand3x_int_pct_major": rand3x_p_m,
        "rand3x_int_pct_ood": rand3x_p_o,
        "rand3x_rank": min(rand3x_rank, orth_dim),
        "enriched_r2": probe["enriched_r2"],
        "joint_diagnostics": probe["joint_diagnostics"],
        "input_loadings": input_loadings,
        "vopt_summary": vopt_summary,
        "r2_filter_threshold": probe["r2_filter_threshold"],
        "filtered_joint_r2": probe["filtered_joint_r2"],
        "filtered_n_dirs": probe["filtered_n_dirs"],
        "filtered_per_dir_r2": probe["filtered_per_dir_r2"],
        "filtered_probe_names": probe["good_feat_names"],
        "filtered_intervened_to_oracle_major": res_filt_major["intervened_to_oracle"],
        "filtered_intervened_to_oracle_ood": res_filt_ood["intervened_to_oracle"],
        "filtered_baseline_to_oracle_major": res_filt_major["baseline_to_oracle"],
        "filtered_baseline_to_oracle_ood": res_filt_ood["baseline_to_oracle"],
        "filtered_delta_loss_major": res_filt_major["delta"],
        "filtered_delta_loss_ood": res_filt_ood["delta"],
        "filtered_intervened_loss_major": res_filt_major["intervened"],
        "filtered_intervened_loss_ood": res_filt_ood["intervened"],
        "filtered_baseline_loss_major": res_filt_major["baseline"],
        "filtered_baseline_loss_ood": res_filt_ood["baseline"],
        "per_task": per_task,
    }

    if print_summary:
        _print_intervention_summary(results)

    return results


# ---------------------------------------------------------------------------
# Layer sweep + plotting
# ---------------------------------------------------------------------------

def plot_optimal_orth_direction_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    figsize: tuple = (14, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
):
    """
    Sweep ``intervene_optimal_orth_direction`` across layers.

    Parameters
    ----------
    **kwargs
        Forwarded to ``intervene_optimal_orth_direction``
        (e.g. ``B``, ``n_directions``, ``scale``, ``opt_source``, etc.).

    Returns ``(fig, fig_loss, fig_r2, fig_filt, fig_per_task, all_results)``.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config

    if layers is None:
        _, _, config = load_model_task_config(exp_name)
        layers = list(range(config.model.n_layer))

    all_results = {}
    for l in layers:
        logger.info(f"[opt-dir sweep linear] layer {l} ...")
        all_results[l] = intervene_optimal_orth_direction(
            exp_name=exp_name, layer=l,
            verbose=False, print_summary=True, **kwargs,
        )

    # ---- helpers ----
    def _extract(key):
        return [all_results[l][key] for l in layers]

    x = np.arange(len(layers))
    bar_w = 0.35
    base_maj = np.mean(_extract("baseline_to_oracle_major"))
    base_ood = np.mean(_extract("baseline_to_oracle_ood"))

    n_directions = kwargs.get("n_directions", 1)
    scale = kwargs.get("scale", 1.0)
    opt_source = kwargs.get("opt_source", "ood")

    pos0_ood = np.mean(_extract("baseline_pos0_to_oracle_ood"))

    def _bar_chart(ax, maj_key, ood_key, maj_label, ood_label, chart_title,
                   show_pos0=False):
        vals_maj, vals_ood = _extract(maj_key), _extract(ood_key)
        ax.bar(x - bar_w / 2, vals_maj, bar_w, label=maj_label,
               color="#2196F3", alpha=0.85)
        ax.bar(x + bar_w / 2, vals_ood, bar_w, label=ood_label,
               color="#FF9800", alpha=0.85)
        for i, (vm, vo) in enumerate(zip(vals_maj, vals_ood)):
            ax.text(x[i] - bar_w / 2, vm, f"{vm:.4f}",
                    ha="center", va="bottom", fontsize=9)
            ax.text(x[i] + bar_w / 2, vo, f"{vo:.4f}",
                    ha="center", va="bottom", fontsize=9)
        ax.axhline(base_maj, color="#2196F3", ls="--", lw=1.5, alpha=0.7,
                   label=f"Major baseline ({base_maj:.4f})")
        ax.axhline(base_ood, color="#FF9800", ls="--", lw=1.5, alpha=0.7,
                   label=f"OOD baseline ({base_ood:.4f})")
        if show_pos0 and not np.isnan(pos0_ood):
            ax.axhline(pos0_ood, color="#4CAF50", ls=":", lw=2, alpha=0.8,
                       label=f"OOD pos-0 MSE ({pos0_ood:.4f})")
        ax.set(xlabel="Layer", ylabel="MSE (model \u2192 oracle)",
               title=chart_title)
        ax.set_xticks(x, [str(l) for l in layers])
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(15)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    # ---- 1. Intervened MSE to oracle ----
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(layers)), 6))
    _bar_chart(
        ax,
        "intervened_to_oracle_major", "intervened_to_oracle_ood",
        "Major (intervened)", "OOD (intervened)",
        title or f"Optimal Rank-{n_directions} Orth Subspace "
                 f"(linear, scale={scale})",
        show_pos0=True,
    )
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_or_close(fig, show)

    # ---- 2. Optimisation loss history ----
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10
    for i, l in enumerate(layers):
        ax_loss.plot(all_results[l]["loss_history"],
                     label=f"Layer {l}", color=cmap(i % 10), alpha=0.85)
    opt_label = "Minor" if opt_source == "minor" else "OOD"
    ax_loss.set(
        xlabel="Optimisation Step",
        ylabel=f"{opt_label} MSE (intervened)",
        title=f"Loss History (linear, rank-{n_directions}, opt={opt_source})",
    )
    ax_loss.legend(fontsize=12)
    ax_loss.grid(alpha=0.3)
    ax_loss.tick_params(labelsize=12)
    _show_or_close(fig_loss, show)

    # ---- 3. R² plot: enriched probes → V_opt (OOD) ----
    fig_r2, ax_r2 = plt.subplots(figsize=(max(8, 1.2 * len(layers)), 5))
    enriched_keys = list(all_results[layers[0]]["enriched_r2"].keys())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
    for ki, feat_name in enumerate(enriched_keys):
        fwd = [all_results[l]["enriched_r2"][feat_name]["fwd_ood"]
               for l in layers]
        ax_r2.plot(layers, fwd, marker=markers[ki % len(markers)], ls="-",
                   label=feat_name, color=f"C{ki}", lw=2, ms=7)
    ax_r2.set(xlabel="Layer", ylabel="R\u00b2",
              title="Probe \u2192 V_opt R\u00b2 (OOD)")
    ax_r2.set_xticks(layers)
    ax_r2.tick_params(labelsize=12)
    ax_r2.legend(fontsize=8, ncol=3, loc="upper left")
    ax_r2.grid(alpha=0.3)
    ax_r2.set_ylim(-0.05, 1.05)
    _show_or_close(fig_r2, show)

    # ---- 4. Filtered V_opt intervention barplot ----
    fig_filt, ax_filt = plt.subplots(figsize=(max(8, 1.4 * len(layers)), 6))
    _bar_chart(
        ax_filt,
        "filtered_intervened_to_oracle_major",
        "filtered_intervened_to_oracle_ood",
        "Major (filtered)", "OOD (filtered)",
        f"V_opt_filtered (scale={scale})",
    )
    _show_or_close(fig_filt, show)

    # ---- 5. Per-task intervention delta ----
    n_major = len(all_results[layers[0]]["per_task"])
    task_colors = [f"C{k}" for k in range(n_major)]
    fig_pt, ax_pt = plt.subplots(figsize=(max(8, 1.4 * len(layers)), 6))
    bw = 0.8 / (n_major + 1)
    for k in range(n_major):
        deltas = [all_results[l]["per_task"][k]["delta"] for l in layers]
        off = (k - n_major / 2 + 0.5) * bw
        ax_pt.bar(x + off, deltas, bw, label=f"Task {k}",
                  color=task_colors[k], alpha=0.85)
        for i, v in enumerate(deltas):
            ax_pt.text(x[i] + off, v, f"{v:.3f}",
                       ha="center", va="bottom", fontsize=8)
    delta_ood = _extract("delta_loss_ood")
    off_ood = (n_major - n_major / 2 + 0.5) * bw
    ax_pt.bar(x + off_ood, delta_ood, bw, label="OOD",
              color="#FF9800", alpha=0.85)
    for i, v in enumerate(delta_ood):
        ax_pt.text(x[i] + off_ood, v, f"{v:.3f}",
                   ha="center", va="bottom", fontsize=8)
    _delta_mse = "\u0394 MSE"
    ax_pt.set(xlabel="Layer", ylabel=_delta_mse,
              title=f"Per-Task {_delta_mse} (rank-{n_directions}, scale={scale})")
    ax_pt.set_xticks(x, [str(l) for l in layers])
    ax_pt.xaxis.label.set_size(15)
    ax_pt.yaxis.label.set_size(15)
    ax_pt.title.set_size(15)
    ax_pt.tick_params(labelsize=14)
    ax_pt.legend(fontsize=12)
    ax_pt.grid(axis="y", alpha=0.3)
    _show_or_close(fig_pt, show)

    return fig, fig_loss, fig_r2, fig_filt, fig_pt, all_results
