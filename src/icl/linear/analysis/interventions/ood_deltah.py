"""OOD delta-h: ablation sweep over SVD components of OOD Δh with
automatic direction diagnostic for the most impactful component."""
import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger
from icl.linear.analysis.interventions._helpers import (
    _cleanup_model, _create_ood_task,
    _extract_baseline_per_position, _extract_hiddens_for_pool,
    _fit_task_subspace,
)

logger = setup_logger(__name__)


def intervene_remove_ood_deltah_subspace(
    exp_name: str,
    layer: int,
    n_ood: int = 30,
    B: int = 64,
    n_samples_eval: int = 500,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    baseline: str = "null_targets",
    min_position: int = 20,
    scale: float = 1.0,
    ablation_max_k: int = 10,
    eval_positions: Optional[list] = None,
    analyze_component: Optional[int] = None,
    show: bool = True,
    figsize: tuple = (14, 6),
    print_summary: bool = True,
    # legacy params (ignored, kept for backward compat)
    method: str = "ablation_sweep",
    alpha: float = 1.0,
    n_components: int = 2,
    **_kwargs,
) -> dict:
    """SVD ablation sweep on OOD Δh with 4-panel direction diagnostic.

    Pipeline:
      1. Δh = h − h_baseline  for major + OOD tasks
      2. SVD on batch-mean late-position OOD Δh  →  UΣVᵀ
      3. Ablation: remove vₖvₖᵀ (individual) and V_{1:k}V_{1:k}ᵀ (cumulative),
         measure causal impact on OOD MSE
      4. Auto-select k* = argmax (individual OOD MSE increase)
      5. 4-panel diagnostic on vₖ*: scatter vs oracle, R² bars,
         per-position r², joint regression with partial R²

    Override k* with ``analyze_component``.
    """
    if baseline not in ("null_targets", "position_0"):
        raise ValueError(
            f"baseline must be 'null_targets' or 'position_0', got {baseline!r}"
        )

    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.linear_ood_analysis import (
        _create_eval_task_pool,
        _setup_eval_task,
    )
    from icl.linear.task_vecs import extract_hidden_multi

    # ── 1. Load model & config ────────────────────────────────────────────
    _, train_task, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    D = int(config.model.n_embd)
    K = int(train_task.n_tasks)
    n_dims = int(config.task.n_dims)
    noise_scale = float(train_task.noise_scale)

    if step is None:
        step = config.training.total_steps
    if eval_positions is None:
        eval_positions = list(range(n_points))

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    task_pos = 2 * torch.arange(n_points, device=device)
    seq_pos_0 = torch.tensor([0], device=device)

    # ── 2. Eval pool [major, OOD] & extract hidden states ─────────────────
    eval_pool, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False,
        device=device, n_minor=0,
    )
    eval_task = _setup_eval_task(config, eval_pool, B, device)
    eval_task.batch_size = B
    n_tasks = eval_task.task_pool.shape[0]
    n_ood_actual = n_tasks - K

    demo_data = eval_task.sample_data(step=step).to(device)

    all_h, _ = _extract_hiddens_for_pool(
        model, eval_task, demo_data,
        step=step, layer=layer, task_pos=task_pos, D=D,
    )

    # ── 3. Baseline hidden states & Δh ────────────────────────────────────
    dummy_tgt = torch.zeros(B, n_points, device=device, dtype=demo_data.dtype)
    if baseline == "null_targets":
        h0 = extract_hidden_multi(
            model=model, demo_data=demo_data, demo_target=dummy_tgt,
            layers=[layer], task_pos=task_pos,
        )
        h_base = h0[0].permute(1, 0, 2).cpu()
        del h0
    else:
        h_base = _extract_baseline_per_position(
            model, demo_data, dummy_tgt,
            layer=layer, n_points=n_points, D=D,
        )

    delta_h = all_h - h_base.unsqueeze(0)       # (n_tasks, T, B, D)

    # ── 4. SVD on batch-mean late-position OOD Δh ─────────────────────────
    # Pool all OOD tasks' late-position Δh (averaged over batch) into a
    # single matrix  M ∈ ℝ^{(n_ood · n_late) × D}  and compute its SVD.
    # The right singular vectors V give the principal directions of
    # variation in OOD Δh; the singular values σₖ quantify the energy
    # along each direction.
    late = torch.arange(n_points) >= min_position
    dh_ood_late = delta_h[K:].float().mean(2)[:, late, :].reshape(-1, D)
    _, S_all, Vt_all = torch.linalg.svd(dh_ood_late, full_matrices=False)
    del dh_ood_late

    # ── 5. Ablation sweep (needs model for forward passes) ────────────────
    ood_task_eval = _create_ood_task(train_task, config, B, n_ood_actual, device)

    from icl.linear.lr_task import NoisyLinearRegression

    def _eval_with_proj(P_proj, task_obj, n_samples):
        """Causal intervention experiment.

        For each batch, compute three MSE curves (vs oracle wᵀxₜ):
          - baseline:   unmodified model output
          - intervened: h′ = h − s · hP  at the target layer (removes the
                        subspace spanned by columns of P)
          - pos0:       model fed zero targets (no in-context signal)
        """
        bl = {p: [] for p in eval_positions}
        iv = {p: [] for p in eval_positions}
        z0 = {p: [] for p in eval_positions}
        n_bat = max(1, (n_samples + B - 1) // B)
        saved_bs = int(task_obj.batch_size)
        task_obj.batch_size = B

        for bi in range(n_bat):
            dd, tb, dt = task_obj.sample_batch(step=bi + 77777, is_eval=True)
            dd, dt, tb = dd.to(device), dt.to(device), tb.to(device)
            oracle = NoisyLinearRegression.evaluate_oracle(dd, tb)

            with torch.no_grad():
                p_base = model(dd, dt)

            # Hook: replace h → h − s·(hP) = (I − sP) h  at the layer
            def hook_fn(module, inp, out, _P=P_proj, _s=scale):
                h = out if torch.is_tensor(out) else out[0]
                h_mod = h - _s * (h @ _P)
                return h_mod if torch.is_tensor(out) else (h_mod,) + out[1:]

            handle = model.transformer.blocks[layer].attn_block.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    p_int = model(dd, dt)
            finally:
                handle.remove()

            with torch.no_grad():
                p_z0 = model(dd, torch.zeros_like(dt))

            for p in eval_positions:
                if p >= p_base.shape[1]:
                    continue
                bl[p].append(((p_base[:, p] - oracle[:, p]) ** 2).mean().item())
                iv[p].append(((p_int[:, p] - oracle[:, p]) ** 2).mean().item())
                z0[p].append(((p_z0[:, p] - oracle[:, p]) ** 2).mean().item())

            del dd, dt, tb, oracle, p_base, p_int, p_z0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        task_obj.batch_size = saved_bs
        bp, ip, zp, vp = [], [], [], []
        for p in eval_positions:
            if bl[p]:
                bp.append(np.mean(bl[p]))
                ip.append(np.mean(iv[p]))
                zp.append(np.mean(z0[p]))
                vp.append(p)
        return {
            "baseline_per_pos": np.array(bp),
            "intervened_per_pos": np.array(ip),
            "pos0_per_pos": np.array(zp),
            "positions": np.array(vp),
        }

    max_k = min(ablation_max_k, len(S_all))
    indiv_maj, indiv_ood = [], []
    cumul_maj, cumul_ood = [], []
    base_maj = base_ood = pos0_maj = pos0_ood = None

    for k in range(max_k):
        logger.info(f"[ablation] component {k} ...")
        v_k = Vt_all[k:k+1].T.to(device)
        P_k = v_k @ v_k.T
        rm = _eval_with_proj(P_k, train_task, n_samples_eval)
        ro = _eval_with_proj(P_k, ood_task_eval, n_samples_eval)
        if base_maj is None:
            base_maj = float(np.mean(rm["baseline_per_pos"]))
            base_ood = float(np.mean(ro["baseline_per_pos"]))
            pos0_maj = float(np.mean(rm["pos0_per_pos"]))
            pos0_ood = float(np.mean(ro["pos0_per_pos"]))
        indiv_maj.append(float(np.mean(rm["intervened_per_pos"])))
        indiv_ood.append(float(np.mean(ro["intervened_per_pos"])))

        V_cum = Vt_all[:k+1].T.to(device)
        P_cum = V_cum @ V_cum.T
        cm = _eval_with_proj(P_cum, train_task, n_samples_eval)
        co = _eval_with_proj(P_cum, ood_task_eval, n_samples_eval)
        cumul_maj.append(float(np.mean(cm["intervened_per_pos"])))
        cumul_ood.append(float(np.mean(co["intervened_per_pos"])))

    _cleanup_model(model)
    del ood_task_eval
    gc.collect()

    # ── 6. Ablation sweep plot ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(max(14, 1.2 * max_k), 7))
    x = np.arange(max_k)
    bw = 0.35

    all_bar_vals = indiv_maj + indiv_ood + cumul_maj + cumul_ood
    y_data_max = max(max(all_bar_vals), base_maj, base_ood)
    y_lim = y_data_max * 1.25
    pos0_in_range = (pos0_ood <= y_lim)

    for ax, mj, od, title_str, xlabel in [
        (axes[0], indiv_maj, indiv_ood,
         "Individual component removal", "SVD component index"),
        (axes[1], cumul_maj, cumul_ood,
         "Cumulative component removal", "Remove components 0..k"),
    ]:
        ax.bar(x - bw/2, mj, bw, label="Major", color="#2196F3", alpha=0.85)
        ax.bar(x + bw/2, od, bw, label="OOD", color="#FF9800", alpha=0.85)
        ax.axhline(base_maj, color="#2196F3", ls="--", lw=1.5, alpha=0.6,
                   label=f"Major baseline ({base_maj:.4f})")
        ax.axhline(base_ood, color="#FF9800", ls="--", lw=1.5, alpha=0.6,
                   label=f"OOD baseline ({base_ood:.4f})")
        if pos0_in_range:
            ax.axhline(pos0_ood, color="#4CAF50", ls=":", lw=2, alpha=0.8,
                       label=f"OOD pos-0 MSE ({pos0_ood:.4f})")
        else:
            p0_lbl = f"OOD pos-0 MSE = {pos0_ood:.2f}"
            ax.annotate(
                p0_lbl, xy=(0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=11, color="#4CAF50",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#4CAF50", alpha=0.9),
            )
        ax.set(xlabel=xlabel, ylabel=r"MSE to oracle $w^\top x$ (avg)",
               title=title_str)
        ax.set_xticks(x)
        ax.set_ylim(0, y_lim)
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    for ax in axes.flat:
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.tick_params(labelsize=11)

    fig.suptitle("", fontsize=18, y=1.01)
    plt.tight_layout()
    if show:
        plt.show()

    if print_summary:
        sv_str = ", ".join(f"{s:.3f}" for s in S_all[:max_k].numpy())
        print(f"\n{'='*70}")
        print(f"SVD Ablation Sweep (layer {layer})")
        print(f"{'='*70}")
        print(f"  Singular values: [{sv_str}]")
        print(f"  {'k':>3}  {'SV':>8}  "
              f"{'Indiv Maj':>10} {'Indiv OOD':>10}  "
              f"{'Cumul Maj':>10} {'Cumul OOD':>10}")
        print("-" * 70)
        for k in range(max_k):
            print(f"  {k:>3}  {S_all[k].item():>8.3f}  "
                  f"{indiv_maj[k]:>10.4f} {indiv_ood[k]:>10.4f}  "
                  f"{cumul_maj[k]:>10.4f} {cumul_ood[k]:>10.4f}")
        print(f"\n  Baseline: Major={base_maj:.4f}, OOD={base_ood:.4f}")
        print(f"  Zero-tgt: Major={pos0_maj:.4f}, OOD={pos0_ood:.4f}")
        print(f"{'='*70}")

    # ── 7. Direction analysis ─────────────────────────────────────────────
    if analyze_component is None:
        analyze_component = int(np.argmax(indiv_ood))
        logger.info(
            f"[auto] component {analyze_component} has largest OOD impact "
            f"(MSE={indiv_ood[analyze_component]:.4f})"
        )

    analysis_out = _analyze_direction(
        exp_name=exp_name, layer=layer, step=step,
        delta_h=delta_h, eval_task=eval_task, demo_data=demo_data,
        S_all=S_all, Vt_all=Vt_all, late=late,
        K=K, n_ood=n_ood_actual, n_points=n_points, D=D, n_dims=n_dims,
        noise_scale=noise_scale, component_idx=analyze_component,
        fit_n_samples=fit_n_samples, fit_positions=fit_positions,
        center_task_vecs=center_task_vecs, min_position=min_position,
        show=show, print_summary=print_summary,
    )

    del delta_h, all_h, h_base, eval_task
    gc.collect()

    # ── 8. Return ─────────────────────────────────────────────────────────
    result = {
        "fig": fig,
        "S_all": S_all.cpu(),
        "Vt_all": Vt_all.cpu(),
        "layer": layer,
        "baseline": baseline,
        "method_info": {
            "method": "ablation_sweep",
            "singular_values": S_all.cpu(),
            "svd_Vt": Vt_all.cpu(),
            "indiv_mse_major": indiv_maj,
            "indiv_mse_ood": indiv_ood,
            "cumul_mse_major": cumul_maj,
            "cumul_mse_ood": cumul_ood,
            "base_mse_major": base_maj,
            "base_mse_ood": base_ood,
            "pos0_mse_major": pos0_maj,
            "pos0_mse_ood": pos0_ood,
        },
    }
    result.update(analysis_out)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Direction analysis  (4-panel diagnostic)
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_direction(
    *, exp_name, layer, step,
    delta_h, eval_task, demo_data,
    S_all, Vt_all, late,
    K, n_ood, n_points, D, n_dims, noise_scale,
    component_idx,
    fit_n_samples, fit_positions, center_task_vecs, min_position,
    show, print_summary,
) -> dict:
    """4-panel diagnostic for SVD component vₖ of OOD Δh.

    Panels:
      (a) Hexbin: vₖᵀΔh  vs  oracle wᵀxₜ  (per task × position × batch)
      (b) Per-feature R²: how well each covariate predicts vₖᵀΔh
      (c) Per-position r²(vₖᵀΔh, wᵀx)  for major vs OOD
      (d) Joint regression partial R²:  vₖᵀΔh ~ [wᵀx, XᵀY/t, xₜ]

    Also prints: task-overlap ‖P_S vₖ‖², variance fraction σₖ²/Σσ²,
    pairwise R² between covariates (collinearity check).
    """
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    if component_idx >= len(S_all):
        raise ValueError(
            f"analyze_component={component_idx} but only {len(S_all)} components"
        )

    # ── task subspace (for overlap metric) ────────────────────────────────
    _, _, basis, _, P_task, _ = _fit_task_subspace(
        exp_name=exp_name, layer=layer, step=step,
        fit_n_samples=fit_n_samples, fit_positions=fit_positions,
        n_points=n_points, center_task_vecs=center_task_vecs,
    )
    P_task = P_task.float()

    # ── Project Δh onto SVD direction vₖ ────────────────────────────
    v_k = Vt_all[component_idx].float()   # k-th right singular vector
    sv = float(S_all[component_idx])       # singular value σₖ
    # Variance fraction: σₖ² / Σᵢ σᵢ² — proportion of total energy
    var_frac = float(S_all[component_idx] ** 2 / (S_all ** 2).sum())
    # Task-subspace overlap: ‖P_S vₖ‖² / ‖vₖ‖² ∈ [0,1]
    # How much of vₖ lies inside the task subspace.
    # ≈0 → vₖ is orthogonal to known task directions.
    frac_task = float((P_task @ v_k).norm() ** 2 / v_k.norm() ** 2)

    logger.info(
        f"[analyse-dir] component {component_idx}: SV={sv:.3f}, "
        f"var_frac={var_frac:.4f}, task_overlap={frac_task:.4f}"
    )

    proj_all = torch.einsum("d,ktbd->ktb", v_k, delta_h.float())
    proj_ood = proj_all[K:]
    proj_maj = proj_all[:K]

    # ── Oracle and running ridge predictions (single pass) ──────────
    # Oracle prediction at position t:  ŷ_oracle = wᵀxₜ  (exact task vector)
    #
    # Running ridge regression uses the Bayesian posterior mean under a
    # Gaussian prior  w ~ N(0, (σ²/λ)I)  and  yₜ|w ~ N(wᵀxₜ, σ²):
    #
    #   ŵ_ridge(t) = (XᵀX + λI)⁻¹ XᵀY
    #
    # where X,Y are accumulated from positions 0…t.  The ridge
    # prediction at position t is  ŷ_ridge = ŵ_ridge(t)ᵀ xₜ.
    #
    # We also record the running sufficient statistic  XᵀY/(t+1)
    # for use in the joint regression panel (d).

    demo_cpu = demo_data.cpu().float()
    ood_w = eval_task.task_pool[K:].squeeze(-1).cpu().float()    # (n_ood, d)
    maj_w = eval_task.task_pool[:K].squeeze(-1).cpu().float()    # (K, d)
    oracle_ood = torch.einsum("kd,btd->kbt", ood_w, demo_cpu)   # wᵀx: (n_ood, B, T)

    all_tgt = torch.empty(eval_task.task_pool.shape[0], demo_data.shape[0],
                          n_points, dtype=torch.float32)
    for i in range(0, all_tgt.shape[0], 8):
        ce = min(i + 8, all_tgt.shape[0])
        dt = eval_task.evaluate(
            demo_data, eval_task.task_pool[i:ce].squeeze(-1).T, step=step,
        )
        if dt.ndim == 3:
            dt = dt.permute(2, 0, 1)
        all_tgt[i:ce] = dt.cpu().float()

    late_pos = torch.where(late)[0].tolist()
    n_late = len(late_pos)
    lam = noise_scale ** 2   # ridge penalty λ = σ²
    B_loc = demo_data.shape[0]

    ridge_pred = torch.zeros(n_ood, B_loc, n_late)
    suff_stat = torch.zeros(n_ood, B_loc, n_late, n_dims)
    w_hat_late = torch.zeros(n_ood, B_loc, n_late, n_dims)
    # Running accumulators: XᵀY and XᵀX updated at each position
    cum_XtY = torch.zeros(n_ood, B_loc, n_dims)
    cum_XtX = torch.zeros(n_ood, B_loc, n_dims, n_dims)
    li = 0
    for t in range(n_points):
        x_t = demo_cpu[:, t, :]   # (B, d)
        for ki in range(n_ood):
            y_t = all_tgt[K + ki, :, t]
            cum_XtY[ki] += x_t * y_t.unsqueeze(-1)                 # XᵀY += xₜ yₜ
            cum_XtX[ki] += torch.einsum("bi,bj->bij", x_t, x_t)   # XᵀX += xₜxₜᵀ
        if t in late_pos:
            for ki in range(n_ood):
                # ŵ = (XᵀX + λI)⁻¹ XᵀY
                reg = cum_XtX[ki] + lam * torch.eye(n_dims).unsqueeze(0).expand(B_loc, -1, -1)
                wh = torch.linalg.solve(reg, cum_XtY[ki].unsqueeze(-1)).squeeze(-1)
                ridge_pred[ki, :, li] = (wh * x_t).sum(-1)   # ŵᵀxₜ
                suff_stat[ki, :, li] = cum_XtY[ki] / (t + 1)
                w_hat_late[ki, :, li] = wh
            li += 1

    # ── flatten late-position data ────────────────────────────────────────
    proj_flat = proj_ood[:, late, :].numpy().reshape(-1)
    orac_flat = np.transpose(oracle_ood[:, :, late].numpy(), (0, 2, 1)).reshape(-1)
    ridge_flat = np.transpose(ridge_pred.numpy(), (0, 2, 1)).reshape(-1)

    r2_oracle = float(np.corrcoef(proj_flat, orac_flat)[0, 1] ** 2)
    r2_ridge = float(np.corrcoef(proj_flat, ridge_flat)[0, 1] ** 2)
    coef = np.polyfit(orac_flat, proj_flat, 1)

    # ── 4-panel figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("", fontsize=18)
    y_label = (r"$v_{" + str(component_idx) + r"}^\top \Delta h$")

    # (a) hexbin scatter
    ax = axes[0, 0]
    hb = ax.hexbin(orac_flat, proj_flat, gridsize=60, cmap="YlOrRd",
                   mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02).set_label("count", fontsize=11)
    xl = np.array([orac_flat.min(), orac_flat.max()])
    ax.plot(xl, np.polyval(coef, xl), "k-", lw=1.5, label=f"slope={coef[0]:.3f}")
    ax.set(xlabel=r"$w^\top x_t$", ylabel=y_label, title="")
    ax.legend(fontsize=11, loc="upper left")

    # (b) feature R² bar chart
    ax = axes[0, 1]
    ood_w_np = ood_w.numpy()
    x_late_np = demo_cpu[:, late, :].numpy()
    x_rep = np.tile(x_late_np[np.newaxis], (n_ood, 1, 1, 1))
    x_flat = np.transpose(x_rep, (0, 2, 1, 3)).reshape(-1, n_dims)
    w_rep = np.repeat(ood_w_np, n_late * B_loc, axis=0)
    suff_flat = suff_stat.numpy().transpose(0, 2, 1, 3).reshape(-1, n_dims)
    what_flat = w_hat_late.numpy().transpose(0, 2, 1, 3).reshape(-1, n_dims)

    def _lr_r2(X, y):
        return float(LinearRegression().fit(X, y).score(X, y))

    r2_feat = {
        r"$x_t$": _lr_r2(x_flat, proj_flat),
        r"$w_k$": _lr_r2(w_rep, proj_flat),
        r"$X^\top Y / t$": _lr_r2(suff_flat, proj_flat),
        r"$\hat w_{\rm ridge}$": _lr_r2(what_flat, proj_flat),
        r"$[w_k, x_t]$": _lr_r2(np.c_[w_rep, x_flat], proj_flat),
        r"$w^\top x$": _lr_r2(orac_flat[:, None], proj_flat),
    }
    sorted_f = sorted(r2_feat.items(), key=lambda kv: kv[1])
    labels, vals = zip(*sorted_f)
    colors = ["tab:green" if v > 0.5 else "tab:gray" for v in vals]
    ax.barh(range(len(labels)), vals, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlim(0, 1.08)
    ax.set(xlabel=r"$R^2$", title="")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=12)

    # (c) per-position r²
    ax = axes[1, 0]
    maj_oracle = torch.einsum("kd,btd->kbt", maj_w, demo_cpu)
    r2_ood_t, r2_maj_t = [], []
    for t in range(n_points):
        for proj_t, orac_t, out in [
            (proj_ood[:, t, :], oracle_ood[:, :, t], r2_ood_t),
            (proj_maj[:, t, :], maj_oracle[:, :, t], r2_maj_t),
        ]:
            a, b = proj_t.numpy().ravel(), orac_t.numpy().ravel()
            out.append(0.0 if a.std() < 1e-8 or b.std() < 1e-8
                       else float(np.corrcoef(a, b)[0, 1] ** 2))

    ax.plot(range(n_points), r2_ood_t, color="tab:red", lw=1.5, label="OOD")
    ax.plot(range(n_points), r2_maj_t, color="tab:blue", lw=1.5, label="Major",
            alpha=0.7)
    ax.axvline(min_position, color="gray", ls="--", alpha=0.5)
    ax.set(xlabel="Position $t$",
           ylabel=r"$r^2$(proj, $w^\top x_t$)",
           title="",
           ylim=(-0.02, 1.05))
    ax.legend(fontsize=11)

    # (d) Joint regression with partial R²
    # Fit:  vₖᵀΔh  ~  [wᵀxₜ,  XᵀY/t,  xₜ]  (all covariates jointly)
    # Then measure partial R² for each group: the fraction of residual
    # variance that only that group can explain, after partialling out
    # the other two (see probes.py for the general formula).
    ax = axes[1, 1]
    X_full = np.c_[orac_flat[:, None], suff_flat, x_flat]
    r2_full = _lr_r2(X_full, proj_flat)

    X_no_wtx = np.c_[suff_flat, x_flat]
    X_no_suff = np.c_[orac_flat[:, None], x_flat]
    X_no_xt = np.c_[orac_flat[:, None], suff_flat]
    r2_no_wtx = _lr_r2(X_no_wtx, proj_flat)
    r2_no_suff = _lr_r2(X_no_suff, proj_flat)
    r2_no_xt = _lr_r2(X_no_xt, proj_flat)

    partial_r2 = {
        r"$w^\top x_t$": (r2_full - r2_no_wtx) / max(1 - r2_no_wtx, 1e-12),
        r"$X^\top Y / t$": (r2_full - r2_no_suff) / max(1 - r2_no_suff, 1e-12),
        r"$x_t$": (r2_full - r2_no_xt) / max(1 - r2_no_xt, 1e-12),
    }
    labels_d = list(partial_r2.keys())
    vals_d = list(partial_r2.values())
    colors_d = ["#E53935", "#1E88E5", "#43A047"]

    ax.barh(range(len(labels_d)), vals_d, color=colors_d, alpha=0.85,
            edgecolor="white", linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(labels_d)))
    ax.set_yticklabels(labels_d, fontsize=14)
    ax.set_xlim(0, 1.08)
    ax.set(xlabel=r"Partial $R^2$", title="")
    for i, v in enumerate(vals_d):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=13)

    # ── global font-size pass ─────────────────────────────────────────────
    for _ax in axes.flat:
        _ax.title.set_fontsize(14)
        _ax.xaxis.label.set_fontsize(13)
        _ax.yaxis.label.set_fontsize(13)
        _ax.tick_params(labelsize=11)
    for lbl in axes[0, 1].get_yticklabels():
        lbl.set_fontsize(13)
    for lbl in axes[1, 1].get_yticklabels():
        lbl.set_fontsize(14)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if show:
        plt.show()

    # ── Covariate collinearity diagnostics ─────────────────────────
    # These pairwise R² values show how redundant the three covariate
    # groups are.  High values mean the covariates share information
    # and their partial R² will be lower than their marginal R².
    #   R²(wᵀx ← XᵀY/t)   : can the sufficient stat predict the oracle?
    #   R²(wᵀx ← xₜ)      : can the current token alone predict the oracle?
    #   R²(XᵀY/t ← xₜ)    : do the suff stats depend on the current token?
    r2_wtx_from_suff = _lr_r2(suff_flat, orac_flat)
    r2_wtx_from_xt = _lr_r2(x_flat, orac_flat)
    r2_suff_from_xt = float(np.mean([
        _lr_r2(x_flat, suff_flat[:, j])
        for j in range(suff_flat.shape[1])
    ]))
    covariate_diag = {
        "r2_wtx_from_suff": r2_wtx_from_suff,
        "r2_wtx_from_xt": r2_wtx_from_xt,
        "r2_suff_from_xt_avg": r2_suff_from_xt,
    }

    # ── summary ───────────────────────────────────────────────────────────
    ood_proj_late_mean = proj_ood[:, late, :].mean(dim=(1, 2)).numpy()
    r2_w = float(LinearRegression().fit(ood_w_np, ood_proj_late_mean)
                 .score(ood_w_np, ood_proj_late_mean))

    summary = {
        "component_idx": component_idx,
        "singular_value": sv,
        "var_fraction": var_frac,
        "frac_in_task_subspace": frac_task,
        "r2_w": r2_w,
        "r2_oracle": r2_oracle,
        "r2_ridge": r2_ridge,
        "feature_r2": r2_feat,
        "joint_r2_full": r2_full,
        "partial_r2": dict(zip(
            ["wtx", "suff_stat", "x_t"], vals_d,
        )),
        "covariate_diag": covariate_diag,
    }

    if print_summary:
        _r2 = "R\u00b2"
        _r2l = "r\u00b2"
        print(f"\n{'='*70}")
        print(f"SVD Component {component_idx}  (layer {layer})")
        print(f"{'='*70}")
        print(f"  Singular value       : {sv:.3f}")
        print(f"  Variance fraction    : {var_frac:.4f}")
        print(f"  Task-subspace overlap: {frac_task:.4f}")
        print(f"  {_r2} (proj ~ w_k)     : {r2_w:.4f}")
        print(f"  {_r2l} (proj ~ w^Tx)    : {r2_oracle:.4f}")
        print(f"  {_r2l} (proj ~ ridge)   : {r2_ridge:.4f}")
        print(f"\n  Joint regression  proj ~ [w^Tx, X^TY/t, x_t]:")
        print(f"    Full {_r2}              : {r2_full:.4f}")
        print(f"    Partial {_r2}(w^Tx)      : {vals_d[0]:.4f}")
        print(f"    Partial {_r2}(X^TY/t)    : {vals_d[1]:.4f}")
        print(f"    Partial {_r2}(x_t)       : {vals_d[2]:.4f}")
        print(f"\n  Covariate relationships:")
        print(f"    {_r2}(w^Tx <- X^TY/t)   : {r2_wtx_from_suff:.4f}")
        print(f"    {_r2}(w^Tx <- x_t)      : {r2_wtx_from_xt:.4f}")
        print(f"    {_r2}(X^TY/t <- x_t) avg: {r2_suff_from_xt:.4f}")
        print(f"\n  Feature {_r2}:")
        for feat, val in sorted(r2_feat.items(), key=lambda kv: -kv[1]):
            tag = "***" if val > 0.5 else "**" if val > 0.1 else ""
            print(f"    {feat:30s}: {val:.4f}  {tag}")
        print(f"{'='*70}")

    return {
        "analysis_fig": fig,
        "analysis_summary": summary,
        "analysis_v_k": v_k.cpu(),
        "P_task": P_task.cpu(),
        "analysis_basis": basis.cpu(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compat wrapper
# ═══════════════════════════════════════════════════════════════════════════

def analyze_ood_deltah_direction(
    exp_name: str,
    layer: int,
    component_idx: int = 3,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    baseline: str = "null_targets",
    min_position: int = 20,
    show: bool = True,
    figsize: tuple = (14, 10),
    print_summary: bool = True,
    precomputed: Optional[dict] = None,
) -> dict:
    """Backward-compat wrapper.  Delegates to
    ``intervene_remove_ood_deltah_subspace(analyze_component=...)``."""
    res = intervene_remove_ood_deltah_subspace(
        exp_name=exp_name, layer=layer, n_ood=n_ood, B=B,
        step=step, baseline=baseline, min_position=min_position,
        fit_n_samples=fit_n_samples, fit_positions=fit_positions,
        center_task_vecs=center_task_vecs,
        analyze_component=component_idx,
        show=show, print_summary=print_summary,
    )
    return {
        "fig": res.get("analysis_fig"),
        "summary": res.get("analysis_summary", {}),
        "v_k": res.get("analysis_v_k"),
        "S_all": res.get("S_all"),
        "Vt_all": res.get("Vt_all"),
        "P_task": res.get("P_task"),
        "basis": res.get("analysis_basis"),
        "layer": layer,
        "component_idx": component_idx,
    }
