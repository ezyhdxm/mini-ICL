"""Direction analysis and subspace decomposition for linear regression."""

import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger
from icl.linear.analysis._helpers import _show_or_close, _task_positions
from icl.linear.analysis.interventions._helpers import (
    _extract_baseline_per_position,
    _extract_hiddens_for_pool,
    _fit_task_subspace,
)

logger = setup_logger(__name__)


def calculation_direction_analysis(
    exp_name: str,
    layer_index: Optional[int] = None,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    baseline: str = "null_targets",
    show: bool = True,
    figsize: tuple = (18, 14),
    title: str = "",
) -> dict:
    """Decompose Δh into task-subspace and orthogonal components.

    1. Probe: h ≈ πW + b   →  W ∈ ℝ^{K×D}
       W̃ = W − w̄1ᵀ,   SVD → V_r,   P_S = V_r V_rᵀ

    2. Baseline subtraction:
       "null_targets":  Δhₜ = h(xₜ) − h_null(xₜ)   (zero targets)
       "position_0":    Δhₜ = h(xₜ) − h₀(xₜ)       (no context)

    3. Decompose:  Δh = P_S Δh  +  (I−P_S) Δh  =  Δh_task + Δh_⊥

    Panels:
      (a) ‖Δh‖², ‖Δh_task‖², ‖Δh_⊥‖²  over positions
      (b) ‖Δh_task‖²/‖Δh‖²  (task-subspace fraction, mean±std)
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
        setup_device,
    )
    from icl.linear.task_vecs import extract_hidden_multi

    # ---- 1. Load config & model ------------------------------------------------
    _, train_task, config = load_model_task_config(exp_name)
    n_layers = config.model.n_layer
    n_points = int(config.task.n_points)
    D = int(config.model.n_embd)
    K_major = int(train_task.n_tasks)

    if layer_index is None:
        layer_index = n_layers - 1
    if step is None:
        step = config.training.total_steps
    if fit_positions is None:
        fit_positions = list(range(max(0, n_points - 20), n_points))

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    task_pos = 2 * torch.arange(n_points, device=device)
    seq_pos_0 = torch.tensor([0], device=device)

    # ---- 2. Fit task subspace ---------------------------------------------------
    W, b, basis, rank, P_task, fit_res = _fit_task_subspace(
        exp_name=exp_name, layer=layer_index, step=step,
        fit_n_samples=fit_n_samples, fit_positions=fit_positions,
        n_points=n_points, center_task_vecs=center_task_vecs,
    )
    P_task = P_task.to(device)

    logger.info(
        f"[calc-dir] Task subspace rank={rank}, D={D}, "
        f"chance fraction={rank / D:.4f}, joint fit R2={fit_res['val_r2']:.4f}"
    )

    # ---- 3. Eval task pool: [major, OOD] ----------------------------------------
    eval_pool, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False,
        device=device, n_minor=0,
    )
    eval_task = _setup_eval_task(config, eval_pool, B, device)
    eval_task.batch_size = B
    n_tasks = eval_task.task_pool.shape[0]
    n_ood_actual = n_tasks - K_major

    # ---- 4. Shared demo data ----------------------------------------------------
    demo_data = eval_task.sample_data(step=step).to(device)  # (B, T, d)

    # ---- 5. h(x_t) for every task -----------------------------------------------
    hiddens, _ = _extract_hiddens_for_pool(
        model, eval_task, demo_data,
        step=step, layer=layer_index, task_pos=task_pos, D=D,
    )

    # ---- 6. Baseline hidden representations (task-independent) -----------------
    dummy_targets = torch.zeros(B, n_points, device=device, dtype=demo_data.dtype)

    if baseline == "null_targets":
        h_null = extract_hidden_multi(
            model=model, demo_data=demo_data, demo_target=dummy_targets,
            layers=[layer_index], task_pos=task_pos,
        )  # (1, B, n_points, D)
        h_baseline = h_null[0].permute(1, 0, 2).cpu()  # (n_points, B, D)
        del h_null
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:  # "position_0"
        h_baseline = _extract_baseline_per_position(
            model, demo_data, dummy_targets,
            layer=layer_index, n_points=n_points, D=D,
        )

    delta_h = hiddens - h_baseline.unsqueeze(0)  # (K, T, B, D)

    # ── Project Δh onto task subspace and its complement ────────────
    # P_S = V_r V_rᵀ   is the orthogonal projector onto the task subspace.
    #
    #   Δh_task = P_S Δh     (component in the task subspace)
    #   Δh_⊥   = (I − P_S) Δh (component orthogonal to the task subspace)
    #
    # Pythagorean decomposition:  ‖Δh‖² = ‖Δh_task‖² + ‖Δh_⊥‖²
    # The fraction  ‖Δh_task‖² / ‖Δh‖²  measures how much of the
    # representation change is captured by the known task directions.
    # Chance level for a rank-r projector in ℝ^D is r/D.
    P_cpu = P_task.cpu().float()
    dh_task = torch.einsum("ktbd,de->ktbe", delta_h, P_cpu)  # P_S Δh: (K, T, B, D)
    dh_orth = delta_h - dh_task                               # (I−P_S) Δh

    task_nsq = (dh_task ** 2).sum(-1)    # ‖Δh_task‖²: (K, T, B)
    orth_nsq = (dh_orth ** 2).sum(-1)    # ‖Δh_⊥‖²
    total_nsq = (delta_h ** 2).sum(-1)   # ‖Δh‖²

    task_nm = task_nsq.mean(-1).numpy()   # batch-averaged: (K, T)
    orth_nm = orth_nsq.mean(-1).numpy()
    total_nm = total_nsq.mean(-1).numpy()
    frac_task_all = (task_nsq / (total_nsq + 1e-12)).numpy()  # (K, T, B)
    frac_task = frac_task_all.mean(-1)     # (K, T)
    frac_task_std = frac_task_all.std(-1)  # (K, T)

    # ---- 9. Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    t_axis = np.arange(n_points)
    major_colors = ["#1f77b4", "#2ca02c", "#d62728"]
    ood_color = "#ff7f0e"
    chance = rank / D

    # (a) Norm decomposition
    ax = axes[0]
    for label, grp, color in [
        ("Major", slice(0, K_major), "#1f77b4"),
        ("OOD", slice(K_major, None), ood_color),
    ]:
        if total_nm[grp].shape[0] == 0:
            continue
        ax.plot(t_axis, total_nm[grp].mean(0), "-", color=color, lw=2,
                label=f"{label} $||\\Delta h||^2$")
        ax.plot(t_axis, task_nm[grp].mean(0), "--", color=color, lw=2,
                label=f"{label} $||\\Delta h_{{\\mathrm{{task}}}}||^2$")
        ax.plot(t_axis, orth_nm[grp].mean(0), ":", color=color, lw=2,
                label=f"{label} $||\\Delta h_{{\\perp}}||^2$")
    ax.set_xlabel("Position $t$", fontsize=13)
    ax.set_ylabel("Squared norm (batch avg)", fontsize=13)
    bl_tag = "h_{\\mathrm{null}}" if baseline == "null_targets" else "h_0"
    ax.set_title(f"Norm decomposition of $\\Delta h = h(x_t) - {bl_tag}(x_t)$",
                 fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # (b) Fraction in task subspace
    ax = axes[1]
    for k in range(K_major):
        c = major_colors[k % len(major_colors)]
        mu_k = frac_task[k]
        sd_k = frac_task_std[k]
        ax.plot(t_axis, mu_k, color=c, lw=2, label=f"Major {k}")
        ax.fill_between(t_axis, mu_k - sd_k, mu_k + sd_k, color=c, alpha=0.12)
    if n_ood_actual > 0:
        ood_mu = frac_task[K_major:].mean(0)
        ood_sd = frac_task[K_major:].std(0)
        ax.plot(t_axis, ood_mu, color=ood_color, lw=2, label="OOD (mean)")
        ax.fill_between(t_axis, ood_mu - ood_sd, ood_mu + ood_sd,
                        color=ood_color, alpha=0.15)
    ax.axhline(chance, color="gray", ls="--", alpha=0.5,
               label=f"Chance ({rank}/{D})")
    ax.set_xlabel("Position $t$", fontsize=13)
    ax.set_ylabel("$\\|\\Delta h_{\\mathrm{task}}\\|^2"
                   " / \\|\\Delta h\\|^2$", fontsize=13)
    ax.set_title("Fraction of $\\Delta h$ in task subspace", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)

    bl_label = "h_{\\mathrm{null}}" if baseline == "null_targets" else "h_0"
    sup = title or (
        f"Calculation-direction analysis (layer {layer_index}, "
        f"baseline={baseline}):  "
        f"$\\Delta h = h(x_t) - {bl_label}(x_t)$"
    )
    fig.suptitle(sup, fontsize=15, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()

    # ---- cleanup ----------------------------------------------------------------
    model.cpu()
    del model, eval_task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig,
        "axes": axes,
        "delta_h": delta_h,
        "hiddens": hiddens,
        "h_baseline": h_baseline,
        "baseline": baseline,
        "fraction_task": frac_task,
        "task_norm_sq_mean": task_nm,
        "orth_norm_sq_mean": orth_nm,
        "total_norm_sq_mean": total_nm,
        "W": W,
        "b": b,
        "P_task": P_task.cpu(),
        "rank": rank,
        "K_major": K_major,
        "n_ood": n_ood_actual,
        "fit_results": fit_res,
    }


def analyze_ood_deltah_direction(
    *args, **kwargs,
) -> dict:
    """Backward-compat wrapper; implementation moved to
    ``icl.linear.analysis.interventions.ood_deltah``."""
    from icl.linear.analysis.interventions.ood_deltah import (
        analyze_ood_deltah_direction as _impl,
    )
    return _impl(*args, **kwargs)


def plot_task_subspace_r2_over_positions(
    exp_name: str,
    layer: int,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    show: bool = True,
    figsize: tuple = (8, 5),
) -> dict:
    """Task-subspace energy fraction over positions.

    For projector P_S = V_r V_rᵀ, compute per sequence (k,b) at position t:

        h′ = h − b_probe   (if center_task_vecs, else h′ = h)
        f_{k,t,b} = ‖P_S h′‖² / ‖h′‖²

    Averaged over batch → f̄_{k,t}.  Chance level = r/D.
    """
    import gc
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.linear_ood_analysis import (
        _create_eval_task_pool,
        _setup_eval_task,
    )
    # ---- 1. Load config & model --------------------------------------------------
    _, train_task, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    D = int(config.model.n_embd)
    K_major = int(train_task.n_tasks)

    if step is None:
        step = config.training.total_steps
    if fit_positions is None:
        fit_positions = list(range(max(0, n_points - 20), n_points))

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    task_pos = 2 * torch.arange(n_points, device=device)

    # ---- 2. Fit task subspace ----------------------------------------------------
    W_fit, b_fit, _, rank, P_task, fit_res = _fit_task_subspace(
        exp_name=exp_name, layer=layer, step=step,
        fit_n_samples=fit_n_samples, fit_positions=fit_positions,
        n_points=n_points, center_task_vecs=center_task_vecs,
    )
    P_task = P_task.float()

    logger.info(
        f"[task-subspace-r2] center={center_task_vecs}, "
        f"rank={rank}, D={D}, chance={rank / D:.4f}"
    )

    # ---- 3. Eval task pool: [major, OOD] -----------------------------------------
    eval_pool, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False,
        device=device, n_minor=0,
    )
    eval_task = _setup_eval_task(config, eval_pool, B, device)
    eval_task.batch_size = B
    n_tasks = eval_task.task_pool.shape[0]
    n_ood_actual = n_tasks - K_major

    # ---- 4. Extract hidden representations --------------------------------------
    demo_data = eval_task.sample_data(step=step).to(device)

    all_hiddens, _ = _extract_hiddens_for_pool(
        model, eval_task, demo_data,
        step=step, layer=layer, task_pos=task_pos, D=D,
    )

    model.cpu()
    del model, demo_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Per-sequence projection fraction ───────────────────────────
    # For each hidden vector h′ (optionally centred by subtracting
    # the probe bias b), compute:
    #
    #   f = ‖P_S h′‖² / ‖h′‖²   ∈ [0, 1]
    #
    # This is the fraction of ‖h′‖² that lies in the task subspace.
    P_cpu = P_task.cpu().float()

    h_all = all_hiddens.float()  # (n_tasks, n_points, B, D)
    if center_task_vecs:
        h_all = h_all - b_fit.cpu().unsqueeze(0).unsqueeze(0).unsqueeze(0)

    h_flat = h_all.reshape(n_tasks * n_points * B, D)
    h_proj = h_flat @ P_cpu                            # P_S h′
    norms_sq = (h_flat ** 2).sum(dim=1)                # ‖h′‖²
    proj_sq = (h_proj ** 2).sum(dim=1)                 # ‖P_S h′‖²
    safe = norms_sq > 0
    frac_flat = torch.zeros_like(norms_sq)
    frac_flat[safe] = proj_sq[safe] / norms_sq[safe]
    frac_all = frac_flat.reshape(n_tasks, n_points, B).numpy()

    r2_per_task = frac_all.mean(axis=2)              # (n_tasks, n_points)
    r2_per_task_std = frac_all.std(axis=2)           # (n_tasks, n_points)

    r2_ood_mean = r2_per_task[K_major:].mean(axis=0)
    r2_ood_std = r2_per_task[K_major:].std(axis=0)
    chance = rank / D

    # ---- 6. Plot -----------------------------------------------------------------
    major_colors = ["tab:blue", "tab:green", "tab:purple", "tab:cyan",
                    "tab:olive", "tab:brown", "tab:pink", "tab:gray"]
    fig, ax = plt.subplots(figsize=figsize)
    positions_arr = np.arange(n_points)

    for k in range(K_major):
        c = major_colors[k % len(major_colors)]
        ax.plot(positions_arr, r2_per_task[k], color=c, linewidth=1.5,
                label=f"Major {k}")
        ax.fill_between(positions_arr,
                        r2_per_task[k] - r2_per_task_std[k],
                        r2_per_task[k] + r2_per_task_std[k],
                        color=c, alpha=0.1)

    ax.plot(positions_arr, r2_ood_mean, color="tab:red", linewidth=1.5,
            label="OOD (mean)")
    ax.fill_between(positions_arr,
                    r2_ood_mean - r2_ood_std,
                    r2_ood_mean + r2_ood_std,
                    color="tab:red", alpha=0.15)
    ax.axhline(chance, color="gray", ls="--", alpha=0.6,
               label=f"Chance ({rank}/{D} = {chance:.4f})")
    ax.set_xlabel("Position $t$")
    ax.set_ylabel(r"$\| P_{\mathrm{task}}\, h \|^2 \;/\; \| h \|^2$"
                  "  (per sequence, mean over batch)")
    centered_tag = ", centered" if center_task_vecs else ""
    ax.set_title(
        f"Fraction of hidden state in major task subspace\n"
        f"(layer {layer}, rank {rank}{centered_tag})"
    )
    y_max = max(r2_per_task[:K_major].max(),
                (r2_ood_mean + r2_ood_std).max())
    ax.set_ylim(0, min(1.0, y_max * 1.15))
    ax.legend(fontsize=9)
    fig.tight_layout()
    if show:
        plt.show()

    del all_hiddens
    gc.collect()

    return {
        "fig": fig,
        "r2_per_task": r2_per_task,
        "r2_per_task_std": r2_per_task_std,
        "r2_ood_mean": r2_ood_mean,
        "r2_ood_std": r2_ood_std,
        "frac_all": frac_all,
        "chance": chance,
        "rank": rank,
        "layer": layer,
        "K_major": K_major,
    }
