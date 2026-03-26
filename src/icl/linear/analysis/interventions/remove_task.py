"""Causal intervention: remove task subspace from hidden states."""
import gc
from typing import Optional

import numpy as np
import torch

from icl.utils.logger import setup_logger
from icl.linear.analysis.interventions._helpers import (
    _load_and_prepare_model,
    _create_ood_task,
    _cleanup_model,
    _fit_task_subspace,
    _run_projection_removal,
    _sweep_layers,
)
from icl.linear.analysis._helpers import _show_or_close

logger = setup_logger(__name__)


def intervene_remove_task_subspace(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_eval: int = 500,
    n_ood: int = 30,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    eval_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    scale: float = 1.0,
    verbose: bool = False,
    print_summary: bool = True,
    extraction_point: str = "post_attn",
    probe_method: str = "ols",
) -> dict:
    """
    Causal intervention: remove the fitted task-subspace component from
    hidden states and measure the effect on prediction MSE.

    Linear regression counterpart of
    ``intervene_remove_task_subspace_latent_nonpadded``.
    """
    from icl.linear.analysis.interventions._helpers import _averaging_fit_task_token
    model, train_task, config, device = _load_and_prepare_model(exp_name, step)
    if step is None:
        step = config.training.total_steps

    n_points = int(config.task.n_points)

    ood_task = _create_ood_task(train_task, config, B, n_ood, device)

    # ---- 1. Fit task subspace ----
    if probe_method == "averaging":
        avg_res = _averaging_fit_task_token(
            model, layer, train_task,
            device=device,
            fit_positions=fit_positions if fit_positions is not None
                else list(range(min(10, n_points), n_points)),
            fit_n_samples=fit_n_samples,
            B=B,
            n_dims=int(config.task.n_dims),
            n_embd=int(config.model.n_embd),
            center_task_vecs=center_task_vecs,
            extraction_point=extraction_point,
        )
        rank = avg_res["rank"]
        basis = avg_res["basis"]  # (D, rank)
        P_task = (basis @ basis.T).to(device)
        if verbose:
            logger.info(
                f"[remove-task linear] Task subspace rank={rank} "
                f"(averaging, centered={center_task_vecs}), "
                f"R²={avg_res['joint_r2']:.4f}"
            )
    else:
        _, _, _, rank, P_task, fit_res = _fit_task_subspace(
            exp_name=exp_name,
            layer=layer,
            step=step,
            fit_n_samples=fit_n_samples,
            fit_positions=fit_positions,
            n_points=n_points,
            center_task_vecs=center_task_vecs,
            extraction_point=extraction_point,
        )
        P_task = P_task.to(device)
        if verbose:
            logger.info(
                f"[remove-task linear] Task subspace rank={rank} "
                f"(centered={center_task_vecs}), joint fit R\u00b2={fit_res['val_r2']:.4f}"
            )

    # ---- 2. Intervention experiment ----
    if eval_positions is None:
        eval_positions = list(range(n_points))

    if verbose:
        logger.info("[remove-task linear] Running major experiment ...")
    res_major = _run_projection_removal(
        model, layer, P_task, scale, train_task, eval_positions,
        n_samples_eval, B, device,
        minor_only=False, track_oracle=False, extraction_point=extraction_point,
    )

    if verbose:
        logger.info("[remove-task linear] Running OOD experiment ...")
    res_ood = _run_projection_removal(
        model, layer, P_task, scale, ood_task, eval_positions,
        n_samples_eval, B, device,
        minor_only=False, track_oracle=False, extraction_point=extraction_point,
    )

    has_minor = (
        train_task.n_minor_tasks > 0 and train_task.minor_pool is not None
    )
    res_minor = None
    if has_minor:
        if verbose:
            logger.info("[remove-task linear] Running minor experiment ...")
        res_minor = _run_projection_removal(
            model, layer, P_task, scale, train_task, eval_positions,
            n_samples_eval, B, device,
            minor_only=True, track_oracle=False, extraction_point=extraction_point,
        )

    _cleanup_model(model)

    pct_major = (
        100.0 * res_major["delta"] / res_major["baseline"]
        if res_major["baseline"] > 0 else float("nan")
    )
    pct_ood = (
        100.0 * res_ood["delta"] / res_ood["baseline"]
        if res_ood["baseline"] > 0 else float("nan")
    )

    results = {
        "baseline_loss_major": res_major["baseline"],
        "intervened_loss_major": res_major["intervened"],
        "delta_loss_major": res_major["delta"],
        "pct_increase_major": pct_major,
        "baseline_loss_ood": res_ood["baseline"],
        "intervened_loss_ood": res_ood["intervened"],
        "delta_loss_ood": res_ood["delta"],
        "pct_increase_ood": pct_ood,
        "baseline_per_pos_major": res_major["baseline_per_pos"],
        "intervened_per_pos_major": res_major["intervened_per_pos"],
        "baseline_per_pos_ood": res_ood["baseline_per_pos"],
        "intervened_per_pos_ood": res_ood["intervened_per_pos"],
        "baseline_loss_at_0_major": res_major["baseline_loss_at_0"],
        "baseline_loss_at_0_ood": res_ood["baseline_loss_at_0"],
        "eval_positions": res_major["positions"],
        "layer": layer,
        "scale": scale,
        "task_subspace_rank": rank,
        "has_minor": has_minor,
    }

    if res_minor is not None:
        pct_minor = (
            100.0 * res_minor["delta"] / res_minor["baseline"]
            if res_minor["baseline"] > 0 else float("nan")
        )
        results.update({
            "baseline_loss_minor": res_minor["baseline"],
            "intervened_loss_minor": res_minor["intervened"],
            "delta_loss_minor": res_minor["delta"],
            "pct_increase_minor": pct_minor,
            "baseline_per_pos_minor": res_minor["baseline_per_pos"],
            "intervened_per_pos_minor": res_minor["intervened_per_pos"],
            "baseline_loss_at_0_minor": res_minor["baseline_loss_at_0"],
        })

    if print_summary:
        print(f"\n{'=' * 65}")
        print(
            f"Causal Intervention: Remove Task Subspace (linear) "
            f"(layer {layer}, scale={scale})"
        )
        print(f"{'=' * 65}")
        print(f"  Task subspace rank: {rank}")
        print(f"  Eval positions: {len(res_major['positions'])} positions\n")
        header = f"{'Metric':<30} {'Major':>12} {'OOD':>12}"
        sep_len = 54
        if has_minor:
            header += f" {'Minor':>12}"
            sep_len += 13
        print(header)
        print("-" * sep_len)
        row_bl = (f"{'Baseline MSE':<30} "
                  f"{res_major['baseline']:>12.4f} "
                  f"{res_ood['baseline']:>12.4f}")
        row_iv = (f"{'Intervened MSE':<30} "
                  f"{res_major['intervened']:>12.4f} "
                  f"{res_ood['intervened']:>12.4f}")
        row_dl = (f"{'Delta MSE':<30} "
                  f"{res_major['delta']:>12.4f} "
                  f"{res_ood['delta']:>12.4f}")
        row_pc = (f"{'Percent increase':<30} "
                  f"{pct_major:>11.1f}% "
                  f"{pct_ood:>11.1f}%")
        if has_minor:
            row_bl += f" {res_minor['baseline']:>12.4f}"
            row_iv += f" {res_minor['intervened']:>12.4f}"
            row_dl += f" {res_minor['delta']:>12.4f}"
            row_pc += f" {pct_minor:>11.1f}%"
        print(row_bl)
        print(row_iv)
        print(row_dl)
        print(row_pc)

    return results


def plot_intervention_remove_task_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    figsize: tuple = (14, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
):
    """
    Sweep ``intervene_remove_task_subspace`` across layers.

    Plots delta-MSE (intervened - baseline) for major, (minor), and OOD.

    **kwargs
        Forwarded to ``intervene_remove_task_subspace``
        (e.g. ``B``, ``scale``, ``fit_positions``, etc.).

    Returns ``(fig, all_results)``.
    """
    import matplotlib.pyplot as plt

    layers, all_results = _sweep_layers(
        exp_name,
        intervene_remove_task_subspace,
        layers,
        log_prefix="task-removal sweep linear",
        **kwargs,
    )

    def _extract_delta_rmse(baseline_key, intervened_key):
        return [
            np.sqrt(all_results[l][intervened_key]) - np.sqrt(all_results[l][baseline_key])
            for l in layers
        ]

    has_minor = all_results[layers[0]].get("has_minor", False)

    COLORS_L = {"maj": "#2166ac", "ood": "#d6604d", "minor": "#1a9850"}
    bw_bar_l = 0.22
    g_step_l = 0.24

    # Offsets: maj left, ood centre, minor right (if present); otherwise maj left, ood right
    if has_minor:
        MC_L = {"maj": -g_step_l, "ood": 0.0, "minor": +g_step_l}
    else:
        MC_L = {"maj": -g_step_l / 2, "ood": +g_step_l / 2}

    # Gains in RMSE units: √MSE_0 − √MSĒ
    ref = all_results[layers[0]]
    maj_info_gain = np.sqrt(ref["baseline_loss_at_0_major"]) - np.sqrt(ref["baseline_loss_major"])
    ood_info_gain = np.sqrt(ref["baseline_loss_at_0_ood"])   - np.sqrt(ref["baseline_loss_ood"])
    min_info_gain = (
        np.sqrt(ref["baseline_loss_at_0_minor"]) - np.sqrt(ref["baseline_loss_minor"])
        if has_minor and "baseline_loss_at_0_minor" in ref else float("nan")
    )

    # Normalized deltas: Δ RMSE / g_RMSE × 100 (% of ICL gain disrupted)
    raw_dm = _extract_delta_rmse("baseline_loss_major", "intervened_loss_major")
    raw_do = _extract_delta_rmse("baseline_loss_ood",   "intervened_loss_ood")
    norm_maj = [v / maj_info_gain * 100 for v in raw_dm]
    norm_ood  = [v / ood_info_gain * 100 for v in raw_do]
    if has_minor:
        raw_dn   = _extract_delta_rmse("baseline_loss_minor", "intervened_loss_minor")
        norm_min = [v / min_info_gain * 100 for v in raw_dn]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    x = np.arange(len(layers))

    ax.bar(x + MC_L["maj"], norm_maj,
           bw_bar_l, label="Maj.", color=COLORS_L["maj"], linewidth=0, zorder=3)
    ax.bar(x + MC_L["ood"], norm_ood,
           bw_bar_l, label="OOD", color=COLORS_L["ood"], linewidth=0, zorder=3)
    if has_minor:
        ax.bar(x + MC_L["minor"], norm_min,
               bw_bar_l, label="Min.", color=COLORS_L["minor"], linewidth=0, zorder=3)

    ax.axhline(100, color="grey", ls="--", lw=1.0, alpha=0.55,
               label="100%")

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Fraction of ICL gain disrupted (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.5, color="grey")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9,
              edgecolor="lightgrey", columnspacing=0.6,
              handlelength=1.2, handletextpad=0.4, borderpad=0.5)
    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_or_close(fig, show)

    # ── Printed Table 1: Per-layer normalized ─────────────────────────────
    _W = 7
    _hd = f"  {'Layer':>5}  {'Maj.%':>{_W}}  {'OOD%':>{_W}}"
    if has_minor:
        _hd += f"  {'Min.%':>{_W}}"
    _hd += f"  |  {'Maj.Δ':>{_W}}  {'OOD Δ':>{_W}}"
    if has_minor:
        _hd += f"  {'Min.Δ':>{_W}}"
    _ln = "  " + "─" * (len(_hd) - 2)
    print(f"\n  Task Subspace Intervention — Δ𝓛/g (% of ICL gain)  [RMSE]")
    print(_ln); print(_hd); print(_ln)
    for _i, _l in enumerate(layers):
        _nm = norm_maj[_i]; _no = norm_ood[_i]
        _row = f"  {_l:>5}  {_nm:>{_W}.1f}%  {_no:>{_W}.1f}%"
        if has_minor:
            _row += f"  {norm_min[_i]:>{_W}.1f}%"
        _row += f"  |  {raw_dm[_i]:>{_W}.4f}  {raw_do[_i]:>{_W}.4f}"
        if has_minor:
            _row += f"  {raw_dn[_i]:>{_W}.4f}"
        print(_row)
    print(_ln)
    _grow = f"  {'Gain':>5}  {'':>{_W}}   {'':>{_W}} "
    if has_minor:
        _grow += f"  {'':>{_W}} "
    _grow += f"  |  {maj_info_gain:{_W}.4f}  {ood_info_gain:{_W}.4f}"
    if has_minor:
        _grow += f"  {min_info_gain:{_W}.4f}"
    print(_grow)
    print()

    # ── Printed Table 2: Layer-averaged ───────────────────────────────────
    _mean_m = float(np.mean(norm_maj)); _std_m = float(np.std(norm_maj))
    _mean_o = float(np.mean(norm_ood)); _std_o = float(np.std(norm_ood))
    _mean_dm = float(np.mean(raw_dm)); _mean_do = float(np.mean(raw_do))
    _WA = 12
    _sep = "  " + "─" * 57
    print(_sep)
    print(f"  Layer-averaged (mean ± std across {len(layers)} layers) — RMSE")
    print(_sep)
    print(f"  {'Mode':<6}  {'Δ/g (%)':>{_WA}}  {'Raw Δ':>{_WA}}  {'g (RMSE)':>{_WA}}")
    print(_sep)
    print(f"  {'Maj.':<6}  {_mean_m:>7.1f}±{_std_m:<4.1f}  {_mean_dm:{_WA}.4f}  {maj_info_gain:{_WA}.4f}")
    print(f"  {'OOD':<6}  {_mean_o:>7.1f}±{_std_o:<4.1f}  {_mean_do:{_WA}.4f}  {ood_info_gain:{_WA}.4f}")
    if has_minor:
        _mean_n = float(np.mean(norm_min)); _std_n = float(np.std(norm_min))
        _mean_dn = float(np.mean(raw_dn))
        print(f"  {'Min.':<6}  {_mean_n:>7.1f}±{_std_n:<4.1f}  {_mean_dn:{_WA}.4f}  {min_info_gain:{_WA}.4f}")
    print(_sep)
    print()
    # ─────────────────────────────────────────────────────────────────────

    return fig, all_results
