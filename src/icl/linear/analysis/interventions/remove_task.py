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
) -> dict:
    """
    Causal intervention: remove the fitted task-subspace component from
    hidden states and measure the effect on prediction MSE.

    Linear regression counterpart of
    ``intervene_remove_task_subspace_latent_nonpadded``.
    """
    model, train_task, config, device = _load_and_prepare_model(exp_name, step)
    if step is None:
        step = config.training.total_steps

    n_points = int(config.task.n_points)

    ood_task = _create_ood_task(train_task, config, B, n_ood, device)

    # ---- 1. Fit task subspace ----
    _, _, _, rank, P_task, fit_res = _fit_task_subspace(
        exp_name=exp_name,
        layer=layer,
        step=step,
        fit_n_samples=fit_n_samples,
        fit_positions=fit_positions,
        n_points=n_points,
        center_task_vecs=center_task_vecs,
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
        minor_only=False, track_oracle=False,
    )

    if verbose:
        logger.info("[remove-task linear] Running OOD experiment ...")
    res_ood = _run_projection_removal(
        model, layer, P_task, scale, ood_task, eval_positions,
        n_samples_eval, B, device,
        minor_only=False, track_oracle=False,
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
            minor_only=True, track_oracle=False,
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

    def _extract(key):
        return [all_results[l][key] for l in layers]

    has_minor = all_results[layers[0]].get("has_minor", False)

    # Build bar groups: [(key, label, color), ...]
    groups = [("delta_loss_major", "Major", "#2196F3")]
    if has_minor:
        groups.append(("delta_loss_minor", "Minor", "#4CAF50"))
    groups.append(("delta_loss_ood", "OOD", "#FF9800"))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(layers))
    bar_w = 0.8 / len(groups)
    offsets = np.arange(len(groups)) * bar_w - (len(groups) - 1) * bar_w / 2

    for off, (key, label, color) in zip(offsets, groups):
        vals = _extract(key)
        ax.bar(x + off, vals, bar_w, label=label, color=color, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(x[i] + off, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # OOD baseline: original model loss on OOD eval positions (no intervention)
    ood_baseline = np.mean(_extract("baseline_loss_ood"))
    ax.axhline(ood_baseline, color="#E53935", ls="--", lw=1.8, alpha=0.8,
               label=f"OOD baseline MSE = {ood_baseline:.3f}")

    scale = kwargs.get("scale", 1.0)
    ax.set(xlabel="Layer", ylabel="Δ MSE",
           title="MSE Increase from Removing Task Subspace")
    ax.set_xticks(x, [str(l) for l in layers])
    ax.xaxis.label.set_size(16); ax.yaxis.label.set_size(15); ax.title.set_size(15)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        title or f"Remove Task Subspace (linear, scale={scale})",
        fontsize=17, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_or_close(fig, show)

    return fig, all_results
