import os
import json

import numpy as np
from typing import Optional

from icl.utils.unified_path_finder import get_exp_name
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def plot_id_ood_loss_dyck(
    k_list,
    logx: bool = True,
    figsize: tuple = (12, 5),
    start_step: Optional[float] = None,
    show: bool = True,
) -> dict:
    """
    Plot ID loss and OOD loss side by side for multiple Dyck experiments.

    Loads ``log.json`` from each Dyck experiment and plots ``eval/IDLoss``
    and ``eval/OODLoss`` against ``eval/step`` (fallback to ``train/step``).

    Parameters
    ----------
    k_list : list
        Dyck experiment k values.
    logx : bool, default=True
        If True, use log scale on x-axis.
    figsize : tuple, default=(12, 5)
    start_step : float, optional
        If provided, only plot points with training step >= ``start_step``.
    show : bool, default=True
    """
    import matplotlib.pyplot as plt

    def _resolve_dyck_exp_dir(exp_name: str) -> str:
        candidates = [
            os.path.join("results", "dyck", exp_name),
            os.path.join("..", "results", "dyck", exp_name),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    results = {}
    for k in k_list:
        exp_name = get_exp_name("dyck", k)
        n_minor_tasks = 2 ** k if k >= 0 else 0

        try:
            exp_dir = _resolve_dyck_exp_dir(exp_name)
            log_path = os.path.join(exp_dir, "log.json")
            with open(log_path, "r") as f:
                data = json.load(f)

            train_steps = np.asarray(data.get("eval/step", data.get("train/step", [])), dtype=float)
            id_loss = data.get("eval/IDLoss", None)
            ood_loss = data.get("eval/OODLoss", None)
            if id_loss is None or ood_loss is None:
                raise KeyError("Missing eval/IDLoss or eval/OODLoss in dyck log.json")

            id_loss = np.asarray(id_loss, dtype=float)
            ood_loss = np.asarray(ood_loss, dtype=float)

            if train_steps.size == 0:
                L = min(len(id_loss), len(ood_loss))
                train_steps = np.arange(1, L + 1, dtype=float)

            L = min(len(train_steps), len(id_loss), len(ood_loss))
            train_steps = train_steps[:L]
            id_loss = id_loss[:L]
            ood_loss = ood_loss[:L]

            results[k] = {
                "n_minor": n_minor_tasks,
                "train_steps": train_steps,
                "id_loss": id_loss,
                "ood_loss": ood_loss,
            }
        except Exception as e:
            logger.warning(f"Could not load dyck k={k}: {e}")

    ks_sorted = sorted(results.keys())
    if not ks_sorted:
        logger.warning("No Dyck experiments loaded successfully.")
        return {}

    cmap = plt.get_cmap("tab20", max(len(ks_sorted), 1))
    color_map = {k: cmap(i) for i, k in enumerate(ks_sorted)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, ys = d["train_steps"], d["id_loss"]
        if start_step is not None:
            mask = xs >= float(start_step)
            xs, ys = xs[mask], ys[mask]
        if logx:
            mask = xs > 0
            xs, ys = xs[mask], ys[mask]
        if xs.size == 0:
            continue
        ax1.plot(xs, ys, color=c, linewidth=2.0)

    if logx:
        ax1.set_xscale("log")
    ax1.set_xlabel("Training Step", fontsize=16)
    ax1.set_ylabel("ID Loss", fontsize=16)
    ax1.tick_params(labelsize=14)
    ax1.grid(True, which="both", alpha=0.25)

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, ys = d["train_steps"], d["ood_loss"]
        if start_step is not None:
            mask = xs >= float(start_step)
            xs, ys = xs[mask], ys[mask]
        if logx:
            mask = xs > 0
            xs, ys = xs[mask], ys[mask]
        if xs.size == 0:
            continue
        ax2.plot(xs, ys, color=c, linewidth=2.0)

    if logx:
        ax2.set_xscale("log")
    ax2.set_xlabel("Training Step", fontsize=16)
    ax2.set_ylabel("OOD Loss", fontsize=16)
    ax2.tick_params(labelsize=14)
    ax2.grid(True, which="both", alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], color=color_map[k], lw=2.5, label=f"k={k}")
        for k in ks_sorted
    ]
    fig.legend(
        handles=legend_handles,
        title="k",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=12,
        title_fontsize=13,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    if show:
        plt.show()

    return {"fig": fig, "ax1": ax1, "ax2": ax2, "results": results}
