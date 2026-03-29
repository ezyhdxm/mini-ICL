"""Shared helpers for posterior-weighted task-vector injection across tasks.

Used by coin, latent_markov, and linear ``inject_posterior`` modules
to avoid duplicating KL helpers, subspace fitting, aggregation, and
plotting code.
"""

from typing import Optional

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────
#  KL divergence helper
# ──────────────────────────────────────────────────────────────────────

def kl_softmax(q_logits: torch.Tensor, p_dist: torch.Tensor) -> torch.Tensor:
    """KL(softmax(q_logits) || p_dist), per sample.

    Args:
        q_logits: Raw logits, shape ``(..., V)``.
        p_dist:   Target probability distribution, shape ``(..., V)``.

    Returns:
        KL divergence per sample, shape ``(...)``.
    """
    lq = torch.log_softmax(q_logits, dim=-1)
    lp = torch.log(p_dist.clamp_min(1e-12))
    return (lq.exp() * (lq - lp)).sum(-1)


# ──────────────────────────────────────────────────────────────────────
#  Task-subspace fitting (SVD + orthogonal projector)
# ──────────────────────────────────────────────────────────────────────

def build_task_subspace(
    W_fit: torch.Tensor,
    center: bool,
    device: torch.device,
) -> tuple:
    """Build the task subspace projector from fitted task vectors.

    Args:
        W_fit:  Probe weight matrix, shape ``(K, D)``.
        center: If True, center the task vectors before SVD.
        device: Target device for the returned tensors.

    Returns:
        ``(P_task, ref_vecs, rank)`` where ``P_task`` is ``(D, D)``
        projector, ``ref_vecs`` is ``(K, D)`` reference vectors (centered
        or raw), and ``rank`` is the effective rank.
    """
    if center:
        tv = W_fit - W_fit.mean(dim=0, keepdim=True)
    else:
        tv = W_fit.clone()
    _, S_tv, Vt_tv = torch.linalg.svd(tv, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    P_task = (Vt_tv[:rank].T @ Vt_tv[:rank]).to(device)

    ref_vecs = (W_fit - W_fit.mean(dim=0) if center
                else W_fit.clone()).to(device)
    return P_task, ref_vecs, rank


# ──────────────────────────────────────────────────────────────────────
#  Per-position metric aggregation
# ──────────────────────────────────────────────────────────────────────

def aggregate_position_metrics(
    eval_positions: list,
    acc_base: dict,
    acc_inj: dict,
    acc_mode: Optional[dict] = None,
    acc_post_std: Optional[dict] = None,
    acc_min_dist: Optional[dict] = None,
) -> dict:
    """Aggregate per-batch accumulators into per-position means.

    Returns a dict with keys ``positions``, ``kl_base``, ``kl_inj``,
    and optionally ``kl_mode``, ``post_std``, ``min_dist`` (as numpy
    arrays).
    """
    positions, kl_base, kl_inj = [], [], []
    kl_mode_list, post_std, min_dist = [], [], []

    for p in eval_positions:
        if acc_base[p]:
            positions.append(p)
            kl_base.append(float(np.mean(acc_base[p])))
            kl_inj.append(float(np.mean(acc_inj[p])))
            if acc_mode is not None:
                kl_mode_list.append(float(np.mean(acc_mode[p])))
            if acc_post_std is not None:
                post_std.append(float(np.mean(acc_post_std[p])))
            if acc_min_dist is not None:
                min_dist.append(float(np.mean(acc_min_dist[p])))

    result = dict(
        positions=positions,
        kl_base=np.array(kl_base),
        kl_inj=np.array(kl_inj),
    )
    if acc_mode is not None:
        result["kl_mode"] = np.array(kl_mode_list)
    if acc_post_std is not None:
        result["post_std"] = np.array(post_std)
    if acc_min_dist is not None:
        result["min_dist"] = np.array(min_dist)
    return result


# ──────────────────────────────────────────────────────────────────────
#  Plotting: per-position KL + concentration
# ──────────────────────────────────────────────────────────────────────

def plot_inject_posterior_per_position(
    result: dict,
    *,
    title: Optional[str] = None,
    figsize: tuple = (5, 3.2),
    show: bool = True,
):
    """Detailed per-position visualisation of a single-layer result.

    *result* is the dict returned by an ``intervene_inject_posterior*``
    or ``intervene_direct_injection*`` function.

    Produces two separate figures:

    1. KL(output || target) vs position for baseline, injected, and mode.
    2. Posterior / alpha concentration metrics vs position.

    Returns ``(fig_kl, fig_conc)``.
    """
    import matplotlib.pyplot as plt

    pos = np.array(result["positions"])
    kl_b = np.array(result["kl_baseline"])
    kl_i = np.array(result["kl_injected"])
    kl_md = np.array(result.get("kl_mode", []))
    post_std = np.array(result.get("posterior_std", []))
    layer = result["layer"]

    sup = title if title is not None else f"Posterior-weighted injection  (layer {layer})"

    # ── Figure 1: KL vs position ─────────────────────────────────────
    fig_kl, ax1 = plt.subplots(figsize=figsize)
    ax1.fill_between(pos, kl_i, kl_b, alpha=0.15, color="#F44336",
                     label="gap closed by injection")
    ax1.plot(pos, kl_b, "o-", color="#F44336", lw=2, ms=4,
             label="unmodified")
    ax1.plot(pos, kl_i, "s-", color="#4CAF50", lw=2, ms=4,
             label=r"$\alpha$-injected")
    if len(kl_md) == len(pos):
        ax1.plot(pos, kl_md, "^--", color="#2196F3", lw=1.5, ms=4,
                 label=r"mode task $q_{k^\star}$")
    ax1.set_xlabel("Position", fontsize=13)
    ax1.set_ylabel(r"$\mathrm{KL}(\mathrm{output}\;\|\;\sum_k \alpha_k q_k)$", fontsize=13)
    if sup:
        ax1.set_title(sup, fontsize=13)
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_kl)

    # ── Figure 2: posterior concentration ─────────────────────────────
    min_d = np.array(result.get("min_dist_basis", []))
    has_std = len(post_std) == len(pos)
    has_dist = len(min_d) == len(pos)

    fig_conc, ax2 = plt.subplots(figsize=figsize)
    if has_std or has_dist:
        if has_std:
            ax2.plot(pos, post_std, "D-", color="#9C27B0", lw=2, ms=4,
                     label="posterior std")
        if has_dist:
            ax2.plot(pos, min_d, "^-", color="#FF9800", lw=2, ms=4,
                     label=r"min $\|\alpha - e_k\|_1$")
            ax2.axhline(0.0, color="#FF9800", ls="--", lw=1, alpha=0.4,
                        label=r"$\delta$ (concentrated)")
        ax2.set_xlabel("Position", fontsize=13)
        ax2.set_ylabel("Posterior concentration", fontsize=13)
        ax2.set_title(r"Spread of $\alpha_{t,k}$", fontsize=13)
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "metrics not available\n(re-run intervention)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_conc)

    return fig_kl, fig_conc


def plot_inject_posterior_across_layers(
    exp_name: str,
    task_name: str,
    intervene_fn,
    layers: Optional[list] = None,
    load_everything_fn=None,
    show: bool = True,
    **kwargs,
):
    """Sweep an inject-posterior function over layers and plot.

    Args:
        exp_name:   Experiment name.
        task_name:  ``"coin"`` or ``"latent"`` (for labels).
        intervene_fn: The single-layer intervention callable.
        layers:     Layers to sweep. ``None`` → all layers from config.
        load_everything_fn: Callable ``(task_name, exp_name) -> (_, _, config)``.
        show:       Whether to display the plot.

    Returns ``(fig, all_results)``.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = load_everything_fn(task_name, exp_name)
        layers = list(range(config.model.num_layers))

    from icl.utils.logger import setup_logger
    _logger = setup_logger(__name__)

    all_res = {}
    for l in layers:
        _logger.info(f"[posterior inj sweep {task_name}] layer {l}")
        all_res[l] = intervene_fn(
            exp_name=exp_name, layer=l, print_summary=True, **kwargs,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    last = layers[-1]
    r = all_res[last]
    ax1.plot(r["positions"], r["kl_baseline"], "o-",
             label="baseline", lw=2, ms=4)
    ax1.plot(r["positions"], r["kl_injected"], "s-",
             label="posterior-injected", lw=2, ms=4)
    ax1.set_xlabel("Position", fontsize=13)
    ax1.set_ylabel(r"KL(output $\|$ misBayes)", fontsize=13)
    ax1.set_title(f"Layer {last}", fontsize=13)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)

    x = np.arange(len(layers))
    bw = 0.35
    avg_base = [np.mean(all_res[l]["kl_baseline"]) for l in layers]
    avg_inj = [np.mean(all_res[l]["kl_injected"]) for l in layers]
    ax2.bar(x - bw / 2, avg_base, bw,
            label="baseline", color="#F44336", alpha=0.85)
    ax2.bar(x + bw / 2, avg_inj, bw,
            label="posterior-injected", color="#4CAF50", alpha=0.85)
    for i, (vb, vi) in enumerate(zip(avg_base, avg_inj)):
        ax2.text(x[i] - bw / 2, vb, f"{vb:.3f}",
                 ha="center", va="bottom", fontsize=8)
        ax2.text(x[i] + bw / 2, vi, f"{vi:.3f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Layer", fontsize=13)
    ax2.set_ylabel(r"Mean KL(output $\|$ misBayes)", fontsize=13)
    ax2.set_xticks(x, [str(l) for l in layers])
    ax2.legend(fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, all_res
