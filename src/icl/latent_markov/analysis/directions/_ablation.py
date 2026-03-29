"""
Head ablation experiment for latent Markov models.
"""

import gc
import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
#  Head ablation experiment
# ---------------------------------------------------------------------------

@torch.no_grad()
def head_ablation_experiment(
    exp_name: str,
    ablations: list,
    B: int = 128,
    n_batches: int = 8,
    step: Optional[int] = None,
    show: bool = True,
    figsize: tuple = (10, 4.5),
    n_minor: int = 256,
    n_ood: int = 40,
    save_path: Optional[str] = None,
) -> dict:
    """
    Measure the effect of zeroing out specific attention heads on
    mean cross-entropy loss, evaluated on major / minor / OOD samples.

    The y-axis uses the same *fraction-of-ICL-gain-disrupted* metric as
    ``plot_optimal_orth_direction_across_layers``:
    ``Δ𝓛 / g × 100 %``, where ``g = 𝓛(pos 0) − 𝓛(model)`` is the
    ICL gain (improvement from seeing context).

    For each ``(layer, head)`` pair in *ablations*, the head's contribution
    is removed by zeroing its slice in the concatenated head output
    **before** the output projection (``MHA.out``).  This is done via a
    ``register_forward_pre_hook`` on the output linear layer, so neither
    the model code nor the flash-attention path needs to be modified.

    Ablation conditions evaluated for every sample mode:

    1. **Separate** – each ``(layer, head)`` is ablated individually.
    2. **Joint** – all heads in *ablations* are ablated simultaneously.

    Parameters
    ----------
    exp_name  : Experiment name understood by ``nu.load_everything``.
    ablations : List of ``(layer_index, head_index)`` tuples to ablate.
    B         : Batch size per forward pass.
    n_batches : Number of batches to average over.
    step      : Checkpoint step; ``None`` → final epoch.
    show      : Call ``plt.show()`` before returning.
    figsize   : Figure size for the plot.
    n_minor   : Number of minor tasks for the sampler.
    n_ood     : Number of OOD tasks for the sampler.
    save_path : If set, save figure to this path.

    Returns
    -------
    dict with keys:
        ``fig``     : matplotlib Figure.
        ``per_sample`` : nested dict of per-sample losses.
        ``summary`` : dict mapping ``(condition, mode)`` → scalar mean CE.
        ``gains``   : dict mapping mode → ICL gain (nats).
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device
    V = config.vocab_size

    sampler, _, _ = get_latent_sampler(exp_name, n_minor=n_minor, n_ood=n_ood)
    seq_len = sampler.seq_len

    modes = ["major"]
    if sampler.n_minor_tasks > 0:
        modes.append("minor")
    modes.append("ood")

    def _compute_loss(mode: str):
        """Return per-sample (mean-over-positions) and pos-0 losses."""
        all_mean = []
        all_pos0 = []
        for _ in range(n_batches):
            gen_out = sampler.generate(mode=mode, num_samples=B)
            tokens = gen_out[0].to(device)
            logits = model(tokens)[:, :-1, :]
            targets = tokens[:, 1:]
            loss_per_pos = F.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1), reduction="none",
            ).reshape(B, seq_len - 1)
            all_mean.append(loss_per_pos.mean(dim=1).cpu())
            all_pos0.append(loss_per_pos[:, 0].cpu())
        return torch.cat(all_mean, dim=0), torch.cat(all_pos0, dim=0)

    def _make_zero_head_hook(head_idx, head_dim):
        """Pre-hook for ``MHA.out`` that zeros out one head's slice."""
        start = head_idx * head_dim
        end = start + head_dim

        def hook_fn(module, inputs):
            x = inputs[0].clone()
            x[:, :, start:end] = 0.0
            return (x,)

        return hook_fn

    # ---- collect all ablation conditions (skip "unablated" in the bar plot) ----
    ablation_conditions = []
    for layer_idx, head_idx in ablations:
        ablation_conditions.append(f"L{layer_idx}H{head_idx}")
    if len(ablations) > 1:
        joint_label = "+".join(f"L{l}H{h}" for l, h in ablations)
        ablation_conditions.append(joint_label)
    else:
        joint_label = None

    all_conditions = ["unablated"] + ablation_conditions

    summary = {}
    per_sample = {m: {} for m in modes}
    gains = {}

    for mode in modes:
        logger.info(f"[head-ablation] evaluating mode={mode} ...")

        # --- unablated ---
        losses, pos0 = _compute_loss(mode)
        per_sample[mode]["unablated"] = losses.numpy()
        summary[("unablated", mode)] = float(losses.mean())
        gains[mode] = float(pos0.mean()) - float(losses.mean())

        # --- separate ablations ---
        for layer_idx, head_idx in ablations:
            mha = model.layers[layer_idx].attn_block.MHA
            handle = mha.out.register_forward_pre_hook(
                _make_zero_head_hook(head_idx, mha.head_dim),
            )
            losses, _ = _compute_loss(mode)
            handle.remove()

            key = f"L{layer_idx}H{head_idx}"
            per_sample[mode][key] = losses.numpy()
            summary[(key, mode)] = float(losses.mean())

        # --- joint ablation ---
        if joint_label is not None:
            handles = []
            for layer_idx, head_idx in ablations:
                mha = model.layers[layer_idx].attn_block.MHA
                handles.append(
                    mha.out.register_forward_pre_hook(
                        _make_zero_head_hook(head_idx, mha.head_dim),
                    )
                )
            losses, _ = _compute_loss(mode)
            for h in handles:
                h.remove()

            per_sample[mode][joint_label] = losses.numpy()
            summary[(joint_label, mode)] = float(losses.mean())

    # ---- print summary table ----
    col_w = 12
    header = f"{'Condition':<25s}" + "".join(f"{m:>{col_w}s}" for m in modes)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"  Head Ablation — Δ𝓛/g (% of ICL gain)  [CE nats]")
    print(sep)
    print(header)
    print(sep)
    for cond in all_conditions:
        row = f"{cond:<25s}"
        for mode in modes:
            val = summary.get((cond, mode), float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)
    print(sep)
    print(f"{'':25s}" + "".join(f"{'Δ/g %':>{col_w}s}" for _ in modes))
    for cond in all_conditions:
        if cond == "unablated":
            continue
        row = f"{cond:<25s}"
        for mode in modes:
            delta = summary.get((cond, mode), 0.0) - summary.get(("unablated", mode), 0.0)
            g = gains.get(mode, 1.0)
            pct = delta / g * 100 if abs(g) > 1e-12 else float("nan")
            row += f"{pct:>{col_w}.1f}"
        print(row)
    print(sep)
    print(f"{'ICL gain (nats)':<25s}" + "".join(
        f"{gains.get(m, float('nan')):>{col_w}.4f}" for m in modes))
    print(f"{sep}\n")

    # ---- grouped bar plot: Δ𝓛/g (%) ----
    MODE_COLORS = {"major": "#2166ac", "ood": "#d6604d", "minor": "#1a9850"}
    MODE_LABELS = {"major": "Maj.", "ood": "OOD", "minor": "Min."}
    n_modes = len(modes)
    n_conds = len(ablation_conditions)

    bw = 0.22
    g_step = 0.24
    offsets = np.linspace(-(n_modes - 1) / 2 * g_step,
                           (n_modes - 1) / 2 * g_step, n_modes)

    x = np.arange(n_conds)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for j, mode in enumerate(modes):
        g = gains[mode] if abs(gains[mode]) > 1e-12 else 1.0
        norm_vals = []
        lo_err = []
        hi_err = []
        bl = summary[("unablated", mode)]
        for cond in ablation_conditions:
            arr = per_sample[mode][cond]
            deltas = arr - bl
            pcts = deltas / g * 100
            m = float(pcts.mean())
            q25, q75 = np.percentile(pcts, [25, 75])
            norm_vals.append(m)
            lo_err.append(m - q25)
            hi_err.append(q75 - m)

        xm = x + offsets[j]
        ax.bar(xm, norm_vals, bw, color=MODE_COLORS[mode], linewidth=0,
               zorder=3, label=MODE_LABELS[mode])
        ax.errorbar(xm, norm_vals, yerr=[lo_err, hi_err], fmt="none",
                    ecolor="black", elinewidth=0.9, capsize=3,
                    capthick=0.9, zorder=5)

    ax.axhline(100, color="grey", ls="--", lw=1.0, alpha=0.55,
               label="100%")

    ax.set_xlabel("Condition", fontsize=9)
    ax.set_ylabel("Fraction of ICL gain disrupted (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_conditions, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.5, color="grey")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=n_modes + 1, framealpha=0.9, edgecolor="lightgrey",
              columnspacing=0.8, handlelength=1.0, handletextpad=0.3,
              borderpad=0.4)
    plt.tight_layout(pad=2.0)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {"fig": fig, "per_sample": per_sample, "summary": summary,
            "gains": gains}
