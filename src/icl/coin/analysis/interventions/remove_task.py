"""Causal intervention: remove the task-subspace projection from hidden states.

Measures how next-token prediction degrades when the task-identifying
component of activations is ablated.
"""

import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.coin.analysis.probes import train_linear_hidden_predictor_coin
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def intervene_remove_task_subspace_coin(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_eval: int = 500,
    n_ood: int = 30,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    eval_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    scale: float = 1.0,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Causal intervention: remove the task-subspace projection from hidden
    representations, then measure the effect on next-token prediction.

    Pipeline:

    1. Fit ``posterior -> hiddens`` on major data at late positions to get
       the task subspace projector ``P_task``.
    2. For each mode (major, OOD):
       a. Baseline forward pass -> per-token cross-entropy.
       b. Intervened forward pass with hook that subtracts
          ``scale * P_task @ h`` from hiddens -> intervened loss.
       c. Report delta_loss = intervened - baseline.

    Returns dict with baseline/intervened/delta losses and percent increase.
    """
    _, _, config = nu.load_everything("coin", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    device = config.device

    sampler_major, _ = get_new_sampler(exp_name, n_minor=0, n_ood=0)

    sampler_ood, _ = get_new_sampler(exp_name, n_minor=0, n_ood=n_ood)

    seq_len = sampler_major.seq_len

    # ---- 1. Fit task vectors ----
    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor_coin(
        exp_name=exp_name,
        layer=layer,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        step=step,
        n_minor=-1,
        print_summary=False,
        skip_baselines=True,
    )
    W_fit = fit_res["model_weight"].float()  # (K_major, D)

    if center_task_vecs:
        task_vecs = W_fit - W_fit.mean(dim=0, keepdim=True)
    else:
        task_vecs = W_fit.clone()
    U_tv, S_tv, Vt_tv = torch.linalg.svd(task_vecs, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    basis = Vt_tv[:rank].T  # (D, rank)

    P_task = (basis @ basis.T).to(device)  # (D, D)

    if verbose:
        logger.info(
            f"[intervene-task] Task subspace rank={rank} "
            f"(centered={center_task_vecs}), "
            f"posterior fit R²={fit_res['val_r2']:.4f}"
        )

    # ---- 2. Intervention experiment ----
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def _run_experiment(sampler, gen_mode, n_samples):
        baseline_losses_by_pos = {p: [] for p in eval_positions}
        intervened_losses_by_pos = {p: [] for p in eval_positions}
        baseline_loss_at_0 = []

        n_batches = (n_samples + B - 1) // B
        for _ in range(n_batches):
            gen_out = sampler.generate(
                mode=gen_mode, task=None, num_samples=B, epochs=1,
            )
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)

            with torch.no_grad():
                logits_base = model(samples)

            def intervention_hook(module, inp, out):
                if torch.is_tensor(out):
                    h = out
                else:
                    h = out[0]
                h_task = h @ P_task
                h_modified = h - scale * h_task
                if torch.is_tensor(out):
                    return h_modified
                return (h_modified,) + out[1:]

            handle = model.layers[layer].attn_block.register_forward_hook(
                intervention_hook,
            )
            try:
                with torch.no_grad():
                    logits_int = model(samples)
            finally:
                handle.remove()

            if samples.shape[1] > 1:
                baseline_loss_at_0.append(
                    ce_loss_fn(logits_base[:, 0, :], samples[:, 1].long()).mean().item()
                )

            for p in eval_positions:
                if p + 1 >= samples.shape[1]:
                    continue
                target = samples[:, p + 1].long()
                loss_base = ce_loss_fn(
                    logits_base[:, p, :], target,
                ).mean().item()
                loss_int = ce_loss_fn(
                    logits_int[:, p, :], target,
                ).mean().item()
                baseline_losses_by_pos[p].append(loss_base)
                intervened_losses_by_pos[p].append(loss_int)

            del samples, logits_base, logits_int
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        baseline_per_pos, intervened_per_pos, valid_positions = [], [], []
        for p in eval_positions:
            if baseline_losses_by_pos[p]:
                baseline_per_pos.append(np.mean(baseline_losses_by_pos[p]))
                intervened_per_pos.append(np.mean(intervened_losses_by_pos[p]))
                valid_positions.append(p)

        baseline_avg = float(np.mean(baseline_per_pos)) if baseline_per_pos else float("nan")
        intervened_avg = float(np.mean(intervened_per_pos)) if intervened_per_pos else float("nan")

        return {
            "baseline": baseline_avg,
            "intervened": intervened_avg,
            "delta": intervened_avg - baseline_avg,
            "baseline_per_pos": baseline_per_pos,
            "intervened_per_pos": intervened_per_pos,
            "positions": valid_positions,
            "baseline_loss_at_0": float(np.mean(baseline_loss_at_0)) if baseline_loss_at_0 else float("nan"),
        }

    if verbose:
        logger.info("[intervene-task] Running major experiment ...")
    res_major = _run_experiment(sampler_major, "major", n_samples_eval)

    if verbose:
        logger.info("[intervene-task] Running OOD experiment ...")
    res_ood = _run_experiment(sampler_ood, "minor", n_samples_eval)

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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
    }

    if print_summary:
        print(f"\n{'=' * 65}")
        print(
            f"Causal Intervention: Remove Task Subspace  "
            f"(layer {layer}, scale={scale})"
        )
        print(f"{'=' * 65}")
        print(f"  Task subspace rank: {rank}")
        print(f"  Eval positions: {len(res_major['positions'])} positions\n")
        print(f"{'Metric':<30} {'Major':>12} {'OOD':>12}")
        print("-" * 54)
        print(
            f"{'Baseline loss':<30} "
            f"{res_major['baseline']:>12.4f} "
            f"{res_ood['baseline']:>12.4f}"
        )
        print(
            f"{'Intervened loss':<30} "
            f"{res_major['intervened']:>12.4f} "
            f"{res_ood['intervened']:>12.4f}"
        )
        print(
            f"{'Delta loss':<30} "
            f"{res_major['delta']:>12.4f} "
            f"{res_ood['delta']:>12.4f}"
        )
        print(
            f"{'Percent increase':<30} "
            f"{pct_major:>11.1f}% "
            f"{pct_ood:>11.1f}%"
        )

    return results


def plot_intervention_remove_task_across_layers_coin(
    exp_name: str,
    layers: Optional[list] = None,
    B: int = 64,
    n_samples_eval: int = 500,
    n_ood: int = 30,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    eval_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    scale: float = 1.0,
    figsize: tuple = (14, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Sweep ``intervene_remove_task_subspace_coin`` across layers.

    Single-panel bar plot of delta loss (intervened - baseline) per layer
    for major vs OOD, with a horizontal line showing the OOD information
    gain (l_0 - mean l_t): loss at first token minus average loss.

    Returns ``(fig, all_results)``.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
        layers = list(range(config.model.num_layers))

    all_results = {}
    for l in layers:
        logger.info(f"[task-removal sweep] layer {l} ...")
        res = intervene_remove_task_subspace_coin(
            exp_name=exp_name,
            layer=l,
            B=B,
            n_samples_eval=n_samples_eval,
            n_ood=n_ood,
            step=step,
            fit_n_samples=fit_n_samples,
            fit_positions=fit_positions,
            eval_positions=eval_positions,
            center_task_vecs=center_task_vecs,
            scale=scale,
            verbose=False,
            print_summary=False,
        )
        all_results[l] = res

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(layers))
    bar_w = 0.35

    delta_maj = [all_results[l]["delta_loss_major"] for l in layers]
    delta_ood = [all_results[l]["delta_loss_ood"] for l in layers]

    ax.bar(
        x - bar_w / 2, delta_maj, bar_w,
        label="Major", color="#2196F3", alpha=0.85,
    )
    ax.bar(
        x + bar_w / 2, delta_ood, bar_w,
        label="OOD", color="#FF9800", alpha=0.85,
    )
    for i, (vm, vo) in enumerate(zip(delta_maj, delta_ood)):
        ax.text(
            x[i] - bar_w / 2, vm, f"{vm:.3f}",
            ha="center", va="bottom", fontsize=10,
        )
        ax.text(
            x[i] + bar_w / 2, vo, f"{vo:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    ref = all_results[layers[0]]
    maj_l0 = ref["baseline_loss_at_0_major"]
    ood_l0 = ref["baseline_loss_at_0_ood"]
    maj_info_gain = maj_l0 - ref["baseline_loss_major"]
    ood_info_gain = ood_l0 - ref["baseline_loss_ood"]

    ax.axhline(maj_info_gain, color="#1565C0", ls="--", lw=1.8, alpha=0.7,
               label=f"Major $l_0 - \\bar{{l}}_t$ = {maj_info_gain:.3f}")
    ax.axhline(ood_info_gain, color="#E65100", ls=":", lw=1.8, alpha=0.7,
               label=f"OOD $l_0 - \\bar{{l}}_t$ = {ood_info_gain:.3f}")

    ax.set_xlabel("Layer", fontsize=16)
    ax.set_ylabel("\u0394 Loss (intervened \u2212 baseline)", fontsize=15)
    ax.set_title("Loss Increase from Removing Task Subspace", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    sup = title or (
        f"Causal Intervention: Remove Task Subspace "
        f"(scale={scale})"
    )
    fig.suptitle(sup, fontsize=17, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, all_results
