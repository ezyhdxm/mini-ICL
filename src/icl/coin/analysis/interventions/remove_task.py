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
    n_minor: int = 30,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    eval_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    scale: float = 1.0,
    verbose: bool = False,
    print_summary: bool = True,
    extraction_point: str = "post_attn",
    probe_method: str = "ols",
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
    sampler_ood, _   = get_new_sampler(exp_name, n_minor=0, n_ood=n_ood)
    sampler_min, _   = get_new_sampler(exp_name, n_minor=n_minor, n_ood=0)

    seq_len = sampler_major.seq_len

    # ---- 1. Fit task vectors ----
    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    if probe_method == "averaging":
        from icl.coin.analysis._helpers import get_token_conditioned_hiddens_coin
        anova_positions = [p for p in fit_positions if p < seq_len - 1]
        all_hiddens_anova, anova_info = get_token_conditioned_hiddens_coin(
            exp_name, layers=[layer], batch_size=B,
            positions_of_interest=anova_positions,
            step=step, n_minor=0,
            extraction_point=extraction_point,
        )
        # all_hiddens_anova: (1, n_pos, V_max, n_major, B, D)
        pos_to_anova = {p: i for i, p in enumerate(anova_info["positions"])}
        n_uniq = anova_info["n_unique_tokens"]
        V_max = all_hiddens_anova.shape[2]
        n_major_anova = all_hiddens_anova.shape[3]
        parts = []
        for p in anova_positions:
            if p not in pos_to_anova:
                continue
            pi = pos_to_anova[p]
            V_p = n_uniq.get(p, V_max)
            parts.append(all_hiddens_anova[0, pi, :V_p, :n_major_anova].float().mean(dim=-2))
        if not parts:
            raise ValueError("probe_method='averaging': no valid fit_positions found.")
        min_V = min(p.shape[0] for p in parts)
        parts = [p[:min_V] for p in parts]
        demeaned = [cm - cm.mean(dim=(0, 1), keepdim=True) for cm in parts]
        pooled = torch.stack(demeaned, dim=0).mean(dim=0)  # (V, K, D)
        grand = pooled.mean(dim=(0, 1))
        W_fit = pooled.mean(dim=0) - grand  # (K, D)
        del all_hiddens_anova
    else:
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
            extraction_point=extraction_point,
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

            hook_target = (
                model.layers[layer]
                if extraction_point == "post_mlp"
                else model.layers[layer].attn_block
            )
            handle = hook_target.register_forward_hook(intervention_hook)
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

    if verbose:
        logger.info("[intervene-task] Running minor experiment ...")
    res_min = _run_experiment(sampler_min, "minor", n_samples_eval)

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
    pct_minor = (
        100.0 * res_min["delta"] / res_min["baseline"]
        if res_min["baseline"] > 0 else float("nan")
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
        "baseline_loss_minor": res_min["baseline"],
        "intervened_loss_minor": res_min["intervened"],
        "delta_loss_minor": res_min["delta"],
        "pct_increase_minor": pct_minor,
        "baseline_per_pos_major": res_major["baseline_per_pos"],
        "intervened_per_pos_major": res_major["intervened_per_pos"],
        "baseline_per_pos_ood": res_ood["baseline_per_pos"],
        "intervened_per_pos_ood": res_ood["intervened_per_pos"],
        "baseline_per_pos_minor": res_min["baseline_per_pos"],
        "intervened_per_pos_minor": res_min["intervened_per_pos"],
        "baseline_loss_at_0_major": res_major["baseline_loss_at_0"],
        "baseline_loss_at_0_ood": res_ood["baseline_loss_at_0"],
        "baseline_loss_at_0_minor": res_min["baseline_loss_at_0"],
        "eval_positions": res_major["positions"],
        "layer": layer,
        "scale": scale,
        "task_subspace_rank": rank,
    }

    if print_summary:
        print(f"\n{'=' * 75}")
        print(
            f"Causal Intervention: Remove Task Subspace  "
            f"(layer {layer}, scale={scale})"
        )
        print(f"{'=' * 75}")
        print(f"  Task subspace rank: {rank}")
        print(f"  Eval positions: {len(res_major['positions'])} positions\n")
        print(f"{'Metric':<30} {'Major':>12} {'OOD':>12} {'Minor':>12}")
        print("-" * 66)
        print(
            f"{'Baseline loss':<30} "
            f"{res_major['baseline']:>12.4f} "
            f"{res_ood['baseline']:>12.4f} "
            f"{res_min['baseline']:>12.4f}"
        )
        print(
            f"{'Intervened loss':<30} "
            f"{res_major['intervened']:>12.4f} "
            f"{res_ood['intervened']:>12.4f} "
            f"{res_min['intervened']:>12.4f}"
        )
        print(
            f"{'Delta loss':<30} "
            f"{res_major['delta']:>12.4f} "
            f"{res_ood['delta']:>12.4f} "
            f"{res_min['delta']:>12.4f}"
        )
        print(
            f"{'Percent increase':<30} "
            f"{pct_major:>11.1f}% "
            f"{pct_ood:>11.1f}% "
            f"{pct_minor:>11.1f}%"
        )

    return results


def plot_intervention_remove_task_across_layers_coin(
    exp_name: str,
    layers: Optional[list] = None,
    B: int = 64,
    n_samples_eval: int = 500,
    n_ood: int = 30,
    n_minor: int = 30,
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
    extraction_point: str = "post_attn",
    probe_method: str = "ols",
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
            n_minor=n_minor,
            step=step,
            fit_n_samples=fit_n_samples,
            fit_positions=fit_positions,
            eval_positions=eval_positions,
            center_task_vecs=center_task_vecs,
            scale=scale,
            verbose=False,
            print_summary=False,
            extraction_point=extraction_point,
            probe_method=probe_method,
        )
        all_results[l] = res

    COLORS_C = {"maj": "#2166ac", "ood": "#d6604d", "minor": "#1a9850"}
    bw_bar_c = 0.22
    g_step_c = 0.24
    MC_C     = {"maj": -g_step_c, "ood": 0.0, "minor": +g_step_c}

    # Gains: baseline at pos-0 minus mean baseline (same for all layers)
    ref0 = all_results[layers[0]]
    g_maj = ref0["baseline_loss_at_0_major"] - ref0["baseline_loss_major"]
    g_ood = ref0["baseline_loss_at_0_ood"]   - ref0["baseline_loss_ood"]
    g_min = ref0["baseline_loss_at_0_minor"]  - ref0["baseline_loss_minor"]

    # Normalized deltas: Δ𝓛 / g × 100  (% of ICL gain disrupted)
    norm_maj = [all_results[l]["delta_loss_major"] / g_maj * 100 for l in layers]
    norm_ood = [all_results[l]["delta_loss_ood"]   / g_ood * 100 for l in layers]
    norm_min = [all_results[l]["delta_loss_minor"]  / g_min * 100 for l in layers]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    x = np.arange(len(layers))

    ax.bar(x + MC_C["maj"],   norm_maj, bw_bar_c,
           label="Maj.", color=COLORS_C["maj"],   linewidth=0, zorder=3)
    ax.bar(x + MC_C["ood"],   norm_ood, bw_bar_c,
           label="OOD",  color=COLORS_C["ood"],   linewidth=0, zorder=3)
    ax.bar(x + MC_C["minor"], norm_min, bw_bar_c,
           label="Min.", color=COLORS_C["minor"], linewidth=0, zorder=3)

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
    ax.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.9,
              edgecolor="lightgrey", columnspacing=0.6,
              handlelength=1.2, handletextpad=0.4, borderpad=0.5)
    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # ── Printed Table 1: Per-layer normalized ─────────────────────────────
    _W = 7
    _hd = (f"  {'Layer':>5}  {'Maj.%':>{_W}}  {'OOD%':>{_W}}  {'Min.%':>{_W}}"
           f"  |  {'Maj.Δ':>{_W}}  {'OOD Δ':>{_W}}  {'Min.Δ':>{_W}}")
    _ln = "  " + "─" * (len(_hd) - 2)
    print(f"\n  Task Subspace Intervention — Δ𝓛/g (% of ICL gain)  [CE nats]")
    print(_ln); print(_hd); print(_ln)
    for _l in layers:
        _r = all_results[_l]
        _nm = _r["delta_loss_major"] / g_maj * 100
        _no = _r["delta_loss_ood"]   / g_ood * 100
        _nn = _r["delta_loss_minor"]  / g_min * 100
        print(f"  {_l:>5}  {_nm:>{_W}.1f}%  {_no:>{_W}.1f}%  {_nn:>{_W}.1f}%  |  "
              f"{_r['delta_loss_major']:>{_W}.4f}  {_r['delta_loss_ood']:>{_W}.4f}"
              f"  {_r['delta_loss_minor']:>{_W}.4f}")
    print(_ln)
    print(f"  {'Gain':>5}  {'':>{_W}}   {'':>{_W}}   {'':>{_W}}   |  "
          f"{g_maj:{_W}.4f}  {g_ood:{_W}.4f}  {g_min:{_W}.4f}")
    print()

    # ── Printed Table 2: Layer-averaged ───────────────────────────────────
    _mean_m = float(np.mean(norm_maj)); _std_m = float(np.std(norm_maj))
    _mean_o = float(np.mean(norm_ood)); _std_o = float(np.std(norm_ood))
    _mean_n = float(np.mean(norm_min)); _std_n = float(np.std(norm_min))
    _mean_dm = float(np.mean([all_results[l]["delta_loss_major"] for l in layers]))
    _mean_do = float(np.mean([all_results[l]["delta_loss_ood"]   for l in layers]))
    _mean_dn = float(np.mean([all_results[l]["delta_loss_minor"]  for l in layers]))
    _WA = 12
    _sep = "  " + "─" * 57
    print(_sep)
    print(f"  Layer-averaged (mean ± std across {len(layers)} layers) — CE nats")
    print(_sep)
    print(f"  {'Mode':<6}  {'Δ/g (%)':>{_WA}}  {'Raw Δ':>{_WA}}  {'g (CE nats)':>{_WA}}")
    print(_sep)
    print(f"  {'Maj.':<6}  {_mean_m:>7.1f}±{_std_m:<4.1f}  {_mean_dm:{_WA}.4f}  {g_maj:{_WA}.4f}")
    print(f"  {'OOD':<6}  {_mean_o:>7.1f}±{_std_o:<4.1f}  {_mean_do:{_WA}.4f}  {g_ood:{_WA}.4f}")
    print(f"  {'Min.':<6}  {_mean_n:>7.1f}±{_std_n:<4.1f}  {_mean_dn:{_WA}.4f}  {g_min:{_WA}.4f}")
    print(_sep)
    print()
    # ─────────────────────────────────────────────────────────────────────

    return fig, all_results
