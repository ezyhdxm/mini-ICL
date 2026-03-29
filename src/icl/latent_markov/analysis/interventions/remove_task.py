"""Causal intervention: remove task subspace (latent Markov, non-padded).

Math
----
At layer l, hidden states are decomposed via joint OLS:

    h_t ≈ π_t W_task + onehot(x_t) W_tok + logit_t W_logit + b      (*)

where π_t ∈ Δ^{K-1} is the Bayesian posterior over K major tasks.
Joint fitting ensures W_task captures task-specific variation after
controlling for token identity and model logits (Frisch–Waugh–Lovell).

The task subspace S = rowspan(W_task) is extracted via SVD:

    W̃ = W_task − mean(W_task)     (optionally centre rows)
    W̃ = U Σ V^T                   → V_r = first r right singular vectors
    P_S = V_r V_r^T                orthogonal projector onto S  ∈ R^{D×D}

Causal intervention at layer l:

    h' = h − s · h P_S             (s = 1 removes S entirely)

If the model encodes task identity in S, removing it raises CE loss on
major sequences (whose task is in-distribution) while having less effect
on OOD sequences (whose task lies outside S).
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
#  Helpers
# ---------------------------------------------------------------------------

def _task_projector(W_task: torch.Tensor, center: bool):
    """SVD of W_task → orthogonal projector P_S = V_r V_r^T.

    Returns (P_S, rank).
    """
    W = W_task.float()
    if center:
        W = W - W.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(W, full_matrices=False)
    rank = int((S > 1e-6 * S[0]).sum().item())
    basis = Vt[:rank].T                     # (D, rank)
    return basis @ basis.T, rank             # (D, D), int


def _generate_eval_batches(sampler, mode, n_samples, B):
    """Generate sequences on CPU. Returns list of (B, seq_len) int tensors."""
    batches = []
    for _ in range((n_samples + B - 1) // B):
        out = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
        s = out[0] if isinstance(out, (tuple, list)) else out
        if s.dim() == 3:
            s = s.squeeze(0)
        batches.append(s.cpu())
    return batches


def _hook_remove_subspace(P, scale):
    """Create a forward hook that removes the P-subspace component.

    h' = h − s · (h P)
    """
    def hook_fn(module, inp, out, _P=P, _s=scale):
        h = out if torch.is_tensor(out) else out[0]
        h_mod = h - _s * (h @ _P)
        return h_mod if torch.is_tensor(out) else (h_mod,) + out[1:]
    return hook_fn


def _compute_ce_over_batches(
    model, batches, eval_positions, device, ce_fn,
    *, layer=None, P=None, scale=1.0, extraction_point="post_attn",
):
    """Average CE loss at *eval_positions* across *batches*.

    If layer/P are provided, registers an intervention hook at that layer.
    Otherwise computes baseline (unhooked) CE.

    Returns (avg_loss_scalar, per_position_list).
    """
    by_pos = {p: [] for p in eval_positions}

    for s_cpu in batches:
        s = s_cpu.to(device)

        handle = None
        if layer is not None and P is not None:
            hook_target = (
                model.layers[layer]
                if extraction_point == "post_mlp"
                else model.layers[layer].attn_block
            )
            handle = hook_target.register_forward_hook(
                _hook_remove_subspace(P, scale),
            )
        try:
            with torch.no_grad():
                logits = model(s)
        finally:
            if handle is not None:
                handle.remove()

        for p in eval_positions:
            if p + 1 >= s.shape[1]:
                continue
            by_pos[p].append(
                ce_fn(logits[:, p, :], s[:, p + 1].long()).mean().item()
            )

        del s, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_pos = [np.mean(v) for v in by_pos.values() if v]
    avg = float(np.mean(per_pos)) if per_pos else float("nan")
    return avg, per_pos


# ---------------------------------------------------------------------------
#  intervene_remove_task_subspace  (single layer, self-contained)
# ---------------------------------------------------------------------------

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
    center_task_vecs: bool = False,
    scale: float = 1.0,
    verbose: bool = False,
    print_summary: bool = True,
    extraction_point: str = "post_attn",
    probe_method: str = "ols",
) -> dict:
    """Remove the task-subspace component from layer-l hiddens.

    Steps
    -----
    1. Fit probe (OLS or averaging)  →  W_task ∈ R^{K×D}
    2. SVD(W_task)  →  projector P_S = V_r V_r^T
    3. Hook layer l:  h' = h − s · (h P_S)
    4. Measure CE loss increase on major and OOD sequences

    Returns dict with baseline/intervened CE losses and metadata.
    """
    from icl.latent_markov.analysis.probes import train_linear_hidden_predictor

    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    device = config.device

    sampler_maj, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)
    seq_len = sampler_maj.seq_len

    # ---- 1–2. Probe → task projector P_S ----
    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    if probe_method == "averaging":
        from icl.latent_markov.analysis.variance import get_token_conditioned_hiddens
        anova_positions = [p for p in fit_positions if p < seq_len - 1]
        all_hiddens_anova, anova_info = get_token_conditioned_hiddens(
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
        W_task = pooled.mean(dim=0) - grand  # (K, D)
        del all_hiddens_anova
    else:
        fit_res = train_linear_hidden_predictor(
            exp_name=exp_name, layer=layer, n_samples=fit_n_samples,
            positions=fit_positions, sample_mode="major", step=step,
            n_minor=-1, print_summary=False, skip_baselines=True,
            extraction_point=extraction_point,
        )
        W_task = fit_res["model_weight"].float()

    P_S, rank = _task_projector(W_task, center_task_vecs)
    P_S = P_S.to(device)

    if verbose:
        logger.info(
            f"[remove-task L{layer}] rank={rank}, R²={fit_res['val_r2']:.4f}"
        )

    # ---- 3–4. Intervention experiments ----
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    ce_fn = torch.nn.CrossEntropyLoss(reduction="none")
    maj_batches = _generate_eval_batches(sampler_maj, "major", n_samples_eval, B)
    ood_batches = _generate_eval_batches(sampler_ood, "minor", n_samples_eval, B)

    base_maj, _ = _compute_ce_over_batches(
        model, maj_batches, eval_positions, device, ce_fn,
        extraction_point=extraction_point,
    )
    int_maj, _ = _compute_ce_over_batches(
        model, maj_batches, eval_positions, device, ce_fn,
        layer=layer, P=P_S, scale=scale, extraction_point=extraction_point,
    )
    base_ood, _ = _compute_ce_over_batches(
        model, ood_batches, eval_positions, device, ce_fn,
        extraction_point=extraction_point,
    )
    int_ood, _ = _compute_ce_over_batches(
        model, ood_batches, eval_positions, device, ce_fn,
        layer=layer, P=P_S, scale=scale, extraction_point=extraction_point,
    )

    model.cpu()
    del model, maj_batches, ood_batches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    d_m, d_o = int_maj - base_maj, int_ood - base_ood
    pct_m = 100 * d_m / base_maj if base_maj > 0 else float("nan")
    pct_o = 100 * d_o / base_ood if base_ood > 0 else float("nan")

    results = {
        "baseline_loss_major": base_maj,
        "intervened_loss_major": int_maj,
        "delta_loss_major": d_m,
        "pct_increase_major": pct_m,
        "baseline_loss_ood": base_ood,
        "intervened_loss_ood": int_ood,
        "delta_loss_ood": d_o,
        "pct_increase_ood": pct_o,
        "layer": layer,
        "scale": scale,
        "task_subspace_rank": rank,
    }

    if print_summary:
        print(f"\n{'='*60}")
        print(f"Remove Task Subspace (latent, layer {layer}, scale={scale})")
        print(f"{'='*60}")
        print(f"  Task subspace rank: {rank}")
        print(f"  {'Metric':<25} {'Major':>12} {'OOD':>12}")
        print(f"  {'-'*49}")
        for lbl, m, o in [
            ("Baseline CE", base_maj, base_ood),
            ("Intervened CE", int_maj, int_ood),
            ("Δ CE", d_m, d_o),
        ]:
            print(f"  {lbl:<25} {m:>12.4f} {o:>12.4f}")
        print(f"  {'% increase':<25} {pct_m:>11.1f}% {pct_o:>11.1f}%")

    return results


# ---------------------------------------------------------------------------
#  plot_intervention_remove_task_across_layers  (efficient multi-layer sweep)
# ---------------------------------------------------------------------------

def plot_intervention_remove_task_across_layers(
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
    figsize: tuple = (10, 4.5),
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    extraction_point: str = "post_attn",
    probe_method: str = "ols",
    verbose: bool = False,
):
    """Sweep task-subspace removal across layers; bar plot of Δ CE loss.

    Efficient implementation:
    - Loads model once (not per layer).
    - Fits probes for all layers in a single forward pass via
      ``_collect_multi_layer_data``.
    - Generates evaluation data once (shared across layers).
    - Computes baseline CE once (no hook, same for all layers).
    - Only the hooked forward pass is per-layer.

    Returns ``(fig, all_results)``.
    """
    import matplotlib.pyplot as plt
    from icl.latent_markov.analysis.probes import (
        _collect_multi_layer_data, _fit_probe,
    )

    # =================================================================
    #  1. Setup: config, samplers, layers
    # =================================================================
    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    if layers is None:
        layers = list(range(config.model.num_layers))
    device = config.device

    sampler_maj, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)
    sampler_min, _, _ = get_latent_sampler(exp_name, n_minor=n_minor, n_ood=0)
    seq_len = sampler_maj.seq_len

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    # =================================================================
    #  2. Fit probes for all layers
    # =================================================================
    projectors, ranks, probe_r2s = {}, {}, {}

    if probe_method == "averaging":
        logger.info(
            f"[remove-task sweep] ANOVA averaging for {len(layers)} layers ..."
        )
        from icl.latent_markov.analysis.variance import get_token_conditioned_hiddens
        anova_positions = [p for p in fit_positions if p < seq_len - 1]
        all_hiddens_anova, anova_info = get_token_conditioned_hiddens(
            exp_name, layers=layers, batch_size=B,
            positions_of_interest=anova_positions,
            step=step, n_minor=0,
            extraction_point=extraction_point,
        )
        pos_to_anova = {p: i for i, p in enumerate(anova_info["positions"])}
        n_uniq = anova_info["n_unique_tokens"]
        V_max = all_hiddens_anova.shape[2]
        n_major_anova = all_hiddens_anova.shape[3]

        for li, l in enumerate(layers):
            parts = []
            for p in anova_positions:
                if p not in pos_to_anova:
                    continue
                pi = pos_to_anova[p]
                V_p = n_uniq.get(p, V_max)
                parts.append(
                    all_hiddens_anova[li, pi, :V_p, :n_major_anova].float().mean(dim=-2)
                )
            if not parts:
                raise ValueError(
                    f"[remove-task sweep] probe_method='averaging': "
                    f"no valid fit_positions for layer {l}."
                )
            min_V = min(p.shape[0] for p in parts)
            parts = [p[:min_V] for p in parts]
            demeaned = [cm - cm.mean(dim=(0, 1), keepdim=True) for cm in parts]
            pooled = torch.stack(demeaned, dim=0).mean(dim=0)  # (V, K, D)
            grand = pooled.mean(dim=(0, 1))
            task_vecs = pooled.mean(dim=0) - grand  # (K, D)
            P, r = _task_projector(task_vecs, center_task_vecs)
            projectors[l] = P.to(device)
            ranks[l] = r
            probe_r2s[l] = float("nan")

        del all_hiddens_anova
    else:
        logger.info(
            f"[remove-task sweep] fitting OLS probes for {len(layers)} layers ..."
        )
        probe_data = _collect_multi_layer_data(
            exp_name, layers, n_samples=fit_n_samples, positions=fit_positions,
            sample_mode="major", n_minor=-1, step=step,
            extraction_point=extraction_point,
        )
        N_seq = probe_data["hiddens_by_layer"][layers[0]].shape[0]
        seq_perm = torch.randperm(N_seq)

        for l in layers:
            res = _fit_probe(
                probe_data["hiddens_by_layer"][l], probe_data["posteriors"],
                probe_data["logits"], probe_data["real_tokens"],
                n_major=probe_data["n_major"], n_tasks=probe_data["n_tasks"],
                layer=l, positions=probe_data["positions"],
                seq_perm=seq_perm, skip_baselines=True, sample_mode="major",
            )
            P, r = _task_projector(res["model_weight"], center_task_vecs)
            projectors[l] = P.to(device)
            ranks[l] = r
            probe_r2s[l] = res["val_r2"]

        del probe_data

    gc.collect()

    # =================================================================
    #  3. Load model for intervention (probe collection freed its copy)
    # =================================================================
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(device)

    # =================================================================
    #  4. Generate evaluation data once (shared across all layers)
    # =================================================================
    maj_batches = _generate_eval_batches(sampler_maj, "major", n_samples_eval, B)
    ood_batches = _generate_eval_batches(sampler_ood, "minor", n_samples_eval, B)
    min_batches = _generate_eval_batches(sampler_min, "minor", n_samples_eval, B)

    ce_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # =================================================================
    #  5. Baseline CE (no hook — identical for every layer)
    # =================================================================
    logger.info("[remove-task sweep] computing baselines ...")
    # Evaluate over all positions (excluding pos-0 if not already in list)
    eval_pos_non0 = [p for p in eval_positions if p != 0]
    if not eval_pos_non0:
        eval_pos_non0 = eval_positions  # fallback: use all

    base_maj, base_maj_pp = _compute_ce_over_batches(
        model, maj_batches, eval_pos_non0, device, ce_fn,
        extraction_point=extraction_point,
    )
    base_ood, base_ood_pp = _compute_ce_over_batches(
        model, ood_batches, eval_pos_non0, device, ce_fn,
        extraction_point=extraction_point,
    )
    base_min, base_min_pp = _compute_ce_over_batches(
        model, min_batches, eval_pos_non0, device, ce_fn,
        extraction_point=extraction_point,
    )

    # Position-0 CE: uninformative prior (no task context seen yet).
    # gain = pos0_loss - base_loss  ≈ how much CE the model saves from context.
    base_maj_0, _ = _compute_ce_over_batches(
        model, maj_batches, [0], device, ce_fn,
        extraction_point=extraction_point,
    )
    base_ood_0, _ = _compute_ce_over_batches(
        model, ood_batches, [0], device, ce_fn,
        extraction_point=extraction_point,
    )
    base_min_0, _ = _compute_ce_over_batches(
        model, min_batches, [0], device, ce_fn,
        extraction_point=extraction_point,
    )
    gain_maj = base_maj_0 - base_maj
    gain_ood = base_ood_0 - base_ood
    gain_min = base_min_0 - base_min

    # =================================================================
    #  6. Per-layer: hooked forward pass
    #
    #     h' = h − s · h P_S   (remove task-subspace projection)
    # =================================================================
    all_results = {}
    for l in layers:
        logger.info(f"[remove-task sweep] layer {l} ...")
        P = projectors[l]

        int_maj, int_maj_pp = _compute_ce_over_batches(
            model, maj_batches, eval_pos_non0, device, ce_fn,
            layer=l, P=P, scale=scale, extraction_point=extraction_point,
        )
        int_ood, int_ood_pp = _compute_ce_over_batches(
            model, ood_batches, eval_pos_non0, device, ce_fn,
            layer=l, P=P, scale=scale, extraction_point=extraction_point,
        )
        int_min, int_min_pp = _compute_ce_over_batches(
            model, min_batches, eval_pos_non0, device, ce_fn,
            layer=l, P=P, scale=scale, extraction_point=extraction_point,
        )

        d_m = int_maj - base_maj
        d_o = int_ood - base_ood
        d_n = int_min - base_min
        all_results[l] = {
            "baseline_loss_major": base_maj,
            "intervened_loss_major": int_maj,
            "delta_loss_major": d_m,
            "pct_increase_major": (
                100 * d_m / base_maj if base_maj > 0 else float("nan")
            ),
            "baseline_loss_ood": base_ood,
            "intervened_loss_ood": int_ood,
            "delta_loss_ood": d_o,
            "pct_increase_ood": (
                100 * d_o / base_ood if base_ood > 0 else float("nan")
            ),
            "baseline_loss_minor": base_min,
            "intervened_loss_minor": int_min,
            "delta_loss_minor": d_n,
            "pct_increase_minor": (
                100 * d_n / base_min if base_min > 0 else float("nan")
            ),
            "task_subspace_rank": ranks[l],
            "probe_r2": probe_r2s[l],
            "baseline_per_pos_major": base_maj_pp,
            "intervened_per_pos_major": int_maj_pp,
            "baseline_per_pos_ood": base_ood_pp,
            "intervened_per_pos_ood": int_ood_pp,
            "baseline_per_pos_minor": base_min_pp,
            "intervened_per_pos_minor": int_min_pp,
        }

    # Cleanup
    model.cpu()
    del model, maj_batches, ood_batches, min_batches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # =================================================================
    #  7. Summary table
    # =================================================================
    if verbose:
        print(f"\n{'='*85}")
        print(f"Remove Task Subspace — Latent (scale={scale})")
        print(f"{'='*85}")
        print(
            f"  {'Layer':<7} {'Rank':<6} {'Probe R²':<10} "
            f"{'Δ CE maj':<11} {'Δ CE ood':<11} {'Δ CE min':<11} "
            f"{'% maj':<9} {'% ood':<9} {'% min':<9}"
        )
        print(f"  {'-'*83}")
        for l in layers:
            r = all_results[l]
            print(
                f"  {l:<7} {ranks[l]:<6} {probe_r2s[l]:<10.4f} "
                f"{r['delta_loss_major']:<11.4f} {r['delta_loss_ood']:<11.4f} "
                f"{r['delta_loss_minor']:<11.4f} "
                f"{r['pct_increase_major']:<8.1f}% "
                f"{r['pct_increase_ood']:<8.1f}% "
                f"{r['pct_increase_minor']:<8.1f}%"
            )
        print()

    # =================================================================
    #  8. Bar plot: Δ𝓛/g (%) per layer — colours / style match Plot 1 of
    #     plot_optimal_orth_direction_across_layers
    # =================================================================
    COLORS = {"maj": "#2166ac", "ood": "#d6604d", "minor": "#1a9850"}
    bw_bar = 0.22
    g_step = 0.24
    MC     = {"maj": -g_step, "ood": 0.0, "minor": +g_step}

    # Normalized deltas: Δ𝓛 / g × 100  (% of ICL gain disrupted)
    norm_maj = [all_results[l]["delta_loss_major"] / gain_maj * 100 for l in layers]
    norm_ood = [all_results[l]["delta_loss_ood"]   / gain_ood * 100 for l in layers]
    norm_min = [all_results[l]["delta_loss_minor"]  / gain_min * 100 for l in layers]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    x = np.arange(len(layers))

    ax.bar(x + MC["maj"],   norm_maj, bw_bar,
           label="Maj.", color=COLORS["maj"],   linewidth=0, zorder=3)
    ax.bar(x + MC["ood"],   norm_ood, bw_bar,
           label="OOD",  color=COLORS["ood"],   linewidth=0, zorder=3)
    ax.bar(x + MC["minor"], norm_min, bw_bar,
           label="Min.", color=COLORS["minor"], linewidth=0, zorder=3)

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
        _nm = _r["delta_loss_major"] / gain_maj * 100
        _no = _r["delta_loss_ood"]   / gain_ood * 100
        _nn = _r["delta_loss_minor"]  / gain_min * 100
        print(f"  {_l:>5}  {_nm:>{_W}.1f}%  {_no:>{_W}.1f}%  {_nn:>{_W}.1f}%  |  "
              f"{_r['delta_loss_major']:>{_W}.4f}  {_r['delta_loss_ood']:>{_W}.4f}"
              f"  {_r['delta_loss_minor']:>{_W}.4f}")
    print(_ln)
    print(f"  {'Gain':>5}  {'':>{_W}}   {'':>{_W}}   {'':>{_W}}   |  "
          f"{gain_maj:{_W}.4f}  {gain_ood:{_W}.4f}  {gain_min:{_W}.4f}")
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
    print(f"  {'Maj.':<6}  {_mean_m:>7.1f}±{_std_m:<4.1f}  {_mean_dm:{_WA}.4f}  {gain_maj:{_WA}.4f}")
    print(f"  {'OOD':<6}  {_mean_o:>7.1f}±{_std_o:<4.1f}  {_mean_do:{_WA}.4f}  {gain_ood:{_WA}.4f}")
    print(f"  {'Min.':<6}  {_mean_n:>7.1f}±{_std_n:<4.1f}  {_mean_dn:{_WA}.4f}  {gain_min:{_WA}.4f}")
    print(_sep)
    print()
    # ─────────────────────────────────────────────────────────────────────

    return fig, all_results
