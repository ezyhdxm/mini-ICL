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
    *, layer=None, P=None, scale=1.0,
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
            handle = model.layers[layer].attn_block.register_forward_hook(
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
) -> dict:
    """Remove the task-subspace component from layer-l hiddens.

    Steps
    -----
    1. Fit joint OLS probe (*)  →  W_task ∈ R^{K×D}
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
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", step=step,
        n_minor=-1, print_summary=False, skip_baselines=True,
    )
    P_S, rank = _task_projector(fit_res["model_weight"], center_task_vecs)
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
    )
    int_maj, _ = _compute_ce_over_batches(
        model, maj_batches, eval_positions, device, ce_fn,
        layer=layer, P=P_S, scale=scale,
    )
    base_ood, _ = _compute_ce_over_batches(
        model, ood_batches, eval_positions, device, ce_fn,
    )
    int_ood, _ = _compute_ce_over_batches(
        model, ood_batches, eval_positions, device, ce_fn,
        layer=layer, P=P_S, scale=scale,
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
    seq_len = sampler_maj.seq_len

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    # =================================================================
    #  2. Fit probes for all layers (one model load + one forward pass)
    #
    #     h_t ≈ π_t W_task + onehot(x_t) W_tok + logit_t W_logit + b
    #     → W_task gives the task-subspace directions per layer.
    # =================================================================
    logger.info(
        f"[remove-task sweep] fitting probes for {len(layers)} layers ..."
    )
    probe_data = _collect_multi_layer_data(
        exp_name, layers, n_samples=fit_n_samples, positions=fit_positions,
        sample_mode="major", n_minor=-1, step=step,
    )
    N_seq = probe_data["hiddens_by_layer"][layers[0]].shape[0]
    seq_perm = torch.randperm(N_seq)

    # Per-layer: SVD(W_task) → projector P_S = V_r V_r^T
    projectors, ranks, probe_r2s = {}, {}, {}
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

    ce_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # =================================================================
    #  5. Baseline CE (no hook — identical for every layer)
    # =================================================================
    logger.info("[remove-task sweep] computing baselines ...")
    base_maj, base_maj_pp = _compute_ce_over_batches(
        model, maj_batches, eval_positions, device, ce_fn,
    )
    base_ood, base_ood_pp = _compute_ce_over_batches(
        model, ood_batches, eval_positions, device, ce_fn,
    )

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
            model, maj_batches, eval_positions, device, ce_fn,
            layer=l, P=P, scale=scale,
        )
        int_ood, int_ood_pp = _compute_ce_over_batches(
            model, ood_batches, eval_positions, device, ce_fn,
            layer=l, P=P, scale=scale,
        )

        d_m = int_maj - base_maj
        d_o = int_ood - base_ood
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
            "task_subspace_rank": ranks[l],
            "probe_r2": probe_r2s[l],
            "baseline_per_pos_major": base_maj_pp,
            "intervened_per_pos_major": int_maj_pp,
            "baseline_per_pos_ood": base_ood_pp,
            "intervened_per_pos_ood": int_ood_pp,
        }

    # Cleanup
    model.cpu()
    del model, maj_batches, ood_batches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # =================================================================
    #  7. Summary table
    # =================================================================
    print(f"\n{'='*72}")
    print(f"Remove Task Subspace — Latent (scale={scale})")
    print(f"{'='*72}")
    print(
        f"  {'Layer':<7} {'Rank':<6} {'Probe R²':<10} "
        f"{'Δ CE maj':<11} {'Δ CE ood':<11} {'% maj':<9} {'% ood':<9}"
    )
    print(f"  {'-'*67}")
    for l in layers:
        r = all_results[l]
        print(
            f"  {l:<7} {ranks[l]:<6} {probe_r2s[l]:<10.4f} "
            f"{r['delta_loss_major']:<11.4f} {r['delta_loss_ood']:<11.4f} "
            f"{r['pct_increase_major']:<8.1f}% "
            f"{r['pct_increase_ood']:<8.1f}%"
        )

    # =================================================================
    #  8. Bar plot: Δ CE per layer
    # =================================================================
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(layers))
    bw = 0.35

    delta_maj = [all_results[l]["delta_loss_major"] for l in layers]
    delta_ood = [all_results[l]["delta_loss_ood"] for l in layers]

    ax.bar(x - bw / 2, delta_maj, bw,
           label="Major", color="#2196F3", alpha=0.85)
    ax.bar(x + bw / 2, delta_ood, bw,
           label="OOD", color="#FF9800", alpha=0.85)

    for i, (vm, vo) in enumerate(zip(delta_maj, delta_ood)):
        ax.text(x[i] - bw / 2, vm, f"{vm:.3f}",
                ha="center", va="bottom", fontsize=10)
        ax.text(x[i] + bw / 2, vo, f"{vo:.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Δ CE Loss", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, all_results
