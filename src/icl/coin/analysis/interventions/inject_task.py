"""Causal injection: replace the task-subspace component with a fitted task vector.

For each (source, target) pair the hook replaces the task-identifying
component of the hidden state with the reference vector from a different task,
measuring whether the model's output shifts to match the injected task.
"""

from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.coin.analysis.probes import train_linear_hidden_predictor_coin
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def intervene_inject_task_vector_coin(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_eval: int = 500,
    n_ood: int = 30,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    scale: float = 1.0,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Causal injection: replace the task-subspace component of hidden states
    with a fitted task vector.

    For every (source_k, target_j) pair with k != j the hook performs::

        h' = h  -  P_task @ h  +  ref_vec[k]

    where ref_vec[k] = W_k (uncentered) or W_k - W̄ (centered).

    For Major→Major data the two are equivalent because the posterior
    sums to 1, making the task-mean W̄ a constant offset.  For OOD
    data, centering matters: the uncentered projector (rank K) also
    removes and replaces the W̄ component, whereas the centered
    projector (rank K-1) leaves it untouched.

    Returns a dict with reference vectors, per-pair metrics (KL, task-ID
    accuracy, loss delta), and OOD-injection results.
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
    K_major = sampler_major.n_major_tasks
    task_probs = sampler_major.major_p.float().to(device)  # (K, V)

    # ---- 1. Fit task subspace ----
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
    _, S_tv, Vt_tv = torch.linalg.svd(task_vecs, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    basis = Vt_tv[:rank].T  # (D, rank)
    P_task = (basis @ basis.T).to(device)  # (D, D)

    if verbose:
        logger.info(
            f"[inject] Task subspace rank={rank} "
            f"(centered={center_task_vecs}), "
            f"posterior fit R²={fit_res['val_r2']:.4f}"
        )

    W_dev = W_fit.to(device)
    if center_task_vecs:
        ref_vecs = W_dev - W_dev.mean(dim=0)
    else:
        ref_vecs = W_dev.clone()

    if verbose:
        for k in range(K_major):
            logger.info(f"[inject] ref_vec[{k}] norm = {ref_vecs[k].norm():.4f}")

    # ---- 2. Injection experiments ----
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def _kl_div(q_logits, p_dist):
        """KL(softmax(q_logits) || p_dist), averaged over batch dim."""
        log_q = torch.log_softmax(q_logits, dim=-1)
        log_p = torch.log(p_dist.clamp_min(1e-12))
        return (torch.exp(log_q) * (log_q - log_p)).sum(dim=-1).mean()

    def _run_injection(sampler, gen_mode, source_k, n_samples,
                       target_task=None):
        """Run baseline + injection for a given source task k.

        When *target_task* is not None the sampler generates data
        conditioned on that task (Major→Major); otherwise it samples
        freely (OOD / minor).
        """
        tv_k = ref_vecs[source_k]

        baseline_losses, injected_losses = [], []
        kl_to_tasks = [[] for _ in range(K_major)]
        kl_base_to_tasks = [[] for _ in range(K_major)]
        task_id_correct = []

        n_batches = (n_samples + B - 1) // B
        for _ in range(n_batches):
            gen_out = sampler.generate(
                mode=gen_mode, task=target_task, num_samples=B, epochs=1,
            )
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)

            with torch.no_grad():
                logits_base = model(samples)

            def _inject_hook(module, inp, out, _tv=tv_k, _P=P_task):
                h = out if torch.is_tensor(out) else out[0]
                h_new = h - (h @ _P) + _tv
                if torch.is_tensor(out):
                    return h_new
                return (h_new,) + out[1:]

            handle = model.layers[layer].attn_block.register_forward_hook(
                _inject_hook,
            )
            try:
                with torch.no_grad():
                    logits_inj = model(samples)
            finally:
                handle.remove()

            for p in eval_positions:
                if p + 1 >= samples.shape[1]:
                    continue
                targets = samples[:, p + 1]

                baseline_losses.append(
                    ce_loss_fn(logits_base[:, p, :], targets).mean().item()
                )
                injected_losses.append(
                    ce_loss_fn(logits_inj[:, p, :], targets).mean().item()
                )

                logits_at_p = logits_inj[:, p, :]
                base_at_p = logits_base[:, p, :]
                kl_per_task_batch = []
                for t in range(K_major):
                    kl_val = _kl_div(logits_at_p, task_probs[t]).item()
                    kl_to_tasks[t].append(kl_val)
                    kl_per_task_batch.append(kl_val)
                    kl_base_to_tasks[t].append(
                        _kl_div(base_at_p, task_probs[t]).item()
                    )
                task_id_correct.append(
                    int(np.argmin(kl_per_task_batch) == source_k)
                )

        bl = float(np.mean(baseline_losses)) if baseline_losses else 0.0
        il = float(np.mean(injected_losses)) if injected_losses else 0.0
        kl_means = [
            float(np.mean(kl_to_tasks[t])) if kl_to_tasks[t] else 0.0
            for t in range(K_major)
        ]
        kl_base_means = [
            float(np.mean(kl_base_to_tasks[t]))
            if kl_base_to_tasks[t] else 0.0
            for t in range(K_major)
        ]
        acc = float(np.mean(task_id_correct)) if task_id_correct else 0.0

        return {
            "baseline_loss": bl,
            "injected_loss": il,
            "delta_loss": il - bl,
            "kl_to_each_task": np.array(kl_means),
            "kl_base_to_each_task": np.array(kl_base_means),
            "task_id_accuracy": acc,
        }

    # --- major -> major injection ---
    injection_results = {}
    for source_k in range(K_major):
        injection_results[source_k] = {}
        for target_j in range(K_major):
            if target_j == source_k:
                continue
            res = _run_injection(
                sampler_major, "major", source_k, n_samples_eval,
                target_task=target_j,
            )
            injection_results[source_k][target_j] = res

    # --- major -> OOD injection ---
    ood_results = {}
    for source_k in range(K_major):
        res = _run_injection(sampler_ood, "minor", source_k, n_samples_eval)
        ood_results[source_k] = res

    # ---- 3. Summary ----
    if print_summary:
        print()
        print("=" * 70)
        print(f"Task-Vector Injection  (layer {layer}, scale={scale})")
        print("=" * 70)
        print(f"  Task subspace rank: {rank}  |  K_major: {K_major}")
        print(f"  Ref vecs: from probe W_k{' - W̄' if center_task_vecs else ''}")
        print()

        print("  Major → Major injection:")
        print(f"  {'src→tgt':<10} {'TaskID acc':>10} "
              + "".join(f"{'KL→t' + str(t):>10}" for t in range(K_major))
              + f"  {'ΔLoss':>8}")
        print(f"  {'-' * (30 + 10 * K_major)}")
        for source_k in range(K_major):
            for target_j in range(K_major):
                if target_j == source_k:
                    continue
                r = injection_results[source_k][target_j]
                kl_str = "".join(f"{r['kl_to_each_task'][t]:>10.4f}"
                                 for t in range(K_major))
                print(f"  {source_k}→{target_j:<7} {r['task_id_accuracy']:>10.1%} "
                      f"{kl_str}  {r['delta_loss']:>8.4f}")

        print()
        print("  Major → OOD injection:")
        print(f"  {'src':<10} {'TaskID acc':>10} "
              + "".join(f"{'KL→t' + str(t):>10}" for t in range(K_major))
              + f"  {'ΔLoss':>8}")
        print(f"  {'-' * (30 + 10 * K_major)}")
        for source_k in range(K_major):
            r = ood_results[source_k]
            kl_str = "".join(f"{r['kl_to_each_task'][t]:>10.4f}"
                             for t in range(K_major))
            print(f"  {source_k:<10} {r['task_id_accuracy']:>10.1%} "
                  f"{kl_str}  {r['delta_loss']:>8.4f}")
        print()

    return {
        "layer": layer,
        "rank": rank,
        "ref_vecs": ref_vecs.cpu(),
        "task_probs": task_probs.cpu(),
        "injection_results": injection_results,
        "ood_results": ood_results,
        "K_major": K_major,
    }


def plot_inject_task_vector_across_layers_coin(
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
    show: bool = True,
):
    """
    Sweep ``intervene_inject_task_vector_coin`` across layers and
    produce three diagnostic figures.

    Returns ``(fig_heatmap, fig_kl, fig_ood, all_results)``.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
        layers = list(range(config.model.num_layers))

    all_results = {}
    for l in layers:
        logger.info(f"[inject sweep] layer {l} ...")
        res = intervene_inject_task_vector_coin(
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
            print_summary=True,
        )
        all_results[l] = res

    K = all_results[layers[0]]["K_major"]

    # ---- 1. Task-ID accuracy heatmap (one subplot per layer) ----
    fig_hm, axes_hm = plt.subplots(
        1, len(layers),
        figsize=(3.5 * len(layers), 3.5),
        squeeze=False,
    )
    for idx, l in enumerate(layers):
        mat = np.full((K, K), np.nan)
        inj = all_results[l]["injection_results"]
        for sk in range(K):
            for tj in range(K):
                if tj == sk:
                    continue
                mat[sk, tj] = inj[sk][tj]["task_id_accuracy"]
        ax = axes_hm[0, idx]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="equal")
        for sk in range(K):
            for tj in range(K):
                if sk == tj:
                    ax.text(tj, sk, "-", ha="center", va="center",
                            fontsize=11, color="gray")
                else:
                    val = mat[sk, tj]
                    ax.text(tj, sk, f"{val:.0%}", ha="center", va="center",
                            fontsize=11,
                            color="white" if val > 0.65 else "black")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xlabel("Target task", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Source (injected) task", fontsize=11)
        ax.set_title(f"Layer {l}", fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig_hm.suptitle("Task-ID Accuracy after Injection", fontsize=15, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_hm)

    # ---- 2. KL divergence chart (hlines for baseline, bars for injection) ----
    fig_kl, (ax_kl1, ax_kl2) = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(layers))
    bw = 0.3

    def _off_diag_kl(layer, key_fn):
        inj = all_results[layer]["injection_results"]
        return float(np.mean([
            key_fn(inj[sk][tj], sk, tj)
            for sk in range(K) for tj in range(K) if tj != sk
        ]))

    kl_base_orig = _off_diag_kl(layers[0], lambda r, sk, tj: r["kl_base_to_each_task"][tj])
    kl_base_inj = _off_diag_kl(layers[0], lambda r, sk, tj: r["kl_base_to_each_task"][sk])
    ax_kl1.axhline(kl_base_orig, color="#9E9E9E", ls="--", lw=1.5, alpha=0.85,
                   label=rf"KL(no inj $\|$ original) = {kl_base_orig:.2f}")
    ax_kl1.axhline(kl_base_inj, color="#FF9800", ls="--", lw=1.5, alpha=0.85,
                   label=rf"KL(no inj $\|$ injected) = {kl_base_inj:.2f}")

    avg_kl_inj_inj, avg_kl_inj_orig = [], []
    for l in layers:
        avg_kl_inj_inj.append(_off_diag_kl(l, lambda r, sk, tj: r["kl_to_each_task"][sk]))
        avg_kl_inj_orig.append(_off_diag_kl(l, lambda r, sk, tj: r["kl_to_each_task"][tj]))

    ax_kl1.bar(x - 0.5 * bw, avg_kl_inj_inj, bw,
               label=r"KL(after inj $\|$ injected)", color="#4CAF50", alpha=0.85)
    ax_kl1.bar(x + 0.5 * bw, avg_kl_inj_orig, bw,
               label=r"KL(after inj $\|$ original)", color="#F44336", alpha=0.85)
    for i, (v3, v4) in enumerate(zip(avg_kl_inj_inj, avg_kl_inj_orig)):
        ax_kl1.text(x[i] - 0.5 * bw, v3, f"{v3:.2f}", ha="center", va="bottom", fontsize=7)
        ax_kl1.text(x[i] + 0.5 * bw, v4, f"{v4:.2f}", ha="center", va="bottom", fontsize=7)
    ax_kl1.set_xlabel("Layer", fontsize=14)
    ax_kl1.set_ylabel("KL divergence", fontsize=14)
    ax_kl1.set_title("KL to Task: Baseline vs After Injection", fontsize=13)
    ax_kl1.set_xticks(x)
    ax_kl1.set_xticklabels([str(l) for l in layers])
    ax_kl1.legend(fontsize=9)
    ax_kl1.grid(axis="y", alpha=0.3)

    avg_acc = []
    for l in layers:
        inj = all_results[l]["injection_results"]
        accs = []
        for sk in range(K):
            for tj in range(K):
                if tj == sk:
                    continue
                accs.append(inj[sk][tj]["task_id_accuracy"])
        avg_acc.append(float(np.mean(accs)))

    ax_kl2.bar(x, avg_acc, 0.5, color="#2196F3", alpha=0.85)
    for i, a in enumerate(avg_acc):
        ax_kl2.text(x[i], a, f"{a:.0%}", ha="center", va="bottom",
                    fontsize=11)
    ax_kl2.set_xlabel("Layer", fontsize=14)
    ax_kl2.set_ylabel("Task-ID accuracy", fontsize=14)
    ax_kl2.set_title("Avg Task-ID Accuracy (matches injected?)", fontsize=13)
    ax_kl2.set_xticks(x)
    ax_kl2.set_xticklabels([str(l) for l in layers])
    ax_kl2.set_ylim(0, 1.15)
    ax_kl2.axhline(1 / K, color="gray", linestyle="--", alpha=0.5,
                    label=f"Chance (1/{K})")
    ax_kl2.legend(fontsize=11)
    ax_kl2.grid(axis="y", alpha=0.3)

    fig_kl.suptitle("Task-Vector Injection: KL & Accuracy (Major→Major)",
                    fontsize=15, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_kl)

    # ---- 3. OOD injection line plot ----
    fig_ood, (ax_o1, ax_o2) = plt.subplots(1, 2, figsize=(14, 5))

    for sk in range(K):
        accs = [all_results[l]["ood_results"][sk]["task_id_accuracy"]
                for l in layers]
        ax_o1.plot(layers, accs, "o-", label=f"Inject task {sk}",
                   linewidth=2, markersize=7)
    ax_o1.axhline(1 / K, color="gray", linestyle="--", alpha=0.5,
                  label=f"Chance (1/{K})")
    ax_o1.set_xlabel("Layer", fontsize=14)
    ax_o1.set_ylabel("Task-ID accuracy", fontsize=14)
    ax_o1.set_title("OOD: Task-ID Accuracy per Injected Task", fontsize=13)
    ax_o1.set_xticks(layers)
    ax_o1.set_ylim(0, 1.1)
    ax_o1.legend(fontsize=11)
    ax_o1.grid(alpha=0.3)

    for sk in range(K):
        kl_inj = [all_results[l]["ood_results"][sk]["kl_to_each_task"][sk]
                  for l in layers]
        ax_o2.plot(layers, kl_inj, "o-", label=f"Inject task {sk}",
                   linewidth=2, markersize=7)
    ax_o2.set_xlabel("Layer", fontsize=14)
    ax_o2.set_ylabel("KL(output || injected task)", fontsize=14)
    ax_o2.set_title("OOD: KL to Injected Task Distribution", fontsize=13)
    ax_o2.set_xticks(layers)
    ax_o2.legend(fontsize=11)
    ax_o2.grid(alpha=0.3)

    fig_ood.suptitle("Task-Vector Injection into OOD Sequences",
                     fontsize=15, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_ood)

    return fig_hm, fig_kl, fig_ood, all_results
