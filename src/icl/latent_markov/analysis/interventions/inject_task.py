"""Task-vector injection intervention (latent Markov, non-padded)."""

import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger
from icl.latent_markov.analysis.probes import train_linear_hidden_predictor

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
#  intervene_inject_task_vector
# ---------------------------------------------------------------------------

def intervene_inject_task_vector(
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
    Latent counterpart of ``intervene_inject_task_vector_coin_nonpadded``.

    Replaces task-subspace content with a reference major-task vector:
        h' = h - (h @ P_task) + ref_vec[k]
    where ref_vec[k] = P_task (W_k + b) is computed directly from the
    fitted probe (no empirical averaging needed).
    """
    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    device = config.device

    sampler_major, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)

    seq_len = sampler_major.seq_len
    K_major = int(sampler_major.n_major_tasks)
    major_trans = sampler_major.major_trans_mat.float().to(device)  # (K, S^order, S)

    # ---- 1. Fit task subspace ----
    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor(
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
    basis = Vt_tv[:rank].T
    P_task = (basis @ basis.T).to(device)

    if verbose:
        logger.info(
            f"[inject latent] Task subspace rank={rank} "
            f"(centered={center_task_vecs}), posterior fit R2={fit_res['val_r2']:.4f}"
        )

    # ---- 2. Per-task reference vectors from fitted W ----
    W_dev = W_fit.to(device)
    if center_task_vecs:
        W_mean = W_dev.mean(dim=0)
        ref_vecs = W_dev - W_mean  # (K, D), each row already in task subspace
    else:
        ref_vecs = W_dev.clone()

    # ---- 3. Injection experiments ----
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def _kl_div_state_cond(q_logits, current_tokens, task_idx):
        """KL(softmax(q_logits) || P(next|current, task)), averaged over batch."""
        p_dist = major_trans[task_idx][current_tokens.long()]  # (B, S)
        log_q = torch.log_softmax(q_logits, dim=-1)
        log_p = torch.log(p_dist.clamp_min(1e-12))
        return (torch.exp(log_q) * (log_q - log_p)).sum(dim=-1).mean()

    def _run_injection(sampler, gen_mode, source_k, n_samples):
        """Inject source_k's vector into OOD/minor data."""
        tv_k = ref_vecs[source_k]
        baseline_losses, injected_losses = [], []
        kl_to_tasks = [[] for _ in range(K_major)]
        kl_base_to_tasks = [[] for _ in range(K_major)]
        task_id_correct = []

        n_batches = (n_samples + B - 1) // B
        for _ in range(n_batches):
            gen_out = sampler.generate(mode=gen_mode, task=None, num_samples=B, epochs=1)
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

            handle = model.layers[layer].attn_block.register_forward_hook(_inject_hook)
            try:
                with torch.no_grad():
                    logits_inj = model(samples)
            finally:
                handle.remove()

            for p in eval_positions:
                if p + 1 >= samples.shape[1]:
                    continue
                targets = samples[:, p + 1]
                current_tokens = samples[:, p]
                baseline_losses.append(ce_loss_fn(logits_base[:, p, :], targets).mean().item())
                injected_losses.append(ce_loss_fn(logits_inj[:, p, :], targets).mean().item())

                logits_at_p = logits_inj[:, p, :]
                base_at_p = logits_base[:, p, :]
                kl_per_task_batch = []
                for t in range(K_major):
                    kl_val = _kl_div_state_cond(logits_at_p, current_tokens, t).item()
                    kl_to_tasks[t].append(kl_val)
                    kl_per_task_batch.append(kl_val)
                    kl_base_to_tasks[t].append(
                        _kl_div_state_cond(base_at_p, current_tokens, t).item()
                    )
                task_id_correct.append(int(np.argmin(kl_per_task_batch) == source_k))

        bl = float(np.mean(baseline_losses)) if baseline_losses else 0.0
        il = float(np.mean(injected_losses)) if injected_losses else 0.0
        kl_means = [float(np.mean(kl_to_tasks[t])) if kl_to_tasks[t] else 0.0 for t in range(K_major)]
        kl_base_means = [float(np.mean(kl_base_to_tasks[t])) if kl_base_to_tasks[t] else 0.0 for t in range(K_major)]
        acc = float(np.mean(task_id_correct)) if task_id_correct else 0.0
        return {
            "baseline_loss": bl,
            "injected_loss": il,
            "delta_loss": il - bl,
            "kl_to_each_task": np.array(kl_means),
            "kl_base_to_each_task": np.array(kl_base_means),
            "task_id_accuracy": acc,
        }

    def _run_injection_specific_target(sampler, target_j, source_k, n_samples):
        """Inject source_k's vector into data generated from target_j (original task)."""
        tv_k = ref_vecs[source_k]
        baseline_losses, injected_losses = [], []
        kl_to_tasks = [[] for _ in range(K_major)]
        kl_base_to_tasks = [[] for _ in range(K_major)]
        task_id_correct = []

        n_batches = (n_samples + B - 1) // B
        for _ in range(n_batches):
            gen_out = sampler.generate(mode="major", task=target_j, num_samples=B, epochs=1)
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

            handle = model.layers[layer].attn_block.register_forward_hook(_inject_hook)
            try:
                with torch.no_grad():
                    logits_inj = model(samples)
            finally:
                handle.remove()

            for p in eval_positions:
                if p + 1 >= samples.shape[1]:
                    continue
                targets = samples[:, p + 1]
                current_tokens = samples[:, p]
                baseline_losses.append(ce_loss_fn(logits_base[:, p, :], targets).mean().item())
                injected_losses.append(ce_loss_fn(logits_inj[:, p, :], targets).mean().item())

                logits_at_p = logits_inj[:, p, :]
                base_at_p = logits_base[:, p, :]
                kl_per_task_batch = []
                for t in range(K_major):
                    kl_val = _kl_div_state_cond(logits_at_p, current_tokens, t).item()
                    kl_to_tasks[t].append(kl_val)
                    kl_per_task_batch.append(kl_val)
                    kl_base_to_tasks[t].append(
                        _kl_div_state_cond(base_at_p, current_tokens, t).item()
                    )
                task_id_correct.append(int(np.argmin(kl_per_task_batch) == source_k))

        bl = float(np.mean(baseline_losses)) if baseline_losses else 0.0
        il = float(np.mean(injected_losses)) if injected_losses else 0.0
        kl_means = [float(np.mean(kl_to_tasks[t])) if kl_to_tasks[t] else 0.0 for t in range(K_major)]
        kl_base_means = [float(np.mean(kl_base_to_tasks[t])) if kl_base_to_tasks[t] else 0.0 for t in range(K_major)]
        acc = float(np.mean(task_id_correct)) if task_id_correct else 0.0
        return {
            "baseline_loss": bl,
            "injected_loss": il,
            "delta_loss": il - bl,
            "kl_to_each_task": np.array(kl_means),
            "kl_base_to_each_task": np.array(kl_base_means),
            "task_id_accuracy": acc,
        }

    injection_results = {}
    for source_k in range(K_major):
        injection_results[source_k] = {}
        for target_j in range(K_major):
            if target_j == source_k:
                continue
            injection_results[source_k][target_j] = _run_injection_specific_target(
                sampler_major, target_j, source_k, n_samples_eval
            )

    ood_results = {}
    for source_k in range(K_major):
        ood_results[source_k] = _run_injection(sampler_ood, "minor", source_k, n_samples_eval)

    if print_summary:
        print()
        print("=" * 70)
        print(f"Task-Vector Injection (latent)  (layer {layer}, scale={scale})")
        print("=" * 70)
        print(f"  Task subspace rank: {rank}  |  K_major: {K_major}")
        print(f"  Ref vecs: from probe W_k{' - W̄' if center_task_vecs else ''}")
        print()

    return {
        "layer": layer,
        "rank": rank,
        "ref_vecs": ref_vecs.cpu(),
        "major_trans_mat": major_trans.cpu(),
        "injection_results": injection_results,
        "ood_results": ood_results,
        "K_major": K_major,
    }


# ---------------------------------------------------------------------------
#  plot_inject_task_vector_across_layers
# ---------------------------------------------------------------------------

def plot_inject_task_vector_across_layers(
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
    verbose: bool = False,
):
    """
    Sweep ``intervene_inject_task_vector`` across layers.

    Returns ``(fig_heatmap, fig_kl, fig_ood, all_results)``.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))

    all_results = {}
    for l in layers:
        if verbose:
            logger.info(f"[inject sweep latent] layer {l} ...")
        all_results[l] = intervene_inject_task_vector(
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
            verbose=verbose,
            print_summary=verbose,
        )

    K = all_results[layers[0]]["K_major"]

    fig_hm, axes_hm = plt.subplots(1, len(layers), figsize=(3.5 * len(layers), 3.5), squeeze=False)
    for idx, l in enumerate(layers):
        mat = np.full((K, K), np.nan)
        inj = all_results[l]["injection_results"]
        for ik in range(K):
            for oj in range(K):
                if oj == ik:
                    continue
                mat[ik, oj] = inj[ik][oj]["task_id_accuracy"]
        ax = axes_hm[0, idx]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="equal")
        for ik in range(K):
            for oj in range(K):
                if ik == oj:
                    ax.text(oj, ik, "-", ha="center", va="center", fontsize=11, color="gray")
                else:
                    val = mat[ik, oj]
                    ax.text(
                        oj, ik, f"{val:.0%}", ha="center", va="center", fontsize=11,
                        color="white" if val > 0.65 else "black",
                    )
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xlabel("Original task (data)", fontsize=14)
        if idx == 0:
            ax.set_ylabel("Injected task (vector)", fontsize=14)
        ax.set_title("", fontsize=18)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_hm.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_hm)

    # ---- 2. KL to injected vs original task  +  Task-ID accuracy ----
    fig_kl, (ax_kl1, ax_kl2) = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(layers))
    bw = 0.3

    def _off_diag_kl(layer, key_fn):
        inj = all_results[layer]["injection_results"]
        return float(np.mean([
            key_fn(inj[ik][oj], ik, oj)
            for ik in range(K) for oj in range(K) if oj != ik
        ]))

    kl_base_orig = _off_diag_kl(layers[0], lambda r, ik, oj: r["kl_base_to_each_task"][oj])
    kl_base_inj = _off_diag_kl(layers[0], lambda r, ik, oj: r["kl_base_to_each_task"][ik])
    ax_kl1.axhline(kl_base_orig, color="#9E9E9E", ls="--", lw=1.5, alpha=0.85,
                   label=rf"KL(no inj $\|$ original) = {kl_base_orig:.2f}")
    ax_kl1.axhline(kl_base_inj, color="#FF9800", ls="--", lw=1.5, alpha=0.85,
                   label=rf"KL(no inj $\|$ injected) = {kl_base_inj:.2f}")

    avg_kl_inj_inj, avg_kl_inj_orig = [], []
    for l in layers:
        avg_kl_inj_inj.append(_off_diag_kl(l, lambda r, ik, oj: r["kl_to_each_task"][ik]))
        avg_kl_inj_orig.append(_off_diag_kl(l, lambda r, ik, oj: r["kl_to_each_task"][oj]))

    ax_kl1.bar(x - 0.5 * bw, avg_kl_inj_inj, bw,
               label=r"KL(after inj $\|$ injected)", color="#4CAF50", alpha=0.85)
    ax_kl1.bar(x + 0.5 * bw, avg_kl_inj_orig, bw,
               label=r"KL(after inj $\|$ original)", color="#F44336", alpha=0.85)
    for i, (v3, v4) in enumerate(zip(avg_kl_inj_inj, avg_kl_inj_orig)):
        ax_kl1.text(x[i] - 0.5 * bw, v3, f"{v3:.2f}", ha="center", va="bottom", fontsize=7)
        ax_kl1.text(x[i] + 0.5 * bw, v4, f"{v4:.2f}", ha="center", va="bottom", fontsize=7)
    ax_kl1.set_xlabel("Layer", fontsize=16)
    ax_kl1.set_ylabel("KL divergence", fontsize=16)
    ax_kl1.set_title("", fontsize=18)
    ax_kl1.set_xticks(x)
    ax_kl1.set_xticklabels([str(l) for l in layers])
    ax_kl1.legend(fontsize=12)
    ax_kl1.grid(axis="y", alpha=0.3)

    avg_acc = []
    for l in layers:
        inj = all_results[l]["injection_results"]
        accs = []
        for ik in range(K):
            for oj in range(K):
                if oj == ik:
                    continue
                accs.append(inj[ik][oj]["task_id_accuracy"])
        avg_acc.append(float(np.mean(accs)))
    ax_kl2.bar(x, avg_acc, 0.5, color="#2196F3", alpha=0.85)
    for i, a in enumerate(avg_acc):
        ax_kl2.text(x[i], a, f"{a:.0%}", ha="center", va="bottom", fontsize=11)
    ax_kl2.set_xlabel("Layer", fontsize=16)
    ax_kl2.set_ylabel("Task-ID accuracy", fontsize=16)
    ax_kl2.set_title("", fontsize=18)
    ax_kl2.set_xticks(x)
    ax_kl2.set_xticklabels([str(l) for l in layers])
    ax_kl2.set_ylim(0, 1.15)
    ax_kl2.axhline(1 / K, color="gray", linestyle="--", alpha=0.5, label=f"Chance (1/{K})")
    ax_kl2.legend(fontsize=12)
    ax_kl2.grid(axis="y", alpha=0.3)
    fig_kl.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_kl)

    # ---- 3. OOD plots ----
    fig_ood, (ax_o1, ax_o2) = plt.subplots(1, 2, figsize=(14, 5))
    for ik in range(K):
        accs = [all_results[l]["ood_results"][ik]["task_id_accuracy"] for l in layers]
        ax_o1.plot(layers, accs, "o-", label=f"Inject task {ik}", linewidth=2, markersize=7)
    ax_o1.axhline(1 / K, color="gray", linestyle="--", alpha=0.5, label=f"Chance (1/{K})")
    ax_o1.set_xlabel("Layer", fontsize=16)
    ax_o1.set_ylabel("Task-ID accuracy", fontsize=16)
    ax_o1.set_title("", fontsize=18)
    ax_o1.set_xticks(layers)
    ax_o1.set_ylim(0, 1.1)
    ax_o1.legend(fontsize=12)
    ax_o1.grid(alpha=0.3)

    for ik in range(K):
        kl_inj = [all_results[l]["ood_results"][ik]["kl_to_each_task"][ik] for l in layers]
        ax_o2.plot(layers, kl_inj, "o-", label=f"Inject task {ik}", linewidth=2, markersize=7)
    ax_o2.set_xlabel("Layer", fontsize=16)
    ax_o2.set_ylabel(r"KL(output $\|$ P(·|s, injected))", fontsize=16)
    ax_o2.set_title("", fontsize=18)
    ax_o2.set_xticks(layers)
    ax_o2.legend(fontsize=12)
    ax_o2.grid(alpha=0.3)
    fig_ood.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_ood)

    return fig_hm, fig_kl, fig_ood, all_results
