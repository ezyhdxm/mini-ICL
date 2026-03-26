"""Causal intervention: remove Gaussian posterior direction from hidden states."""
import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger
from icl.linear.analysis.probes import train_linear_hidden_predictor, probe_gaussian_posterior
from icl.linear.analysis._helpers import _show_or_close
from icl.linear.analysis.interventions._helpers import (
    _load_and_prepare_model,
    _create_ood_task,
    _cleanup_model,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
#  plot_gaussian_posterior_probe
# ---------------------------------------------------------------------------


def plot_gaussian_posterior_probe(
    exp_name: str,
    layers: Optional[list] = None,
    B: int = 64,
    n_samples_major: int = 1000,
    n_samples_ood: int = 1000,
    n_ood: int = 30,
    step: Optional[int] = None,
    positions: Optional[list] = None,
    p_gaussian: Optional[float] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    validation_split: float = 0.2,
    figsize: tuple = (16, 5),
    show: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = False,
):
    """
    Sweep ``probe_gaussian_posterior`` across layers and plot:

    1. Probe KL loss by layer (major val, OOD val, baseline)
    2. Mean P(Z=K+1) predicted by probe: major vs OOD
    3. Histogram of predicted P(Z=K+1) for major vs OOD at best layer

    Returns ``(fig, all_results)``.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config

    if layers is None:
        _, _, config = load_model_task_config(exp_name)
        layers = list(range(config.model.n_layer))

    all_results = {}
    for l in layers:
        logger.info(f"[gauss-probe sweep] layer {l} ...")
        res = probe_gaussian_posterior(
            exp_name=exp_name,
            layer=l,
            B=B,
            n_samples_major=n_samples_major,
            n_samples_ood=n_samples_ood,
            n_ood=n_ood,
            step=step,
            positions=positions,
            p_gaussian=p_gaussian,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            validation_split=validation_split,
            verbose=verbose,
        )
        all_results[l] = res

    x = np.arange(len(layers))
    bar_w = 0.25

    # ---- Panel 1: KL loss by layer ----
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax1 = axes[0]
    kl_maj = [all_results[l]["kl_major_val"] for l in layers]
    kl_ood = [all_results[l]["kl_ood_val"] for l in layers]
    ax1.bar(x - bar_w / 2, kl_maj, bar_w, label="Major (val)", color="#2196F3", alpha=0.85)
    ax1.bar(x + bar_w / 2, kl_ood, bar_w, label="OOD (val)", color="#FF9800", alpha=0.85)
    ax1.set_xlabel("Layer", fontsize=13)
    ax1.set_ylabel("KL Divergence", fontsize=13)
    ax1.set_title("", fontsize=18)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in layers])
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(labelsize=11)

    # ---- Panel 2: Mean P(Z=K+1) predicted by probe ----
    ax2 = axes[1]
    pred_maj = [all_results[l]["mean_pred_pgauss_major"] for l in layers]
    pred_ood = [all_results[l]["mean_pred_pgauss_ood"] for l in layers]
    true_maj = [all_results[l]["mean_true_pgauss_major"] for l in layers]
    true_ood = [all_results[l]["mean_true_pgauss_ood"] for l in layers]

    ax2.plot(x, pred_maj, "s-", label="Pred Major", color="#2196F3", markersize=7, linewidth=2)
    ax2.plot(x, pred_ood, "o-", label="Pred OOD", color="#FF9800", markersize=7, linewidth=2)
    ax2.axhline(true_maj[0], ls="--", color="#2196F3", alpha=0.5, label=f"Oracle Major ({true_maj[0]:.3f})")
    ax2.axhline(true_ood[0], ls="--", color="#FF9800", alpha=0.5, label=f"Oracle OOD ({true_ood[0]:.3f})")
    ax2.set_xlabel("Layer", fontsize=13)
    ax2.set_ylabel("Mean P(Z=K+1)", fontsize=13)
    ax2.set_title("", fontsize=18)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in layers])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=11)
    ax2.set_ylim(bottom=-0.02)

    # ---- Panel 3: Histogram at best layer ----
    best_layer = min(layers, key=lambda l: all_results[l]["kl_ood_val"])
    ax3 = axes[2]
    pg_maj = all_results[best_layer]["pred_pgauss_major"]
    pg_ood = all_results[best_layer]["pred_pgauss_ood"]

    bins = np.linspace(0, 1, 40)
    if len(pg_maj) > 0:
        ax3.hist(pg_maj, bins=bins, alpha=0.6, label="Major", color="#2196F3", density=True)
    if len(pg_ood) > 0:
        ax3.hist(pg_ood, bins=bins, alpha=0.6, label="OOD", color="#FF9800", density=True)
    ax3.set_xlabel("Predicted P(Z=K+1)", fontsize=13)
    ax3.set_ylabel("Density", fontsize=13)
    ax3.set_title("", fontsize=18)
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=11)

    fig.suptitle("", fontsize=18, y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_or_close(fig, show)

    return fig, all_results


# ---------------------------------------------------------------------------
#  intervene_remove_gaussian_direction
# ---------------------------------------------------------------------------


def intervene_remove_gaussian_direction(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_major: int = 1000,
    n_samples_ood: int = 1000,
    n_ood: int = 30,
    step: Optional[int] = None,
    positions: Optional[list] = None,
    p_gaussian: Optional[float] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    scale: float = 1.0,
    n_samples_probe: int = 1000,
    n_samples_eval: int = 500,
    fit_positions: Optional[list] = None,
    fit_n_samples: int = 5000,
    orth_to_task: bool = True,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Remove the Gaussian posterior direction from hidden states and measure
    the effect on major vs OOD prediction MSE.

    1. Train a (K+1)-way Gaussian posterior probe at ``layer``.
    2. Extract the probe weight for Z=K+1 (the Gaussian class).
    3. Optionally project onto orthogonal complement of the task subspace.
    4. Remove this direction from hidden states via a forward hook.
    5. Evaluate MSE on major and OOD data.
    """
    model, train_task, config, device = _load_and_prepare_model(exp_name, step=step)
    if step is None:
        step = config.training.total_steps

    n_points = int(config.task.n_points)

    ood_task = _create_ood_task(train_task, config, B, n_ood, device)

    if positions is None:
        positions = list(range(n_points))
    eval_positions = positions

    # ---- 1. Train Gaussian posterior probe ----
    if verbose:
        logger.info(f"[gauss-dir] Training Gaussian posterior probe at layer {layer} ...")

    probe_res = probe_gaussian_posterior(
        exp_name=exp_name,
        layer=layer,
        B=B,
        n_samples_major=n_samples_probe,
        n_samples_ood=n_samples_probe,
        n_ood=n_ood,
        step=step,
        positions=positions,
        p_gaussian=p_gaussian,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        validation_split=0.2,
        verbose=verbose,
    )

    probe_model = probe_res["probe"]
    probe_kl = probe_res["final_val_kl"]

    # ---- 2. Extract Gaussian direction ----
    W_probe = probe_model[0].weight.detach().float()  # (K+1, D)
    w_gauss = W_probe[-1].clone()  # (D,)

    if verbose:
        logger.info(
            f"[gauss-dir] Probe val KL={probe_kl:.4f}, "
            f"||w_gauss||={w_gauss.norm():.4f}"
        )

    # ---- 3. Optionally project onto orth complement of task subspace ----
    if orth_to_task:
        if verbose:
            logger.info("[gauss-dir] Projecting w_gauss onto orth complement of task subspace ...")

        if fit_positions is None:
            fit_positions = list(range(min(10, n_points), n_points))

        fit_res = train_linear_hidden_predictor(
            exp_name=exp_name,
            layer=layer,
            n_samples=fit_n_samples,
            positions=fit_positions,
            sample_mode="major",
            step=step,
            n_minor=None,
            print_summary=False,
            skip_baselines=True,
            include_position_bias=False,
            include_logit=False,
        )
        W_task = fit_res["model_weight"].float()  # (K, D)
        task_vecs = W_task - W_task.mean(dim=0, keepdim=True)
        U_task, S_task, _ = torch.linalg.svd(task_vecs.T, full_matrices=False)
        rank = int((S_task > 1e-6 * S_task[0]).sum().item())
        U_task = U_task[:, :rank]  # (D, rank)

        proj_task = U_task @ (U_task.T @ w_gauss)
        w_gauss_orth = w_gauss - proj_task

        if verbose:
            overlap = proj_task.norm() / w_gauss.norm()
            logger.info(
                f"[gauss-dir] Task subspace rank={rank}, "
                f"overlap with w_gauss: {overlap:.4f}, "
                f"||w_gauss_orth||={w_gauss_orth.norm():.4f}"
            )

        w_gauss = w_gauss_orth

    norm = w_gauss.norm()
    if norm < 1e-10:
        logger.warning("[gauss-dir] w_gauss is near-zero after projection; skipping.")
        _cleanup_model(model)
        return {"error": "w_gauss collapsed to zero after task-subspace projection"}

    v = (w_gauss / norm).to(device)  # (D,) unit vector

    # ---- 4. Evaluate intervention: remove v from hidden states ----
    P_v = torch.outer(v, v)  # (D, D) rank-1 projector

    def _run_experiment(task_obj, n_samples):
        baseline_losses_by_pos = {p: [] for p in eval_positions}
        intervened_losses_by_pos = {p: [] for p in eval_positions}

        n_batches = max(1, (n_samples + B - 1) // B)
        orig_bs = int(task_obj.batch_size)
        task_obj.batch_size = B

        for bi in range(n_batches):
            demo_data, _, demo_target = task_obj.sample_batch(
                step=bi + 33333, is_eval=True,
            )
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)

            with torch.no_grad():
                preds_base = model(demo_data, demo_target)

            def hook_fn(module, inp, out, _P=P_v, _s=scale):
                h = out if torch.is_tensor(out) else out[0]
                h_mod = h - _s * (h @ _P)
                if torch.is_tensor(out):
                    return h_mod
                return (h_mod,) + out[1:]

            handle = model.transformer.blocks[layer].attn_block.register_forward_hook(
                hook_fn,
            )
            try:
                with torch.no_grad():
                    preds_int = model(demo_data, demo_target)
            finally:
                handle.remove()

            for p in eval_positions:
                if p >= preds_base.shape[1]:
                    continue
                mse_b = ((preds_base[:, p] - demo_target[:, p]) ** 2).mean().item()
                mse_i = ((preds_int[:, p] - demo_target[:, p]) ** 2).mean().item()
                baseline_losses_by_pos[p].append(mse_b)
                intervened_losses_by_pos[p].append(mse_i)

            del demo_data, demo_target, preds_base, preds_int
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        task_obj.batch_size = orig_bs

        bp, ip, vp = [], [], []
        for p in eval_positions:
            if baseline_losses_by_pos[p]:
                bp.append(np.mean(baseline_losses_by_pos[p]))
                ip.append(np.mean(intervened_losses_by_pos[p]))
                vp.append(p)

        baseline_avg = float(np.mean(bp)) if bp else float("nan")
        intervened_avg = float(np.mean(ip)) if ip else float("nan")
        return {
            "baseline": baseline_avg,
            "intervened": intervened_avg,
            "delta": intervened_avg - baseline_avg,
        }

    # random direction baseline
    v_rand = torch.randn_like(v)
    v_rand = v_rand / v_rand.norm()
    P_rand = torch.outer(v_rand, v_rand)

    def _run_random(task_obj, n_samples):
        losses = []
        n_batches = max(1, (n_samples + B - 1) // B)
        orig_bs = int(task_obj.batch_size)
        task_obj.batch_size = B

        for bi in range(n_batches):
            demo_data, _, demo_target = task_obj.sample_batch(
                step=bi + 33333, is_eval=True,
            )
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)

            def hook_fn(module, inp, out, _P=P_rand, _s=scale):
                h = out if torch.is_tensor(out) else out[0]
                h_mod = h - _s * (h @ _P)
                if torch.is_tensor(out):
                    return h_mod
                return (h_mod,) + out[1:]

            handle = model.transformer.blocks[layer].attn_block.register_forward_hook(
                hook_fn,
            )
            try:
                with torch.no_grad():
                    preds_int = model(demo_data, demo_target)
            finally:
                handle.remove()

            batch_mses = []
            for p in eval_positions:
                if p >= preds_int.shape[1]:
                    continue
                batch_mses.append(
                    ((preds_int[:, p] - demo_target[:, p]) ** 2).mean().item()
                )
            if batch_mses:
                losses.append(np.mean(batch_mses))

            del demo_data, demo_target, preds_int
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        task_obj.batch_size = orig_bs
        return float(np.mean(losses)) if losses else float("nan")

    if verbose:
        logger.info("[gauss-dir] Evaluating intervention on major tasks ...")
    res_major = _run_experiment(train_task, n_samples_eval)
    if verbose:
        logger.info("[gauss-dir] Evaluating intervention on OOD tasks ...")
    res_ood = _run_experiment(ood_task, n_samples_eval)

    rand_major = _run_random(train_task, n_samples_eval)
    rand_ood = _run_random(ood_task, n_samples_eval)

    pct_major = (
        100.0 * res_major["delta"] / res_major["baseline"]
        if res_major["baseline"] > 0 else float("nan")
    )
    pct_ood = (
        100.0 * res_ood["delta"] / res_ood["baseline"]
        if res_ood["baseline"] > 0 else float("nan")
    )
    rand_delta_major = rand_major - res_major["baseline"]
    rand_delta_ood = rand_ood - res_ood["baseline"]
    rand_pct_major = (
        100.0 * rand_delta_major / res_major["baseline"]
        if res_major["baseline"] > 0 else float("nan")
    )
    rand_pct_ood = (
        100.0 * rand_delta_ood / res_ood["baseline"]
        if res_ood["baseline"] > 0 else float("nan")
    )

    results = {
        "layer": layer,
        "baseline_loss_major": res_major["baseline"],
        "intervened_loss_major": res_major["intervened"],
        "delta_loss_major": res_major["delta"],
        "pct_increase_major": pct_major,
        "baseline_loss_ood": res_ood["baseline"],
        "intervened_loss_ood": res_ood["intervened"],
        "delta_loss_ood": res_ood["delta"],
        "pct_increase_ood": pct_ood,
        "rand_delta_major": rand_delta_major,
        "rand_pct_major": rand_pct_major,
        "rand_delta_ood": rand_delta_ood,
        "rand_pct_ood": rand_pct_ood,
        "probe_val_kl": probe_kl,
        "probe_mean_pgauss_major": probe_res["mean_pred_pgauss_major"],
        "probe_mean_pgauss_ood": probe_res["mean_pred_pgauss_ood"],
        "orth_to_task": orth_to_task,
        "scale": scale,
    }

    if print_summary:
        print(f"\n{'=' * 65}")
        print(
            f"Remove Gaussian Direction  (layer {layer}, scale={scale}, "
            f"orth_to_task={orth_to_task})"
        )
        print(f"{'=' * 65}")
        print(f"  Probe val KL: {probe_kl:.4f}")
        print(
            f"  Probe P(gauss):  major={probe_res['mean_pred_pgauss_major']:.4f}  "
            f"OOD={probe_res['mean_pred_pgauss_ood']:.4f}"
        )
        print()
        print(f"  {'Metric':<30} {'Major':>12} {'OOD':>12}")
        print(f"  {'-' * 54}")
        print(
            f"  {'Baseline MSE':<30} "
            f"{res_major['baseline']:>12.4f} "
            f"{res_ood['baseline']:>12.4f}"
        )
        print(
            f"  {'Intervened MSE':<30} "
            f"{res_major['intervened']:>12.4f} "
            f"{res_ood['intervened']:>12.4f}"
        )
        print(
            f"  {'Delta MSE':<30} "
            f"{res_major['delta']:>12.4f} "
            f"{res_ood['delta']:>12.4f}"
        )
        print(
            f"  {'Percent increase':<30} "
            f"{pct_major:>11.1f}% "
            f"{pct_ood:>11.1f}%"
        )
        _rand_dir_label = 'Rand dir \u0394 MSE'
        print(
            f"  {_rand_dir_label:<30} "
            f"{rand_delta_major:>12.4f} "
            f"{rand_delta_ood:>12.4f}"
        )
        print(
            f"  {'Rand dir % increase':<30} "
            f"{rand_pct_major:>11.1f}% "
            f"{rand_pct_ood:>11.1f}%"
        )

    _cleanup_model(model)
    return results


# ---------------------------------------------------------------------------
#  plot_remove_gaussian_direction_across_layers
# ---------------------------------------------------------------------------


def plot_remove_gaussian_direction_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    B: int = 64,
    n_samples_major: int = 1000,
    n_samples_ood: int = 1000,
    n_ood: int = 30,
    step: Optional[int] = None,
    positions: Optional[list] = None,
    p_gaussian: Optional[float] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    scale: float = 1.0,
    n_samples_probe: int = 1000,
    n_samples_eval: int = 500,
    fit_n_samples: int = 5000,
    orth_to_task: bool = True,
    figsize: tuple = (14, 5),
    show: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = False,
):
    """
    Sweep ``intervene_remove_gaussian_direction`` across layers and plot:

    - **Left**: % MSE increase (Major vs OOD bars per layer)
    - **Right**: Probe quality (val KL by layer)

    Returns ``(fig, all_results)``.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config

    if layers is None:
        _, _, config = load_model_task_config(exp_name)
        layers = list(range(config.model.n_layer))

    all_results = {}
    for l in layers:
        logger.info(f"[gauss-dir sweep] layer {l} ...")
        res = intervene_remove_gaussian_direction(
            exp_name=exp_name,
            layer=l,
            B=B,
            n_samples_major=n_samples_major,
            n_samples_ood=n_samples_ood,
            n_ood=n_ood,
            step=step,
            positions=positions,
            p_gaussian=p_gaussian,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            scale=scale,
            n_samples_probe=n_samples_probe,
            n_samples_eval=n_samples_eval,
            fit_n_samples=fit_n_samples,
            orth_to_task=orth_to_task,
            verbose=verbose,
            print_summary=True,
        )
        all_results[l] = res

    x = np.arange(len(layers))
    bar_w = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ---- Left panel: % RMSE increase ----
    pct_maj = [
        100.0 * (np.sqrt(all_results[l]["intervened_loss_major"])
                 - np.sqrt(all_results[l]["baseline_loss_major"]))
        / np.sqrt(all_results[l]["baseline_loss_major"])
        if all_results[l]["baseline_loss_major"] > 0 else float("nan")
        for l in layers
    ]
    pct_ood = [
        100.0 * (np.sqrt(all_results[l]["intervened_loss_ood"])
                 - np.sqrt(all_results[l]["baseline_loss_ood"]))
        / np.sqrt(all_results[l]["baseline_loss_ood"])
        if all_results[l]["baseline_loss_ood"] > 0 else float("nan")
        for l in layers
    ]

    ax1.bar(x - bar_w / 2, pct_maj, bar_w, label="Major", color="#2196F3", alpha=0.85)
    ax1.bar(x + bar_w / 2, pct_ood, bar_w, label="OOD", color="#FF9800", alpha=0.85)

    for i, (vm, vo) in enumerate(zip(pct_maj, pct_ood)):
        ax1.text(x[i] - bar_w / 2, vm, f"{vm:.1f}%", ha="center", va="bottom", fontsize=9)
        ax1.text(x[i] + bar_w / 2, vo, f"{vo:.1f}%", ha="center", va="bottom", fontsize=9)

    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_xlabel("Layer", fontsize=13)
    ax1.set_ylabel("% RMSE Increase", fontsize=13)
    ax1.set_title("", fontsize=18)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in layers])
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(labelsize=11)

    # ---- Right panel: Probe quality ----
    probe_kls = [all_results[l]["probe_val_kl"] for l in layers]

    ax2.bar(x, probe_kls, bar_w, label="Probe val KL", color="#4CAF50", alpha=0.85)

    for i, v in enumerate(probe_kls):
        ax2.text(x[i], v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax2.set_xlabel("Layer", fontsize=13)
    ax2.set_ylabel("KL Divergence", fontsize=13)
    ax2.set_title("", fontsize=18)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in layers])
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(labelsize=11)

    orth_str = "orth-to-task" if orth_to_task else "raw"
    fig.suptitle("", fontsize=18, y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_or_close(fig, show)

    return fig, all_results
