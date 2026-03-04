"""
Calculation-direction analysis and task-subspace R² for latent Markov
(non-padded sequences).

Extracted from ``icl.utils.latent_nonpadded``.
"""

import gc
import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.latent_markov.analysis.probes import train_linear_hidden_predictor
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
#  calculation_direction_analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def calculation_direction_analysis(
    exp_name: str,
    layer_index: Optional[int] = None,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    show: bool = True,
    figsize: tuple = (16, 5),
    title: str = "",
) -> dict:
    """
    Decompose accumulated information in hidden representations for latent
    Markov tasks.

    Baseline: ``h_0(x_t)``
        For each position *t* with token value ``x_t``, the baseline is the
        hidden representation at **position 0** when the same token ``x_t``
        is placed there.  Because causal attention at position 0 sees only
        itself, ``h_0(x_t)`` carries no contextual information.  Using the
        *same* token eliminates token-identity effects, so

            ``delta_h = h_t(x_t) - h_0(x_t)``

        isolates exactly:

        * context accumulated from positions 0..t-1  (task-dependent), and
        * the positional-encoding shift from pos 0 to pos t (task-independent).

    ``delta_h`` is decomposed into:

    * **task-subspace component** -- aligned with the posterior-derived task
      vectors ``W`` from ``train_linear_hidden_predictor``
    * **orthogonal component** -- the *calculation direction* that carries
      non-task-specific processing (including positional encoding shift)

    Two panels are produced:

    (a) squared-norm decomposition over positions,
    (b) R² of ``delta_h`` explained by the task vectors.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.latent_markov.legacy.coin_latent_task_vecs import extract_hidden_multi_coin_latent

    # ---- 1. Load config & model ------------------------------------------------
    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    sampler, _k_minor, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)

    K_major = sampler.n_major_tasks
    seq_len = sampler.seq_len
    n_layers = len(model.layers)
    D = int(config.model.emb_dim)
    num_states = sampler.num_states

    if layer_index is None:
        layer_index = n_layers - 1
    if fit_positions is None:
        fit_positions = list(range(max(0, seq_len // 2), seq_len))

    n_total_tasks = K_major + n_ood

    # ---- 2. Fit task subspace ---------------------------------------------------
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name,
        layer=layer_index,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        n_minor=-1,
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"].float()   # (K_major, D)
    b = fit_res["model_bias"].float()     # (D,)

    W_for_svd = (W - W.mean(dim=0, keepdim=True)) if center_task_vecs else W.clone()
    _, S_tv, Vt_tv = torch.linalg.svd(W_for_svd, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    basis = Vt_tv[:rank].T                           # (D, rank)
    P_task = (basis @ basis.T).to(device)             # (D, D)

    logger.info(
        f"[calc-dir-latent] Task subspace rank={rank}, D={D}, "
        f"chance fraction={rank / D:.4f}, fit R2={fit_res['val_r2']:.4f}"
    )

    # ---- 3. Compute h_0(v) for each vocabulary token v --------------------------
    dummy_seq = torch.zeros(num_states, seq_len, dtype=torch.long, device=device)
    dummy_seq[:, 0] = torch.arange(num_states, device=device)

    task_pos_0 = torch.tensor([0], device=device, dtype=torch.long)
    h0_all = extract_hidden_multi_coin_latent(
        model, dummy_seq, layers=[layer_index], task_pos=task_pos_0,
    )  # (1, num_states, 1, D)
    h0_lookup = h0_all[0, :, 0, :].cpu().float()  # (num_states, D)

    # ---- 4. For each task, generate data and extract hiddens --------------------
    task_pos_all = torch.arange(seq_len, device=device, dtype=torch.long)

    hiddens = torch.empty(n_total_tasks, seq_len, B, D, dtype=torch.float32)
    all_tokens = torch.empty(n_total_tasks, B, seq_len, dtype=torch.long)

    for task_idx in range(n_total_tasks):
        gen_out = sampler.generate(
            mode="testing", task=task_idx, num_samples=B,
        )
        samples = gen_out[0].to(device)  # (B, seq_len)
        all_tokens[task_idx] = samples.cpu()

        h = extract_hidden_multi_coin_latent(
            model, samples, layers=[layer_index], task_pos=task_pos_all,
        )  # (1, B, seq_len, D)
        hiddens[task_idx] = h[0].permute(1, 0, 2).cpu().float()  # (seq_len, B, D)

        del samples, h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 5. delta_h = h_t(x_t) - h_0(x_t) using token lookup -------------------
    h_baseline = h0_lookup[all_tokens.reshape(-1)].reshape(
        n_total_tasks, B, seq_len, D,
    )
    h_baseline = h_baseline.permute(0, 2, 1, 3)  # (K, T, B, D)

    delta_h = hiddens - h_baseline  # (K, T, B, D)

    # ---- 6. Project onto task subspace ------------------------------------------
    P_cpu = P_task.cpu().float()
    dh_task = torch.einsum("ktbd,de->ktbe", delta_h, P_cpu)  # (K, T, B, D)
    dh_orth = delta_h - dh_task

    task_nsq = (dh_task ** 2).sum(-1)    # (K, T, B)
    orth_nsq = (dh_orth ** 2).sum(-1)
    total_nsq = (delta_h ** 2).sum(-1)

    task_nm = task_nsq.mean(-1).numpy()   # (K, T)
    orth_nm = orth_nsq.mean(-1).numpy()
    total_nm = total_nsq.mean(-1).numpy()
    frac_task = task_nm / (total_nm + 1e-12)

    # ---- 7. Lambda estimation (unconstrained) -----------------------------------
    dh_mean = delta_h.mean(dim=2)  # (K, T, D)
    lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
        W.cpu(), dh_mean.cpu(), is_zero_mean=False,
    )  # lambdas: (K, T, K_major), r2_scores: (K, T)

    # ---- 8. Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    t_axis = np.arange(seq_len)
    major_colors = ["#1f77b4", "#2ca02c", "#d62728"]
    ood_color = "#ff7f0e"
    chance = rank / D

    # (a) Norm decomposition
    ax = axes[0]
    for label, grp, color in [
        ("Major", slice(0, K_major), "#1f77b4"),
        ("OOD", slice(K_major, None), ood_color),
    ]:
        if total_nm[grp].shape[0] == 0:
            continue
        ax.plot(t_axis, total_nm[grp].mean(0), "-", color=color, lw=2,
                label=f"{label} $||\\Delta h||^2$")
        ax.plot(t_axis, task_nm[grp].mean(0), "--", color=color, lw=2,
                label=f"{label} $||\\Delta h_{{\\mathrm{{task}}}}||^2$")
        ax.plot(t_axis, orth_nm[grp].mean(0), ":", color=color, lw=2,
                label=f"{label} $||\\Delta h_{{\\perp}}||^2$")
    ax.set_xlabel("Position $t$", fontsize=16)
    ax.set_ylabel("Squared norm (batch avg)", fontsize=16)
    ax.set_title("", fontsize=18)
    ax.legend(fontsize=12, loc="best")
    ax.grid(alpha=0.3)

    # (c) R^2 of delta_h explained by task vectors
    ax = axes[1]
    for k in range(K_major):
        ax.plot(t_axis, r2_scores[k],
                color=major_colors[k % len(major_colors)],
                lw=2, label=f"Major {k}")
    if n_ood > 0:
        r2_mu = r2_scores[K_major:].mean(0)
        r2_sd = r2_scores[K_major:].std(0)
        ax.plot(t_axis, r2_mu, color=ood_color, lw=2, label="OOD (mean)")
        ax.fill_between(t_axis, r2_mu - r2_sd, r2_mu + r2_sd,
                        color=ood_color, alpha=0.15)
    ax.set_xlabel("Position $t$", fontsize=16)
    ax.set_ylabel("$R^2$", fontsize=16)
    ax.set_title("", fontsize=18)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12, loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=14)

    fig.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()

    # ---- cleanup ----------------------------------------------------------------
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig,
        "axes": axes,
        "delta_h": delta_h,
        "hiddens": hiddens,
        "h_baseline": h_baseline,
        "all_tokens": all_tokens,
        "fraction_task": frac_task,
        "task_norm_sq_mean": task_nm,
        "orth_norm_sq_mean": orth_nm,
        "total_norm_sq_mean": total_nm,
        "lambdas": lambdas,
        "r2_scores": r2_scores,
        "W": W,
        "b": b,
        "P_task": P_task.cpu(),
        "rank": rank,
        "K_major": K_major,
        "n_ood": n_ood,
        "fit_results": fit_res,
    }


# ---------------------------------------------------------------------------
#  plot_task_subspace_r2_over_positions
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_task_subspace_r2_over_positions(
    exp_name: str,
    layer: int,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    show: bool = True,
    figsize: tuple = (8, 5),
) -> dict:
    """
    At each position *t*, compute the fraction of each individual hidden
    vector's energy that lies in the major-task subspace, separately for
    Major and OOD tasks.

    When ``center_task_vecs=True`` (default), the task vectors are centered
    before SVD (rank K-1 instead of K), and the probe bias is subtracted
    from hidden representations before projecting.

    Produces a single plot with per-Major-task curves, OOD mean +/- std,
    and the chance level.

    Latent Markov counterpart of
    ``plot_task_subspace_r2_over_positions_linear_nonpadded``.
    """
    import matplotlib.pyplot as plt
    from icl.latent_markov.legacy.coin_latent_task_vecs import extract_hidden_multi_coin_latent

    # ---- 1. Load config & model --------------------------------------------------
    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    D = int(config.model.emb_dim)

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    sampler, _k_minor, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)

    K_major = sampler.n_major_tasks
    seq_len = sampler.seq_len
    n_total_tasks = K_major + n_ood

    if fit_positions is None:
        fit_positions = list(range(max(0, seq_len // 2), seq_len))

    # ---- 2. Fit task subspace ----------------------------------------------------
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name,
        layer=layer,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        n_minor=-1,
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W_fit = fit_res["model_weight"].float()   # (K_major, D)
    b_fit = fit_res["model_bias"].float()     # (D,)

    W_for_svd = (W_fit - W_fit.mean(0, keepdim=True)) if center_task_vecs else W_fit.clone()
    _, S_tv, Vt_tv = torch.linalg.svd(W_for_svd, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    basis = Vt_tv[:rank].T  # (D, rank)
    P_task = (basis @ basis.T).float()

    logger.info(
        f"[task-subspace-r2-latent] center={center_task_vecs}, "
        f"rank={rank}, D={D}, chance={rank / D:.4f}"
    )

    # ---- 3. Extract hidden representations per task ------------------------------
    task_pos_all = torch.arange(seq_len, device=device, dtype=torch.long)
    all_hiddens = torch.empty(n_total_tasks, seq_len, B, D, dtype=torch.float32)

    for task_idx in range(n_total_tasks):
        gen_out = sampler.generate(
            mode="testing", task=task_idx, num_samples=B,
        )
        samples = gen_out[0].to(device)  # (B, seq_len)
        h = extract_hidden_multi_coin_latent(
            model, samples, layers=[layer], task_pos=task_pos_all,
        )  # (1, B, seq_len, D)
        all_hiddens[task_idx] = h[0].permute(1, 0, 2).cpu().float()  # (seq_len, B, D)
        del samples, h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 4. Per-sequence projection fraction -------------------------------------
    P_cpu = P_task.cpu().float()
    h_all = all_hiddens.float()  # (n_tasks, seq_len, B, D)
    if center_task_vecs:
        h_all = h_all - b_fit.cpu().unsqueeze(0).unsqueeze(0).unsqueeze(0)

    h_flat = h_all.reshape(n_total_tasks * seq_len * B, D)
    h_proj = h_flat @ P_cpu
    norms_sq = (h_flat ** 2).sum(dim=1)
    proj_sq = (h_proj ** 2).sum(dim=1)
    safe = norms_sq > 0
    frac_flat = torch.zeros_like(norms_sq)
    frac_flat[safe] = proj_sq[safe] / norms_sq[safe]
    frac_all = frac_flat.reshape(n_total_tasks, seq_len, B).numpy()

    r2_per_task = frac_all.mean(axis=2)        # (n_tasks, seq_len)
    r2_per_task_std = frac_all.std(axis=2)     # (n_tasks, seq_len)

    r2_ood_mean = r2_per_task[K_major:].mean(axis=0)
    r2_ood_std = r2_per_task[K_major:].std(axis=0)
    chance = rank / D

    # ---- 5. Plot -----------------------------------------------------------------
    major_colors = ["tab:blue", "tab:green", "tab:purple", "tab:cyan",
                    "tab:olive", "tab:brown", "tab:pink", "tab:gray"]
    fig, ax = plt.subplots(figsize=figsize)
    positions_arr = np.arange(seq_len)

    for k in range(K_major):
        c = major_colors[k % len(major_colors)]
        ax.plot(positions_arr, r2_per_task[k], color=c, linewidth=1.5,
                label=f"Major {k}")
        ax.fill_between(positions_arr,
                        r2_per_task[k] - r2_per_task_std[k],
                        r2_per_task[k] + r2_per_task_std[k],
                        color=c, alpha=0.1)

    ax.plot(positions_arr, r2_ood_mean, color="tab:red", linewidth=1.5,
            label="OOD (mean)")
    ax.fill_between(positions_arr,
                    r2_ood_mean - r2_ood_std,
                    r2_ood_mean + r2_ood_std,
                    color="tab:red", alpha=0.15)
    ax.axhline(chance, color="gray", ls="--", alpha=0.6,
               label=f"Chance ({rank}/{D} = {chance:.4f})")
    ax.set_xlabel("Position $t$", fontsize=16)
    ax.set_ylabel(r"$\| P_{\mathrm{task}}\, h \|^2 \;/\; \| h \|^2$"
                  "  (per sequence, mean over batch)", fontsize=16)
    ax.set_title("", fontsize=18)
    y_max = max(r2_per_task[:K_major].max(),
                (r2_ood_mean + r2_ood_std).max())
    ax.set_ylim(0, min(1.0, y_max * 1.15))
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    if show:
        plt.show()

    del all_hiddens
    gc.collect()

    return {
        "fig": fig,
        "r2_per_task": r2_per_task,
        "r2_per_task_std": r2_per_task_std,
        "r2_ood_mean": r2_ood_mean,
        "r2_ood_std": r2_ood_std,
        "frac_all": frac_all,
        "chance": chance,
        "rank": rank,
        "layer": layer,
        "K_major": K_major,
    }
