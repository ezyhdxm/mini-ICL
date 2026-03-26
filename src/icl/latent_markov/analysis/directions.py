"""
Calculation-direction analysis and task-subspace R² for latent Markov
(non-padded sequences).

Extracted from ``icl.utils.latent_nonpadded``.
"""

import gc
import os
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


# ---------------------------------------------------------------------------
#  Attention-head probes (non-padded sequences)
# ---------------------------------------------------------------------------

def previous_token_head_score(
    tokens: torch.Tensor,
    attn: torch.Tensor,
) -> torch.Tensor:
    """
    Previous-Token Head (PTH) score for non-padded sequences.

    For each query position t ∈ {1, …, L-1} the PTH score is the attention
    weight placed on the immediately preceding key t-1.  The score is
    averaged over all valid positions and over the batch.

    Parameters
    ----------
    tokens : (B, L) long tensor — token ids (used only to infer B and L).
    attn   : (B, H, L, L) float tensor — causal attention weights.

    Returns
    -------
    score : (H,) float tensor — per-head mean PTH score.
    """
    # attn[:, h, t, t-1] for t = 1..L-1 is the sub-diagonal with offset -1
    # torch.diagonal(offset=-1, dim1=-2, dim2=-1) extracts exactly this.
    sub_diag = torch.diagonal(attn, offset=-1, dim1=-2, dim2=-1)  # (B, H, L-1)
    return sub_diag.mean(dim=-1).mean(dim=0)  # (H,)


def induction_head_score(
    tokens: torch.Tensor,
    attn: torch.Tensor,
    include_self: bool = True,
) -> torch.Tensor:
    """
    Induction Head (IH) score for non-padded sequences.

    For query position t with token s_t, let J(t) = {j < t : s_j = s_t} be
    the set of matching historical positions.  If J(t) is non-empty the IH
    score at t is the **total attention mass** placed on the union of

        * successors  {j+1 : j ∈ J(t), j+1 < t}  (strictly past positions
          that followed a prior occurrence of s_t), and
        * (if ``include_self=True``) the matches themselves {j : j ∈ J(t)}.

    Successors are restricted to j+1 < t to avoid counting self-attention
    at the query position (which would occur when s_{t-1} = s_t).

    The mass is **not** divided by |J(t)|; the score is the raw fraction
    of attention in [0, 1] directed at induction-relevant positions.
    Under uniform attention this baseline is approximately
    ``|targets| / t ≈ 1/V`` (or ``2/V`` with ``include_self``), so any
    value substantially above that indicates genuine induction-like behaviour.
    Positions t with no match are skipped.  The per-batch score is averaged
    over valid t and then over the batch.

    Parameters
    ----------
    tokens       : (B, L) long tensor — token ids.
    attn         : (B, H, L, L) float tensor — causal attention weights.
    include_self : If True (default), also count attention at the matching
                   positions j in addition to their successors j+1.

    Returns
    -------
    score : (H,) float tensor — per-head mean IH score in [0, 1].
    """
    B, H, L, _ = attn.shape
    device = attn.device
    dtype = attn.dtype

    sum_mass = torch.zeros(B, H, device=device, dtype=dtype)
    cnt = torch.zeros(B, device=device, dtype=torch.long)

    for t in range(1, L):
        s_t = tokens[:, t]                               # (B,)
        matches = (tokens[:, :t] == s_t.unsqueeze(1))   # (B, t)
        n_matches = matches.sum(dim=1)                   # (B,)
        has_match = n_matches > 0

        if not has_match.any():
            continue

        rows = attn[:, :, t, :]                          # (B, H, L)

        # Successor mask: position j+1 for each match at j, but only where
        # j+1 < t so we never count self-attention at the query position.
        succ_mask = torch.zeros(B, L, device=device, dtype=dtype)
        if t > 1:
            succ_mask[:, 1:t] = matches[:, :t - 1].to(dtype)

        if include_self:
            self_mask = torch.zeros(B, L, device=device, dtype=dtype)
            self_mask[:, :t] = matches.to(dtype)
            combined = (succ_mask + self_mask).clamp_(0.0, 1.0)
        else:
            combined = succ_mask

        mass = (rows * combined.unsqueeze(1)).sum(dim=-1)  # (B, H)

        sum_mass[has_match] += mass[has_match]
        cnt[has_match] += 1

    score = torch.zeros(B, H, device=device, dtype=dtype)
    nz = cnt > 0
    if nz.any():
        score[nz] = sum_mass[nz] / cnt[nz].float().unsqueeze(1)

    return score.mean(dim=0)  # (H,)


@torch.no_grad()
def get_attention_scores_nonpadded(
    exp_name: str,
    B: int = 128,
    n_batches: int = 4,
    mode: str = "major",
    step: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force_recompute: bool = False,
    include_self: bool = True,
    modes: Optional[list] = None,
    n_minor: int = 256,
    n_ood: int = 40,
) -> dict:
    """
    Compute PTH and IH scores for every (layer, head) pair.

    Generates ``n_batches × B`` non-padded sequences from the latent Markov
    sampler, runs a forward pass with flash attention disabled to capture raw
    attention maps, then calls :func:`previous_token_head_score` and
    :func:`induction_head_score` on each layer's attention tensor.

    When ``modes`` is provided (e.g. ``["major", "minor", "ood"]``), scores
    are computed separately for each mode and returned in a nested dict.
    When ``modes`` is ``None`` (the default), the legacy single-mode behaviour
    is preserved and ``mode`` is used.

    Results are optionally cached so that subsequent calls with the same
    arguments can skip the forward passes.

    Parameters
    ----------
    exp_name        : Experiment name understood by ``nu.load_everything``.
    B               : Batch size per forward pass.
    n_batches       : Number of batches to average over.
    mode            : Sampling mode (legacy single-mode API).
    step            : Checkpoint step; ``None`` → final epoch.
    cache_dir       : Directory for cached results.  ``None`` disables caching.
    force_recompute : Ignore existing cache and recompute.
    include_self    : Passed to :func:`induction_head_score`.
    modes           : List of modes to evaluate (e.g. ``["major", "minor", "ood"]``).
                      If provided, overrides ``mode`` and returns per-mode scores.
    n_minor         : Number of minor tasks for the sampler.
    n_ood           : Number of OOD tasks for the sampler.

    Returns
    -------
    If ``modes`` is ``None`` (legacy):
        dict with ``pth``, ``ih``, ``n_layers``, ``n_heads``.
    If ``modes`` is a list:
        dict mapping each mode to a sub-dict with ``pth``, ``ih``, plus
        top-level ``n_layers`` and ``n_heads``.
    """
    from icl.utils.train_utils import get_attn_base

    multi_mode = modes is not None
    if not multi_mode:
        modes_to_run = [mode]
    else:
        modes_to_run = list(modes)

    # --- cache (only for legacy single-mode) ---
    cache_path = None
    if cache_dir is not None and not multi_mode:
        os.makedirs(cache_dir, exist_ok=True)
        suffix = "_incl_self" if include_self else ""
        cache_path = os.path.join(cache_dir, f"{exp_name}_attn_scores{suffix}_v3.npz")
        if not force_recompute and os.path.exists(cache_path):
            data = np.load(cache_path)
            logger.info(f"[attn-probes] Loaded cached scores from {cache_path}")
            return {
                "pth": data["pth"],
                "ih": data["ih"],
                "n_layers": int(data["n_layers"]),
                "n_heads": int(data["n_heads"]),
            }

    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    need_minor_ood = any(m in ("minor", "ood") for m in modes_to_run)
    if need_minor_ood:
        sampler, _, _ = get_latent_sampler(exp_name, n_minor=n_minor, n_ood=n_ood)
    else:
        sampler, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)

    n_layers = len(model.layers)
    all_mode_results = {}

    for cur_mode in modes_to_run:
        logger.info(f"[attn-probes] computing scores for mode={cur_mode} ...")
        pth_acc = None
        ih_acc = None

        for batch_idx in range(n_batches):
            gen_out = sampler.generate(mode=cur_mode, num_samples=B)
            tokens = gen_out[0].to(device)

            attn_maps = get_attn_base(model, tokens)

            for l_idx in range(n_layers):
                attn_l = attn_maps[l_idx].to(device)

                pth_l = previous_token_head_score(tokens, attn_l)
                ih_l = induction_head_score(tokens, attn_l, include_self=include_self)

                if pth_acc is None:
                    H = pth_l.shape[0]
                    pth_acc = torch.zeros(n_layers, H, dtype=torch.float32)
                    ih_acc = torch.zeros(n_layers, H, dtype=torch.float32)

                pth_acc[l_idx] += pth_l.cpu().float()
                ih_acc[l_idx] += ih_l.cpu().float()

            del tokens, attn_maps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[attn-probes] mode={cur_mode} batch {batch_idx + 1}/{n_batches} done")

        pth_acc /= n_batches
        ih_acc /= n_batches

        all_mode_results[cur_mode] = {
            "pth": pth_acc.numpy(),
            "ih": ih_acc.numpy(),
        }

    n_heads = all_mode_results[modes_to_run[0]]["pth"].shape[1]

    # --- cache (legacy) ---
    if cache_path is not None and not multi_mode:
        r = all_mode_results[modes_to_run[0]]
        np.savez(cache_path, pth=r["pth"], ih=r["ih"],
                 n_layers=n_layers, n_heads=n_heads)
        logger.info(f"[attn-probes] Cached scores to {cache_path}")

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if not multi_mode:
        r = all_mode_results[modes_to_run[0]]
        return {"pth": r["pth"], "ih": r["ih"],
                "n_layers": n_layers, "n_heads": n_heads}
    else:
        result = {"n_layers": n_layers, "n_heads": n_heads}
        for m in modes_to_run:
            result[m] = all_mode_results[m]
        return result


def plot_head_scores(
    pth: np.ndarray = None,
    ih: np.ndarray = None,
    figsize: Optional[tuple] = None,
    cmap: str = "viridis",
    show: bool = True,
    shared_scale: bool = False,
    ih_random_baseline: Optional[float] = None,
    vocab_size: Optional[int] = None,
    include_self: bool = True,
    multi_mode_scores: Optional[dict] = None,
) -> "matplotlib.figure.Figure":
    """
    Side-by-side heatmaps of PTH and IH scores across layers and heads.

    Supports two calling conventions:

    **Legacy (single-mode):**  pass ``pth`` and ``ih`` arrays directly.
    Produces one row of two heatmaps (PTH | IH).

    **Multi-mode:**  pass the dict returned by
    ``get_attention_scores_nonpadded(..., modes=[...])`` as
    ``multi_mode_scores``.  Produces one row per mode, each containing
    PTH and IH heatmaps with the mode name as a row label.

    All panels share a fixed colour scale of [0, 1] so colours directly
    represent absolute attention mass.

    Parameters
    ----------
    pth                 : (n_layers, n_heads) array — PTH scores (legacy).
    ih                  : (n_layers, n_heads) array — IH scores (legacy).
    figsize             : Figure size; auto-sized from data shape if ``None``.
    cmap                : Matplotlib colour map (default ``"viridis"``).
    show                : Call ``plt.show()`` before returning.
    shared_scale        : Kept for API compatibility (ignored; scale is always
                          fixed to [0, 1]).
    ih_random_baseline  : Override for the IH random baseline shown in the
                          subtitle.  Computed from ``vocab_size`` if not given.
    vocab_size          : Used to compute the random-chance IH baseline.
    include_self        : Matches the flag used in :func:`induction_head_score`.
    multi_mode_scores   : Dict returned by ``get_attention_scores_nonpadded``
                          with ``modes=[...]``.  Keys include mode names
                          mapping to sub-dicts with ``pth`` and ``ih``.

    Returns
    -------
    fig : matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    MODE_LABELS = {"major": "Major", "minor": "Minor", "ood": "OOD"}

    ih_sub = None

    # --- build list of (mode_label, pth_data, ih_data) rows ---
    if multi_mode_scores is not None:
        mode_keys = [k for k in multi_mode_scores
                     if k not in ("n_layers", "n_heads")]
        rows = [(MODE_LABELS.get(m, m),
                 multi_mode_scores[m]["pth"],
                 multi_mode_scores[m]["ih"]) for m in mode_keys]
    else:
        if pth is None or ih is None:
            raise ValueError("Provide pth/ih arrays or multi_mode_scores dict.")
        rows = [(None, pth, ih)]

    n_rows = len(rows)
    n_layers, n_heads = rows[0][1].shape

    import matplotlib.patheffects as pe
    from matplotlib.gridspec import GridSpec

    multi = n_rows > 1

    if figsize is None:
        col_w = max(2.2, n_heads * 1.1)
        row_h = max(1.8, n_layers * 0.55)
        left_margin = 0.7 if multi else 0.05
        fig_w = left_margin + col_w * 2 + 0.9
        fig_h = row_h * n_rows + (0.6 if multi else 0.4)
        figsize = (fig_w, fig_h)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, 2, figure=fig,
                  wspace=0.15,
                  hspace=0.12 if multi else 0.05)

    last_im = None
    all_axes = []
    for row_idx, (mode_label, pth_data, ih_data) in enumerate(rows):
        col_titles = ["PTH", "IH"]
        col_subtitles = [None, ih_sub]
        col_data = [pth_data, ih_data]

        for col_idx, (data, title, subtitle) in enumerate(
                zip(col_data, col_titles, col_subtitles)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            all_axes.append(ax)

            im = ax.imshow(
                data, aspect="auto", cmap=cmap,
                vmin=0.0, vmax=1.0, origin="upper",
            )
            last_im = im

            ax.set_xticks(range(n_heads))
            ax.set_yticks(range(n_layers))

            if row_idx == n_rows - 1:
                ax.set_xticklabels([str(h) for h in range(n_heads)],
                                   fontsize=9)
                ax.set_xlabel("Head", fontsize=10)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='x', length=2, pad=2)

            if col_idx == 0:
                ax.set_yticklabels([str(l) for l in range(n_layers)],
                                   fontsize=9)
                ax.set_ylabel("Layer", fontsize=10, labelpad=2)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis='y', length=2, pad=2)

            if row_idx == 0:
                full_title = (title if subtitle is None
                              else f"{title}\n{subtitle}")
                ax.set_title(full_title, fontsize=10, fontweight="bold",
                             pad=4)

            if col_idx == 0 and mode_label is not None:
                ax.annotate(
                    mode_label, xy=(0, 0.5),
                    xytext=(-0.35, 0.5), textcoords="axes fraction",
                    xycoords="axes fraction",
                    fontsize=11, fontweight="bold",
                    ha="center", va="center", rotation=90,
                )

            for li in range(n_layers):
                for hi in range(n_heads):
                    val = data[li, hi]
                    txt_color = "white" if val > 0.45 else "black"
                    stroke_color = ("black" if txt_color == "white"
                                    else "white")
                    ax.text(
                        hi, li, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=9, fontweight="semibold",
                        color=txt_color,
                        path_effects=[
                            pe.withStroke(linewidth=1.2,
                                         foreground=stroke_color),
                        ],
                    )

    left = 0.13 if multi else 0.08
    fig.subplots_adjust(left=left, right=0.88, top=0.88, bottom=0.08)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.72])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Attention mass", fontsize=9)

    fig.subplots_adjust(top=0.92)
    if show:
        plt.show()
    return fig


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
    figsize: tuple = (14, 6),
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
