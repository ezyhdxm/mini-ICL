"""Posterior-weighted task-vector injection for the Latent Markov task.

For OOD samples, replaces the task-subspace component with the
posterior-weighted combination of task vectors and compares the model
output to the misspecified Bayesian prediction.

Also provides a *direct* variant that skips OOD generation entirely:
sample α ~ Dirichlet, inject α @ W_task, and compare to α-weighted
mixture q(s_{t+1} | s_t) = Σ_k α_k P_k(s_{t+1} | s_t).
"""

from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.probes import train_linear_hidden_predictor
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.latent_markov.analysis.bayes import task_posterior_over_time
from icl.utils.logger import setup_logger
from icl.utils.inject_common import (
    kl_softmax as _kl,
    build_task_subspace,
    aggregate_position_metrics,
    plot_inject_posterior_per_position,
    plot_inject_posterior_across_layers as _plot_across_layers,
)

logger = setup_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Direct injection (no OOD, no posterior computation)
# ──────────────────────────────────────────────────────────────────────

def intervene_direct_injection(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 2000,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    include_position_bias: bool = False,
    include_logit: bool = False,
    dirichlet_alpha: float = 1.0,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """Direct α-injection test for the linear-interpolation hypothesis.

    Instead of generating OOD sequences and computing posteriors, this
    function:

    1. Samples input sequences from major tasks.
    2. Samples α ~ Dirichlet(dirichlet_alpha · 1_K) independently per
       sample (constant across all positions).
    3. Injects  h' = h − h P_task + (α W_task)  at every position.
    4. Compares model output to  q(s_{t+1} | s_t) = Σ_k α_k P_k(s_{t+1} | s_t).

    Returns a dict compatible with ``plot_inject_posterior_per_position``.
    """
    _, sampler, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    device = config.device

    K_major = int(sampler.n_major_tasks)
    major_trans = sampler.major_trans_mat.float().to(device)  # (K, S^order, S)
    order = int(sampler.order)
    powers = sampler.powers.to(device)
    seq_len = sampler.seq_len

    # ── Fit task subspace ────────────────────────────────────────────
    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", step=step,
        n_minor=-1, print_summary=False, skip_baselines=True,
        include_position_bias=include_position_bias,
        include_logit=include_logit,
    )
    W_fit = fit_res["model_weight"].float()            # (K, D)
    P_task, ref_vecs, rank = build_task_subspace(W_fit, center_task_vecs, device)

    if verbose:
        logger.info(
            f"[direct inj latent] rank={rank}, "
            f"R²={fit_res['val_r2']:.4f}"
        )

    # ── Helpers ──────────────────────────────────────────────────────
    def _build_history_idx(samples_batch):
        B_cur, L = samples_batch.shape
        if order == 1:
            return samples_batch.long()
        idx = torch.zeros(B_cur, L, dtype=torch.long, device=device)
        for t in range(order - 1, L):
            for j in range(order):
                idx[:, t] += (
                    samples_batch[:, t - (order - 1) + j].long()
                    * int(powers[j].item())
                )
        return idx

    # ── Evaluate ─────────────────────────────────────────────────────
    if eval_positions is None:
        eval_positions = list(range(order, seq_len))

    dirichlet = torch.distributions.Dirichlet(
        torch.ones(K_major, device=device) * dirichlet_alpha,
    )

    acc_base = {p: [] for p in eval_positions}
    acc_inj = {p: [] for p in eval_positions}
    acc_mode = {p: [] for p in eval_positions}
    acc_post_std = {p: [] for p in eval_positions}
    acc_min_dist = {p: [] for p in eval_positions}

    eye_K = torch.eye(K_major, device=device)

    n_done = 0
    bi = 0
    while n_done < n_samples:
        gen = sampler.generate(
            mode="major", task=None, num_samples=B, epochs=1,
        )
        samples = gen[0] if isinstance(gen, (tuple, list)) else gen
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples = samples.to(device)
        cur_B = samples.shape[0]
        bi += 1

        alpha = dirichlet.sample((cur_B,))                # (B, K)

        hist_idx = _build_history_idx(samples)
        trans_at = major_trans[:, hist_idx].permute(1, 2, 0, 3)  # (B, T, K, S)
        q_mis = torch.einsum("bk,btks->bts", alpha, trans_at)   # (B, T, S)

        mode_idx = alpha.argmax(dim=-1)                          # (B,)
        q_mode_trans = major_trans[mode_idx]                     # (B, S^order, S)
        q_mode = q_mode_trans[
            torch.arange(cur_B, device=device).unsqueeze(1),
            hist_idx,
        ]                                                        # (B, T, S)

        with torch.no_grad():
            logits_base = model(samples)

        def _hook(mod, inp, out,
                  _a=alpha, _W=ref_vecs, _P=P_task):
            h = out if torch.is_tensor(out) else out[0]
            B_h = h.shape[0]
            inj = (_a[:B_h] @ _W).unsqueeze(1)            # (B, 1, D)
            h_new = h - (h @ _P) + inj
            return h_new if torch.is_tensor(out) else (h_new,) + out[1:]

        handle = model.layers[layer].attn_block.register_forward_hook(
            _hook,
        )
        try:
            with torch.no_grad():
                logits_inj = model(samples)
        finally:
            handle.remove()

        for p in eval_positions:
            if p + 1 >= samples.shape[1]:
                continue
            q_p = q_mis[:, p]                              # (B, S)
            acc_base[p].append(_kl(logits_base[:, p], q_p).mean().item())
            acc_inj[p].append(_kl(logits_inj[:, p], q_p).mean().item())
            lq_md = torch.log(q_mode[:, p].clamp_min(1e-12))
            lq_ms = torch.log(q_p.clamp_min(1e-12))
            kl_md = (q_mode[:, p] * (lq_md - lq_ms)).sum(-1).mean().item()
            acc_mode[p].append(kl_md)
            acc_post_std[p].append(alpha.std(dim=-1).mean().item())
            dists = (alpha.unsqueeze(1) - eye_K.unsqueeze(0)).abs().sum(-1)
            acc_min_dist[p].append(dists.min(dim=-1).values.mean().item())

        n_done += cur_B

    # ── Aggregate ────────────────────────────────────────────────────
    positions, kl_base, kl_inj, kl_mode_list = [], [], [], []
    post_std, min_dist = [], []
    for p in eval_positions:
        if acc_base[p]:
            positions.append(p)
            kl_base.append(float(np.mean(acc_base[p])))
            kl_inj.append(float(np.mean(acc_inj[p])))
            kl_mode_list.append(float(np.mean(acc_mode[p])))
            post_std.append(float(np.mean(acc_post_std[p])))
            min_dist.append(float(np.mean(acc_min_dist[p])))

    if print_summary:
        print(f"\n{'=' * 70}")
        print(f"Direct Injection  (latent, layer {layer})")
        print(f"{'=' * 70}")
        print(f"  rank={rank}  K={K_major}  "
              f"dirichlet_alpha={dirichlet_alpha}")
        print(f"  Mean KL(baseline \u2016 q_\u03b1) = {np.mean(kl_base):.4f}")
        print(f"  Mean KL(injected \u2016 q_\u03b1) = {np.mean(kl_inj):.4f}")
        print(f"  Mean KL(mode \u2016 q_\u03b1)     = {np.mean(kl_mode_list):.4f}")
        print()

    return dict(
        layer=layer, rank=rank, K_major=K_major,
        positions=positions,
        kl_baseline=np.array(kl_base),
        kl_injected=np.array(kl_inj),
        kl_mode=np.array(kl_mode_list),
        posterior_std=np.array(post_std),
        min_dist_basis=np.array(min_dist),
    )


# ──────────────────────────────────────────────────────────────────────
#  OOD-based posterior injection (original approach)
# ──────────────────────────────────────────────────────────────────────


def intervene_inject_posterior(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_eval: int = 500,
    n_ood: int = 30,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    include_position_bias: bool = False,
    include_logit: bool = False,
    filter_convergent: bool = True,
    convergence_threshold: float = 0.75,
    filter_positions: Optional[list] = None,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Posterior-weighted task-vector injection on OOD data (latent Markov).

    For each OOD sample at each position *t*:

      1. Compute misspecified posterior α_{t,k} = P(Z=k | s_{≤t})
         using only major transition matrices.
      2. Replace task component::

             h' = h − h P_task + α_t W_task

      3. Compare intervened model output to the misspecified Bayesian
         prediction  q(s_{t+1} | s_t) = Σ_k α_{t,k} P_k(s_{t+1} | s_t).

    Returns
    -------
    dict with keys:
        positions   : evaluated positions
        kl_baseline : KL(baseline_output || misspecified_Bayesian) per position
        kl_injected : KL(injected_output || misspecified_Bayesian) per position
    """
    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    device = config.device

    sampler_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)
    K_major = int(sampler_ood.n_major_tasks)
    major_trans = sampler_ood.major_trans_mat.float().to(device)  # (K, S^order, S)
    order = int(sampler_ood.order)
    powers = sampler_ood.powers.to(device)
    seq_len = sampler_ood.seq_len

    # ── Fit task subspace ────────────────────────────────────────────
    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", step=step,
        n_minor=-1, print_summary=False, skip_baselines=True,
        include_position_bias=include_position_bias,
        include_logit=include_logit,
    )
    W_fit = fit_res["model_weight"].float()          # (K, D)
    P_task, ref_vecs, rank = build_task_subspace(W_fit, center_task_vecs, device)

    if verbose:
        logger.info(
            f"[posterior inj latent] rank={rank}, "
            f"R²={fit_res['val_r2']:.4f}"
        )

    # ── Helpers ──────────────────────────────────────────────────────
    def _build_history_idx(samples_batch):
        """Build Markov history index for predicting next token.

        At position t the model predicts s_{t+1}, so the relevant
        history is (s_{t-order+1}, …, s_t).
        """
        B_cur, L = samples_batch.shape
        if order == 1:
            return samples_batch.long()
        idx = torch.zeros(B_cur, L, dtype=torch.long, device=device)
        for t in range(order - 1, L):
            for j in range(order):
                idx[:, t] += (
                    samples_batch[:, t - (order - 1) + j].long()
                    * int(powers[j].item())
                )
        return idx

    # ── Evaluate ─────────────────────────────────────────────────────
    if eval_positions is None:
        eval_positions = list(range(order, seq_len))

    # ── Collect OOD samples (optionally filtering convergent ones) ─
    all_samples_list, all_alpha_list = [], []
    n_generated = 0
    max_generate = n_samples_eval * 10

    while (sum(s.shape[0] for s in all_samples_list) < n_samples_eval
           and n_generated < max_generate):
        gen = sampler_ood.generate(
            mode="minor", task=None, num_samples=B, epochs=1,
        )
        samples = gen[0] if isinstance(gen, (tuple, list)) else gen
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples = samples.to(device)
        n_generated += samples.shape[0]

        alpha = task_posterior_over_time(
            sampler_ood, samples, include_minor=False,
        ).to(device)

        if filter_convergent:
            if filter_positions is not None:
                fp = [min(p, alpha.shape[1] - 1) for p in filter_positions]
            else:
                fp = [max(eval_positions[0] - 10, 0),
                      min(eval_positions[-1] + 10, alpha.shape[1] - 1)]
            keep = torch.ones(alpha.shape[0], dtype=torch.bool, device=device)
            for fp_i in fp:
                keep &= alpha[:, fp_i, :].max(dim=-1).values < convergence_threshold
            if keep.any():
                all_samples_list.append(samples[keep])
                all_alpha_list.append(alpha[keep])
        else:
            all_samples_list.append(samples)
            all_alpha_list.append(alpha)

    if not all_samples_list:
        if print_summary:
            print(f"  WARNING: No non-convergent OOD samples found "
                  f"(threshold={convergence_threshold})")
        return dict(
            layer=layer, rank=rank, K_major=K_major,
            positions=[], kl_baseline=np.array([]),
            kl_injected=np.array([]),
            posterior_std=np.array([]), min_dist_basis=np.array([]),
        )

    all_samples = torch.cat(all_samples_list)[:n_samples_eval]
    all_alpha = torch.cat(all_alpha_list)[:n_samples_eval]

    if print_summary and filter_convergent:
        kept = all_samples.shape[0]
        print(f"  Filtered: kept {kept}/{n_generated} OOD samples "
              f"({kept / n_generated * 100:.0f}%, "
              f"threshold={convergence_threshold})")

    # ── Run intervention in batches ───────────────────────────────
    acc_base = {p: [] for p in eval_positions}
    acc_inj = {p: [] for p in eval_positions}
    acc_post_std = {p: [] for p in eval_positions}
    acc_min_dist = {p: [] for p in eval_positions}

    eye_K = torch.eye(K_major, device=device)

    for start in range(0, all_samples.shape[0], B):
        samples = all_samples[start:start + B]
        alpha = all_alpha[start:start + B]

        hist_idx = _build_history_idx(samples)
        trans_at = major_trans[:, hist_idx].permute(1, 2, 0, 3)
        q_mis = torch.einsum("btk,btks->bts", alpha, trans_at)

        with torch.no_grad():
            logits_base = model(samples)

        def _hook(mod, inp, out, _a=alpha, _W=ref_vecs, _P=P_task):
            h = out if torch.is_tensor(out) else out[0]
            h_new = h - (h @ _P) + (_a[:h.shape[0]] @ _W)
            return h_new if torch.is_tensor(out) else (h_new,) + out[1:]

        handle = model.layers[layer].attn_block.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                logits_inj = model(samples)
        finally:
            handle.remove()

        for p in eval_positions:
            if p + 1 >= samples.shape[1]:
                continue
            q_p = q_mis[:, p]
            acc_base[p].append(_kl(logits_base[:, p], q_p).mean().item())
            acc_inj[p].append(_kl(logits_inj[:, p], q_p).mean().item())
            a_p = alpha[:, p, :]
            acc_post_std[p].append(a_p.std(dim=-1).mean().item())
            dists = (a_p.unsqueeze(1) - eye_K.unsqueeze(0)).abs().sum(dim=-1)
            acc_min_dist[p].append(dists.min(dim=-1).values.mean().item())

    # ── Aggregate ────────────────────────────────────────────────────
    positions, kl_base, kl_inj, post_std, min_dist = [], [], [], [], []
    for p in eval_positions:
        if acc_base[p]:
            positions.append(p)
            kl_base.append(float(np.mean(acc_base[p])))
            kl_inj.append(float(np.mean(acc_inj[p])))
            post_std.append(float(np.mean(acc_post_std[p])))
            min_dist.append(float(np.mean(acc_min_dist[p])))

    if print_summary:
        print(f"\n{'=' * 70}")
        print(f"Posterior-Weighted Injection  (latent, layer {layer})")
        print(f"{'=' * 70}")
        print(f"  rank={rank}  K={K_major}  n_ood={n_ood}")
        print(f"  Mean KL(baseline \u2016 misBayes) = {np.mean(kl_base):.4f}")
        print(f"  Mean KL(injected \u2016 misBayes) = {np.mean(kl_inj):.4f}")
        print()

    return dict(
        layer=layer, rank=rank, K_major=K_major,
        positions=positions,
        kl_baseline=np.array(kl_base),
        kl_injected=np.array(kl_inj),
        posterior_std=np.array(post_std),
        min_dist_basis=np.array(min_dist),
    )


# ──────────────────────────────────────────────────────────────────────
#  Layer sweep + plotting
# ──────────────────────────────────────────────────────────────────────

def plot_inject_posterior_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    show: bool = True,
    **kwargs,
):
    """Sweep ``intervene_inject_posterior`` over layers and plot.

    Returns ``(fig, all_results)``.
    """
    return _plot_across_layers(
        exp_name=exp_name,
        task_name="latent",
        intervene_fn=intervene_inject_posterior,
        layers=layers,
        load_everything_fn=nu.load_everything,
        show=show,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────
#  Averaging-based injection (per-position, simplex plot)
# ──────────────────────────────────────────────────────────────────────

def intervene_averaging_injection(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 2000,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    estimation_positions: Optional[list] = None,
    estimation_B: int = 128,
    per_position_mean: bool = True,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    dirichlet_alpha: float = 1.0,
    figsize: tuple = (10, 3.5),
    show: bool = True,
    verbose: bool = False,
    print_summary: bool = True,
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    use_per_position_vecs: bool = False,
) -> dict:
    r"""Direct α-injection using averaging-based task vectors (latent Markov).

    Like ``intervene_direct_injection`` but estimates task vectors by
    averaging hidden states at late positions rather than fitting an OLS
    probe, and performs **separate per-position injection**.

    Steps:
      1. Extract hidden states for all major tasks at *layer*.
      2. Estimate task vectors via ``estimate_task_vectors_by_averaging``.
      3. Build orthogonal projector P_task from the centred task vectors.
      4. For each eval position p, run a *separate* forward pass that
         injects only at position p:
         ``h'[p] = h[p] − (h[p]−μ_p)P + α·θ``.
         Compare output at p to q_α(s_{p+1}|s_p) = Σ_k α_k P_k(s_{p+1}|s_p).
      5. Plot KL divergence on the simplex.

    Parameters
    ----------
    per_position_mean : bool
        If True, subtract per-position grand mean when removing the task
        component (handles position-dependent rotations).
    estimation_B : int
        Batch size for hidden-state extraction (task vector estimation).
    use_per_position_vecs : bool
        If True, estimate task vectors independently at each position and
        perform direct subtraction in the hook:
        ``h'[p] = h[p] − θ_k(p) + α·θ(p)`` where k is the true task.
        Data is generated per-task so the true task label is known.
    """
    import gc
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.utils.separability import (
        estimate_task_vectors_by_averaging,
        per_position_task_vectors,
    )

    # ── Load model & config ──────────────────────────────────────────
    _, sampler, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    device = config.device

    K_major = int(sampler.n_major_tasks)
    major_trans = sampler.major_trans_mat.float().to(device)  # (K, S^order, S)
    order = int(sampler.order)
    powers = sampler.powers.to(device)
    seq_len = sampler.seq_len

    T_max = seq_len - 1  # hiddens are 0-indexed up to seq_len-2
    if estimation_positions is None:
        estimation_positions = list(range(max(0, T_max - 30), T_max))
    if eval_positions is None:
        eval_positions = list(range(order, T_max))

    # ── Extract hidden states for major tasks ────────────────────────
    hiddens_all, _k_minor = _get_hiddens_at_real_positions(
        task_name="latent", exp_name=exp_name,
        n_minor=0, n_ood=0, B=estimation_B, step=step,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )
    # hiddens_all: (n_layers, K, T, B_est, D)
    hiddens_major = hiddens_all[layer].float()  # (K_major, T, B_est, D)
    T_full = hiddens_major.shape[1]
    D = hiddens_major.shape[-1]

    if use_per_position_vecs:
        tv_pos_all, _grand_means = per_position_task_vectors(
            hiddens_major, per_position_mean=per_position_mean,
        )
        tv_pos_all = tv_pos_all.float().to(device)  # (K, T, D)

        if verbose:
            logger.info(
                f"[avg inj latent per-pos] K={K_major}, "
                f"direct subtraction with per-position task vectors"
            )
    else:
        task_vecs, grand_mean = estimate_task_vectors_by_averaging(
            hiddens_major, estimation_positions,
        )

        tv = task_vecs.float()
        _, S_tv, Vt_tv = torch.linalg.svd(tv, full_matrices=False)
        rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
        P_task = (Vt_tv[:rank].T @ Vt_tv[:rank]).to(device)
        ref_vecs = task_vecs.to(device)  # (K, D)

        tv_pos_all = None
        P_task_per_pos = None

        if per_position_mean:
            mu_per_point = hiddens_major.mean(dim=(0, 2)).to(device)
        else:
            mu_global = grand_mean.to(device)

        if verbose:
            logger.info(
                f"[avg inj latent] rank={rank}, K={K_major}, "
                f"estimation_positions={estimation_positions[:3]}..{estimation_positions[-1]}"
            )

    del hiddens_all, hiddens_major
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Helpers ──────────────────────────────────────────────────────
    def _build_history_idx(samples_batch):
        B_cur, L = samples_batch.shape
        if order == 1:
            return samples_batch.long()
        idx = torch.zeros(B_cur, L, dtype=torch.long, device=device)
        for t in range(order - 1, L):
            for j in range(order):
                idx[:, t] += (
                    samples_batch[:, t - (order - 1) + j].long()
                    * int(powers[j].item())
                )
        return idx

    # ── Injection loop (per-position) ────────────────────────────────
    dirichlet = torch.distributions.Dirichlet(
        torch.ones(K_major, device=device) * dirichlet_alpha,
    )

    all_alphas, all_kl_base, all_kl_inj, all_kl_mode = [], [], [], []

    fwd_chunk = B
    n_done = 0
    bi = 0

    def _forward_chunked(model_fn, data, chunk_size):
        outs = []
        for i in range(0, data.shape[0], chunk_size):
            outs.append(model_fn(data[i:i + chunk_size]))
        return torch.cat(outs, dim=0)

    def _forward_chunked_with_hook(model_fn, data, hook_maker, chunk_size):
        outs = []
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i + chunk_size]
            hook_fn = hook_maker(i)
            handle = model.layers[layer].attn_block.register_forward_hook(hook_fn)
            try:
                outs.append(model_fn(chunk))
            finally:
                handle.remove()
        return torch.cat(outs, dim=0)

    while n_done < n_samples:
        # ── Generate samples ──────────────────────────────────────────
        if use_per_position_vecs:
            B_per_task = max(1, B // K_major)
            all_samp, all_tl = [], []
            for k in range(K_major):
                gen_k = sampler.generate(
                    mode="major", task=k, num_samples=B_per_task, epochs=1,
                )
                s_k = gen_k[0] if isinstance(gen_k, (tuple, list)) else gen_k
                if s_k.dim() == 3:
                    s_k = s_k.squeeze(0)
                all_samp.append(s_k)
                all_tl.append(
                    torch.full((s_k.shape[0],), k, dtype=torch.long),
                )
            samples = torch.cat(all_samp, dim=0).to(device)
            task_labels = torch.cat(all_tl).to(device)
        else:
            gen = sampler.generate(
                mode="major", task=None, num_samples=B, epochs=1,
            )
            samples = gen[0] if isinstance(gen, (tuple, list)) else gen
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)
            task_labels = None

        cur_B = samples.shape[0]
        bi += 1

        alpha = dirichlet.sample((cur_B,))  # (B, K)
        hist_idx = _build_history_idx(samples)
        trans_at = major_trans[:, hist_idx].permute(1, 2, 0, 3)  # (B, T, K, S)
        q_mis = torch.einsum("bk,btks->bts", alpha, trans_at)  # (B, T, S)

        mode_idx = alpha.argmax(dim=-1)  # (B,)
        q_mode_trans = major_trans[mode_idx]  # (B, S^order, S)
        q_mode = q_mode_trans[
            torch.arange(cur_B, device=device).unsqueeze(1), hist_idx,
        ]  # (B, T, S)

        # Baseline forward (with OOM retry)
        while True:
            try:
                with torch.no_grad():
                    logits_base = _forward_chunked(model, samples, fwd_chunk)
                break
            except torch.cuda.OutOfMemoryError:
                fwd_chunk = max(1, fwd_chunk // 2)
                logger.warning(f"OOM baseline, chunk -> {fwd_chunk}")
                torch.cuda.empty_cache()

        # Per-position injected forwards
        kl_inj_per_pos = []
        kl_base_per_pos = []
        kl_mode_per_pos = []
        valid_pos = []
        for p in eval_positions:
            if p + 1 >= samples.shape[1]:
                continue
            valid_pos.append(p)
            q_p = q_mis[:, p]  # (B, S)
            kl_base_per_pos.append(_kl(logits_base[:, p], q_p).cpu())

            def _hook_maker(offset, _alpha=alpha, _p=p):
                if use_per_position_vecs:
                    _rv = tv_pos_all[:, _p, :]
                    _tl = task_labels
                else:
                    _mu = (mu_per_point[_p] if per_position_mean
                           else mu_global)
                    _P = P_task
                    _rv = ref_vecs
                def _hook(mod, inp, out):
                    h = out if torch.is_tensor(out) else out[0]
                    h_new = h.clone()
                    B_h = h.shape[0]
                    a_sub = _alpha[offset:offset + B_h]
                    h_at = h_new[:, _p, :]
                    if use_per_position_vecs:
                        tl_sub = _tl[offset:offset + B_h]
                        theta_k = _rv[tl_sub]
                        tc = a_sub @ _rv
                        h_new[:, _p, :] = h_at - theta_k + tc
                    else:
                        h_centered = h_at - _mu.unsqueeze(0)
                        tc = a_sub @ _rv
                        h_new[:, _p, :] = h_at - (h_centered @ _P) + tc
                    return h_new if torch.is_tensor(out) else (h_new,) + out[1:]
                return _hook

            while True:
                try:
                    with torch.no_grad():
                        logits_inj_p = _forward_chunked_with_hook(
                            model, samples, _hook_maker, fwd_chunk,
                        )
                    kl_inj_per_pos.append(
                        _kl(logits_inj_p[:, p], q_p).cpu()
                    )
                    del logits_inj_p
                    break
                except torch.cuda.OutOfMemoryError:
                    fwd_chunk = max(1, fwd_chunk // 2)
                    logger.warning(f"OOM inj pos {p}, chunk -> {fwd_chunk}")
                    torch.cuda.empty_cache()

            lq_md = torch.log(q_mode[:, p].clamp_min(1e-12))
            lq_ms = torch.log(q_p.clamp_min(1e-12))
            kl_mode_per_pos.append(
                (q_mode[:, p] * (lq_md - lq_ms)).sum(-1).cpu()
            )

        if kl_inj_per_pos:
            kl_b_stack = torch.stack(kl_base_per_pos, dim=1)  # (B, n_eval)
            kl_i_stack = torch.stack(kl_inj_per_pos, dim=1)
            kl_m_stack = torch.stack(kl_mode_per_pos, dim=1)
            all_alphas.append(alpha.cpu())
            all_kl_base.append(kl_b_stack.mean(dim=1))
            all_kl_inj.append(kl_i_stack.mean(dim=1))
            all_kl_mode.append(kl_m_stack.mean(dim=1))

        n_done += cur_B
        del samples, logits_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Aggregate ────────────────────────────────────────────────────
    all_alphas_np = torch.cat(all_alphas, dim=0).numpy()  # (N, K)
    kl_base_np = torch.cat(all_kl_base).numpy()           # (N,)
    kl_inj_np = torch.cat(all_kl_inj).numpy()
    kl_mode_np = torch.cat(all_kl_mode).numpy()

    # ── Simplex plots ────────────────────────────────────────────────
    from icl.utils.plot_config import plot_simplex_panel_pairs

    fig1, fig2 = plot_simplex_panel_pairs(
        all_alphas_np, kl_base_np, kl_inj_np, kl_mode_np,
        metric_label="KL divergence",
        mean_fmt="mean KL = {:.4f}",
        color_vmin=color_vmin, color_vmax=color_vmax,
        figsize=figsize,
    )
    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    _rank = rank if not use_per_position_vecs else None
    result = dict(
        layer=layer, rank=_rank, K_major=K_major,
        alphas=all_alphas_np,
        kl_baseline=kl_base_np,
        kl_injected=kl_inj_np,
        kl_mode=kl_mode_np,
        fig=fig1, fig_mode=fig2,
    )

    if print_summary:
        print(f"\n{'=' * 70}")
        print(f"Averaging Injection  (latent, layer {layer})")
        print(f"{'=' * 70}")
        print(f"  rank={_rank}  K={K_major}  "
              f"dirichlet_alpha={dirichlet_alpha}")
        print(f"  Mean KL(baseline || q_α) = {kl_base_np.mean():.4f}")
        print(f"  Mean KL(injected || q_α) = {kl_inj_np.mean():.4f}")
        print(f"  Mean KL(mode || q_α)     = {kl_mode_np.mean():.4f}")
        print()

    return result
