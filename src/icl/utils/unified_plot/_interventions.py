import icl.utils.notebook_utils as nu
from icl.utils.unified_interface import _get_hiddens_at_real_positions

from typing import Optional
import numpy as np
import torch


def _get_layer_module(model, layer_index: int, is_linear: bool):
    """Return the attention block module for a given layer."""
    if is_linear:
        return model.transformer.blocks[layer_index].attn_block
    return model.layers[layer_index].attn_block


# ======================================================================
#  Task-component scaling intervention
# ======================================================================

@torch.no_grad()
def intervene_scale_task_component(
    task_name: str,
    exp_name: str,
    layer: int,
    scale_factors: Optional[list] = None,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    estimation_positions: Optional[list] = None,
    estimation_B: int = 128,
    per_position_mean: bool = True,
    late_fraction: float = 0.5,
    figsize: tuple = (14, 5),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    r"""Scale the task-subspace component and measure output sensitivity.

    For each scale factor *c*, modifies the hidden state at *layer*:

    .. math::

        h'_t = h_t + (c - 1)\, P_{\text{task}}(h_t - \mu_t)

    When *c* = 1 this is the identity.  When *c* = 0 the task component
    is removed entirely.  When *c* > 1 the task component is amplified.

    **Metric**: KL(softmax(logits_scaled) || softmax(logits_base)) for
    the coin / latent tasks; MSE(preds_scaled, preds_base) for linear
    regression.

    Parameters
    ----------
    task_name : ``"coin"`` | ``"latent"`` | ``"linear"``
    exp_name : str
        Experiment folder name.
    layer : int
        Transformer layer to intervene on.
    scale_factors : list[float], optional
        Values of *c* to sweep.  Default ``[0, .25, .5, .75, 1, 1.25,
        1.5, 2, 3]``.
    B : int
        Batch size per forward pass.
    n_samples : int
        Total number of samples to evaluate.
    step : int, optional
        Checkpoint step (``None`` = final).
    estimation_positions : list[int], optional
        Positions used to estimate task vectors (default: last 10).
    estimation_B : int
        Batch size for hidden-state extraction (task vector estimation).
    per_position_mean : bool
        If True, centre with a position-specific grand mean.
    late_fraction : float
        Fraction of positions considered "late" for the summary panel.
    figsize : tuple
        ``(width, height)`` for the figure.
    show : bool
        Whether to call ``plt.show()``.
    verbose : bool
        Extra logging.

    Returns
    -------
    dict
        ``scale_factors``, ``metric_by_c`` (c → (n_eval_pos,) array),
        ``eval_positions``, ``fig``, ``axes``.
    """
    import gc
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from icl.utils.separability import estimate_task_vectors_by_averaging

    if scale_factors is None:
        scale_factors = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

    is_linear = task_name == "linear"

    # ── 1. Load model & sampler ───────────────────────────────────────
    if is_linear:
        from icl.linear.analysis.interventions._helpers import (
            _load_and_prepare_model,
        )
        from icl.linear.analysis._helpers import _task_positions

        model, train_task, config, device = _load_and_prepare_model(
            exp_name, step=step,
        )
        if step is None:
            step = config.training.total_steps
        n_points = int(config.task.n_points)
        pad_mode = getattr(model, "pad", "mapsto")
        task_pos = _task_positions(pad_mode, n_points, device)
    else:
        _, sampler, config = nu.load_everything(task_name, exp_name)
        if step is None:
            step = config.training.num_epochs
        model, _ = nu.load_checkpoint(
            config, step=step, exp_name=exp_name, return_actual_step=True,
        )
        model.eval().to(config.device)
        device = config.device

    # ── 2. Estimate task vectors & build projector ─────────────────────
    hiddens_all = _get_hiddens_at_real_positions(
        task_name, exp_name, n_minor=0, n_ood=0,
        B=estimation_B, step=step,
    )
    if isinstance(hiddens_all, tuple):
        hiddens_all = hiddens_all[0]

    hiddens_layer = hiddens_all[layer].float()       # (K, T, B_est, D)
    K, T_est, _, D = hiddens_layer.shape

    if estimation_positions is None:
        estimation_positions = list(range(max(0, T_est - 10), T_est))

    task_vecs, grand_mean = estimate_task_vectors_by_averaging(
        hiddens_layer, estimation_positions,
    )

    tv = task_vecs.float()
    _, S_tv, Vt_tv = torch.linalg.svd(tv, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    P_task = (Vt_tv[:rank].T @ Vt_tv[:rank]).to(device)

    if per_position_mean:
        mu_per_pos = hiddens_layer.mean(dim=(0, 2)).to(device)  # (T_est, D)
    else:
        mu_global = grand_mean.to(device)                       # (D,)

    if verbose:
        print(f"[scale] rank={rank}, K={K}, T_est={T_est}, D={D}")

    del hiddens_all, hiddens_layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── 3. Determine eval positions ───────────────────────────────────
    #  For linear, the model output is (B, n_points) — already indexed
    #  by point, not by sequence position.  So eval_positions_seq should
    #  be point indices.  The hook, however, must modify hidden states at
    #  the *sequence*-level positions given by task_pos.
    if is_linear:
        eval_positions_seq = list(range(n_points))
        eval_positions_label = list(range(n_points))
        task_pos_list = task_pos.cpu().tolist()
    else:
        T_max = T_est
        eval_positions_seq = list(range(T_max))
        eval_positions_label = eval_positions_seq
        task_pos_list = None

    n_eval = len(eval_positions_seq)

    # ── 4. Hook factory ───────────────────────────────────────────────
    def _make_hook(c):
        if c == 1.0:
            return None

        def _hook(mod, inp, out):
            h = out if torch.is_tensor(out) else out[0]
            h_new = h.clone()
            if is_linear:
                _tp = task_pos.to(h.device)
                h_sub = h_new[:, _tp, :]             # (B, n_points, D)
                if per_position_mean:
                    mu = mu_per_pos.unsqueeze(0)      # (1, n_points, D)
                else:
                    mu = mu_global.unsqueeze(0).unsqueeze(0)
                tc = (h_sub - mu) @ P_task
                h_new[:, _tp, :] = h_sub + (c - 1.0) * tc
            else:
                T_h = min(T_est, h_new.shape[1])
                h_sub = h_new[:, :T_h, :]
                if per_position_mean:
                    mu = mu_per_pos[:T_h].unsqueeze(0)
                else:
                    mu = mu_global.unsqueeze(0).unsqueeze(0)
                tc = (h_sub - mu) @ P_task
                h_new[:, :T_h, :] = h_sub + (c - 1.0) * tc
            return h_new if torch.is_tensor(out) else (h_new,) + out[1:]
        return _hook

    # ── 5. KL / MSE helpers ───────────────────────────────────────────
    def _kl_per_pos(logits_a, logits_b, positions):
        """KL(softmax(a) || softmax(b)) at given positions, (B, n_pos)."""
        la = torch.log_softmax(logits_a[:, positions], dim=-1)
        lb = torch.log_softmax(logits_b[:, positions], dim=-1)
        return (la.exp() * (la - lb)).sum(-1)

    def _mse_per_pos(preds_a, preds_b, positions):
        """MSE at given positions, (B, n_pos)."""
        return (preds_a[:, positions] - preds_b[:, positions]).pow(2)

    # ── 6. Evaluation loop ────────────────────────────────────────────
    accum = {c: [] for c in scale_factors}
    fwd_chunk = B
    n_done = 0
    bi = 0

    if is_linear:
        orig_bs = int(train_task.batch_size)
        train_task.batch_size = B

    while n_done < n_samples:
        # Generate data
        if is_linear:
            demo_data, _, demo_target = train_task.sample_batch(
                step=bi + 99999, is_eval=False,
            )
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)
            cur_B = demo_data.shape[0]
        else:
            gen = sampler.generate(
                mode="major", task=None, num_samples=B, epochs=1,
            )
            samples = gen[0] if isinstance(gen, (tuple, list)) else gen
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)
            cur_B = samples.shape[0]

        # Baseline forward
        if is_linear:
            logits_base = model(demo_data, demo_target)
        else:
            logits_base = model(samples)

        # Scaled forwards
        for c in scale_factors:
            hook_fn = _make_hook(c)
            if hook_fn is None:
                metric_c = torch.zeros(cur_B, n_eval, device=device)
            else:
                handle = _get_layer_module(model, layer, is_linear).register_forward_hook(hook_fn)
                try:
                    if is_linear:
                        logits_scaled = model(demo_data, demo_target)
                    else:
                        logits_scaled = model(samples)
                finally:
                    handle.remove()

                if is_linear:
                    metric_c = _mse_per_pos(
                        logits_scaled, logits_base, eval_positions_seq,
                    )
                else:
                    metric_c = _kl_per_pos(
                        logits_scaled, logits_base, eval_positions_seq,
                    )
                del logits_scaled

            accum[c].append(metric_c.cpu())

        n_done += cur_B
        bi += 1
        if is_linear:
            del demo_data, demo_target
        else:
            del samples
        del logits_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_linear:
        train_task.batch_size = orig_bs

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── 7. Aggregate ──────────────────────────────────────────────────
    metric_by_c = {}
    for c in scale_factors:
        stacked = torch.cat(accum[c], dim=0)          # (N, n_eval)
        metric_by_c[c] = stacked.mean(dim=0).numpy()  # (n_eval,)

    positions_arr = np.array(eval_positions_label)
    metric_name = "MSE" if is_linear else "KL"

    # ── 8. Plot ───────────────────────────────────────────────────────
    fw, fh = figsize
    norm = mcolors.TwoSlopeNorm(vmin=min(scale_factors), vcenter=1.0,
                                vmax=max(scale_factors))
    cmap = cm.coolwarm

    # Figure 1: per-position output change
    fig1, ax1 = plt.subplots(figsize=(fw, fh))
    for c in scale_factors:
        col = cmap(norm(c))
        lw = 2.5 if c == 1.0 else 1.4
        ls = "--" if c == 1.0 else "-"
        ax1.plot(positions_arr, metric_by_c[c], color=col, lw=lw, ls=ls,
                 label=f"c={c:.2g}")
    ax1.set_xlabel("Position $t$", fontsize=13)
    ax1.set_ylabel(
        rf"${metric_name}(\mathrm{{scaled}} \| \mathrm{{baseline}})$",
        fontsize=13,
    )
    ax1.set_title(
        f"{task_name} — layer {layer} — per-position output change",
        fontsize=14,
    )
    ax1.legend(fontsize=12, ncol=2, loc="best")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # Figure 2: mean metric vs c
    n_late = max(1, int(n_eval * late_fraction))
    late_idx = slice(n_eval - n_late, n_eval)
    mean_late = [metric_by_c[c][late_idx].mean() for c in scale_factors]
    mean_all = [metric_by_c[c].mean() for c in scale_factors]

    fig2, ax2 = plt.subplots(figsize=(fw, fh))
    ax2.plot(scale_factors, mean_all, "o-", color="#555555", lw=2, ms=6,
             label="all positions")
    ax2.plot(scale_factors, mean_late, "s-", color="#D32F2F", lw=2, ms=6,
             label=f"late {n_late} positions")
    ax2.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.5)
    ax2.set_xlabel("Scale factor $c$", fontsize=13)
    ax2.set_ylabel(
        rf"Mean ${metric_name}(\mathrm{{scaled}} \| \mathrm{{baseline}})$",
        fontsize=13,
    )
    ax2.set_title(
        f"{task_name} — layer {layer} — output sensitivity to scaling",
        fontsize=14,
    )
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return {
        "scale_factors": scale_factors,
        "metric_by_c": metric_by_c,
        "metric_name": metric_name,
        "eval_positions": positions_arr,
        "late_fraction": late_fraction,
        "layer": layer,
        "rank": rank,
        "figs": (fig1, fig2),
        "axes": (ax1, ax2),
    }


# ──────────────────────────────────────────────────────────────────────
#  Residual-removal intervention (reference task + token subspace)
# ──────────────────────────────────────────────────────────────────────

def intervene_residual_removal(
    task_name: str,
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 2000,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    estimation_positions: Optional[list] = None,
    estimation_B: int = 128,
    per_position_mean: bool = True,
    per_position_tokens: bool = False,
    per_position_covariate: bool = False,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    task_batch_size: int = 8,
    marker_every: int = 5,
    figsize: tuple = (8, 4),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    r"""Remove the residual from the additive decomposition and measure output change.

    Tests whether the additive model
    :math:`h_t \approx \mu_t + \theta_k + \nu_{s_t}` captures everything
    functionally important **using reference (late-position) task vectors**.

    At each evaluation position *p*, the hidden state is replaced with:

    .. math::

        \hat{h}_p = \mu(p) + P(p)\,(h_p - \mu(p))

    where :math:`P(p)` is the orthogonal projector onto
    ``span(θ_1^ref, ..., θ_K^ref, ν_1(p), ..., ν_V(p))`` when
    ``per_position_tokens=True``, or a single fixed projector onto
    ``span(θ_1^ref, ..., θ_K^ref, ν_1^ref, ..., ν_V^ref)`` when
    ``per_position_tokens=False``.  Task vectors are always the
    reference ones estimated from late positions.

    For the **linear** task the projector spans task vectors *and*
    covariate directions from the OLS slope ``B``.  When
    ``per_position_covariate=False`` (default), a single ``B`` is pooled
    across estimation positions; when ``True``, a separate ``B(p)`` is
    fitted at every evaluation position to form per-position projectors.

    **Metric**: KL (coin/latent) or MSE (linear).

    Parameters
    ----------
    task_name : str
        ``"coin"``, ``"latent"``, or ``"linear"``.
    layer : int
        Layer at which to intervene.
    estimation_positions : list, optional
        Positions to pool for reference task/token vectors.  ``None`` ->
        last 30 positions (coin/latent) or last 10 positions (linear).
    estimation_B : int
        Batch size for hidden-state extraction (task vector estimation).
    task_batch_size : int
        Samples per (task, token) cell for token-vector estimation
        (coin/latent only).
    per_position_mean : bool
        If True, use position-specific grand mean when centering.
    per_position_tokens : bool
        If True, token vectors vary by position (combined with fixed
        reference task vectors to form per-position projectors).
        Ignored for ``"linear"`` (no discrete tokens).
    per_position_covariate : bool
        If True, fit a separate covariate slope ``B(p)`` at each
        evaluation position and build per-position projectors.
        Only affects ``"linear"``; ignored for coin/latent.
    marker_every : int
        Place a marker every *n* positions on the plot lines.

    Returns
    -------
    dict with keys: layer, eval_positions, metric_per_pos, metric_means, ...
    """
    import gc
    import matplotlib.pyplot as plt
    from icl.utils.separability import (
        estimate_task_vectors_by_averaging,
        per_position_token_vectors_balanced,
    )

    assert task_name in ("coin", "latent", "linear"), (
        f"residual removal supports coin/latent/linear, got {task_name!r}"
    )
    is_linear = task_name == "linear"

    # ── Load model & config ──────────────────────────────────────────
    if is_linear:
        from icl.linear.analysis.interventions._helpers import (
            _load_and_prepare_model, _extract_hiddens_for_pool,
        )
        from icl.linear.analysis._helpers import _task_positions
        from icl.linear.linear_ood_analysis import (
            _setup_eval_task,
        )

        model, train_task, config, device = _load_and_prepare_model(
            exp_name, step=step,
        )
        if step is None:
            step = config.training.total_steps
        n_points = int(config.task.n_points)
        K_major = int(train_task.n_tasks)
        D_model = int(config.model.n_embd)
        pad_mode = getattr(model, "pad", "mapsto")
        task_pos = _task_positions(pad_mode, n_points, device)
        T_max = n_points

        if estimation_positions is None:
            estimation_positions = list(range(max(0, T_max - 10), T_max))
        if eval_positions is None:
            eval_positions = list(range(1, T_max))
    else:
        _, sampler, config = nu.load_everything(task_name, exp_name)
        if step is None:
            step = config.training.num_epochs
        model, _ = nu.load_checkpoint(
            config, step=step, exp_name=exp_name, return_actual_step=True,
        )
        model.eval().to(config.device)
        device = config.device

        K_major = sampler.n_major_tasks
        seq_len = sampler.seq_len
        T_max = seq_len - 1

        if estimation_positions is None:
            estimation_positions = list(range(max(0, T_max - 30), T_max))
        if eval_positions is None:
            eval_positions = list(range(1, T_max))

    # ── Extract hidden states (natural sampling) ─────────────────────
    if is_linear:
        major_pool = train_task.task_pool.squeeze(-1).to(device)[:K_major]
        eval_task_est = _setup_eval_task(config, major_pool, estimation_B, device)
        eval_task_est.batch_size = estimation_B
        demo_data_est = eval_task_est.sample_data(step=step).to(device)
        hiddens_layer, _ = _extract_hiddens_for_pool(
            model, eval_task_est, demo_data_est,
            step=step, layer=layer, task_pos=task_pos, D=D_model,
            n_tasks=K_major, chunk=8,
            post_layernorm=post_layernorm,
            extraction_point=extraction_point,
        )
        hiddens_layer = hiddens_layer.float()  # (K, T, B_est, D)
        demo_data_est = demo_data_est.cpu().float()  # (B_est, T, D_x)
        del eval_task_est
    else:
        hiddens_all, _ = _get_hiddens_at_real_positions(
            task_name=task_name, exp_name=exp_name,
            n_minor=0, n_ood=0, B=estimation_B, step=step,
            post_layernorm=post_layernorm,
            extraction_point=extraction_point,
        )
        hiddens_layer = hiddens_all[layer].float()  # (K, T, B_est, D)
    K, T, _B_est, D = hiddens_layer.shape

    # ── Reference task vectors (natural, late positions) ──────────────
    ref_task_vecs, _ = estimate_task_vectors_by_averaging(
        hiddens_layer, estimation_positions,
    )  # (K, D), grand_mean (D,)

    # ── Per-position grand means (natural) ───────────────────────────
    if per_position_mean:
        grand_means = hiddens_layer.mean(dim=(0, 2))   # (T, D)
    else:
        grand_means = hiddens_layer.mean(dim=(0, 1, 2)).unsqueeze(0).expand(T, D)
    grand_means = grand_means.cpu().float()

    # ── Token vectors (balanced) — coin/latent only ────────────────
    V = 0
    tok_vecs_pos = None
    ref_token_vecs = None

    if is_linear:
        per_position_tokens = False
        D_x = demo_data_est.shape[-1]
        ref_covariate_vecs = None

        if per_position_covariate:
            pass  # B(p) fitted per position in projector-building section
        else:
            # Estimate a single covariate slope B (no task-covariate interaction):
            #   h_{k,t,b} ≈ μ_t + θ_k + B x_{t,b}
            # Pool residuals across ALL tasks and estimation positions.
            est_pos_t = estimation_positions
            X_one = (demo_data_est[:, est_pos_t, :]           # (B_est, n_est, D_x)
                     .permute(1, 0, 2)                         # (n_est, B_est, D_x)
                     .reshape(-1, D_x))                        # (n_est*B_est, D_x)
            all_residuals = []
            for k in range(K):
                h_k = hiddens_layer[k, est_pos_t, :, :]       # (n_est, B_est, D)
                mu_est = grand_means[est_pos_t]                # (n_est, D)
                r_k = (h_k
                       - mu_est.unsqueeze(1)
                       - ref_task_vecs[k].unsqueeze(0).unsqueeze(0))
                all_residuals.append(r_k.reshape(-1, D))       # (n_est*B_est, D)
            R_pool = torch.cat(all_residuals, dim=0)           # (K*n_est*B_est, D)
            X_pool = X_one.repeat(K, 1)                        # (K*n_est*B_est, D_x)
            B_ref = torch.linalg.lstsq(X_pool, R_pool).solution  # (D_x, D)
            ref_covariate_vecs = B_ref.float()                 # (D_x, D)
            del demo_data_est
    else:
        if task_name == "coin":
            from icl.coin.analysis._helpers import (
                get_token_conditioned_hiddens_coin,
            )
            tc_hiddens_all, _ = get_token_conditioned_hiddens_coin(
                exp_name,
                layers=[layer],
                batch_size=estimation_B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
        else:
            from icl.latent_markov.analysis.variance import (
                get_token_conditioned_hiddens,
            )
            tc_hiddens_all, _ = get_token_conditioned_hiddens(
                exp_name,
                layers=[layer],
                batch_size=estimation_B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )

        cell_h = tc_hiddens_all[0].float()  # (T, V, K, B, D)
        V = cell_h.shape[1]

        if per_position_tokens:
            tok_vecs_pos = per_position_token_vectors_balanced(
                cell_h, per_position_mean=per_position_mean,
            ).cpu().float()
        else:
            cell_means = cell_h.mean(dim=3)                            # (T, V, K, D)
            est_cell_means = cell_means[estimation_positions]          # (T_est, V, K, D)
            ref_token_means = est_cell_means.mean(dim=(0, 2))          # (V, D)
            ref_token_grand = ref_token_means.mean(dim=0)              # (D,)
            ref_token_vecs = (ref_token_means - ref_token_grand.unsqueeze(0)).cpu().float()

        del cell_h, tc_hiddens_all

    if not is_linear:
        del hiddens_all
    _keep_hiddens = is_linear and per_position_covariate
    if not _keep_hiddens:
        del hiddens_layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Build projector(s) ───────────────────────────────────────────
    ref_task_vecs_f = ref_task_vecs.cpu().float()       # (K, D)
    eps = 1e-6
    use_per_pos_proj = per_position_tokens or (is_linear and per_position_covariate)

    def _build_projector(vecs):
        """SVD-based orthogonal projector from a (N, D) matrix."""
        _, S, Vt = torch.linalg.svd(vecs, full_matrices=False)
        r = int((S > eps * S[0].clamp_min(eps)).sum().item())
        if r == 0:
            return torch.zeros(D, D), 0
        return (Vt[:r].T @ Vt[:r]), r

    valid_positions = [p for p in eval_positions if p < T]

    if is_linear and per_position_covariate:
        P_by_pos = {}
        ranks_by_pos = {}
        hiddens_f = hiddens_layer.cpu().float()          # (K, T, B_est, D)
        demo_data_f = demo_data_est.cpu().float()        # (B_est, T, D_x)
        for p in valid_positions:
            x_p = demo_data_f[:, p, :]                   # (B_est, D_x)
            X_p = x_p.repeat(K, 1)                       # (K*B_est, D_x)
            all_r = []
            for k in range(K):
                r_k = (hiddens_f[k, p, :, :]
                       - grand_means[p].unsqueeze(0)
                       - ref_task_vecs_f[k].unsqueeze(0))
                all_r.append(r_k)                        # (B_est, D)
            R_p = torch.cat(all_r, dim=0)                # (K*B_est, D)
            B_p = torch.linalg.lstsq(X_p, R_p).solution  # (D_x, D)
            all_vecs = torch.cat(
                [ref_task_vecs_f, B_p.float()], dim=0,
            )                                            # (K + D_x, D)
            P_by_pos[p], ranks_by_pos[p] = _build_projector(all_vecs)
        del hiddens_f, demo_data_f, hiddens_layer, demo_data_est
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    elif per_position_tokens:
        P_by_pos = {}
        ranks_by_pos = {}
        for p in valid_positions:
            all_vecs = torch.cat(
                [ref_task_vecs_f, tok_vecs_pos[:, p, :]], dim=0,
            )
            P_by_pos[p], ranks_by_pos[p] = _build_projector(all_vecs)
    elif is_linear:
        all_linear_vecs = torch.cat(
            [ref_task_vecs_f, ref_covariate_vecs], dim=0,
        )  # (K + D_x, D)
        P_ref, rank = _build_projector(all_linear_vecs)
    else:
        all_ref_vecs = torch.cat([ref_task_vecs_f, ref_token_vecs], dim=0)
        P_ref, rank = _build_projector(all_ref_vecs)

    if verbose:
        v_str = f", V={V}" if not is_linear else f", D_x={D_x}"
        print(f"[residual removal] layer={layer}, K={K}{v_str}, D={D}")
        if use_per_pos_proj:
            ranks_list = [ranks_by_pos[p] for p in valid_positions]
            label = ("per-pos covariate (OLS)" if per_position_covariate
                     else "per-position token vectors")
            print(f"  {label}: rank range "
                  f"[{min(ranks_list)}, {max(ranks_list)}]")
        else:
            mode = "task + covariate (OLS)" if is_linear else "task + token"
            print(f"  reference subspace rank = {rank} ({mode})")
        print(f"  estimation_positions = "
              f"{estimation_positions[:3]}..{estimation_positions[-1]}")

    # ── Metric helpers ──────────────────────────────────────────────
    def _kl(logits_a, logits_b):
        """KL(softmax(a) || softmax(b))."""
        lp_a = torch.log_softmax(logits_a, dim=-1)
        lp_b = torch.log_softmax(logits_b, dim=-1)
        return (lp_a.exp() * (lp_a - lp_b)).sum(-1)

    metric_name = "MSE" if is_linear else "KL"

    # ── Intervention loop ────────────────────────────────────────────
    fwd_chunk = B
    n_done = 0
    metric_accum = {p: [] for p in valid_positions}
    bi = 0

    if not use_per_pos_proj:
        P_ref_dev = P_ref.to(device)

    if is_linear:
        orig_bs = int(train_task.batch_size)
        train_task.batch_size = B
        task_pos_list = task_pos.cpu().tolist()

    while n_done < n_samples:
        # ── Generate data ────────────────────────────────────────────
        if is_linear:
            demo_data, _, demo_target = train_task.sample_batch(
                step=bi + 99999, is_eval=False,
            )
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)
            cur_B = demo_data.shape[0]
        else:
            gen = sampler.generate(
                mode="major", task=None, num_samples=B, epochs=1,
            )
            samples = gen[0] if isinstance(gen, (tuple, list)) else gen
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)
            cur_B = samples.shape[0]

        with torch.no_grad():
            if is_linear:
                preds_base = model(demo_data, demo_target)
            else:
                logits_base = model(samples)

        for p in valid_positions:
            if is_linear:
                seq_pos = task_pos_list[p]
            else:
                if p + 1 >= samples.shape[1]:
                    continue
                seq_pos = p

            _mu_p = grand_means[p].to(device)
            if use_per_pos_proj:
                _P_p = P_by_pos[p].to(device)
            else:
                _P_p = P_ref_dev

            def _hook_fn(_mu=_mu_p, _P=_P_p, _sp=seq_pos):
                def _hook(mod, inp, out):
                    h = out if torch.is_tensor(out) else out[0]
                    h_new = h.clone()
                    h_centered = h_new[:, _sp, :] - _mu.unsqueeze(0)
                    h_new[:, _sp, :] = _mu.unsqueeze(0) + h_centered @ _P
                    return (h_new if torch.is_tensor(out)
                            else (h_new,) + out[1:])
                return _hook

            while True:
                try:
                    handle = _get_layer_module(model, layer, is_linear).register_forward_hook(
                        _hook_fn(),
                    )
                    try:
                        with torch.no_grad():
                            if is_linear:
                                preds_proj = model(demo_data, demo_target)
                            else:
                                logits_proj = model(samples)
                    finally:
                        handle.remove()

                    if is_linear:
                        mse_vals = (
                            (preds_base[:, p] - preds_proj[:, p]) ** 2
                        ).cpu()
                        metric_accum[p].append(mse_vals)
                        del preds_proj
                    else:
                        kl_vals = _kl(
                            logits_base[:, p], logits_proj[:, p],
                        ).cpu()
                        metric_accum[p].append(kl_vals)
                        del logits_proj
                    break
                except torch.cuda.OutOfMemoryError:
                    fwd_chunk = max(1, fwd_chunk // 2)
                    torch.cuda.empty_cache()

        n_done += cur_B
        bi += 1
        if is_linear:
            del demo_data, demo_target, preds_base
        else:
            del samples, logits_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_linear:
        train_task.batch_size = orig_bs

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Aggregate ────────────────────────────────────────────────────
    metric_per_pos = {}
    for p in valid_positions:
        if metric_accum[p]:
            metric_per_pos[p] = torch.cat(metric_accum[p]).numpy()

    positions_arr = np.array(sorted(metric_per_pos.keys()))
    m_means = np.array([metric_per_pos[p].mean() for p in positions_arr])
    m_medians = np.array([np.median(metric_per_pos[p]) for p in positions_arr])
    m_q75 = np.array([np.percentile(metric_per_pos[p], 75) for p in positions_arr])
    m_q90 = np.array([np.percentile(metric_per_pos[p], 90) for p in positions_arr])
    zeros = np.zeros_like(positions_arr, dtype=float)

    # ── Marker indices ───────────────────────────────────────────────
    me = max(1, marker_every)
    mark_idx = list(range(0, len(positions_arr), me))

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.fill_between(positions_arr, zeros, m_q90, alpha=0.12,
                     color="#1976D2", label="0–90 pctl")
    ax1.fill_between(positions_arr, zeros, m_q75, alpha=0.22,
                     color="#1976D2", label="0–75 pctl")
    ax1.plot(positions_arr, m_medians, "-o", color="#1976D2", lw=2,
             label="median", markevery=mark_idx, ms=5)
    ax1.plot(positions_arr, m_means, "--s", color="#D32F2F", lw=1.5,
             label="mean", markevery=mark_idx, ms=4)
    ax1.set_xlabel("Position $t$", fontsize=13)
    if is_linear:
        ax1.set_ylabel(
            r"$\mathrm{MSE}(\text{original},\, \text{projected})$",
            fontsize=13,
        )
    else:
        ax1.set_ylabel(
            r"$\mathrm{KL}(\text{original} \| \text{projected})$",
            fontsize=13,
        )

    if is_linear and per_position_covariate:
        tok_mode = "per-pos cov"
    elif is_linear:
        tok_mode = "task + cov"
    elif per_position_tokens:
        tok_mode = "per-pos tokens"
    else:
        tok_mode = "ref tokens"

    if use_per_pos_proj:
        ranks_list = [ranks_by_pos[p] for p in valid_positions]
        rank_str = f"rank {min(ranks_list)}-{max(ranks_list)}"
    else:
        rank_str = f"rank {rank}"
    ax1.set_title(
        f"{task_name} -- layer {layer} -- residual removal "
        f"({tok_mode}, {rank_str})",
        fontsize=13,
    )
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    overall_mean = m_means.mean()
    print(f"\n{'=' * 60}")
    print(f"Residual Removal  ({task_name}, layer {layer})")
    print(f"{'=' * 60}")
    if is_linear:
        cov_label = "per-position" if per_position_covariate else "pooled"
        print(f"  Subspace: task + covariate (OLS {cov_label}, D_x={D_x})")
    else:
        print(f"  Token vectors: {'per-position' if per_position_tokens else 'reference (pooled)'}")
    if use_per_pos_proj:
        print(f"  Subspace rank range: [{min(ranks_list)}, {max(ranks_list)}]")
    else:
        print(f"  Reference subspace rank = {rank}")
    print(f"  Overall mean {metric_name}(original, projected) = {overall_mean:.6f}")
    print(f"  Per-position mean {metric_name} range: "
          f"[{m_means.min():.6f}, {m_means.max():.6f}]")
    print()

    result = {
        "layer": layer,
        "task_name": task_name,
        "eval_positions": positions_arr,
        "metric_per_pos": metric_per_pos,
        "metric_means": m_means,
        "metric_medians": m_medians,
        "metric_overall_mean": overall_mean,
        "metric_name": metric_name,
        "per_position_tokens": per_position_tokens,
        "per_position_covariate": per_position_covariate,
        "fig": fig,
        "ax": ax1,
    }
    if use_per_pos_proj:
        result["subspace_ranks"] = np.array(
            [ranks_by_pos.get(p, 0) for p in positions_arr]
        )
    else:
        result["subspace_rank"] = rank
    return result
