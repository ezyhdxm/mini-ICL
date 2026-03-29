from icl.utils.unified_interface import _get_hiddens_at_real_positions

from typing import Optional
import numpy as np
import torch


# -----------------------------------------------------------------------
# Task-vector geometry across positions
# -----------------------------------------------------------------------

@torch.no_grad()
def plot_task_vector_geometry(
    task_name: str,
    exp_name: str,
    layer_index: int,
    reference_positions: Optional[list] = None,
    plot_positions: Optional[list] = None,
    B: int = 1024,
    step: Optional[int] = None,
    per_position_mean: bool = True,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    balanced: bool = False,
    task_batch_size: int = 8,
    task_colors: tuple = ("#0072B2", "#E69F00", "#009E73"),
    marker_every: int = 5,
    figsize: tuple = (7, 10),
    show: bool = True,
) -> dict:
    """Compare task vectors estimated at each position.

    Produces a 4-panel figure (5 panels for discrete tasks ``"coin"``/``"latent"``):

    1. **Norms** -- ||theta_k(t)|| vs position t for each task k.
    2. **Cosine with reference** -- cos(theta_k(t), theta_k^ref) vs t.
    3. **Inter-task cosine** -- cos(theta_i(t), theta_j(t)) for all pairs.
    4. **Subspace membership** -- cosine of theta_k(t) with reference subspace.
    5. **Task-token overlap** (discrete tasks only) -- cosine of theta_k(t)
       with the token subspace span({nu_a(t)}), verifying orthogonality.
       Token vectors are always estimated from a balanced (interventional)
       design regardless of the ``balanced`` flag.

    Parameters
    ----------
    task_name : str
        One of ``"coin"``, ``"linear"``, ``"latent"``.
    exp_name : str
        Experiment folder name.
    layer_index : int
        Which transformer layer to analyse.
    reference_positions : list[int], optional
        Positions used to estimate reference task vectors.  Defaults to
        the last 10 real-token positions.
    plot_positions : list[int], optional
        Which positions to show on the x-axis.  Defaults to all
        available positions.
    B : int
        Batch size per task (natural) or per cell (balanced).
    step : int, optional
        Checkpoint step (None = final).
    per_position_mean : bool
        If True, centre with a position-specific grand mean.
    balanced : bool
        If True, use an orthogonal (token-conditioned) design so that
        task vectors are uniformly averaged over tokens, eliminating
        token leakage.  Only supported for discrete-token tasks
        (``"coin"``, ``"latent"``).
    task_batch_size : int
        Number of tasks batched per forward pass (balanced mode only).
    task_colors : tuple of str
        Colours for each major task.
    marker_every : int
        Place a marker every this many positions (0 to disable).
    figsize : tuple
        Figure size ``(width, height)``.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    dict
        ``task_vecs_by_pos``, ``reference_vecs``, ``norms``,
        ``cos_with_ref``, ``inter_task_cos``, ``fig``, ``axes``.
    """
    import itertools
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from icl.utils.separability import (
        per_position_task_vectors,
        estimate_task_vectors_by_averaging,
        per_position_task_vectors_balanced,
        estimate_task_vectors_by_averaging_balanced,
        per_position_token_vectors_balanced,
    )

    if balanced:
        # -- Balanced (orthogonal) design: token-conditioned hiddens -----
        if task_name == "coin":
            from icl.coin.analysis._helpers import (
                get_token_conditioned_hiddens_coin,
            )
            all_hiddens_tc, _tok_info = get_token_conditioned_hiddens_coin(
                exp_name,
                layers=[layer_index],
                batch_size=B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
        elif task_name == "latent":
            from icl.latent_markov.analysis.variance import (
                get_token_conditioned_hiddens,
            )
            all_hiddens_tc, _tok_info = get_token_conditioned_hiddens(
                exp_name,
                layers=[layer_index],
                batch_size=B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
        else:
            raise ValueError(
                f"balanced=True is not supported for task_name='{task_name}' "
                "(requires discrete tokens)"
            )

        # all_hiddens_tc: (1, T, V, K, B, D) -> (T, V, K, B, D)
        hiddens_tc = all_hiddens_tc[0]
        T, V, K, _B, D = hiddens_tc.shape

        tv_pos, grand_means = per_position_task_vectors_balanced(
            hiddens_tc, per_position_mean=per_position_mean,
        )

        if reference_positions is None:
            reference_positions = list(range(max(0, T - 10), T))
        ref_vecs, _ = estimate_task_vectors_by_averaging_balanced(
            hiddens_tc, reference_positions,
        )

        tok_vecs_pos = per_position_token_vectors_balanced(
            hiddens_tc, per_position_mean=per_position_mean,
        )  # (V, T, D)
    else:
        # -- Natural sampling (original path) ----------------------------
        result = _get_hiddens_at_real_positions(
            task_name, exp_name, n_minor=0, n_ood=0, B=B, step=step,
            post_layernorm=post_layernorm,
            extraction_point=extraction_point,
        )
        if isinstance(result, tuple):
            all_hiddens = result[0]
        else:
            all_hiddens = result

        hiddens_layer = all_hiddens[layer_index]        # (K, T, B, D)
        K, T, _B, D = hiddens_layer.shape

        tv_pos, grand_means = per_position_task_vectors(
            hiddens_layer, per_position_mean=per_position_mean,
        )

        if reference_positions is None:
            reference_positions = list(range(max(0, T - 10), T))
        ref_vecs, _ = estimate_task_vectors_by_averaging(
            hiddens_layer, reference_positions,
        )

        # For discrete tasks, fetch token-conditioned data for token subspace
        tok_vecs_pos = None
        if task_name == "coin":
            from icl.coin.analysis._helpers import (
                get_token_conditioned_hiddens_coin,
            )
            _tc_hiddens, _ = get_token_conditioned_hiddens_coin(
                exp_name,
                layers=[layer_index],
                batch_size=B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
            tok_vecs_pos = per_position_token_vectors_balanced(
                _tc_hiddens[0], per_position_mean=per_position_mean,
            )
        elif task_name == "latent":
            from icl.latent_markov.analysis.variance import (
                get_token_conditioned_hiddens,
            )
            _tc_hiddens, _ = get_token_conditioned_hiddens(
                exp_name,
                layers=[layer_index],
                batch_size=B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
            tok_vecs_pos = per_position_token_vectors_balanced(
                _tc_hiddens[0], per_position_mean=per_position_mean,
            )

    # -- 4. metrics (all on CPU) -----------------------------------------
    tv_pos = tv_pos.cpu().float()
    ref_vecs = ref_vecs.cpu().float()
    grand_means = grand_means.cpu().float()

    eps = 1e-12
    norms = tv_pos.norm(dim=-1)                     # (K, T)

    ref_norm = ref_vecs.norm(dim=-1, keepdim=True).clamp_min(eps)  # (K, 1)
    pos_norm = norms.clamp_min(eps)                                # (K, T)
    cos_ref = (tv_pos * ref_vecs.unsqueeze(1)).sum(dim=-1) / (
        pos_norm * ref_norm
    )  # (K, T)

    pairs = list(itertools.combinations(range(K), 2))
    inter_cos = torch.zeros(len(pairs), T)
    for idx, (i, j) in enumerate(pairs):
        dot = (tv_pos[i] * tv_pos[j]).sum(dim=-1)
        inter_cos[idx] = dot / (norms[i].clamp_min(eps) * norms[j].clamp_min(eps))

    # Subspace membership: fraction of ||θ_k(t)||² in span(ref_vecs)
    rv = ref_vecs.float()                                    # (K, D)
    _, _S_rv, Vt_rv = torch.linalg.svd(rv, full_matrices=False)
    sub_rank = int((_S_rv > 1e-6 * _S_rv[0]).sum().item())
    P_ref = Vt_rv[:sub_rank].T @ Vt_rv[:sub_rank]           # (D, D)

    proj_norms = (tv_pos @ P_ref).pow(2).sum(dim=-1).sqrt()    # (K, T)
    subspace_cos = proj_norms / norms.clamp_min(eps)           # (K, T)

    # Task-token subspace overlap (discrete tasks: coin/latent)
    task_token_overlap = None
    if tok_vecs_pos is not None:
        tok_vp = tok_vecs_pos.cpu().float()              # (V, T, D)
        V_tok = tok_vp.shape[0]
        task_token_overlap = torch.zeros(K, T)
        for t_idx in range(T):
            tok_mat = tok_vp[:, t_idx, :]                # (V_tok, D)
            _, _S_tok, Vt_tok = torch.linalg.svd(tok_mat, full_matrices=False)
            tok_rank = int((_S_tok > 1e-6 * _S_tok[0].clamp_min(eps)).sum().item())
            if tok_rank == 0:
                continue
            P_tok = Vt_tok[:tok_rank].T @ Vt_tok[:tok_rank]  # (D, D)
            for k in range(K):
                v = tv_pos[k, t_idx]                     # (D,)
                v_norm = v.norm().clamp_min(eps)
                proj = P_tok @ v
                task_token_overlap[k, t_idx] = proj.norm() / v_norm

    if plot_positions is not None:
        pp = np.array(plot_positions)
        norms = norms[:, pp]
        cos_ref = cos_ref[:, pp]
        inter_cos = inter_cos[:, pp]
        subspace_cos = subspace_cos[:, pp]
        if task_token_overlap is not None:
            task_token_overlap = task_token_overlap[:, pp]
        ts = pp
    else:
        ts = np.arange(T)

    colors = list(task_colors[:K])
    task_markers = ["o", "s", "D", "^", "v", "P"]
    pair_markers = ["o", "s", "D"]
    me = max(marker_every, 1) if marker_every > 0 else len(ts) + 1

    # -- 5. plot ---------------------------------------------------------
    n_panels = 5 if task_token_overlap is not None else 4
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(figsize[0], figsize[1] * n_panels / 3),
    )

    # Panel 1: norms
    ax = axes[0]
    for k in range(K):
        ax.plot(ts, norms[k].numpy(), color=colors[k], lw=2,
                marker=task_markers[k % len(task_markers)],
                markevery=me, markersize=5,
                label=rf"$\theta_{{{k+1}}}$")
    ax.set_xlabel("Position $t$", fontsize=12)
    ax.set_ylabel(r"$\|\theta_k(t)\|$", fontsize=12)
    ax.set_title("Task vector norm", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: cosine with reference
    ax = axes[1]
    for k in range(K):
        ax.plot(ts, cos_ref[k].numpy(), color=colors[k], lw=2,
                marker=task_markers[k % len(task_markers)],
                markevery=me, markersize=5,
                label=rf"$\theta_{{{k+1}}}$")
    ax.set_xlabel("Position $t$", fontsize=12)
    ax.set_ylabel(r"$\cos(\theta_k(t),\;\theta_k^{\mathrm{ref}})$", fontsize=12)
    ax.set_title("Directional stability", fontsize=13)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: inter-task cosine
    ax = axes[2]
    pair_styles = ["-", "--", ":"]
    for idx, (i, j) in enumerate(pairs):
        blend = tuple(
            0.5 * c1 + 0.5 * c2
            for c1, c2 in zip(
                mcolors.to_rgb(colors[i]),
                mcolors.to_rgb(colors[j]),
            )
        )
        ax.plot(
            ts, inter_cos[idx].numpy(), color=blend,
            lw=2, ls=pair_styles[idx % len(pair_styles)],
            marker=pair_markers[idx % len(pair_markers)],
            markevery=me, markersize=5,
            label=rf"$(\theta_{{{i+1}}},\theta_{{{j+1}}})$",
        )
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Position $t$", fontsize=12)
    ax.set_ylabel(r"$\cos(\theta_i(t),\;\theta_j(t))$", fontsize=12)
    ax.set_title("Inter-task cosine", fontsize=13)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 4: subspace membership (cosine with task subspace)
    ax = axes[3]
    for k in range(K):
        ax.plot(ts, subspace_cos[k].numpy(), color=colors[k], lw=2,
                marker=task_markers[k % len(task_markers)],
                markevery=me, markersize=5,
                label=rf"$\theta_{{{k+1}}}$")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Position $t$", fontsize=12)
    ax.set_ylabel(
        r"$\|P_{\mathrm{task}}\,\theta_k(t)\| \,/\, \|\theta_k(t)\|$",
        fontsize=12,
    )
    ax.set_title(
        r"Subspace membership (cosine with reference task subspace)",
        fontsize=13,
    )
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 5 (balanced only): task-token subspace overlap
    if task_token_overlap is not None:
        ax = axes[4]
        for k in range(K):
            ax.plot(ts, task_token_overlap[k].numpy(), color=colors[k], lw=2,
                    marker=task_markers[k % len(task_markers)],
                    markevery=me, markersize=5,
                    label=rf"$\theta_{{{k+1}}}$")
        ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Position $t$", fontsize=12)
        ax.set_ylabel(
            r"$\|P_{\mathrm{tok}}\,\theta_k(t)\| \,/\, \|\theta_k(t)\|$",
            fontsize=12,
        )
        ax.set_title(
            r"Task–token subspace overlap "
            r"(cosine of $\theta_k(t)$ with token subspace)",
            fontsize=13,
        )
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "task_vecs_by_pos": tv_pos,
        "reference_vecs": ref_vecs,
        "grand_means": grand_means,
        "norms": norms,
        "cos_with_ref": cos_ref,
        "inter_task_cos": inter_cos,
        "subspace_cos": subspace_cos,
        "task_token_overlap": task_token_overlap,
        "pairs": pairs,
        "fig": fig,
        "axes": axes,
    }
