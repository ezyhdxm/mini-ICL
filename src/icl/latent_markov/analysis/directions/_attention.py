"""
Attention-head probes (PTH / IH scores) for non-padded latent Markov sequences.
"""

import gc
import os
import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


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
