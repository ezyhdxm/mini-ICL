#!/usr/bin/env python3
"""Generate the combined Dyck figure (E4):
  Left  — Residual variance ratio across Dyck positions (layers 0–5).
  Right — 2D projection of final-layer hidden states at prefix length l=7.

Usage (from project root):
    uv run python scripts/fig_dyck_combined.py
    uv run python scripts/fig_dyck_combined.py --k -1 --probe-k 10 --layer 5 --target-length 7
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "dyck_combined.png"
SAVE_PATH_PROJECTION = PROJECT_ROOT / "paper_figs" / "dyck_projection.png"


def _check_experiment_exists(task_name: str, exp_name: str) -> None:
    """Raise a clear error if the experiment directory is missing."""
    config_path = PROJECT_ROOT / "results" / task_name / exp_name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment not found: {config_path}\n"
            f"The Dyck model for k={exp_name!r} has not been trained yet (or was "
            "trained with different hyper-parameters).\n"
            "Run training first:  uv run python scripts/run_dyck.py"
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Monkey-patch the missing get_dyck_sampler stub ───────────────────────────
# variance.py and probes.py both declare `get_dyck_sampler = None` as a
# placeholder that was never wired up.  We provide a working implementation
# here so all downstream calls work without modifying the library files.

def _patch_get_dyck_sampler():
    import icl.utils.notebook_utils as _nu
    import icl.dyck.analysis.variance as _v_mod
    import icl.dyck.analysis.probes as _p_mod

    def _get_dyck_sampler(exp_name, n_minor, n_ood):
        _, sampler, _ = _nu.load_everything("dyck", exp_name)
        return sampler, n_minor

    _v_mod.get_dyck_sampler = _get_dyck_sampler
    _p_mod.get_dyck_sampler = _get_dyck_sampler


# ── Left panel: variance ratio R² ────────────────────────────────────────────

def _draw_variance_r2(ax, exp_name, *, prefix_k=3, batch_size=2048,
                      n_masks=30, layers=None):
    """Draw residual variance ratio curves onto *ax*."""
    from icl.dyck.analysis.variance import compute_p1_variance_dyck, _compute_dyck_r2
    import icl.utils.notebook_utils as nu
    from icl.utils.separability._task_vector_r2 import _layer_style
    from matplotlib.ticker import MaxNLocator

    if layers is None:
        layers = list(range(6))

    results = compute_p1_variance_dyck(
        exp_name=exp_name,
        layers=list(layers),
        B=batch_size,
        n_masks=n_masks,
        n_minor=0,
        n_ood=0,
        verbose=False,
    )
    _, sampler, _ = nu.load_everything("dyck", exp_name)
    positions = results["positions"]
    r2_dict = _compute_dyck_r2(results, prefix_k, sampler)

    for li in results["layers"]:
        vals = [1.0 - v for v in r2_dict[li]]
        ax.plot(positions, vals, label=str(li),
                **_layer_style(li, len(positions)))

    ax.set_xlabel("Dyck position")
    ax.set_ylabel("Residual variance ratio")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Layer", fontsize=9, title_fontsize=10,
              framealpha=0.9, loc="upper right",
              borderaxespad=0.3, handlelength=1.5)
    ax.grid(True, alpha=0.25, linewidth=0.5)


# ── Right panel: 2D prefix scatter ───────────────────────────────────────────

def _draw_prefix_scatter(ax, probe, viz_data, probe_results, *,
                         target_length=7, grid_res=300,
                         color_by="depth", intra_depth_variation=0.3,
                         depth_cmap="viridis",
                         centroid_label_style="hex"):
    """Draw the 2D prefix scatter for *target_length* onto *ax*.

    ``centroid_label_style`` controls the text drawn at each cluster
    centroid: ``"hex"`` (compact hex encoding), ``"paren"`` (the actual
    Dyck prefix, e.g. ``"(()("``), or ``"none"`` to omit labels.
    """
    from icl.dyck.prefix_probe._visualize import (
        _hierarchical_colors, _depth_colors, _darken,
        _prefix_to_hex, _prefix_to_paren,
    )
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    data_by_len        = viz_data["data_by_len"]
    prefix_to_class    = viz_data["prefix_to_class"]
    n_classes_per_length = probe_results["n_classes_per_length"]
    device = viz_data.get("device") or "cpu"

    l = target_length
    class_to_prefix = {v: k for k, v in prefix_to_class[l].items()}
    n_cls = n_classes_per_length[l]
    prefixes = [class_to_prefix[c] for c in range(n_cls)]
    if color_by == "depth":
        color_map, depth_norm, depth_cmap_obj = _depth_colors(
            prefixes,
            cmap_name=depth_cmap,
            intra_depth_variation=intra_depth_variation,
        )
    elif color_by == "hierarchical":
        color_map = _hierarchical_colors(prefixes)
        depth_norm = depth_cmap_obj = None
    else:
        raise ValueError(
            f"color_by must be 'hierarchical' or 'depth', got {color_by!r}"
        )

    h_all = data_by_len[l]["hiddens"]
    y_all = data_by_len[l]["labels"]

    probe.eval()
    with torch.no_grad():
        z_all = probe.project(h_all.to(device)).cpu().numpy()

    pca = None
    if z_all.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_all = pca.fit_transform(z_all)

    y_np = y_all.numpy()

    # Decision-boundary background
    margin = 0.08
    x_min, x_max = z_all[:, 0].min(), z_all[:, 0].max()
    y_min, y_max = z_all[:, 1].min(), z_all[:, 1].max()
    x_pad = (x_max - x_min) * margin
    y_pad = (y_max - y_min) * margin
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, grid_res),
        np.linspace(y_min - y_pad, y_max + y_pad, grid_res),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_in = torch.tensor(
        pca.inverse_transform(grid_2d) if pca is not None else grid_2d,
        dtype=torch.float32,
    )
    with torch.no_grad():
        grid_pred = probe.classify(grid_in.to(device), l).argmax(-1).cpu().numpy()
    grid_pred = grid_pred.reshape(xx.shape)

    region_rgba = np.zeros((*xx.shape, 4))
    for c in range(n_cls):
        rgba = list(color_map.get(prefixes[c], (0.5, 0.5, 0.5))) + [1.0]
        region_rgba[grid_pred == c] = rgba[:4]
    region_rgba[..., 3] = 0.28
    ax.imshow(region_rgba,
              extent=[xx.min(), xx.max(), yy.min(), yy.max()],
              origin="lower", aspect="auto", interpolation="bilinear")

    if centroid_label_style == "hex":
        _label_fn = _prefix_to_hex
    elif centroid_label_style == "paren":
        _label_fn = _prefix_to_paren
    elif centroid_label_style == "none":
        _label_fn = None
    else:
        raise ValueError(
            "centroid_label_style must be 'hex', 'paren' or 'none', "
            f"got {centroid_label_style!r}"
        )

    centroids = []
    for c in range(n_cls):
        pts = z_all[y_np == c]
        if len(pts) == 0:
            continue
        color = color_map[prefixes[c]]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[color], s=12, alpha=0.6,
                   edgecolors="white", linewidths=0.3)
        if _label_fn is not None:
            centroids.append((pts[:, 0].mean(), pts[:, 1].mean(),
                              _label_fn(prefixes[c]), color))

    if centroids:
        txt_size = max(7, 10 - n_cls * 0.02)
        bbox_props = dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.7)
        font_kwargs = {}
        if centroid_label_style == "paren":
            font_kwargs["family"] = "monospace"
        for cx, cy, lbl, color in centroids:
            ax.annotate(lbl, (cx, cy),
                        fontsize=txt_size, fontweight="bold",
                        ha="center", va="center",
                        color=_darken(color), bbox=bbox_props,
                        **font_kwargs)

    ax.set_xlabel("Projection dim 1")
    ax.set_ylabel("Projection dim 2")

    if color_by == "depth":
        depths_present = sorted({sum(p) for p in prefixes})
        discrete_colors = [
            depth_cmap_obj(depth_norm(d)) for d in depths_present
        ]
        discrete_cmap = mcolors.ListedColormap(discrete_colors)
        if len(depths_present) == 1:
            d0 = depths_present[0]
            bounds = [d0 - 0.5, d0 + 0.5]
        else:
            bounds = [depths_present[0] - 0.5]
            for a, b in zip(depths_present[:-1], depths_present[1:]):
                bounds.append((a + b) / 2.0)
            bounds.append(depths_present[-1] + 0.5)
        discrete_norm = mcolors.BoundaryNorm(
            bounds, ncolors=len(discrete_colors)
        )
        sm = mpl.cm.ScalarMappable(norm=discrete_norm, cmap=discrete_cmap)
        sm.set_array([])
        cbar = ax.figure.colorbar(
            sm, ax=ax,
            ticks=depths_present,
            boundaries=bounds,
            spacing="uniform",
            pad=0.02,
            drawedges=True,
        )
        cbar.set_label("number of unmatched '('", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_linewidth(0.5)
        cbar.dividers.set_color("white")
        cbar.dividers.set_linewidth(1.0)


# ── CLI & main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate combined Dyck (E4) figure.")
    p.add_argument("--k",             type=int, default=-1,
                   help="k for variance R² panel (default: -1)")
    p.add_argument("--probe-k",       type=int, default=10,
                   help="k for prefix probe / scatter panel (default: 10)")
    p.add_argument("--prefix-k",      type=int, default=3,
                   help="Prefix length for variance R² plot (default: 3)")
    p.add_argument("--target-length", type=int, default=7,
                   help="Prefix length shown in scatter (default: 7)")
    p.add_argument("--layer",         type=int, default=5,
                   help="Transformer layer for prefix probe (default: 5)")
    p.add_argument("--layers",        type=int, nargs="+", default=list(range(6)),
                   help="Layers for variance R² (default: 0 1 2 3 4 5)")
    p.add_argument("--batch-size",    type=int, default=2048,
                   help="Batch size for variance R² (default: 2048)")
    p.add_argument("--n-masks",       type=int, default=30,
                   help="Number of masks for variance R² (default: 30)")
    p.add_argument("--probe-masks",   type=int, default=10,
                   help="Number of masks for prefix probe (default: 10)")
    p.add_argument("--probe-epochs",  type=int, default=2000,
                   help="Probe training epochs (default: 2000)")
    p.add_argument("--probe-batch",   type=int, default=32,
                   help="Probe data-collection batch size (default: 32)")
    return p.parse_args()


def main():
    args = parse_args()

    # Must patch before any icl.dyck.analysis imports that use get_dyck_sampler
    _patch_get_dyck_sampler()

    from icl.utils.unified_interface import get_exp_name
    from icl.utils.plot_config import apply_paper_style
    from icl.dyck.prefix_probe import run_prefix_probe

    apply_paper_style()

    exp_name       = get_exp_name("dyck", args.k)
    probe_exp_name = get_exp_name("dyck", args.probe_k)
    log.info(f"Variance R² experiment : {exp_name}  (k={args.k})")
    log.info(f"Prefix probe experiment: {probe_exp_name}  (k={args.probe_k})")

    # Verify both experiments exist before doing any computation.
    _check_experiment_exists("dyck", exp_name)
    _check_experiment_exists("dyck", probe_exp_name)

    t_total = time.time()

    # ── Variance R² (left panel data collected inside _draw_variance_r2) ──────
    log.info("Computing variance R² …")

    # ── Prefix probe (right panel) ────────────────────────────────────────────
    log.info("Training prefix probe …")
    t0 = time.time()
    probe, probe_results, viz_data = run_prefix_probe(
        k_value=args.probe_k,
        exp_name=probe_exp_name,  # explicit hash so probe uses the correct experiment
        layer_index=args.layer,
        n_masks=args.probe_masks,
        batch_size=args.probe_batch,
        max_prefix=args.target_length,
        proj_dim=2,
        num_epochs=args.probe_epochs,
        loss_threshold=6e-3,
        mlp_hidden=64,
        lr=3e-3,
        samples_per_class=512,
        refresh_every=2,
        curriculum_threshold=0.1,
        verbose_every=200,
    )
    log.info(f"Prefix probe done in {time.time() - t0:.1f}s")

    # ── Compose combined figure ───────────────────────────────────────────────
    log.info("Composing figure …")
    fig = plt.figure(figsize=(11, 3.2))
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        wspace=0.22,
        left=0.08, right=0.97,
        top=0.95, bottom=0.16,
    )
    ax_left  = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    log.info("Drawing variance R² …")
    t0 = time.time()
    _draw_variance_r2(
        ax_left, exp_name,
        prefix_k=args.prefix_k,
        batch_size=args.batch_size,
        n_masks=args.n_masks,
        layers=args.layers,
    )
    log.info(f"Variance R² drawn in {time.time() - t0:.1f}s")

    log.info("Drawing prefix scatter …")
    _draw_prefix_scatter(
        ax_right, probe, viz_data, probe_results,
        target_length=args.target_length,
    )

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH}")
    plt.close(fig)

    # ── Standalone projection figure (parenthesis centroid labels) ────────────
    log.info("Drawing standalone projection figure …")
    fig_proj, ax_proj = plt.subplots(figsize=(7.5, 6.5))
    _draw_prefix_scatter(
        ax_proj, probe, viz_data, probe_results,
        target_length=args.target_length,
        centroid_label_style="paren",
    )
    fig_proj.tight_layout()
    fig_proj.savefig(SAVE_PATH_PROJECTION, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH_PROJECTION}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig_proj)


if __name__ == "__main__":
    main()
