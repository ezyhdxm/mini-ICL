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

    ax.set_xlabel("Dyck position", fontsize=12)
    ax.set_ylabel("Residual variance ratio", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=11)
    ax.legend(title="Layer", fontsize=11, title_fontsize=11,
              framealpha=0.9, loc="upper right",
              borderaxespad=0.3, handlelength=2.2)
    ax.grid(True, alpha=0.25, linewidth=0.5)


# ── Right panel: 2D prefix scatter ───────────────────────────────────────────

def _draw_prefix_scatter(ax, probe, viz_data, probe_results, *,
                         target_length=7, grid_res=300):
    """Draw the 2D prefix scatter for *target_length* onto *ax*."""
    from icl.dyck.prefix_probe._visualize import (
        _hierarchical_colors, _darken, _prefix_to_hex,
    )

    data_by_len        = viz_data["data_by_len"]
    prefix_to_class    = viz_data["prefix_to_class"]
    n_classes_per_length = probe_results["n_classes_per_length"]
    device = viz_data.get("device") or "cpu"

    l = target_length
    class_to_prefix = {v: k for k, v in prefix_to_class[l].items()}
    n_cls = n_classes_per_length[l]
    prefixes = [class_to_prefix[c] for c in range(n_cls)]
    color_map = _hierarchical_colors(prefixes)

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

    # Scatter points + centroid labels
    centroids = []
    for c in range(n_cls):
        pts = z_all[y_np == c]
        if len(pts) == 0:
            continue
        color = color_map[prefixes[c]]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[color], s=12, alpha=0.6,
                   edgecolors="white", linewidths=0.3)
        centroids.append((pts[:, 0].mean(), pts[:, 1].mean(),
                          _prefix_to_hex(prefixes[c]), color))

    txt_size = max(7, 10 - n_cls * 0.02)
    bbox_props = dict(boxstyle="round,pad=0.15",
                      facecolor="white", edgecolor="none", alpha=0.7)
    for cx, cy, lbl, color in centroids:
        ax.annotate(lbl, (cx, cy),
                    fontsize=txt_size, fontweight="bold",
                    ha="center", va="center",
                    color=_darken(color), bbox=bbox_props)

    ax.set_xlabel("Projection dim 1", fontsize=12)
    ax.set_ylabel("Projection dim 2", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)


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
    fig = plt.figure(figsize=(11, 4.5))
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        wspace=0.20,
        left=0.08, right=0.97,
        top=0.95, bottom=0.13,
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
    log.info(f"Saved → {SAVE_PATH}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
