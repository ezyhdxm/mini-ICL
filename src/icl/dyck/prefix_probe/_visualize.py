"""Visualisation helpers for the Dyck prefix probe."""

import numpy as np
import torch

from icl.utils.device_utils import get_default_device


def _prefix_to_hex(prefix_tuple):
    """Compact hex label: +1 → 1, -1 → 0, right-padded to a multiple of 4.

    The leading step (always +1) is dropped since it carries no information.
    """
    bits = [1 if s == 1 else 0 for s in prefix_tuple[1:]]
    pad = (-len(bits)) % 4
    bits.extend([0] * pad)
    val = 0
    for b in bits:
        val = (val << 1) | b
    n_digits = len(bits) // 4
    return f"{val:0{n_digits}X}"


def _prefix_to_paren(prefix_tuple):
    """Render a Dyck prefix as parentheses: +1 → '(', -1 → ')'."""
    return "".join("(" if s == 1 else ")" for s in prefix_tuple)


def plot_accuracy_bar(active_lengths, val_accs_history, n_classes_per_length,
                      layer_index, proj_dim, best_val_accs=None,
                      best_epoch=None, save_path=None):
    """Bar chart: best validation accuracy vs prefix length."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if best_val_accs is not None:
        accs = [best_val_accs[l] * 100 for l in active_lengths]
    else:
        accs = [val_accs_history[l][-1][1] * 100 for l in active_lengths]
    chances = [100.0 / n_classes_per_length[l] for l in active_lengths]
    x = np.arange(len(active_lengths))

    bars = ax.bar(x, accs, 0.5, color="steelblue", label="Probe accuracy")
    ax.bar(x, chances, 0.5, alpha=0.3, color="gray", label="Chance level")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"l={l}\n({n_classes_per_length[l]} cls)" for l in active_lengths]
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Dyck prefix length")
    epoch_str = f", best epoch {best_epoch}" if best_epoch is not None else ""
    ax.set_title("")
    ax.legend()
    ax.set_ylim(0, 105)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{a:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_loss(train_losses, save_path=None):
    """Simple training-loss curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(train_losses, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_2d_scatter(
    probe,
    data_by_len,
    prefix_to_class,
    active_lengths,
    n_classes_per_length,
    *,
    layer_index,
    k_value,
    device=None,
    max_pts_per_class=2000,
    legend_max_classes=12,
    show_boundary=True,
    grid_res=500,
    save_dir=None,
    save_path=None,
):
    """Produce one figure per prefix length with hierarchical colouring.

    Prefixes sharing a longer common ancestor receive more similar hues
    (binary subdivision of the colour wheel at each tree level).

    Parameters
    ----------
    show_boundary : bool, default True
        If True, render the MLP decision boundary as a semi-transparent
        coloured background behind the scatter points.
    grid_res : int, default 500
        Resolution of the decision boundary grid (grid_res x grid_res).
    save_dir : str | None
        If given, each figure is saved as ``<save_dir>/scatter_l<l>.png``.
    save_path : str | None
        Legacy single-file path (ignored when *save_dir* is set).

    Returns
    -------
    figs : dict[int, Figure]
        One matplotlib figure per prefix length.
    """
    if device is None:
        device = get_default_device()
    import matplotlib.pyplot as plt
    import os

    class_to_prefix = {
        l: {v: k for k, v in prefix_to_class[l].items()}
        for l in active_lengths
    }

    probe.eval()
    figs = {}

    for l in active_lengths:
        h_all = data_by_len[l]["hiddens"]
        y_all = data_by_len[l]["labels"]

        with torch.no_grad():
            z_all = probe.project(h_all.to(device)).cpu().numpy()

        pca = None
        if z_all.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z_all = pca.fit_transform(z_all)

        n_cls = n_classes_per_length[l]
        y_np = y_all.numpy()

        prefixes = [class_to_prefix[l][c] for c in range(n_cls)]
        color_map = _hierarchical_colors(prefixes)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.tick_params(axis="both", labelsize=14)

        if show_boundary:
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
            if pca is not None:
                grid_proj = pca.inverse_transform(grid_2d)
            else:
                grid_proj = grid_2d
            grid_z = torch.tensor(grid_proj, dtype=torch.float32)
            with torch.no_grad():
                grid_logits = probe.classify(grid_z.to(device), l)
                grid_pred = grid_logits.argmax(-1).cpu().numpy()
            grid_pred = grid_pred.reshape(xx.shape)

            region_rgba = np.zeros((*xx.shape, 4))
            for c in range(n_cls):
                rgba = list(color_map.get(prefixes[c], (0.5, 0.5, 0.5)))
                if len(rgba) == 3:
                    rgba.append(1.0)
                mask = grid_pred == c
                region_rgba[mask] = rgba
            region_rgba[..., 3] = 0.28
            ax.imshow(
                region_rgba,
                extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                origin="lower", aspect="auto", interpolation="bilinear",
            )

        centroids = []
        for c in range(n_cls):
            mask_c = y_np == c
            prefix = prefixes[c]
            hex_lbl = _prefix_to_hex(prefix)
            paren_lbl = _prefix_to_paren(prefix)
            color = color_map[prefix]
            pts = z_all[mask_c]
            if len(pts) == 0:
                continue

            if len(pts) > max_pts_per_class:
                sub = np.random.choice(len(pts), max_pts_per_class,
                                       replace=False)
                pts = pts[sub]

            use_legend = n_cls <= legend_max_classes
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=[color], s=12, alpha=0.6,
                edgecolors="white", linewidths=0.3,
                label=paren_lbl if use_legend else None,
            )
            centroids.append((pts[:, 0].mean(), pts[:, 1].mean(),
                              hex_lbl, color))

        if n_cls > legend_max_classes:
            txt_size = max(7, 10 - n_cls * 0.02)
            bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                        edgecolor="none", alpha=0.7)
            for cx, cy, lbl, color in centroids:
                ax.annotate(
                    lbl, (cx, cy),
                    fontsize=txt_size,
                    fontweight="bold",
                    ha="center", va="center",
                    color=_darken(color),
                    bbox=bbox,
                )

        if pca is not None:
            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]:.0%} var)", fontsize=16)
            ax.set_ylabel(f"PC2 ({ev[1]:.0%} var)", fontsize=16)
        else:
            ax.set_xlabel("Projection dim 1", fontsize=16)
            ax.set_ylabel("Projection dim 2", fontsize=16)

        if n_cls <= legend_max_classes:
            ax.legend(fontsize=10, markerscale=2.5, loc="best",
                      framealpha=0.85, ncol=max(1, (n_cls + 4) // 5))

        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"scatter_l{l}.png"),
                        dpi=150, bbox_inches="tight")
        elif save_path:
            fig.savefig(save_path.replace(".png", f"_l{l}.png"),
                        dpi=150, bbox_inches="tight")

        figs[l] = fig

    return figs


def plot_3d_scatter(
    probe,
    data_by_len,
    prefix_to_class,
    active_lengths,
    n_classes_per_length,
    *,
    layer_index,
    k_value,
    device=None,
    max_pts_per_class=2000,
    legend_max_classes=12,
    save_dir=None,
    elev=25,
    azim=135,
):
    """Produce one 3D figure per prefix length with hierarchical colouring.

    When proj_dim > 3, PCA reduces to 3D. When proj_dim == 3, plots directly.

    Parameters
    ----------
    elev, azim : float
        Elevation and azimuth angles for the 3D view.
    """
    if device is None:
        device = get_default_device()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import os

    class_to_prefix = {
        l: {v: k for k, v in prefix_to_class[l].items()}
        for l in active_lengths
    }

    probe.eval()
    figs = {}

    for l in active_lengths:
        h_all = data_by_len[l]["hiddens"]
        y_all = data_by_len[l]["labels"]

        with torch.no_grad():
            z_all = probe.project(h_all.to(device)).cpu().numpy()

        pca = None
        if z_all.shape[1] > 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            z_all = pca.fit_transform(z_all)
        elif z_all.shape[1] < 3:
            continue

        n_cls = n_classes_per_length[l]
        y_np = y_all.numpy()

        prefixes = [class_to_prefix[l][c] for c in range(n_cls)]
        color_map = _hierarchical_colors(prefixes)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        centroids = []
        for c in range(n_cls):
            mask_c = y_np == c
            prefix = prefixes[c]
            hex_lbl = _prefix_to_hex(prefix)
            paren_lbl = _prefix_to_paren(prefix)
            color = color_map[prefix]
            pts = z_all[mask_c]
            if len(pts) == 0:
                continue

            if len(pts) > max_pts_per_class:
                sub = np.random.choice(len(pts), max_pts_per_class, replace=False)
                pts = pts[sub]

            use_legend = n_cls <= legend_max_classes
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=[color], s=8, alpha=0.5,
                label=paren_lbl if use_legend else None,
            )
            centroids.append((
                pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean(),
                hex_lbl, color,
            ))

        if n_cls > legend_max_classes:
            txt_size = max(5.0, 7.5 - n_cls * 0.02)
            for cx, cy, cz, lbl, color in centroids:
                ax.text(
                    cx, cy, cz, lbl,
                    fontsize=txt_size, fontweight="bold",
                    ha="center", va="center",
                    color=_darken(color),
                )

        if pca is not None:
            ev = pca.explained_variance_ratio_
            proj_label = "PCA of learned projection"
            ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
            ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
            ax.set_zlabel(f"PC3 ({ev[2]:.0%})")
        else:
            proj_label = "shared projection"
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")

        ax.set_title("", fontsize=14)
        ax.view_init(elev=elev, azim=azim)

        if n_cls <= legend_max_classes:
            ax.legend(fontsize=7, markerscale=2.5, loc="best",
                      framealpha=0.85, ncol=max(1, (n_cls + 4) // 5))

        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"scatter3d_l{l}.png"),
                        dpi=150, bbox_inches="tight")

        figs[l] = fig

    return figs


def _hierarchical_colors(prefixes, n_hue_levels=3):
    """Assign colours using prefix-tree structure in HSV space.

    The first *n_hue_levels* prefix steps determine the **hue** via binary
    subdivision of [0, 1].  Deeper steps cycle through three channels —
    hue perturbation, saturation, and value — so siblings sharing a long
    common prefix remain in the same colour family but are visually
    distinct across all three perceptual dimensions.
    """
    import matplotlib.colors as mcolors

    n_hue_levels = min(n_hue_levels, max(len(p) for p in prefixes))

    color_map = {}
    for prefix in prefixes:
        hue_lo, hue_hi = 0.0, 1.0
        for step in prefix[:n_hue_levels]:
            mid = (hue_lo + hue_hi) / 2.0
            if step == 1:
                hue_hi = mid
            else:
                hue_lo = mid
        hue_base = (hue_lo + hue_hi) / 2.0
        hue_band = (hue_hi - hue_lo) / 2.0

        hue_off_lo, hue_off_hi = -hue_band * 0.8, hue_band * 0.8
        sat_lo, sat_hi = 0.35, 1.0
        val_lo, val_hi = 0.40, 1.0

        for i, step in enumerate(prefix[n_hue_levels:]):
            channel = i % 3
            if channel == 0:
                mid = (hue_off_lo + hue_off_hi) / 2.0
                if step == 1:
                    hue_off_hi = mid
                else:
                    hue_off_lo = mid
            elif channel == 1:
                mid = (sat_lo + sat_hi) / 2.0
                if step == 1:
                    sat_hi = mid
                else:
                    sat_lo = mid
            else:
                mid = (val_lo + val_hi) / 2.0
                if step == 1:
                    val_hi = mid
                else:
                    val_lo = mid

        hue = (hue_base + (hue_off_lo + hue_off_hi) / 2.0) % 1.0
        sat = (sat_lo + sat_hi) / 2.0
        val = (val_lo + val_hi) / 2.0

        color_map[prefix] = mcolors.hsv_to_rgb([hue, sat, val])
    return color_map


def _darken(color, factor=0.4):
    """Return a darker version of *color* for readable text on white bg."""
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    return tuple(c * factor for c in rgb)
