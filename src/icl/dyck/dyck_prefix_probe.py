"""
Dyck prefix history probe.

Tests whether the transformer's hidden representations encode the full Dyck
prefix history by training a shared 2D linear projection with separate MLP
classification heads per prefix length.

Architecture (jointly trained):
    h in R^d  -->  z = Wh + b in R^proj_dim           (shared projection)
              -->  MLP_l(z) in R^n_classes(l)          (per-length MLP head)
"""

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import icl.utils.notebook_utils as nu
from icl.utils.unified_interface import get_exp_name
from icl.dyck.dyck_utils import sample_binary_mask


# ──────────────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────────────

class PrefixProbe(nn.Module):
    """
    Shared linear projection + separate MLP head per prefix length.

    The linear projection is trained jointly across all prefix lengths.
    Each prefix length has its own full MLP classifier on top.
    """

    def __init__(self, d_model, proj_dim, n_classes_per_length, mlp_hidden=64):
        super().__init__()
        self.proj = nn.Linear(d_model, proj_dim)
        self.heads = nn.ModuleDict()
        for l, n_cls in n_classes_per_length.items():
            self.heads[str(l)] = nn.Sequential(
                nn.Linear(proj_dim, mlp_hidden),
                nn.SiLU(),
                nn.Linear(mlp_hidden, n_cls),
            )

    def project(self, h):
        return self.proj(h)

    def classify(self, z, prefix_len):
        return self.heads[str(prefix_len)](z)

    def forward(self, h, prefix_len):
        z = self.project(h)
        return self.classify(z, prefix_len)


# ──────────────────────────────────────────────────────────────────
#  Data collection
# ──────────────────────────────────────────────────────────────────

def _load_model_and_sampler(k_value, device=None):
    """Load the trained Dyck model, sampler, and config for a given k."""
    exp_name = get_exp_name("dyck", k_value)
    _, sampler_orig, config = nu.load_everything("dyck", exp_name)
    if device is not None:
        config.device = device
    step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval()
    model = model.to(config.device)
    sampler = copy.deepcopy(sampler_orig)
    return model, sampler, config, exp_name


def _get_task_paths(sampler):
    """Return a list of Dyck paths (as +1/-1 step tensors) for all tasks."""
    one_tok = int(sampler.one)
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    paths = []
    for t in range(n_tasks):
        raw = sampler.get_task_dyck_path(t)
        steps = torch.tensor([+1 if int(v) == one_tok else -1 for v in raw])
        paths.append(steps)
    return paths


@torch.no_grad()
def collect_hidden_states(
    model,
    sampler,
    config,
    *,
    layer_index,
    n_masks=8,
    batch_size=64,
    verbose=True,
):
    """
    Generate samples under several random Dyck masks and extract hidden
    states from the specified transformer layer.

    Returns
    -------
    all_data : list[dict]
        Each entry has keys 'hiddens' (B, seq_len, d), 'dyck_mask' (seq_len,),
        'task_id' (int).
    """
    device = config.device
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    all_data = []

    for mask_idx in range(n_masks):
        dyck_mask = sample_binary_mask(config).to(device)

        cache = {}

        def _hook_fn(module, inp, out_tensor):
            cache["h"] = out_tensor.detach()

        handle = model.layers[layer_index].attn_block.register_forward_hook(_hook_fn)
        try:
            for t in range(n_tasks):
                demo_data, _ = sampler.generate(
                    mode="testing", task=t, num_samples=batch_size,
                    dyck_mask=dyck_mask.clone(),
                )
                demo_data = demo_data.to(device, non_blocking=True)
                cache.clear()
                model(demo_data)

                all_data.append({
                    "hiddens": cache["h"].cpu(),
                    "dyck_mask": dyck_mask.cpu(),
                    "task_id": t,
                })
        finally:
            handle.remove()

        if verbose and ((mask_idx + 1) % 2 == 0 or mask_idx == 0):
            print(
                f"  Mask {mask_idx + 1}/{n_masks} done "
                f"({n_tasks} tasks x {batch_size} samples)"
            )

    if verbose:
        print(f"\nTotal entries: {len(all_data)} "
              f"(= {n_masks} masks x {n_tasks} tasks)")
    return all_data


# ──────────────────────────────────────────────────────────────────
#  Data organisation by prefix length
# ──────────────────────────────────────────────────────────────────

def build_prefix_datasets(all_data, task_paths, max_prefix, verbose=True):
    """
    Group hidden states by the number of Dyck tokens seen (prefix length)
    and label them by the actual prefix observed.

    Returns
    -------
    data_by_len : dict[int, dict]
        ``data_by_len[l]`` has keys ``'hiddens'`` (N, d) and ``'labels'`` (N,).
    prefix_to_class : dict[int, dict]
        ``prefix_to_class[l]`` maps prefix tuples to class indices.
    active_lengths : list[int]
        Prefix lengths with >= 2 distinct classes.
    """
    n_tasks = len(task_paths)

    prefix_to_class = {}
    for l in range(1, max_prefix + 1):
        seen = {}
        for t in range(n_tasks):
            pref = tuple(task_paths[t][:l].tolist())
            if pref not in seen:
                seen[pref] = len(seen)
        prefix_to_class[l] = seen

    if verbose:
        print("Prefix diversity per length:")
        for l in range(1, max_prefix + 1):
            print(f"  l={l}: {len(prefix_to_class[l])} distinct prefixes")

    data_by_len = defaultdict(lambda: {"hiddens": [], "labels": []})

    for entry in all_data:
        h = entry["hiddens"]
        mask = entry["dyck_mask"]
        t = entry["task_id"]
        cum_dyck = mask.cumsum(dim=0)

        for l in range(1, max_prefix + 1):
            if len(prefix_to_class[l]) < 2:
                continue

            pref = tuple(task_paths[t][:l].tolist())
            cls = prefix_to_class[l][pref]

            pos_mask = cum_dyck == l
            pos_mask[-1] = False
            if not pos_mask.any():
                continue

            positions = torch.nonzero(pos_mask, as_tuple=True)[0]
            h_at = h[:, positions, :]
            B, n_pos, d = h_at.shape

            data_by_len[l]["hiddens"].append(h_at.reshape(B * n_pos, d))
            data_by_len[l]["labels"].append(
                torch.full((B * n_pos,), cls, dtype=torch.long)
            )

    for l in data_by_len:
        data_by_len[l]["hiddens"] = torch.cat(data_by_len[l]["hiddens"], 0)
        data_by_len[l]["labels"] = torch.cat(data_by_len[l]["labels"], 0)

    active_lengths = sorted(
        l for l in data_by_len if len(prefix_to_class[l]) >= 2
    )

    if verbose:
        print("\nDataset sizes per prefix length:")
        for l in active_lengths:
            n_cls = len(prefix_to_class[l])
            n_pts = data_by_len[l]["hiddens"].shape[0]
            d = data_by_len[l]["hiddens"].shape[1]
            print(f"  l={l}: {n_pts} samples, {n_cls} classes, dim={d}")

    return dict(data_by_len), prefix_to_class, active_lengths


def _subsample_by_len(data_by_len, active_lengths, samples_per_class):
    """Cap each prefix length to ``samples_per_class * n_classes`` samples."""
    for l in active_lengths:
        h = data_by_len[l]["hiddens"]
        y = data_by_len[l]["labels"]
        n_cls = int(y.max().item()) + 1
        target = samples_per_class * n_cls
        if h.shape[0] > target:
            idx = torch.randperm(h.shape[0])[:target]
            data_by_len[l]["hiddens"] = h[idx]
            data_by_len[l]["labels"] = y[idx]
    return data_by_len


def _resample_from_pool(pool, active_lengths, samples_per_class):
    """Draw a fresh random subsample from *pool* without modifying it."""
    out = {}
    for l in active_lengths:
        h, y = pool[l]["hiddens"], pool[l]["labels"]
        n_cls = int(y.max().item()) + 1
        target = min(samples_per_class * n_cls, h.shape[0])
        idx = torch.randperm(h.shape[0])[:target]
        out[l] = {"hiddens": h[idx], "labels": y[idx]}
    return out


# ──────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────

def _evaluate_probe(probe, val_sets, active_lengths, device):
    """Compute per-length val accuracy, loss, and their means.

    Expects val_sets tensors already on *device*.
    """
    probe.eval()
    criterion = nn.CrossEntropyLoss()
    accs, losses = {}, {}
    with torch.no_grad():
        for l in active_lengths:
            h_v, y_v = val_sets[l]
            logits = probe(h_v, l)
            accs[l] = (logits.argmax(-1) == y_v).float().mean().item()
            losses[l] = criterion(logits, y_v).item()
    mean_acc = sum(accs.values()) / len(accs) if accs else 0.0
    mean_loss = sum(losses.values()) / len(losses) if losses else float("inf")
    return accs, mean_acc, losses, mean_loss


def train_prefix_probe(
    data_by_len,
    prefix_to_class,
    active_lengths,
    *,
    d_model,
    proj_dim=2,
    mlp_hidden=64,
    num_epochs=300,
    lr=1e-3,
    proj_lr_scale=0.1,
    mini_batch=512,
    val_data=None,
    val_frac=0.2,
    device="cuda",
    verbose_every=10,
    loss_threshold=0.01,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    refresh_every=3,
    refresh_fn=None,
    curriculum_threshold=None,
):
    """
    Build and train the ``PrefixProbe`` end-to-end.

    Training stops early once the mean validation cross-entropy loss
    across all active prefix lengths drops below *loss_threshold*.
    The model with the lowest mean val loss is always restored at the end.

    Parameters
    ----------
    val_data : dict or None
        Pre-split validation data (same format as *data_by_len*).
        If None, validation data is split from *data_by_len* using
        *val_frac*.
    loss_threshold : float, default 0.01
        Stop training when mean val loss <= this value.
        Early stopping is only allowed after all prefix lengths have
        been introduced (i.e. after curriculum completes).
    refresh_every : int, default 3
        Re-generate training data every this many epochs.
        Requires *refresh_fn* to be set; ignored otherwise.
    refresh_fn : callable or None
        ``refresh_fn()`` → ``data_by_len`` (same format as input).
        When provided, fresh training data is generated every
        *refresh_every* epochs.  Validation data is kept fixed.
    curriculum_threshold : float or None
        If not None, use curriculum learning: start training with
        only the two shortest prefix lengths and add the next length
        once the mean val loss on current lengths drops below this
        threshold.  Set to None to disable (train all lengths from
        the start).

    Returns
    -------
    probe : PrefixProbe  (eval mode, on *device*, best-epoch weights)
    results : dict
    """
    n_classes_per_length = {
        l: len(prefix_to_class[l]) for l in active_lengths
    }

    probe = PrefixProbe(d_model, proj_dim, n_classes_per_length,
                        mlp_hidden=mlp_hidden).to(device)
    proj_lr = lr * proj_lr_scale
    print(f"Active lengths  : {active_lengths}")
    print(f"Classes/length  : {n_classes_per_length}")
    print(f"Probe params    : {sum(p.numel() for p in probe.parameters()):,}")
    print(f"LR heads={lr}, proj={proj_lr} ({proj_lr_scale}x)")
    print(f"Loss threshold  : {loss_threshold}")
    if curriculum_threshold is not None:
        print(f"Curriculum      : start with 2 lengths, "
              f"promote when val_loss < {curriculum_threshold}")
    if refresh_fn is not None:
        print(f"Refresh training data every {refresh_every} epochs")

    # Pre-move data to GPU
    train_sets, val_sets = {}, {}
    if val_data is not None:
        for l in active_lengths:
            h_v, y_v = val_data[l]["hiddens"], val_data[l]["labels"]
            val_sets[l] = (h_v.to(device), y_v.to(device))
            h_t, y_t = data_by_len[l]["hiddens"], data_by_len[l]["labels"]
            train_sets[l] = (h_t.to(device), y_t.to(device))
    else:
        for l in active_lengths:
            h = data_by_len[l]["hiddens"]
            y = data_by_len[l]["labels"]
            n = h.shape[0]
            perm = torch.randperm(n)
            n_val = int(n * val_frac)
            val_sets[l] = (h[perm[:n_val]].to(device),
                           y[perm[:n_val]].to(device))
            train_sets[l] = (h[perm[n_val:]].to(device),
                             y[perm[n_val:]].to(device))

    optimizer = optim.AdamW([
        {"params": probe.proj.parameters(), "lr": proj_lr},
        {"params": probe.heads.parameters(), "lr": lr},
    ], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs_history = {l: [] for l in active_lengths}
    val_losses_history = {l: [] for l in active_lengths}

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_val_accs = {}

    if curriculum_threshold is not None:
        cur_n_active = min(2, len(active_lengths))
    else:
        cur_n_active = len(active_lengths)
    prev_n_active = 0
    all_lengths_active = (cur_n_active >= len(active_lengths))

    for epoch in range(num_epochs):
        train_lengths = active_lengths[:cur_n_active]

        if cur_n_active != prev_n_active:
            print(f"[Epoch {epoch+1}] Training on lengths: "
                  f"{list(train_lengths)}")
            prev_n_active = cur_n_active

        # Refresh training data periodically (only for active lengths)
        if (refresh_fn is not None
                and epoch > 0
                and epoch % refresh_every == 0):
            cur_max = max(train_lengths)
            fresh = refresh_fn(cur_max_prefix=cur_max)
            for l in train_lengths:
                if l in fresh:
                    train_sets[l] = (fresh[l]["hiddens"].to(device),
                                     fresh[l]["labels"].to(device))

        probe.train()
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for l in train_lengths:
            h_tr, y_tr = train_sets[l]
            n = h_tr.shape[0]
            perm = torch.randperm(n, device=device)
            for start in range(0, n, mini_batch):
                idx = perm[start : start + mini_batch]
                logits = probe(h_tr[idx], l)
                loss = criterion(logits, y_tr[idx])
                loss.backward()
                epoch_loss += loss.item()
                n_batches += 1

        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(probe.parameters(), max_grad_norm)
        optimizer.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            accs, mean_acc, vloss, mean_vloss = _evaluate_probe(
                probe, val_sets, train_lengths, device,
            )
            for l in train_lengths:
                val_accs_history[l].append((epoch + 1, accs[l]))
                val_losses_history[l].append((epoch + 1, vloss[l]))

            if mean_vloss < best_val_loss:
                best_val_loss = mean_vloss
                best_epoch = epoch + 1
                best_val_accs = dict(accs)
                best_state = copy.deepcopy(probe.state_dict())

            if (epoch + 1) % verbose_every == 0 or epoch == 0:
                s = ", ".join(
                    f"l={l}: {accs[l]*100:.1f}%" for l in train_lengths
                )
                worst = max(vloss.values()) if vloss else float("inf")
                print(f"Epoch {epoch+1:3d} | train_loss={avg_loss:.4f} "
                      f"| val_loss={mean_vloss:.4f} "
                      f"| worst={worst:.4f} | {s}"
                      f" | lengths={list(train_lengths)}")

            # Curriculum promotion: add next length when ALL active
            # lengths (including the newest) are below threshold.
            max_vloss = max(vloss.values()) if vloss else float("inf")
            if (curriculum_threshold is not None
                    and not all_lengths_active
                    and max_vloss <= curriculum_threshold):
                cur_n_active = min(cur_n_active + 1, len(active_lengths))
                all_lengths_active = (cur_n_active >= len(active_lengths))
                if all_lengths_active:
                    best_val_loss = float("inf")
                    best_state = None
                    print(f"[Epoch {epoch+1}] All lengths now active — "
                          f"resetting best-model tracking")

            # Early stopping only after all lengths are active
            if all_lengths_active and mean_vloss <= loss_threshold:
                print(f"\nStopping at epoch {epoch+1}: mean val loss "
                      f"{mean_vloss:.4f} <= threshold {loss_threshold}")
                break

    # Restore best model
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()

    # Final evaluation on ALL lengths with best model
    final_accs, _, final_vloss, final_mean_vloss = _evaluate_probe(
        probe, val_sets, active_lengths, device,
    )
    best_val_accs = final_accs

    s = ", ".join(
        f"l={l}: {final_accs[l]*100:.1f}%" for l in active_lengths
    )
    print(f"\nBest epoch {best_epoch} | mean val loss (training)={best_val_loss:.4f}"
          f" | mean val loss (all)={final_mean_vloss:.4f}")
    print(f"  Per-length acc: {s}")

    return probe, {
        "train_losses": train_losses,
        "val_losses_history": val_losses_history,
        "val_accs_history": val_accs_history,
        "train_sets": train_sets,
        "val_sets": val_sets,
        "n_classes_per_length": n_classes_per_length,
        "best_epoch": best_epoch,
        "best_val_accs": best_val_accs,
        "best_val_loss": best_val_loss,
    }


# ──────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ──────────────────────────────────────────────────────────────────

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
    ax.set_title(
        f"Prefix Classification Accuracy "
        f"(layer {layer_index}, {proj_dim}D projection{epoch_str})"
    )
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
    ax.set_title("Joint Probe Training Loss")
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
    device="cuda",
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

        # If projection dim > 2, reduce to 2D via PCA for visualization
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

        # Decision boundary background
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
            txt_size = max(5.5, 8 - n_cls * 0.02)
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

        proj_label = "PCA of learned projection" if pca is not None else "shared projection"
        ax.set_title(
            f"Prefix length l={l}  ({n_cls} classes)\n"
            f"layer {layer_index}, {proj_label}, k={k_value}",
            fontsize=11,
        )
        if pca is not None:
            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]:.0%} var)")
            ax.set_ylabel(f"PC2 ({ev[1]:.0%} var)")
        else:
            ax.set_xlabel("Projection dim 1")
            ax.set_ylabel("Projection dim 2")

        if n_cls <= legend_max_classes:
            ax.legend(fontsize=8, markerscale=2.5, loc="best",
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
    device="cuda",
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

        ax.set_title(
            f"Prefix length l={l}  ({n_cls} classes)\n"
            f"layer {layer_index}, {proj_label}, k={k_value}",
            fontsize=11,
        )
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


# ── Hierarchical colour helpers ──────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
#  Convenience: run everything
# ──────────────────────────────────────────────────────────────────

def run_prefix_probe(
    k_value=5,
    layer_index=5,
    n_masks=8,
    batch_size=64,
    max_prefix=7,
    proj_dim=2,
    mlp_hidden=64,
    num_epochs=300,
    lr=1e-3,
    proj_lr_scale=0.1,
    mini_batch=512,
    val_frac=0.2,
    loss_threshold=0.01,
    refresh_every=3,
    curriculum_threshold=None,
    samples_per_class=500,
    verbose_every=10,
    device=None,
):
    """
    End-to-end: load model, collect data, train probe, return everything
    needed for visualisation.

    Data is collected **once** from the frozen model into a large pool.
    Training-data "refreshes" simply re-sample from this pool (no model
    forward passes), making them essentially free.

    Parameters
    ----------
    n_masks : int
        Number of random Dyck masks for the one-time data collection.
    samples_per_class : int
        Target samples per class per prefix length for each training
        refresh.  Validation uses the same budget, drawn once upfront.
    refresh_every : int
        Re-sample training data from the pool every this many epochs.
    curriculum_threshold : float or None
        Promote to next prefix length when mean val loss drops below
        this value.  None disables curriculum learning.
    verbose_every : int
        Print training progress every this many epochs.

    Returns
    -------
    probe, results, viz_data
    """
    model, sampler, config, exp_name = _load_model_and_sampler(
        k_value, device=device,
    )
    if device is None:
        device = config.device

    task_paths = _get_task_paths(sampler)
    d_model = config.model.emb_dim

    print(f"Experiment : {exp_name}")
    print(f"Tasks      : {sampler.n_major_tasks} major + "
          f"{sampler.n_minor_tasks} minor = "
          f"{sampler.n_major_tasks + sampler.n_minor_tasks}")
    print(f"d_model    : {d_model}, layer: {layer_index}, "
          f"proj_dim: {proj_dim}")
    print(f"samples/cls: {samples_per_class}, masks: {n_masks}\n")

    # ── One-time collection: build the full pool ──
    all_data = collect_hidden_states(
        model, sampler, config,
        layer_index=layer_index, n_masks=n_masks, batch_size=batch_size,
    )
    pool, prefix_to_class, active_lengths = build_prefix_datasets(
        all_data, task_paths, max_prefix,
    )
    del all_data

    print("\nPool sizes:")
    for l in active_lengths:
        n = pool[l]["hiddens"].shape[0]
        n_cls = len(prefix_to_class[l])
        print(f"  l={l}: {n} samples, {n_cls} classes "
              f"({n // max(n_cls, 1)} per class)")

    # ── Split fixed validation set from pool ──
    val_data = {}
    for l in active_lengths:
        h, y = pool[l]["hiddens"], pool[l]["labels"]
        n_cls = int(y.max().item()) + 1
        val_budget = samples_per_class * n_cls
        perm = torch.randperm(h.shape[0])
        n_val = min(val_budget, h.shape[0] // 3)
        val_data[l] = {"hiddens": h[perm[:n_val]], "labels": y[perm[:n_val]]}
        pool[l]["hiddens"] = h[perm[n_val:]]
        pool[l]["labels"] = y[perm[n_val:]]

    # ── Initial training subsample from remaining pool ──
    data_by_len = _resample_from_pool(pool, active_lengths, samples_per_class)

    print("\nTraining / validation sizes:")
    for l in active_lengths:
        n_tr = data_by_len[l]["hiddens"].shape[0]
        n_va = val_data[l]["hiddens"].shape[0]
        n_cls = len(prefix_to_class[l])
        print(f"  l={l}: train={n_tr}, val={n_va}, {n_cls} classes")

    def _refresh(cur_max_prefix=None):
        """Re-sample from pool — no model forward pass needed."""
        mp = cur_max_prefix or max_prefix
        lengths = [l for l in active_lengths if l <= mp]
        return _resample_from_pool(pool, lengths, samples_per_class)

    probe, results = train_prefix_probe(
        data_by_len, prefix_to_class, active_lengths,
        d_model=d_model, proj_dim=proj_dim, mlp_hidden=mlp_hidden,
        num_epochs=num_epochs, lr=lr, proj_lr_scale=proj_lr_scale,
        mini_batch=mini_batch,
        val_data=val_data, device=device, verbose_every=verbose_every,
        loss_threshold=loss_threshold,
        refresh_every=refresh_every, refresh_fn=_refresh,
        curriculum_threshold=curriculum_threshold,
    )

    viz_data = {
        "data_by_len": data_by_len,
        "prefix_to_class": prefix_to_class,
        "active_lengths": active_lengths,
        "layer_index": layer_index,
        "k_value": k_value,
        "proj_dim": proj_dim,
        "device": device,
    }

    return probe, results, viz_data
