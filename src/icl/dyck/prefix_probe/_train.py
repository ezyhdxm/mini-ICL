"""Training loop for the Dyck prefix probe."""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from icl.utils.device_utils import get_default_device

from ._model import PrefixProbe


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
    verbose_every=10,
    loss_threshold=0.01,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    refresh_every=3,
    refresh_fn=None,
    curriculum_threshold=None,
    device=None,
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
    if device is None:
        device = get_default_device()
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

            if all_lengths_active and mean_vloss <= loss_threshold:
                print(f"\nStopping at epoch {epoch+1}: mean val loss "
                      f"{mean_vloss:.4f} <= threshold {loss_threshold}")
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()

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
