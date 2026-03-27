"""End-to-end pipeline: load model, collect data, train probe."""

import torch

from ._data import (
    _load_model_and_sampler,
    _get_task_paths,
    collect_hidden_states,
    build_prefix_datasets,
    _resample_from_pool,
)
from ._train import train_prefix_probe


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
