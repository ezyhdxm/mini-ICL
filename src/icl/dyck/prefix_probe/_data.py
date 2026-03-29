"""Data collection and organisation for the Dyck prefix probe."""

import copy
from collections import defaultdict

import torch

import icl.utils.notebook_utils as nu
from icl.utils.unified_interface import get_exp_name
from icl.dyck.dyck_utils import sample_binary_mask


def _load_model_and_sampler(k_value, device=None, exp_name=None):
    """Load the trained Dyck model, sampler, and config for a given k.

    Parameters
    ----------
    exp_name : str, optional
        Override the experiment name / hash.  When omitted the name is
        derived from the default config for ``k_value``.
    """
    if exp_name is None:
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
