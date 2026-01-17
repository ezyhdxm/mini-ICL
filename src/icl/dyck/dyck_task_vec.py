from typing import Tuple, Union, Optional
import torch
import copy

import icl.utils.notebook_utils as nu

def get_dyck_sampler(exp_name, n_minor=64, n_ood=30):
    _, sampler, _ = nu.load_everything("dyck", exp_name)
    sampler_clone = copy.deepcopy(sampler)

    orig = sampler_clone.minor_task_pool
    k_minor = min(n_minor, sampler.n_minor_tasks)
    n_tasks = n_ood + k_minor
    sampler_clone.n_minor_tasks = n_tasks

    ood = sampler_clone._random_dyck_path(n_ood)
    if orig is not None:
        sampler_clone.minor_task_pool = torch.cat([ood, orig[:k_minor]], dim=0)
    else:
        sampler_clone.minor_task_pool = ood

    return sampler_clone, k_minor

@torch.no_grad()
def compute_hiddens_dyck(
    config,
    model: torch.nn.Module,
    sampler,
    dyck_mask,
    *,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Extract hidden representations from *all layers'* `attn_block`.

    Returns:
      - (n_layers, n_tasks, P, B, d_model)
        where P = seq_len-1, from positions 2*arange(P)+1
    """
    if device is None:
        device = getattr(config, "device", None)
        if device is None:
            device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    # ----- task / shape bookkeeping -----
    n_tasks = int(sampler.n_major_tasks + sampler.n_minor_tasks)

    seq_len = (int(sampler.seq_len) + 1) // 2
    d_model = int(config.model.emb_dim)

    if not hasattr(model, "layers"):
        raise AttributeError("Model has no attribute `layers`.")
    n_layers = len(model.layers)

    # positions (tensor)
    P = seq_len - 1
    task_pos = 2 * torch.arange(P, device=device) + 1  # (P,)

    # output: (n_layers, n_tasks, P, B, d)
    out = torch.empty((n_layers, n_tasks, P, batch_size, d_model), device=device)

    # ----- register hooks on all layers -----
    cache = {}

    def make_hook(layer_i: int):
        def hook_fn(module, inp, out_tensor):
            # out_tensor: (B, L, d)
            cache[layer_i] = out_tensor.index_select(dim=1, index=task_pos).detach()  # (B, P, d)
        return hook_fn

    handles = []
    for l in range(n_layers):
        handles.append(model.layers[l].attn_block.register_forward_hook(make_hook(l)))

    try:
        for t in range(n_tasks):
            demo_data, _ = sampler.generate(
                mode="testing",
                task=t,
                num_samples=batch_size,
                dyck_mask=dyck_mask.clone(),
            )
            idx = torch.nonzero(dyck_mask == 1, as_tuple=True)[0]
            # print(t, demo_data[:,::2][:, idx])

            if demo_data.device != device:
                demo_data = demo_data.to(device, non_blocking=True)

            cache.clear()
            _ = model(demo_data)

            # fill per-layer
            for l in range(n_layers):
                if l not in cache:
                    raise RuntimeError(
                        f"Hook did not capture activations for layer {l}. "
                        "Check that model.layers[l].attn_block is executed and returns (B, L, d)."
                    )
                vecs = cache[l]  # (B, P, d)
                out[l, t].copy_(vecs.permute(1, 0, 2))  # (P, B, d)

            if verbose and (t == 0 or (t + 1) % 10 == 0 or t == n_tasks - 1):
                print(f"[compute_hiddens_dyck] task {t+1}/{n_tasks} done")
    finally:
        for h in handles:
            h.remove()

    return out
