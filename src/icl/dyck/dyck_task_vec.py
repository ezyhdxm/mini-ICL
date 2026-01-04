from typing import Tuple, Union, Optional
import torch


@torch.no_grad()
def compute_hiddens_dyck(
    config,
    model: torch.nn.Module,
    sampler,
    dyck_mask,
    *,
    layer_index: int = 1,
    batch_size: int = 64,
    max_tasks: int = 80,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Extract hidden representations from a given layer's `attn_block`.

    Returns:
      - (n_tasks, P, B, d_model) where P = seq_len-1, from positions 2*arange(P)+1
    """
    if device is None:
        device = getattr(config, "device", None)
        if device is None:
            device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    # ----- task / shape bookkeeping -----
    n_total = int(sampler.n_major_tasks + sampler.n_minor_tasks)
    n_tasks = min(int(max_tasks), n_total)

    # Your sampler seems to use "full length" with separators; you map to Dyck positions
    seq_len = (int(sampler.seq_len) + 1) // 2
    d_model = int(config.model.emb_dim)

    if not (0 <= layer_index < len(model.layers)):
        raise IndexError(f"layer_index={layer_index} out of range for model.layers (len={len(model.layers)})")

    # multiple positions (tensor)
    P = seq_len - 1
    task_pos = 2 * torch.arange(P, device=device) + 1  # (P,)
    out = torch.empty((n_tasks, P, batch_size, d_model), device=device)

    # ----- one hook, reused across tasks -----
    cache = {}

    def hook_fn(module, inp, out_tensor):
        # out_tensor: (B, L, d_model)
        if isinstance(task_pos, torch.Tensor):
            cache["vecs"] = out_tensor.index_select(dim=1, index=task_pos).detach()  # (B, P, d)
        else:
            cache["vecs"] = out_tensor[:, task_pos, :].detach()  # (B, d)

    handle = model.layers[layer_index].attn_block.register_forward_hook(hook_fn)
    try:
        for t in range(n_tasks):
            # clone dyck_mask per task if sampler mutates it internally
            demo_data, _ = sampler.generate(
                mode="testing",
                task=t,
                num_samples=batch_size,
                dyck_mask=dyck_mask.clone(),
            )  # expected (B, L_in)

            if demo_data.device != device:
                demo_data = demo_data.to(device, non_blocking=True)

            cache.clear()
            _ = model(demo_data)

            if "vecs" not in cache:
                raise RuntimeError(
                    "Hook did not capture activations. "
                    "Check that model.layers[layer_index].attn_block is executed and returns (B, L, d)."
                )

            vecs = cache["vecs"]
            # (B, P, d) -> (P, B, d)
            out[t].copy_(vecs.permute(1, 0, 2))

            if verbose and (t == 0 or (t + 1) % 10 == 0 or t == n_tasks - 1):
                print(f"[compute_hiddens_dyck] task {t+1}/{n_tasks} done")
    finally:
        handle.remove()

    return out
