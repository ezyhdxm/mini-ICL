import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def extract_task_vector_markov(
        model, demo_data, l=0, task_pos=-1
    ):
    extracted_vector = {}

    def hook_fn(module, input, output):
        extracted_vector['vector'] = output[:, task_pos, :].detach().clone()

    hook_handle = model.layers[l].attn_block.register_forward_hook(hook_fn)
    with torch.no_grad(): _ = model(demo_data)
    hook_handle.remove()
    return extracted_vector['vector']

def predict_with_task_vector_markov(
        model, query_data, task_vector, l=0, pad="mapsto", pos=1
    ):
    task_pos=0 if pad == "bos" else pos
    
    def inject_hook(module, input, output):
        output[:, task_pos, :] = task_vector
        return output
    
    hook_handle = model.layers[l].attn_block.register_forward_hook(inject_hook)

    with torch.no_grad():
        preds = model(query_data)
    hook_handle.remove()

    return preds


def weighted_average_favor_late_batched(
    x: torch.Tensor, 
    mode="linear", 
    alpha=0.1, 
    cap_threshold: int = None
):
    B, T, D = x.shape
    t = torch.arange(T, dtype=x.dtype, device=x.device)

    if mode == "linear":
        weights = t + 1
    elif mode == "quadratic":
        weights = (t + 1) ** 2
    elif mode == "exp":
        weights = torch.exp(alpha * t)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if cap_threshold is not None:
        # Compute the cap value at threshold index
        cap_index = min(cap_threshold, T - 1)  # prevent out-of-bounds
        cap_value = weights[cap_index]
        weights = torch.where(t <= cap_threshold, weights, cap_value)

    weighted_sum = (weights[None, :, None] * x).sum(dim=1)  # (B, D)
    weights_sum = weights.sum()  # scalar
    return weighted_sum / weights_sum  # (B, D)

def moving_l2_distance_from_mean_sumD(x: torch.Tensor, window_size: int, sqrt: bool = False) -> torch.Tensor:
    B, T, D = x.shape
    x_perm = x.permute(0, 2, 1)  # (B, D, T)
    x_unfold = x_perm.unfold(dimension=2, size=window_size, step=1)  # (B, D, T-w+1, w)

    mean = x_unfold.mean(dim=-1, keepdim=True)  # (B, D, T-w+1, 1)
    sq_dev = ((x_unfold - mean) ** 2).mean(dim=-1)  # (B, D, T-w+1)

    reduced = sq_dev.sum(dim=1)  # sum over D → (B, T-w+1)

    if sqrt:
        reduced = reduced.sqrt()

    return reduced