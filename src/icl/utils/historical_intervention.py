import torch
import torch.nn.functional as F
import icl.utils.notebook_utils as nu

from icl.utils.coin_ood_analysis import get_new_sampler
from icl.utils.latent_ood_analysis import get_latent_sampler
from icl.linear.linear_path_utils import load_model_task_config
from icl.linear.sampling import sample_points_from_balls


# ============================================================
# Shared helpers
# ============================================================

def _unwrap_logits(model_out):
    # model_out can be:
    # - Tensor logits
    # - HF-style object with .logits
    # - tuple where first element is logits
    if torch.is_tensor(model_out):
        return model_out
    if hasattr(model_out, "logits"):
        return model_out.logits
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0 and torch.is_tensor(model_out[0]):
        return model_out[0]
    raise RuntimeError(f"Cannot unwrap logits from type={type(model_out)}")


def _get_pad_id(sampler):
    # Coins convention: pad_id = sampler.num_states (when pad=True)
    # Latent sampler might follow same convention; if it has pad_id, prefer that.
    if hasattr(sampler, "pad_id"):
        return int(getattr(sampler, "pad_id"))
    if hasattr(sampler, "num_states"):
        return int(getattr(sampler, "num_states"))
    # fallback: try vocab_size-1
    if hasattr(sampler, "vocab_size"):
        return int(getattr(sampler, "vocab_size")) - 1
    raise AttributeError("Could not infer pad token id from sampler (need pad_id/num_states/vocab_size).")


def _generate_batch(sampler, *, mode: str, task: int, B: int):
    # Different samplers sometimes return (batch, meta) or just batch
    out = sampler.generate(mode=mode, task=task, num_samples=B)
    if isinstance(out, (tuple, list)):
        batch = out[0]
    else:
        batch = out
    return batch


# ============================================================
# Core: Historical perturbation for coin/latent
# ============================================================

def historical_injection_test_next_token(
    config,
    model: torch.nn.Module,
    sampler,
    *,
    layer_idx: int,
    p_idx: int,
    task: int,
    B: int = 64,
    mode: str = "testing",
    seed: int = 0,
    p: float = 0.1,  # probability of perturbing each historical position
    rand_mode: str = "standard_normal",  # {"standard_normal", "match_std", "match_mean_std"}
    eps: float = 1e-12,
    hook_attr: str = "attn_block",
    return_logits: bool = False,
):
    """
    Perturb historical hidden representations (before padded token and token before it).
    
    Assumes "even/odd" padding layout when sampler.pad is True:
      - even positions  0,2,4,...   are real tokens
      - odd positions   1,3,5,...   are PAD tokens
    We inject at token_pos = 2*p_idx + 1 (odd, padded position).
    We keep positions token_pos-1 and token_pos unchanged.
    We randomly perturb positions [0, ..., token_pos-2] with probability p.
    We evaluate next-token prediction at token_pos against target batch[:, token_pos+1].
    
    Works for both coin + latent samplers as long as:
      - sampler.pad == True
      - sampler.seq_len is the number of *real* tokens
      - sampler.generate(mode=..., task=..., num_samples=...) returns a token batch
    """
    if not getattr(sampler, "pad", False):
        raise ValueError("This function is for padded-only injection (sampler.pad must be True).")
    
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(device)

    if not hasattr(model, "layers"):
        raise AttributeError("model has no attribute 'layers' (expected transformer-like model.layers).")
    n_layers = len(model.layers)

    seq_len = int(getattr(sampler, "seq_len"))
    P = seq_len - 1
    if not (0 <= p_idx < P):
        raise ValueError(f"p_idx out of range: {p_idx} not in [0, {P-1}]")
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} not in [0, {n_layers-1}]")

    # ---- padded token position ----
    token_pos = 2 * p_idx + 1  # odd index => PAD token
    token_before_pad = token_pos - 1  # even index => real token before PAD
    
    # Historical positions to potentially perturb: [0, ..., token_pos-2]
    historical_end = token_pos - 1  # exclusive end
    if historical_end <= 0:
        raise ValueError(f"No historical positions to perturb for p_idx={p_idx} (token_pos={token_pos})")

    # ---- batch ----
    batch_data = _generate_batch(sampler, mode=mode, task=task, B=B).to(device)

    L = batch_data.size(1)
    if token_pos + 1 >= L:
        raise ValueError(f"token_pos+1 out of bounds: token_pos={token_pos}, L={L}")

    # sanity: confirm we're injecting at PAD tokens
    pad_id = _get_pad_id(sampler)
    if (batch_data[:, token_pos] != pad_id).any():
        bad = (batch_data[:, token_pos] != pad_id).nonzero(as_tuple=False)[:5].flatten().tolist()
        raise RuntimeError(
            f"Expected PAD token at position {token_pos}, but found non-pad tokens for some rows. "
            f"First bad row indices: {bad}. (pad_id={pad_id})"
        )

    def _get_next_token_slice(logits: torch.Tensor):
        logits_at_pos = logits[:, token_pos, :]      # predicts token at token_pos+1
        target = batch_data[:, token_pos + 1]        # next real token
        return logits_at_pos, target

    @torch.no_grad()
    def _run_baseline():
        out = model(batch_data)
        return _unwrap_logits(out)

    def _replace_historical_hiddens(hid: torch.Tensor, cache: dict):
        """
        Randomly perturbs historical hidden states (before token_before_pad and token_pos).
        Keeps token_before_pad and token_pos unchanged.
        """
        if hid.dim() != 3:
            raise RuntimeError(f"Expected 3D hidden, got shape={tuple(hid.shape)}")

        B_local = batch_data.size(0)

        if hid.size(0) == B_local:
            # (B, L, d)
            if token_pos >= hid.size(1):
                raise RuntimeError(f"token_pos={token_pos} >= L={hid.size(1)} in (B,L,d)")
            out2 = hid.clone()
            cache["layout"] = "BLD"
            
            # Generate random mask for historical positions
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            # Mask shape: (B, historical_end) - one mask per batch item for reproducibility
            # We use the same seed but different offsets per position to ensure determinism
            mask = torch.zeros((B_local, historical_end), dtype=torch.bool, device=device)
            for pos in range(historical_end):
                g_pos = torch.Generator(device=device)
                g_pos.manual_seed(seed + pos)  # Different seed per position
                mask[:, pos] = torch.rand(B_local, generator=g_pos, device=device) < p
            
            cache["perturb_mask"] = mask.detach()
            cache["n_perturbed"] = mask.sum().item()
            
            # Perturb historical positions according to mask
            for pos in range(historical_end):
                if mask[:, pos].any():
                    h_pos = out2[:, pos, :]  # (B, d)
                    h_perturbed = h_pos.clone()
                    
                    # Only perturb rows where mask is True
                    mask_rows = mask[:, pos]  # (B,)
                    h_to_perturb = h_pos[mask_rows]  # (n_perturbed, d)
                    
                    g_perturb = torch.Generator(device=device)
                    g_perturb.manual_seed(seed + pos + 1000)  # Different seed for perturbation
                    
                    # Compute std from ALL batch items at this position (consistent with random injection)
                    # This ensures noise magnitude is comparable across different perturbation probabilities p
                    if rand_mode == "standard_normal":
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb)
                    elif rand_mode == "match_std":
                        std = h_pos.std(unbiased=False) + eps  # Use h_pos (all batch items), not h_to_perturb
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb) * std
                    elif rand_mode == "match_mean_std":
                        mean = h_pos.mean()  # Use h_pos (all batch items), not h_to_perturb
                        std = h_pos.std(unbiased=False) + eps
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb) * std + mean
                    else:
                        raise ValueError(f"Unknown rand_mode={rand_mode!r}")
                    
                    h_perturbed[mask_rows] = r
                    out2[:, pos, :] = h_perturbed

        elif hid.size(1) == B_local:
            # (L, B, d)
            if token_pos >= hid.size(0):
                raise RuntimeError(f"token_pos={token_pos} >= L={hid.size(0)} in (L,B,d)")
            out2 = hid.clone()
            cache["layout"] = "LBD"
            
            # Generate random mask for historical positions
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            mask = torch.zeros((historical_end, B_local), dtype=torch.bool, device=device)
            for pos in range(historical_end):
                g_pos = torch.Generator(device=device)
                g_pos.manual_seed(seed + pos)
                mask[pos, :] = torch.rand(B_local, generator=g_pos, device=device) < p
            
            cache["perturb_mask"] = mask.detach()
            cache["n_perturbed"] = mask.sum().item()
            
            # Perturb historical positions according to mask
            for pos in range(historical_end):
                if mask[pos, :].any():
                    h_pos = out2[pos, :, :]  # (B, d)
                    h_perturbed = h_pos.clone()
                    
                    mask_cols = mask[pos, :]  # (B,)
                    h_to_perturb = h_pos[mask_cols, :]  # (n_perturbed, d)
                    
                    g_perturb = torch.Generator(device=device)
                    g_perturb.manual_seed(seed + pos + 1000)
                    
                    # Compute std from ALL batch items at this position (consistent with random injection)
                    # This ensures noise magnitude is comparable across different perturbation probabilities p
                    if rand_mode == "standard_normal":
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb)
                    elif rand_mode == "match_std":
                        std = h_pos.std(unbiased=False) + eps  # Use h_pos (all batch items), not h_to_perturb
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb) * std
                    elif rand_mode == "match_mean_std":
                        mean = h_pos.mean()  # Use h_pos (all batch items), not h_to_perturb
                        std = h_pos.std(unbiased=False) + eps
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb) * std + mean
                    else:
                        raise ValueError(f"Unknown rand_mode={rand_mode!r}")
                    
                    h_perturbed[mask_cols, :] = r
                    out2[pos, :, :] = h_perturbed

        else:
            raise RuntimeError(f"Cannot infer layout. hid={tuple(hid.shape)}, batch B={B_local}")

        return out2

    @torch.no_grad()
    def _run_injected():
        cache = {}
        layer = model.layers[layer_idx]
        if not hasattr(layer, hook_attr):
            raise AttributeError(f"Layer {layer_idx} has no attribute {hook_attr!r}.")
        mod = getattr(layer, hook_attr)

        def hook_fn(module, inp, out):
            # out can be Tensor or tuple(Tensor, ...)
            if torch.is_tensor(out):
                return _replace_historical_hiddens(out, cache)
            if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                new0 = _replace_historical_hiddens(out[0], cache)
                return (new0,) + out[1:]
            raise RuntimeError(f"Unsupported hook output type: {type(out)}")

        handle = mod.register_forward_hook(hook_fn)
        try:
            out = model(batch_data)
            logits = _unwrap_logits(out)
        finally:
            handle.remove()

        return logits, cache

    # ---- run both passes ----
    base_logits = _run_baseline()
    inj_logits, cache = _run_injected()

    base_slice, target = _get_next_token_slice(base_logits)
    inj_slice, _ = _get_next_token_slice(inj_logits)

    base_logp = F.log_softmax(base_slice, dim=-1)
    inj_logp = F.log_softmax(inj_slice, dim=-1)

    base_tlp = base_logp.gather(1, target[:, None]).squeeze(1)
    inj_tlp = inj_logp.gather(1, target[:, None]).squeeze(1)
    delta_tlp = inj_tlp - base_tlp

    base_top1 = base_slice.argmax(dim=-1)
    inj_top1 = inj_slice.argmax(dim=-1)

    base_acc = (base_top1 == target).float().mean().item()
    inj_acc = (inj_top1 == target).float().mean().item()
    top1_flip_rate = (base_top1 != inj_top1).float().mean().item()

    base_p = base_logp.exp()
    kl = (base_p * (base_logp - inj_logp)).sum(dim=-1)

    results = {
        "layer_idx": int(layer_idx),
        "p_idx": int(p_idx),
        "token_pos": int(token_pos),
        "token_before_pad": int(token_before_pad),
        "historical_end": int(historical_end),
        "task": int(task),
        "B": int(B),
        "mode": mode,
        "hook_attr": hook_attr,
        "rand_mode": rand_mode,
        "seed": int(seed),
        "p": float(p),
        "n_perturbed": int(cache.get("n_perturbed", 0)),
        "hidden_layout": cache.get("layout", None),

        "base_acc": float(base_acc),
        "inj_acc": float(inj_acc),
        "top1_flip_rate": float(top1_flip_rate),

        "delta_target_logprob_mean": float(delta_tlp.mean().item()),
        "delta_target_logprob_std": float(delta_tlp.std(unbiased=False).item()),
        "kl_mean": float(kl.mean().item()),
        "kl_std": float(kl.std(unbiased=False).item()),
    }

    if return_logits:
        results.update({
            "base_logits_at_pos": base_slice.detach(),
            "inj_logits_at_pos": inj_slice.detach(),
            "target_tokens": target.detach(),
            "perturb_mask": cache.get("perturb_mask", None),
            "delta_target_logprob": delta_tlp.detach(),
            "kl_per_example": kl.detach(),
        })

    return results


# ============================================================
# Core: Historical perturbation for linear regression
# ============================================================

def historical_injection_test_next_token_linear(
    config,
    model: torch.nn.Module,
    train_task,
    *,
    layer_idx: int,
    p_idx: int,
    task_idx: int,
    B: int = 64,
    step: int = 1008600,
    seed: int = 0,
    p: float = 0.1,  # probability of perturbing each historical position
    rand_mode: str = "standard_normal",  # {"standard_normal", "match_std", "match_mean_std"}
    eps: float = 1e-12,
    hook_attr: str = "attn_block",
    return_logits: bool = False,
):
    """
    Perturb historical hidden representations for linear regression task.
    
    Assumes "mapsto" padding layout:
      - Pattern: [data_0, PAD, target_0, data_1, PAD, target_1, ...]
      - Padding positions: 1, 4, 7, ... (every 3rd position starting at 1)
      - We inject at token_pos = 3 * p_idx + 1
      - We keep positions token_pos-1 and token_pos unchanged.
      - We randomly perturb positions [0, ..., token_pos-2] with probability p.
      - Evaluate next-token prediction at that position against the next target
    
    The model forward takes (data, targets) as two separate arguments.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(device)
    
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "blocks"):
        raise AttributeError("model.transformer.blocks not found (expected TransformerLin model).")
    n_layers = len(model.transformer.blocks)
    
    n_points = int(getattr(config.task, "n_points", train_task.n_points))
    P = n_points  # number of padded locations (one per data point)
    
    if not (0 <= p_idx < P):
        raise ValueError(f"p_idx out of range: {p_idx} not in [0, {P-1}]")
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} not in [0, {n_layers-1}]")
    
    # ---- padded token position (mapsto format: positions 1, 4, 7, ...) ----
    token_pos = 3 * p_idx + 1
    token_before_pad = token_pos - 1  # position before PAD
    
    # Historical positions to potentially perturb: [0, ..., token_pos-2]
    historical_end = token_pos - 1  # exclusive end
    if historical_end <= 0:
        raise ValueError(f"No historical positions to perturb for p_idx={p_idx} (token_pos={token_pos})")
    
    # ---- get task from task_pool ----
    if train_task.task_pool is None:
        raise ValueError("train_task.task_pool is None. Cannot sample from task.")
    if task_idx >= train_task.task_pool.shape[0]:
        raise ValueError(f"task_idx {task_idx} >= n_tasks {train_task.task_pool.shape[0]}")
    
    task_vector = train_task.task_pool[task_idx:task_idx+1]  # (1, n_dims, 1)
    
    # ---- generate batch ----
    original_batch_size = train_task.batch_size
    train_task.batch_size = B
    try:
        demo_data, demo_target = train_task.sample_from_task(task_vector, step=step)
        # demo_data: (B, n_points, n_dims)
        # demo_target: (B, n_points)
    finally:
        train_task.batch_size = original_batch_size
    
    demo_data = demo_data.to(device)
    demo_target = demo_target.to(device)
    
    seq_len_with_pad = 3 * n_points  # (data, PAD, target) * n_points
    
    if token_pos >= seq_len_with_pad:
        raise ValueError(f"token_pos out of bounds: token_pos={token_pos}, seq_len={seq_len_with_pad}")
    
    @torch.no_grad()
    def _run_baseline():
        out = model(demo_data, demo_target)
        return out
    
    def _replace_historical_hiddens(hid: torch.Tensor, cache: dict):
        """
        Randomly perturbs historical hidden states (before token_before_pad and token_pos).
        Keeps token_before_pad and token_pos unchanged.
        """
        if hid.dim() != 3:
            raise RuntimeError(f"Expected 3D hidden, got shape={tuple(hid.shape)}")
        
        B_local = demo_data.size(0)
        
        if hid.size(0) == B_local:
            # (B, L, d)
            if token_pos >= hid.size(1):
                raise RuntimeError(f"token_pos={token_pos} >= L={hid.size(1)} in (B,L,d)")
            out2 = hid.clone()
            cache["layout"] = "BLD"
            
            # Generate random mask for historical positions
            mask = torch.zeros((B_local, historical_end), dtype=torch.bool, device=device)
            for pos in range(historical_end):
                g_pos = torch.Generator(device=device)
                g_pos.manual_seed(seed + pos)
                mask[:, pos] = torch.rand(B_local, generator=g_pos, device=device) < p
            
            cache["perturb_mask"] = mask.detach()
            cache["n_perturbed"] = mask.sum().item()
            
            # Perturb historical positions according to mask
            for pos in range(historical_end):
                if mask[:, pos].any():
                    h_pos = out2[:, pos, :]  # (B, d)
                    h_perturbed = h_pos.clone()
                    
                    mask_rows = mask[:, pos]  # (B,)
                    h_to_perturb = h_pos[mask_rows]  # (n_perturbed, d)
                    
                    g_perturb = torch.Generator(device=device)
                    g_perturb.manual_seed(seed + pos + 1000)
                    
                    # Compute std from ALL batch items at this position (consistent with random injection)
                    # This ensures noise magnitude is comparable across different perturbation probabilities p
                    if rand_mode == "standard_normal":
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb)
                    elif rand_mode == "match_std":
                        std = h_pos.std(unbiased=False) + eps  # Use h_pos (all batch items), not h_to_perturb
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb) * std
                    elif rand_mode == "match_mean_std":
                        mean = h_pos.mean()  # Use h_pos (all batch items), not h_to_perturb
                        std = h_pos.std(unbiased=False) + eps
                        r = torch.randn(h_to_perturb.shape, device=h_to_perturb.device, 
                                       dtype=h_to_perturb.dtype, generator=g_perturb) * std + mean
                    else:
                        raise ValueError(f"Unknown rand_mode={rand_mode!r}")
                    
                    h_perturbed[mask_rows] = r
                    out2[:, pos, :] = h_perturbed
        else:
            raise RuntimeError(f"Cannot infer layout. hid={tuple(hid.shape)}, batch B={B_local}")
        
        return out2
    
    @torch.no_grad()
    def _run_injected():
        cache = {}
        block = model.transformer.blocks[layer_idx]
        if not hasattr(block, hook_attr):
            raise AttributeError(f"Block {layer_idx} has no attribute {hook_attr!r}.")
        mod = getattr(block, hook_attr)
        
        def hook_fn(module, inp, out):
            # out can be Tensor or tuple(Tensor, ...)
            if torch.is_tensor(out):
                return _replace_historical_hiddens(out, cache)
            if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                new0 = _replace_historical_hiddens(out[0], cache)
                return (new0,) + out[1:]
            raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = mod.register_forward_hook(hook_fn)
        try:
            out = model(demo_data, demo_target)
        finally:
            handle.remove()
        
        return out, cache
    
    # ---- run both passes ----
    base_preds = _run_baseline()
    inj_preds, cache = _run_injected()
    
    # Extract prediction at position p_idx
    base_pred_at_pos = base_preds[:, p_idx]  # (B,)
    inj_pred_at_pos = inj_preds[:, p_idx]  # (B,)
    target = demo_target[:, p_idx]  # (B,)
    
    # Compute MSE differences
    base_mse = ((base_pred_at_pos - target) ** 2).mean().item()
    inj_mse = ((inj_pred_at_pos - target) ** 2).mean().item()
    delta_mse = inj_mse - base_mse
    
    # For consistency with discrete tasks API
    delta_target_logprob_mean = -delta_mse
    
    results = {
        "layer_idx": int(layer_idx),
        "p_idx": int(p_idx),
        "token_pos": int(token_pos),
        "token_before_pad": int(token_before_pad),
        "historical_end": int(historical_end),
        "task_idx": int(task_idx),
        "B": int(B),
        "hook_attr": hook_attr,
        "rand_mode": rand_mode,
        "seed": int(seed),
        "p": float(p),
        "n_perturbed": int(cache.get("n_perturbed", 0)),
        "hidden_layout": cache.get("layout", None),
        
        "base_mse": float(base_mse),
        "inj_mse": float(inj_mse),
        "delta_mse": float(delta_mse),
        "delta_target_logprob_mean": float(delta_target_logprob_mean),
    }
    
    if return_logits:
        results.update({
            "base_pred_at_pos": base_pred_at_pos.detach(),
            "inj_pred_at_pos": inj_pred_at_pos.detach(),
            "target": target.detach(),
            "perturb_mask": cache.get("perturb_mask", None),
        })
    
    return results


# ============================================================
# Coin wrapper: aggregate over tasks at each padded position
# ============================================================

def historical_injection_coin(
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    step: int | None = None,
    layer_idx: int = 5,
    hook_attr: str = "attn_block",
    agg: str = "mean",           # {"mean","median"}
    mode: str = "testing",
    seed: int = 0,
    p: float = 0.1,
    rand_mode: str = "standard_normal",
    verbose: bool = False,
):
    """
    Historical perturbation analysis for Coin task.
    
    For each padded position p_idx (starting from 1), randomly perturbs historical hidden states
    (before the padded token and token before it) with probability p, and measures
    the effect on next-token prediction.
    
    Note: p_idx=0 is skipped since there are no historical positions to perturb.
    
    Returns:
      out: list[float] length (seq_len-2), corresponding to p_idx values [1, ..., seq_len-2]
      info: dict with task list and per-task values
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("coin", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

    # Clone sampler (major+minor)
    sampler, _ = get_new_sampler(exp_name, n_minor, n_ood)

    if not getattr(sampler, "pad", False):
        raise ValueError(
            "historical_injection_coin requires padded coin sequences (sampler.pad must be True)."
        )

    n_tasks = int(getattr(sampler, "total_tasks", sampler.n_major_tasks + sampler.n_minor_tasks))
    task_ids = list(range(n_tasks))

    P = int(sampler.seq_len) - 1  # number of padded locations

    out = []
    per_task = []

    # Skip p_idx=0 since there are no historical positions to perturb
    # (token_pos=1, historical_end=0 means no positions before it)
    for p_idx in range(1, P):
        vals = []
        for tid in task_ids:
            r = historical_injection_test_next_token(
                config=config,
                model=model,
                sampler=sampler,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                seed=seed,
                p=p,
                rand_mode=rand_mode,
                hook_attr=hook_attr,
            )
            vals.append(r["delta_target_logprob_mean"])

        vals_t = torch.tensor(vals, dtype=torch.float32)
        if agg == "mean":
            agg_val = float(vals_t.mean().item())
        elif agg == "median":
            agg_val = float(vals_t.median().item())
        else:
            raise ValueError(f"Unknown agg={agg!r}. Expected 'mean' or 'median'.")

        out.append(agg_val)
        per_task.append(vals)

        if verbose and ((p_idx - 1) % 5 == 0 or p_idx == P - 1):
            print(
                f"[coin] p_idx={p_idx:02d}/{P-1:02d} "
                f"(padded token_pos={2*p_idx+1:03d}) "
                f"Δlogp({agg} over {n_tasks} tasks, p={p})={agg_val:+.4f}"
            )

    info = {"task_ids": task_ids, "per_task": per_task, "p": float(p)}
    return out, info


# ============================================================
# Latent wrapper: aggregate over tasks at each padded position
# ============================================================

def historical_injection_latent(
    exp_name: str,
    n_minor: int = 256,
    n_ood: int = 40,
    B: int = 96,
    step: int | None = None,
    layer_idx: int = 5,
    hook_attr: str = "attn_block",
    agg: str = "mean",           # {"mean","median"}
    mode: str = "testing",
    seed: int = 0,
    p: float = 0.1,
    rand_mode: str = "standard_normal",
    verbose: bool = False,
):
    """
    Historical perturbation analysis for Latent task.
    
    Same as historical_injection_coin but for latent task sampler.
    
    Note: p_idx=0 is skipped since there are no historical positions to perturb.
    
    Returns:
      out: list[float] length (seq_len-2), corresponding to p_idx values [1, ..., seq_len-2]
      info: dict with task list and per-task values
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

    # Get latent sampler + task count
    # Note: get_latent_sampler returns n_tasks that only includes minor+ood, not major
    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood)

    if not getattr(sampler, "pad", False):
        raise ValueError(
            "historical_injection_latent requires padded latent sequences (sampler.pad must be True)."
        )

    # Ensure we include both major and minor tasks
    # The sampler's generate method expects task IDs in range [0, n_major_tasks + n_minor_tasks)
    n_major = int(getattr(sampler, "n_major_tasks", 0))
    n_minor_actual = int(getattr(sampler, "n_minor_tasks", 0))
    n_tasks = n_major + n_minor_actual
    task_ids = list(range(n_tasks))
    P = int(sampler.seq_len) - 1

    out = []
    per_task = []

    # Skip p_idx=0 since there are no historical positions to perturb
    # (token_pos=1, historical_end=0 means no positions before it)
    for p_idx in range(1, P):
        vals = []
        for tid in task_ids:
            r = historical_injection_test_next_token(
                config=config,
                model=model,
                sampler=sampler,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                seed=seed,
                p=p,
                rand_mode=rand_mode,
                hook_attr=hook_attr,
            )
            vals.append(r["delta_target_logprob_mean"])

        vals_t = torch.tensor(vals, dtype=torch.float32)
        if agg == "mean":
            agg_val = float(vals_t.mean().item())
        elif agg == "median":
            agg_val = float(vals_t.median().item())
        else:
            raise ValueError(f"Unknown agg={agg!r}. Expected 'mean' or 'median'.")

        out.append(agg_val)
        per_task.append(vals)

        if verbose and ((p_idx - 1) % 5 == 0 or p_idx == P - 1):
            print(
                f"[latent] p_idx={p_idx:02d}/{P-1:02d} "
                f"(padded token_pos={2*p_idx+1:03d}) "
                f"Δlogp({agg} over {len(task_ids)} tasks, p={p})={agg_val:+.4f}"
            )

    info = {"task_ids": task_ids, "per_task": per_task, "k_minor": int(k_minor), 
            "n_tasks": int(n_tasks), "p": float(p)}
    return out, info


# ============================================================
# Linear wrapper: aggregate over tasks at each padded position
# ============================================================

def historical_injection_linear(
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    step: int | None = None,
    layer_idx: int = 5,
    hook_attr: str = "attn_block",
    agg: str = "mean",           # {"mean","median"}
    seed: int = 0,
    p: float = 0.1,
    rand_mode: str = "standard_normal",
    verbose: bool = False,
    radius: float = 2.0,  # radius for OOD task sampling
):
    """
    Historical perturbation analysis for Linear regression task.
    
    For each padded position p_idx (starting from 1), randomly perturbs historical hidden states
    (before the padded token and token before it) with probability p, and measures
    the effect on prediction accuracy (MSE).
    
    Note: p_idx=0 is skipped since there are no historical positions to perturb.
    
    Returns:
      out: list[float] length (n_points-1), corresponding to p_idx values [1, ..., n_points-1]
      info: dict with task list and per-task values
    """
    # Load config/model/task
    model, train_task, config = load_model_task_config(exp_name)
    
    if step is None:
        step = getattr(config.training, "num_epochs", 1008600)
    
    # Check if model uses "mapsto" padding
    if not hasattr(model, "pad") or model.pad != "mapsto":
        raise ValueError(
            "historical_injection_linear requires model with pad='mapsto' "
            f"(got pad={getattr(model, 'pad', None)})."
        )
    
    if train_task.task_pool is None:
        raise ValueError("train_task.task_pool is None. Cannot iterate over tasks.")
    
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Create expanded task pool: major + minor + OOD
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)
    n_major = anchor_pool.shape[0]
    
    eval_task_pool_list = [anchor_pool]
    n_major_tasks = n_major
    
    # Add minor tasks if available
    n_minor_sampled = 0
    if train_task.minor_pool is not None and train_task.n_minor_tasks > 0:
        minor_pool = train_task.minor_pool.squeeze(-1).to(device)
        k_minor = min(n_minor, train_task.n_minor_tasks)
        if train_task.n_minor_tasks > n_minor:
            indices = torch.randperm(train_task.n_minor_tasks, device=device)[:k_minor]
            minor_pool_sampled = minor_pool[indices]
        else:
            minor_pool_sampled = minor_pool
        eval_task_pool_list.append(minor_pool_sampled)
        n_minor_sampled = minor_pool_sampled.shape[0]
    
    # Generate OOD tasks
    if n_ood > 0:
        M = anchor_pool.shape[0]
        base = n_ood // M
        rem = n_ood % M
        n_per_ball = torch.full((M,), base, dtype=torch.long, device=device)
        n_per_ball[:rem] += 1
        
        ood_pool, _ = sample_points_from_balls(
            anchor_pool,
            r=radius,
            n_per_ball=n_per_ball,
        )
        eval_task_pool_list.append(ood_pool)
    
    # Combine all task pools
    eval_task_pool = torch.cat(eval_task_pool_list, dim=0)
    n_tasks = eval_task_pool.shape[0]
    task_ids = list(range(n_tasks))
    
    n_points = int(getattr(config.task, "n_points", train_task.n_points))
    P = n_points
    
    # Temporarily replace task pool
    original_task_pool = train_task.task_pool
    train_task.task_pool = eval_task_pool.unsqueeze(-1)
    
    try:
        out = []
        per_task = []
        
        # Skip p_idx=0 since there are no historical positions to perturb
        # (token_pos=1, historical_end=0 means no positions before it)
        for p_idx in range(1, P):
            vals = []
            for tid in task_ids:
                r = historical_injection_test_next_token_linear(
                    config=config,
                    model=model,
                    train_task=train_task,
                    layer_idx=layer_idx,
                    p_idx=p_idx,
                    task_idx=tid,
                    B=B,
                    step=step,
                    seed=seed,
                    p=p,
                    rand_mode=rand_mode,
                    hook_attr=hook_attr,
                )
                vals.append(r["delta_target_logprob_mean"])
            
            vals_t = torch.tensor(vals, dtype=torch.float32)
            if agg == "mean":
                agg_val = float(vals_t.mean().item())
            elif agg == "median":
                agg_val = float(vals_t.median().item())
            else:
                raise ValueError(f"Unknown agg={agg!r}. Expected 'mean' or 'median'.")
            
            out.append(agg_val)
            per_task.append(vals)
            
            if verbose and ((p_idx - 1) % 5 == 0 or p_idx == P - 1):
                print(
                    f"[linear] p_idx={p_idx:02d}/{P-1:02d} "
                    f"(padded token_pos={3*p_idx+1:03d}) "
                    f"ΔMSE({agg} over {n_tasks} tasks, p={p})={-agg_val:+.4f}"
                )
        
        info = {
            "task_ids": task_ids,
            "per_task": per_task,
            "n_tasks": int(n_tasks),
            "n_major": int(n_major_tasks),
            "n_minor": int(n_minor_sampled),
            "n_ood": int(n_ood),
            "p": float(p),
        }
        return out, info
    finally:
        train_task.task_pool = original_task_pool


# ============================================================
# Plotting functions
# ============================================================

import matplotlib.pyplot as plt

def plot_historical_injection_results(out, p: float, task_name: str = ""):
    """
    Plot historical injection results.
    
    Args:
        out: List of aggregated delta_target_logprob_mean values
             Note: This list corresponds to p_idx values [1, 2, ..., len(out)]
        p: Probability parameter used for perturbation
        task_name: Optional task name for title
    """
    # x-axis maps to p_idx values starting from 1
    x = list(range(1, len(out) + 1))
    y = out

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("p_idx (padded position index)")
    plt.ylabel("mean Δ target log-prob (inj - base)")
    title = f"Historical perturbation at padded positions"
    if task_name:
        title += f" ({task_name})"
    plt.title(title)
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, f"p={p}", transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()


def plot_historical_injection_results_linear(out, p: float):
    """
    Plot historical injection results for linear regression.
    
    Args:
        out: List of aggregated delta_target_logprob_mean values (-delta_mse)
             Note: This list corresponds to p_idx values [1, 2, ..., len(out)]
        p: Probability parameter used for perturbation
    """
    # x-axis maps to p_idx values starting from 1
    x = list(range(1, len(out) + 1))
    y = out

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("p_idx (padded position index)")
    plt.ylabel("mean Δ (-MSE) (positive = better, negative = worse)")
    plt.title("Historical perturbation at padded positions (linear regression)")
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, f"p={p}", transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()

