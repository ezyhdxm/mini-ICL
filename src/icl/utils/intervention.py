import torch
import torch.nn.functional as F
import icl.utils.notebook_utils as nu

from icl.utils.coin_ood_analysis import get_new_sampler
from icl.utils.latent_ood_analysis import get_latent_sampler
from icl.linear.linear_path_utils import load_model_task_config
from icl.linear.lr_utils import to_seq
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
# Core: inject ONLY at padded positions, measure next-token logp
# ============================================================

def injection_test_next_token_padded_only(
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
    rand_mode: str = "standard_normal",  # {"standard_normal", "match_std", "match_mean_std"}
    eps: float = 1e-12,
    hook_attr: str = "attn_block",
    return_logits: bool = False,
):
    """
    Inject ONLY at the padded location for the given p_idx.

    Assumes "even/odd" padding layout when sampler.pad is True:
      - even positions  0,2,4,...   are real tokens
      - odd positions   1,3,5,...   are PAD tokens
    We inject at token_pos = 2*p_idx + 1 (odd, padded position),
    and evaluate next-token prediction at that position, i.e. logits[:, token_pos, :]
    against target batch[:, token_pos+1] (the next real token at even index).

    Works for both coin + latent samplers as long as:
      - sampler.pad == True
      - sampler.seq_len is the number of *real* tokens
      - sampler.generate(mode=..., task=..., num_samples=...) returns a token batch
      - pad token id can be inferred via pad_id or num_states or vocab_size-1
    """
    if not getattr(sampler, "pad", False):
        raise ValueError("This function is for padded-only injection (sampler.pad must be True).")

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

    def _replace_hidden(hid: torch.Tensor, cache: dict):
        """
        Supports hid shaped either (B, L, d) or (L, B, d).
        Replaces the hidden vector at the *padded* position token_pos ONLY.
        
        Important: This function clones the hidden tensor and only modifies position token_pos.
        Due to causal attention masking, positions before token_pos are unaffected by this change
        (they cannot attend to token_pos). Only token_pos itself and positions after it may be affected.
        """
        if hid.dim() != 3:
            raise RuntimeError(f"Expected 3D hidden, got shape={tuple(hid.shape)}")

        B_local = batch_data.size(0)

        if hid.size(0) == B_local:
            # (B, L, d)
            if token_pos >= hid.size(1):
                raise RuntimeError(f"token_pos={token_pos} >= L={hid.size(1)} in (B,L,d)")
            out2 = hid.clone()
            h = out2[:, token_pos, :]  # (B, d)
            setter = lambda out2_, r_: out2_.__setitem__((slice(None), token_pos, slice(None)), r_)
            cache["layout"] = "BLD"

        elif hid.size(1) == B_local:
            # (L, B, d)
            if token_pos >= hid.size(0):
                raise RuntimeError(f"token_pos={token_pos} >= L={hid.size(0)} in (L,B,d)")
            out2 = hid.clone()
            h = out2[token_pos, :, :]  # (B, d)
            setter = lambda out2_, r_: out2_.__setitem__((token_pos, slice(None), slice(None)), r_)
            cache["layout"] = "LBD"

        else:
            raise RuntimeError(f"Cannot infer layout. hid={tuple(hid.shape)}, batch B={B_local}")

        cache["orig_h"] = h.detach()

        g = torch.Generator(device=device)
        g.manual_seed(seed)

        if rand_mode == "standard_normal":
            r = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=g)
        elif rand_mode == "match_std":
            std = h.std(unbiased=False) + eps
            r = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=g) * std
        elif rand_mode == "match_mean_std":
            mean = h.mean()
            std = h.std(unbiased=False) + eps
            r = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=g) * std + mean
        else:
            raise ValueError(f"Unknown rand_mode={rand_mode!r}")

        cache["rand_h"] = r.detach()
        setter(out2, r)
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
                return _replace_hidden(out, cache)
            if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                new0 = _replace_hidden(out[0], cache)
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
        "task": int(task),
        "B": int(B),
        "mode": mode,
        "hook_attr": hook_attr,
        "rand_mode": rand_mode,
        "seed": int(seed),
        "pad_id": int(pad_id),
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
            "orig_hidden": cache.get("orig_h", None),
            "rand_hidden": cache.get("rand_h", None),
            "delta_target_logprob": delta_tlp.detach(),
            "kl_per_example": kl.detach(),
        })

    return results


# ============================================================
# Core: Linear regression injection (different padding pattern)
# ============================================================

def injection_test_next_token_linear_padded_only(
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
    rand_mode: str = "standard_normal",  # {"standard_normal", "match_std", "match_mean_std"}
    eps: float = 1e-12,
    hook_attr: str = "attn_block",
    return_logits: bool = False,
):
    """
    Inject ONLY at the padded location for linear regression task.
    
    Assumes "mapsto" padding layout:
      - Pattern: [data_0, PAD, target_0, data_1, PAD, target_1, ...]
      - Padding positions: 1, 4, 7, ... (every 3rd position starting at 1)
      - We inject at token_pos = 3 * p_idx + 1
      - Evaluate next-token prediction at that position against the next target
    
    The model forward takes (data, targets) as two separate arguments.
    """
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
    
    # ---- get task from task_pool ----
    if train_task.task_pool is None:
        raise ValueError("train_task.task_pool is None. Cannot sample from task.")
    if task_idx >= train_task.task_pool.shape[0]:
        raise ValueError(f"task_idx {task_idx} >= n_tasks {train_task.task_pool.shape[0]}")
    
    task_vector = train_task.task_pool[task_idx:task_idx+1]  # (1, n_dims, 1)
    
    # ---- generate batch ----
    # Use sample_from_task to get (data, targets) for this specific task
    # We need B samples, so we'll call it multiple times or adjust batch_size
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
    
    # Convert to sequence format to understand the structure
    # to_seq creates: [data_0, target_0, data_1, target_1, ...] -> length 2*n_points
    # With "mapsto" padding: [data_0, PAD, target_0, data_1, PAD, target_1, ...] -> length 3*n_points
    # No BOS token prepended in "mapsto" format
    seq_len_with_pad = 3 * n_points  # (data, PAD, target) * n_points
    
    if token_pos >= seq_len_with_pad:
        raise ValueError(f"token_pos out of bounds: token_pos={token_pos}, seq_len={seq_len_with_pad}")
    
    
    @torch.no_grad()
    def _run_baseline():
        out = model(demo_data, demo_target)
        # out is (B, n_points) predictions
        # We need to get the prediction at position p_idx
        # But we also need logits for the token position, so we need to hook into the transformer
        # Actually, for linear regression, the "logits" are the continuous predictions
        # We'll treat the output as logits for comparison purposes
        return out
    
    def _replace_hidden(hid: torch.Tensor, cache: dict):
        """
        Supports hid shaped (B, L, d).
        Replaces the hidden vector at the *padded* position token_pos ONLY.
        
        Important: This function clones the hidden tensor and only modifies position token_pos.
        Due to causal attention masking, positions before token_pos are unaffected by this change
        (they cannot attend to token_pos). Only token_pos itself and positions after it may be affected.
        """
        if hid.dim() != 3:
            raise RuntimeError(f"Expected 3D hidden, got shape={tuple(hid.shape)}")
        
        B_local = demo_data.size(0)
        
        if hid.size(0) == B_local:
            # (B, L, d)
            if token_pos >= hid.size(1):
                raise RuntimeError(f"token_pos={token_pos} >= L={hid.size(1)} in (B,L,d)")
            out2 = hid.clone()
            h = out2[:, token_pos, :]  # (B, d)
            setter = lambda out2_, r_: out2_.__setitem__((slice(None), token_pos, slice(None)), r_)
            cache["layout"] = "BLD"
        else:
            raise RuntimeError(f"Cannot infer layout. hid={tuple(hid.shape)}, batch B={B_local}")
        
        cache["orig_h"] = h.detach()
        
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        
        if rand_mode == "standard_normal":
            r = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=g)
        elif rand_mode == "match_std":
            std = h.std(unbiased=False) + eps
            r = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=g) * std
        elif rand_mode == "match_mean_std":
            mean = h.mean()
            std = h.std(unbiased=False) + eps
            r = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=g) * std + mean
        else:
            raise ValueError(f"Unknown rand_mode={rand_mode!r}")
        
        cache["rand_h"] = r.detach()
        setter(out2, r)
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
                return _replace_hidden(out, cache)
            if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                new0 = _replace_hidden(out[0], cache)
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
    
    # For linear regression, we compare predictions (continuous values)
    # Extract prediction at position p_idx
    base_pred_at_pos = base_preds[:, p_idx]  # (B,)
    inj_pred_at_pos = inj_preds[:, p_idx]  # (B,)
    target = demo_target[:, p_idx]  # (B,)
    
    # Compute MSE differences
    # For continuous regression, MSE is the appropriate metric
    base_mse = ((base_pred_at_pos - target) ** 2).mean().item()
    inj_mse = ((inj_pred_at_pos - target) ** 2).mean().item()
    delta_mse = inj_mse - base_mse
    
    # For consistency with discrete tasks API, we use -delta_mse as delta_target_logprob_mean
    # This way: positive = injection helped (lower error), negative = injection hurt (higher error)
    # This matches the convention where positive delta_target_logprob_mean means improvement
    delta_target_logprob_mean = -delta_mse
    
    # Also compute log probability differences for reference (treating predictions as logits in a Gaussian)
    noise_scale = getattr(config.task, "noise_scale", 1.0)
    base_logp = -0.5 * ((base_pred_at_pos - target) / noise_scale) ** 2
    inj_logp = -0.5 * ((inj_pred_at_pos - target) / noise_scale) ** 2
    delta_logp = (inj_logp - base_logp).mean().item()
    
    results = {
        "layer_idx": int(layer_idx),
        "p_idx": int(p_idx),
        "token_pos": int(token_pos),
        "task_idx": int(task_idx),
        "B": int(B),
        "hook_attr": hook_attr,
        "rand_mode": rand_mode,
        "seed": int(seed),
        "hidden_layout": cache.get("layout", None),
        
        "base_mse": float(base_mse),
        "inj_mse": float(inj_mse),
        "delta_mse": float(delta_mse),
        "delta_target_logprob_mean": float(delta_target_logprob_mean),  # -delta_mse for consistency
        "delta_logp_gaussian": float(delta_logp),  # Gaussian log-likelihood for reference
    }
    
    if return_logits:
        results.update({
            "base_pred_at_pos": base_pred_at_pos.detach(),
            "inj_pred_at_pos": inj_pred_at_pos.detach(),
            "target": target.detach(),
            "orig_hidden": cache.get("orig_h", None),
            "rand_hidden": cache.get("rand_h", None),
        })
    
    return results


# ============================================================
# Coin wrapper: aggregate over tasks at each padded position
# ============================================================

def random_injection_coin_padded_only(
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
    rand_mode: str = "standard_normal",
    verbose: bool = False,
):
    """
    For padded Coin sequences only:
      - inject only at padded token positions (odd indices 1,3,5,...)
      - iterate over all tasks in the cloned sampler (major+minor)
      - for each padded position p_idx in [0, seq_len-2], aggregate delta_target_logprob_mean over tasks

    Returns:
      out: list[float] length (seq_len-1)
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
            "random_injection_coin_padded_only requires padded coin sequences (sampler.pad must be True)."
        )

    # Ensure we include both major and minor tasks
    # total_tasks property returns n_major_tasks + n_minor_tasks
    n_major = int(getattr(sampler, "n_major_tasks", 0))
    n_minor = int(getattr(sampler, "n_minor_tasks", 0))
    n_tasks = n_major + n_minor
    task_ids = list(range(n_tasks))

    P = int(sampler.seq_len) - 1  # number of padded locations

    out = []
    per_task = []  # per_task[p_idx][task_i] = delta_target_logprob_mean

    for p_idx in range(P):
        vals = []
        for tid in task_ids:
            r = injection_test_next_token_padded_only(
                config=config,
                model=model,
                sampler=sampler,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                seed=seed,
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

        if verbose and (p_idx % 5 == 0 or p_idx == P - 1):
            print(
                f"[coin] p_idx={p_idx:02d}/{P-1:02d} "
                f"(padded token_pos={2*p_idx+1:03d}) "
                f"Δlogp({agg} over {n_tasks} tasks)={agg_val:+.4f}"
            )

    info = {"task_ids": task_ids, "per_task": per_task}
    return out, info


# ============================================================
# Latent wrapper: aggregate over tasks at each padded position
# ============================================================

def random_injection_latent_padded_only(
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
    rand_mode: str = "standard_normal",
    verbose: bool = False,
):
    """
    Same analysis as coin, but for the latent task sampler returned by get_latent_sampler.
    Requires sampler.pad == True (padded-only injection).
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

    # Get latent sampler + task count
    # Expected return: sampler, k_minor, n_tasks (but n_tasks only includes minor+ood, not major)
    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood)

    if not getattr(sampler, "pad", False):
        raise ValueError(
            "random_injection_latent_padded_only requires padded latent sequences (sampler.pad must be True)."
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

    for p_idx in range(P):
        vals = []
        for tid in task_ids:
            r = injection_test_next_token_padded_only(
                config=config,
                model=model,
                sampler=sampler,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                seed=seed,
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

        if verbose and (p_idx % 5 == 0 or p_idx == P - 1):
            print(
                f"[latent] p_idx={p_idx:02d}/{P-1:02d} "
                f"(padded token_pos={2*p_idx+1:03d}) "
                f"Δlogp({agg} over {len(task_ids)} tasks)={agg_val:+.4f}"
            )

    info = {"task_ids": task_ids, "per_task": per_task, "k_minor": int(k_minor), "n_tasks": int(n_tasks)}
    return out, info


# ============================================================
# Linear wrapper: aggregate over tasks at each padded position
# ============================================================

def random_injection_linear_padded_only(
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    step: int | None = None,
    layer_idx: int = 5,
    hook_attr: str = "attn_block",
    agg: str = "mean",           # {"mean","median"}
    seed: int = 0,
    rand_mode: str = "standard_normal",
    verbose: bool = False,
    radius: float = 2.0,  # radius for OOD task sampling
):
    """
    For linear regression task with "mapsto" padding:
      - inject only at padded token positions (positions 1, 4, 7, ...)
      - iterate over tasks: major tasks + minor tasks (up to n_minor) + OOD tasks (n_ood)
      - for each padded position p_idx in [0, n_points-1], aggregate delta_target_logprob_mean over tasks
    
    Args:
        exp_name: Experiment name
        n_minor: Number of minor tasks to include (from minor_pool)
        n_ood: Number of OOD tasks to generate
        B: Batch size
        step: Training step (default: final step)
        layer_idx: Layer index for injection
        hook_attr: Hook attribute name (default: "attn_block")
        agg: Aggregation method ("mean" or "median")
        seed: Random seed
        rand_mode: Random injection mode
        verbose: Print progress
        radius: Radius for OOD task sampling
    
    Returns:
      out: list[float] length n_points
      info: dict with task list and per-task values
    """
    # Load config/model/task
    model, train_task, config = load_model_task_config(exp_name)
    
    if step is None:
        step = getattr(config.training, "num_epochs", 1008600)
    
    # Check if model uses "mapsto" padding
    if not hasattr(model, "pad") or model.pad != "mapsto":
        raise ValueError(
            "random_injection_linear_padded_only requires model with pad='mapsto' "
            f"(got pad={getattr(model, 'pad', None)})."
        )
    
    if train_task.task_pool is None:
        raise ValueError("train_task.task_pool is None. Cannot iterate over tasks.")
    
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Create expanded task pool: major + minor + OOD
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)  # (n_major, n_dims)
    n_major = anchor_pool.shape[0]
    
    # Start with major tasks
    eval_task_pool_list = [anchor_pool]
    n_major_tasks = n_major
    
    # Add minor tasks if available
    n_minor_sampled = 0
    if train_task.minor_pool is not None and train_task.n_minor_tasks > 0:
        minor_pool = train_task.minor_pool.squeeze(-1).to(device)  # (n_minor_total, n_dims)
        k_minor = min(n_minor, train_task.n_minor_tasks)
        if train_task.n_minor_tasks > n_minor:
            # Sample subset of minor tasks
            indices = torch.randperm(train_task.n_minor_tasks, device=device)[:k_minor]
            minor_pool_sampled = minor_pool[indices]
        else:
            minor_pool_sampled = minor_pool
        eval_task_pool_list.append(minor_pool_sampled)
        n_minor_sampled = minor_pool_sampled.shape[0]
    
    # Generate OOD tasks
    if n_ood > 0:
        # Sample OOD tasks around anchor tasks
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
    eval_task_pool = torch.cat(eval_task_pool_list, dim=0)  # (n_total, n_dims)
    n_tasks = eval_task_pool.shape[0]
    task_ids = list(range(n_tasks))
    
    n_points = int(getattr(config.task, "n_points", train_task.n_points))
    P = n_points  # number of padded locations
    
    # Create a temporary task object with the expanded task pool for sampling
    # We'll modify train_task temporarily to use eval_task_pool
    original_task_pool = train_task.task_pool
    train_task.task_pool = eval_task_pool.unsqueeze(-1)  # (n_total, n_dims, 1)
    
    try:
        out = []
        per_task = []  # per_task[p_idx][task_i] = delta_target_logprob_mean
        
        for p_idx in range(P):
            vals = []
            for tid in task_ids:
                r = injection_test_next_token_linear_padded_only(
                    config=config,
                    model=model,
                    train_task=train_task,
                    layer_idx=layer_idx,
                    p_idx=p_idx,
                    task_idx=tid,
                    B=B,
                    step=step,
                    seed=seed,
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
            
            if verbose and (p_idx % 5 == 0 or p_idx == P - 1):
                print(
                    f"[linear] p_idx={p_idx:02d}/{P-1:02d} "
                    f"(padded token_pos={3*p_idx+1:03d}) "
                    f"ΔMSE({agg} over {n_tasks} tasks)={-agg_val:+.4f} "
                    f"(positive=worse, negative=better)"
                )
        
        info = {
            "task_ids": task_ids,
            "per_task": per_task,
            "n_tasks": int(n_tasks),
            "n_major": int(n_major_tasks),
            "n_minor": int(n_minor_sampled),
            "n_ood": int(n_ood),
        }
        return out, info
    finally:
        # Restore original task pool
        train_task.task_pool = original_task_pool






import matplotlib.pyplot as plt

def plot_injection_results(out):
    """
    Plot injection results.
    
    Args:
        out: Either a list of floats, or a dict with keys "major", "minor", "ood", "all"
             (for projection removal results)
    """
    # Handle dictionary format (from projection removal interventions)
    if isinstance(out, dict):
        if "all" in out:
            # Use "all" for backward compatibility
            out = out["all"]
        elif "major" in out:
            # Fallback to major if "all" not available
            out = out["major"]
        else:
            raise ValueError(f"Dictionary format not recognized. Expected keys 'all' or 'major', got {list(out.keys())}")
    
    # x-axis: which padded slot (between token i and i+1)
    x = list(range(len(out)))          # p_idx = 0..(seq_len-2)
    y = out

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("p_idx (padded slot between token i and i+1)")
    plt.ylabel("mean Δ target log-prob (inj - base)")
    plt.title("Random injection at padded positions only")
    plt.axhline(0.0, linestyle="--")
    plt.show()


def plot_injection_results_linear(out):
    """
    Plot injection results for linear regression task.
    
    Args:
        out: List of aggregated delta_target_logprob_mean values from random_injection_linear_padded_only
             Note: These are -delta_mse values, so positive = injection helped (lower error),
             negative = injection hurt (higher error)
    """
    # x-axis: which padded position (p_idx)
    x = list(range(len(out)))          # p_idx = 0..(n_points-1)
    y = out

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("p_idx (padded position index)")
    plt.ylabel("mean Δ (-MSE) (positive = better, negative = worse)")
    plt.title("Random injection at padded positions (linear regression)")
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.show()
