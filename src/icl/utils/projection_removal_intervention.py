"""
Projection removal intervention for coin, linear regression, and latent Markov tasks.

This module implements an intervention that removes the projection of hidden states
onto the majority task subspace (defined by final_task_vecs) at padded token positions.

Supported tasks:
- Coin task: even/odd padding pattern (token_pos = 2*p_idx + 1)
- Linear regression: "mapsto" padding pattern (token_pos = 3*p_idx + 1)
- Latent Markov: even/odd padding pattern (token_pos = 2*p_idx + 1, same as coin)
"""

import torch
import torch.nn.functional as F
import icl.utils.notebook_utils as nu
from icl.utils.coin_ood_analysis import get_new_sampler
from icl.utils.latent_ood_analysis import get_latent_sampler
from icl.linear.linear_path_utils import load_model_task_config


def _unwrap_logits(model_out):
    """Unwrap logits from model output (handles different output formats)."""
    if torch.is_tensor(model_out):
        return model_out
    if hasattr(model_out, "logits"):
        return model_out.logits
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0 and torch.is_tensor(model_out[0]):
        return model_out[0]
    raise RuntimeError(f"Cannot unwrap logits from type={type(model_out)}")


def _get_pad_id(sampler):
    """Get pad token ID from sampler."""
    if hasattr(sampler, "pad_id"):
        return int(getattr(sampler, "pad_id"))
    if hasattr(sampler, "num_states"):
        return int(getattr(sampler, "num_states"))
    if hasattr(sampler, "vocab_size"):
        return int(getattr(sampler, "vocab_size")) - 1
    raise AttributeError("Could not infer pad token id from sampler.")


def _generate_batch(sampler, *, mode: str, task: int, B: int):
    """Generate a batch from the sampler."""
    out = sampler.generate(mode=mode, task=task, num_samples=B)
    if isinstance(out, (tuple, list)):
        batch = out[0]
    else:
        batch = out
    return batch


def compute_final_task_vecs_coin(
    config,
    model: torch.nn.Module,
    sampler,
    layer_idx: int,
    B: int = 64,
):
    """
    Compute final_task_vecs for coin task from hidden states at the final timestep.
    
    Args:
        config: Configuration object
        model: The model to extract hidden states from
        sampler: Coin task sampler
        layer_idx: Layer index to extract hidden states from
        B: Batch size for computing task vectors
        
    Returns:
        final_task_vecs: Tensor of shape (3, n_embd) containing the final task vectors
                        for the first 3 (major) tasks at the last timestep.
    """
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(device)
    
    n_major_tasks = int(getattr(sampler, "n_major_tasks", 3))
    if n_major_tasks < 3:
        raise ValueError(f"Need at least 3 major tasks, got {n_major_tasks}")
    
    seq_len = int(getattr(sampler, "seq_len"))
    n_embd = int(config.model.emb_dim)
    
    # Extract hidden states at the final padded position for all major tasks
    # Final position: p_idx = seq_len - 2, token_pos = 2 * (seq_len - 2) + 1 = 2 * seq_len - 3
    final_p_idx = seq_len - 2
    final_token_pos = 2 * final_p_idx + 1
    
    # Collect hidden states for first 3 major tasks at final position
    hiddens_final = []  # Will be list of (B, n_embd) tensors
    
    for task_id in range(3):
        batch_data = _generate_batch(sampler, mode="testing", task=task_id, B=B).to(device)
        
        # Hook to extract hidden state at final padded position
        cache = {}
        layer = model.layers[layer_idx]
        
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                # out: (B, L, d)
                if final_token_pos >= out.size(1):
                    raise RuntimeError(f"final_token_pos={final_token_pos} >= L={out.size(1)}")
                cache["hidden"] = out[:, final_token_pos, :].detach()  # (B, d)
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                if final_token_pos >= out[0].size(1):
                    raise RuntimeError(f"final_token_pos={final_token_pos} >= L={out[0].size(1)}")
                cache["hidden"] = out[0][:, final_token_pos, :].detach()  # (B, d)
            else:
                raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = layer.attn_block.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _ = model(batch_data)
            hiddens_final.append(cache["hidden"])  # (B, n_embd)
        finally:
            handle.remove()
    
    # Stack: (3, B, n_embd)
    hiddens_final = torch.stack(hiddens_final, dim=0)
    
    # Compute mean over batch dimension: (3, n_embd)
    task_vecs = hiddens_final.mean(dim=1)
    
    # Center by subtracting global mean (mean over all 3 tasks)
    global_mean = task_vecs.mean(dim=0, keepdim=True)  # (1, n_embd)
    final_task_vecs = task_vecs - global_mean  # (3, n_embd)
    
    return final_task_vecs


def _compute_orthonormal_basis(final_task_vecs, eps=1e-8):
    """
    Compute an orthonormal basis Q for the span of final_task_vecs using QR decomposition.
    
    Args:
        final_task_vecs: Tensor of shape (3, D) where D is embedding dimension
        eps: Small value for numerical stability
        
    Returns:
        Q: Tensor of shape (D, r) where r is the rank of final_task_vecs (at most 3)
           Columns of Q form an orthonormal basis for the span of final_task_vecs.
    """
    # final_task_vecs: (3, D)
    # Transpose to (D, 3) for QR decomposition
    F = final_task_vecs.T  # (D, 3)
    
    # QR decomposition: F = Q @ R
    # Q will have shape (D, 3) with orthonormal columns
    Q, R = torch.linalg.qr(F, mode='reduced')
    
    # Handle rank deficiency: find columns of Q that correspond to non-zero diagonal of R
    # R is (3, 3) upper triangular
    diag_R = torch.diag(R).abs()
    rank = (diag_R > eps).sum().item()
    
    if rank == 0:
        # All vectors are zero (shouldn't happen in practice)
        # Return zero basis with correct shape
        D = final_task_vecs.size(1)
        return torch.zeros((D, 1), device=final_task_vecs.device, dtype=final_task_vecs.dtype)
    
    # Take first 'rank' columns of Q
    Q = Q[:, :rank]  # (D, rank)
    
    return Q


def projection_removal_test_next_token_padded_only(
    config,
    model: torch.nn.Module,
    sampler,
    final_task_vecs: torch.Tensor,
    *,
    layer_idx: int,
    p_idx: int,
    task: int,
    B: int = 64,
    mode: str = "testing",
    hook_attr: str = "attn_block",
    return_logits: bool = False,
    verbose: bool = False,
):
    """
    Remove projection onto majority task subspace at padded token positions.
    
    For each hidden vector h at a padded position:
    1. Compute its projection onto the span of final_task_vecs
    2. Subtract that projection from h
    3. Use the modified hidden state for next-token prediction
    
    Args:
        config: Configuration object
        model: The model to intervene on
        sampler: Coin task sampler (must have pad=True)
        final_task_vecs: Tensor of shape (3, n_embd) defining the majority task subspace
        layer_idx: Layer index to apply intervention
        p_idx: Padded position index (maps to token_pos = 2*p_idx + 1)
        task: Task ID to evaluate
        B: Batch size
        mode: Sampling mode
        hook_attr: Attribute name of the module to hook (default: "attn_block")
        return_logits: If True, return logits and other detailed outputs
        verbose: If True, print sanity check information
        
    Returns:
        Dictionary with metrics and results (same format as injection_test_next_token_padded_only)
    """
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded coin sequences (sampler.pad must be True).")
    
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
    
    # Validate final_task_vecs shape
    if final_task_vecs.dim() != 2 or final_task_vecs.size(0) != 3:
        raise ValueError(f"final_task_vecs must have shape (3, n_embd), got {final_task_vecs.shape}")
    
    # Ensure final_task_vecs is on the correct device and dtype
    final_task_vecs = final_task_vecs.to(device=device, dtype=torch.float32)
    
    # Compute orthonormal basis Q for the span of final_task_vecs
    Q = _compute_orthonormal_basis(final_task_vecs)  # (D, rank)
    
    # Padded token position
    token_pos = 2 * p_idx + 1  # odd index => PAD token
    
    # Generate batch
    batch_data = _generate_batch(sampler, mode=mode, task=task, B=B).to(device)
    
    L = batch_data.size(1)
    if token_pos + 1 >= L:
        raise ValueError(f"token_pos+1 out of bounds: token_pos={token_pos}, L={L}")
    
    # Sanity: confirm we're intervening at PAD tokens
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
    
    def _remove_projection(hid: torch.Tensor, cache: dict):
        """
        Remove projection onto majority task subspace from hidden states at padded position.
        
        Supports hid shaped either (B, L, d) or (L, B, d).
        Only modifies the hidden vector at the *padded* position token_pos.
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
        
        cache["orig_h"] = h.detach()  # (B, d)
        
        # Ensure Q has same dtype as h for matmul
        Q_dtype = Q.to(dtype=h.dtype)
        
        # Compute projection: proj = Q @ (Q.T @ h)
        # Q: (D, rank), h: (B, D) -> Q.T @ h: (rank, B) -> Q @ (Q.T @ h): (D, B) -> transpose to (B, D)
        QTh = Q_dtype.T @ h.T  # (rank, B)
        proj = (Q_dtype @ QTh).T  # (B, D)
        
        # Remove projection: h_new = h - proj
        h_new = h - proj
        
        cache["proj_h"] = proj.detach()
        cache["h_new"] = h_new.detach()
        
        # Sanity check: verify Q.T @ h_new is approximately zero
        # Q.T @ h_new should be approximately zero since we removed the projection
        # We compute the norm for each sample in the batch, then take mean/max
        QTh_new = Q_dtype.T @ h_new.T  # (rank, B) - projection of h_new onto Q subspace
        QTh_new_norm = QTh_new.norm(dim=0)  # (B,) - norm per sample in batch
        cache["QTh_new_norm"] = QTh_new_norm.detach()
        cache["QTh_new_norm_mean"] = float(QTh_new_norm.mean().item())
        cache["QTh_new_norm_max"] = float(QTh_new_norm.max().item())
        
        # Note: verbose printing is now handled at the wrapper level to avoid repetition
        
        setter(out2, h_new)
        return out2
    
    @torch.no_grad()
    def _run_intervened():
        cache = {}
        layer = model.layers[layer_idx]
        if not hasattr(layer, hook_attr):
            raise AttributeError(f"Layer {layer_idx} has no attribute {hook_attr!r}.")
        mod = getattr(layer, hook_attr)
        
        def hook_fn(module, inp, out):
            # out can be Tensor or tuple(Tensor, ...)
            if torch.is_tensor(out):
                return _remove_projection(out, cache)
            if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                new0 = _remove_projection(out[0], cache)
                return (new0,) + out[1:]
            raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = mod.register_forward_hook(hook_fn)
        try:
            out = model(batch_data)
            logits = _unwrap_logits(out)
        finally:
            handle.remove()
        
        return logits, cache
    
    # Run both passes
    base_logits = _run_baseline()
    int_logits, cache = _run_intervened()
    
    base_slice, target = _get_next_token_slice(base_logits)
    int_slice, _ = _get_next_token_slice(int_logits)
    
    base_logp = F.log_softmax(base_slice, dim=-1)
    int_logp = F.log_softmax(int_slice, dim=-1)
    
    base_tlp = base_logp.gather(1, target[:, None]).squeeze(1)
    int_tlp = int_logp.gather(1, target[:, None]).squeeze(1)
    delta_tlp = int_tlp - base_tlp
    
    base_top1 = base_slice.argmax(dim=-1)
    int_top1 = int_slice.argmax(dim=-1)
    
    base_acc = (base_top1 == target).float().mean().item()
    int_acc = (int_top1 == target).float().mean().item()
    top1_flip_rate = (base_top1 != int_top1).float().mean().item()
    
    base_p = base_logp.exp()
    kl = (base_p * (base_logp - int_logp)).sum(dim=-1)
    
    # Count padded tokens modified (should be B, one per batch item)
    n_padded_modified = B
    
    results = {
        "layer_idx": int(layer_idx),
        "p_idx": int(p_idx),
        "token_pos": int(token_pos),
        "task": int(task),
        "B": int(B),
        "mode": mode,
        "hook_attr": hook_attr,
        "pad_id": int(pad_id),
        "hidden_layout": cache.get("layout", None),
        "n_padded_modified": int(n_padded_modified),
        
        "base_acc": float(base_acc),
        "int_acc": float(int_acc),
        "top1_flip_rate": float(top1_flip_rate),
        
        "delta_target_logprob_mean": float(delta_tlp.mean().item()),
        "delta_target_logprob_std": float(delta_tlp.std(unbiased=False).item()),
        "kl_mean": float(kl.mean().item()),
        "kl_std": float(kl.std(unbiased=False).item()),
        
        # Sanity check metrics
        "QTh_new_norm_mean": cache.get("QTh_new_norm_mean", 0.0),
        "QTh_new_norm_max": cache.get("QTh_new_norm_max", 0.0),
    }
    
    if return_logits:
        results.update({
            "base_logits_at_pos": base_slice.detach(),
            "int_logits_at_pos": int_slice.detach(),
            "target_tokens": target.detach(),
            "orig_hidden": cache.get("orig_h", None),
            "proj_hidden": cache.get("proj_h", None),
            "h_new": cache.get("h_new", None),
            "delta_target_logprob": delta_tlp.detach(),
            "kl_per_example": kl.detach(),
            "QTh_new_norm": cache.get("QTh_new_norm", None),
        })
    
    return results


def projection_removal_coin_padded_only(
    exp_name: str,
    n_major: int | None = None,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    step: int | None = None,
    layer_idx: int = 5,
    hook_attr: str = "attn_block",
    agg: str = "mean",           # {"mean","median"}
    mode: str = "testing",
    verbose: bool = False,
    final_task_vecs: torch.Tensor | None = None,
    compute_final_task_vecs: bool = True,
):
    """
    Apply projection removal intervention at all padded positions for coin task.
    
    For each padded position p_idx, removes the projection of hidden states onto
    the majority task subspace (defined by final_task_vecs) and measures the effect
    on next-token prediction.
    
    Args:
        exp_name: Experiment name
        n_major: Number of major tasks to use (default: all available, typically 3)
        n_minor: Number of minor tasks
        n_ood: Number of OOD tasks
        B: Batch size
        step: Training step (default: final step)
        layer_idx: Layer index for intervention
        hook_attr: Hook attribute name (default: "attn_block")
        agg: Aggregation method ("mean" or "median")
        mode: Sampling mode
        verbose: Print progress and sanity checks
        final_task_vecs: Pre-computed final_task_vecs of shape (3, n_embd).
                        If None, will be computed from the model.
        compute_final_task_vecs: If True and final_task_vecs is None, compute them.
        
    Returns:
        out: dict with keys "major", "minor", "ood", "all" - each is list[float] length (seq_len-1)
        info: dict with task lists and per-task values for each group
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("coin", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    
    # Clone sampler (major+minor+OOD)
    # get_new_sampler returns: (sampler, k_minor) where k_minor is the number of original minor tasks kept
    # The sampler structure after get_new_sampler:
    #   - Major tasks: 0 to n_major_tasks-1
    #   - Minor pool (n_minor_tasks total): [OOD tasks (first n_ood), original minor tasks (last k_minor)]
    sampler, k_minor = get_new_sampler(exp_name, n_minor, n_ood)
    
    if not getattr(sampler, "pad", False):
        raise ValueError(
            "projection_removal_coin_padded_only requires padded coin sequences (sampler.pad must be True)."
        )
    
    # Compute or use provided final_task_vecs
    # IMPORTANT: final_task_vecs are ALWAYS computed from the first 3 major tasks,
    # regardless of the n_major parameter. The n_major parameter only controls
    # which tasks we evaluate, not which tasks define the majority subspace.
    if final_task_vecs is None:
        if compute_final_task_vecs:
            if verbose:
                print("Computing final_task_vecs from model (using first 3 major tasks)...")
            final_task_vecs = compute_final_task_vecs_coin(
                config=config,
                model=model,
                sampler=sampler,
                layer_idx=layer_idx,
                B=B,
            )
            if verbose:
                print(f"Computed final_task_vecs: shape {final_task_vecs.shape}")
        else:
            raise ValueError("final_task_vecs must be provided or compute_final_task_vecs must be True")
    else:
        final_task_vecs = final_task_vecs.to(device=config.device, dtype=torch.float32)
        if verbose:
            print(f"Using provided final_task_vecs: shape {final_task_vecs.shape}")
    
    # Separate tasks into major, minor, and OOD
    # Note: n_major parameter only affects which tasks we evaluate, not final_task_vecs computation
    n_major_available = int(getattr(sampler, "n_major_tasks", 0))
    n_minor_total = int(getattr(sampler, "n_minor_tasks", 0))  # Total in minor pool (OOD + original minor)
    
    if n_major is None:
        n_major_use = n_major_available
    else:
        n_major_use = min(n_major, n_major_available)
    
    # Task ID ranges
    # get_new_sampler structure: minor_pool = [OOD tasks (first n_ood), original minor tasks (last k_minor)]
    # So task IDs are:
    #   - Major: 0 to n_major_use-1
    #   - OOD: n_major_available to n_major_available + n_ood - 1 (first n_ood in minor pool)
    #   - Minor: n_major_available + n_ood to n_major_available + n_minor_total - 1 (remaining k_minor in minor pool)
    major_task_ids = list(range(n_major_use))
    ood_start = n_major_available
    ood_end = ood_start + n_ood
    ood_task_ids = list(range(ood_start, ood_end)) if n_ood > 0 else []
    minor_start = ood_end
    minor_end = minor_start + k_minor
    minor_task_ids = list(range(minor_start, minor_end)) if k_minor > 0 else []
    all_task_ids = major_task_ids + ood_task_ids + minor_task_ids
    
    P = int(sampler.seq_len) - 1  # number of padded locations
    
    out_major = []
    out_minor = []
    out_ood = []
    out_all = []
    per_task_major = []
    per_task_minor = []
    per_task_ood = []
    per_task_all = []
    
    for p_idx in range(P):
        vals_major = []
        vals_minor = []
        vals_ood = []
        vals_all = []
        # Collect sanity check metrics for verbose output
        sanity_checks = [] if (verbose and p_idx == 0) else None
        
        # Process major tasks
        for tid in major_task_ids:
            r = projection_removal_test_next_token_padded_only(
                config=config,
                model=model,
                sampler=sampler,
                final_task_vecs=final_task_vecs,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                hook_attr=hook_attr,
                verbose=False,
            )
            vals_major.append(r["delta_target_logprob_mean"])
            vals_all.append(r["delta_target_logprob_mean"])
            
            if sanity_checks is not None:
                sanity_checks.append({
                    "task": tid,
                    "group": "major",
                    "mean": r["QTh_new_norm_mean"],
                    "max": r["QTh_new_norm_max"],
                })
        
        # Process OOD tasks
        for tid in ood_task_ids:
            r = projection_removal_test_next_token_padded_only(
                config=config,
                model=model,
                sampler=sampler,
                final_task_vecs=final_task_vecs,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                hook_attr=hook_attr,
                verbose=False,
            )
            vals_ood.append(r["delta_target_logprob_mean"])
            vals_all.append(r["delta_target_logprob_mean"])
            
            if sanity_checks is not None:
                sanity_checks.append({
                    "task": tid,
                    "group": "ood",
                    "mean": r["QTh_new_norm_mean"],
                    "max": r["QTh_new_norm_max"],
                })
        
        # Process minor tasks
        for tid in minor_task_ids:
            r = projection_removal_test_next_token_padded_only(
                config=config,
                model=model,
                sampler=sampler,
                final_task_vecs=final_task_vecs,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                hook_attr=hook_attr,
                verbose=False,
            )
            vals_minor.append(r["delta_target_logprob_mean"])
            vals_all.append(r["delta_target_logprob_mean"])
            
            if sanity_checks is not None:
                sanity_checks.append({
                    "task": tid,
                    "group": "minor",
                    "mean": r["QTh_new_norm_mean"],
                    "max": r["QTh_new_norm_max"],
                })
        
        # Aggregate for each group
        def aggregate_vals(vals):
            if len(vals) == 0:
                return None
            vals_t = torch.tensor(vals, dtype=torch.float32)
            if agg == "mean":
                return float(vals_t.mean().item())
            elif agg == "median":
                return float(vals_t.median().item())
            else:
                raise ValueError(f"Unknown agg={agg!r}. Expected 'mean' or 'median'.")
        
        agg_major = aggregate_vals(vals_major)
        agg_minor = aggregate_vals(vals_minor)
        agg_ood = aggregate_vals(vals_ood)
        agg_all = aggregate_vals(vals_all)
        
        out_major.append(agg_major)
        out_minor.append(agg_minor)
        out_ood.append(agg_ood)
        out_all.append(agg_all)
        per_task_major.append(vals_major)
        per_task_minor.append(vals_minor)
        per_task_ood.append(vals_ood)
        per_task_all.append(vals_all)
        
        # Print aggregated sanity check info for first position
        if sanity_checks is not None:
            all_means = [sc["mean"] for sc in sanity_checks]
            all_maxs = [sc["max"] for sc in sanity_checks]
            print(
                f"  [p_idx={p_idx}] Q.T @ h_new norm (over {len(sanity_checks)} tasks): "
                f"mean={sum(all_means)/len(all_means):.6f} (range: {min(all_means):.6f}-{max(all_means):.6f}), "
                f"max={max(all_maxs):.6f}"
            )
        
        if verbose and (p_idx % 20 == 0 or p_idx == P - 1):
            major_str = f"major={agg_major:+.4f}" if agg_major is not None else "major=N/A"
            minor_str = f"minor={agg_minor:+.4f}" if agg_minor is not None else "minor=N/A"
            ood_str = f"ood={agg_ood:+.4f}" if agg_ood is not None else "ood=N/A"
            print(
                f"[coin] p_idx={p_idx:02d}/{P-1:02d} "
                f"(padded token_pos={2*p_idx+1:03d}) "
                f"Δlogp({agg}): {major_str}, {minor_str}, {ood_str}, all={agg_all:+.4f}"
            )
    
    out = {
        "major": out_major,
        "minor": out_minor,
        "ood": out_ood,
        "all": out_all,
    }
    
    info = {
        "major_task_ids": major_task_ids,
        "minor_task_ids": minor_task_ids,
        "ood_task_ids": ood_task_ids,
        "all_task_ids": all_task_ids,
        "per_task_major": per_task_major,
        "per_task_minor": per_task_minor,
        "per_task_ood": per_task_ood,
        "per_task_all": per_task_all,
        "final_task_vecs_shape": tuple(final_task_vecs.shape),
        "n_major": n_major_use,
        "n_minor": len(minor_task_ids),
        "n_ood": len(ood_task_ids),
        "k_minor": int(k_minor),  # Number of original minor tasks kept
    }
    return out, info


# ============================================================
# Linear Regression Task Implementation
# ============================================================

def compute_final_task_vecs_linear(
    config,
    model: torch.nn.Module,
    train_task,
    layer_idx: int,
    B: int = 64,
    step: int = 1008600,
):
    """
    Compute final_task_vecs for linear regression task from hidden states at the final timestep.
    
    Args:
        config: Configuration object
        model: The model to extract hidden states from
        train_task: Linear regression task object
        layer_idx: Layer index to extract hidden states from
        B: Batch size for computing task vectors
        step: Step for sampling data
        
    Returns:
        final_task_vecs: Tensor of shape (3, n_embd) containing the final task vectors
                        for the first 3 (major) tasks at the last timestep.
    """
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(device)
    
    if train_task.task_pool is None:
        raise ValueError("train_task.task_pool is None. Cannot compute final_task_vecs.")
    
    n_tasks = train_task.task_pool.shape[0]
    if n_tasks < 3:
        raise ValueError(f"Need at least 3 tasks, got {n_tasks}")
    
    n_points = int(getattr(config.task, "n_points", train_task.n_points))
    n_embd = int(config.model.n_embd)
    
    # Final padded position: token_pos = 3 * n_points - 2
    final_token_pos = 3 * n_points - 2
    
    # Collect hidden states for first 3 tasks at final position
    hiddens_final = []  # Will be list of (B, n_embd) tensors
    
    original_batch_size = train_task.batch_size
    train_task.batch_size = B
    
    try:
        for task_idx in range(3):
            task_vector = train_task.task_pool[task_idx:task_idx+1]  # (1, n_dims, 1)
            demo_data, demo_target = train_task.sample_from_task(task_vector, step=step)
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)
            
            # Hook to extract hidden state at final padded position
            cache = {}
            block = model.transformer.blocks[layer_idx]
            
            def hook_fn(module, inp, out):
                if torch.is_tensor(out):
                    # out: (B, L, d)
                    if final_token_pos >= out.size(1):
                        raise RuntimeError(f"final_token_pos={final_token_pos} >= L={out.size(1)}")
                    cache["hidden"] = out[:, final_token_pos, :].detach()  # (B, d)
                elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                    if final_token_pos >= out[0].size(1):
                        raise RuntimeError(f"final_token_pos={final_token_pos} >= L={out[0].size(1)}")
                    cache["hidden"] = out[0][:, final_token_pos, :].detach()  # (B, d)
                else:
                    raise RuntimeError(f"Unsupported hook output type: {type(out)}")
            
            handle = block.attn_block.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    _ = model(demo_data, demo_target)
                hiddens_final.append(cache["hidden"])  # (B, n_embd)
            finally:
                handle.remove()
    finally:
        train_task.batch_size = original_batch_size
    
    # Stack: (3, B, n_embd)
    hiddens_final = torch.stack(hiddens_final, dim=0)
    
    # Compute mean over batch dimension: (3, n_embd)
    task_vecs = hiddens_final.mean(dim=1)
    
    # Center by subtracting global mean (mean over all 3 tasks)
    global_mean = task_vecs.mean(dim=0, keepdim=True)  # (1, n_embd)
    final_task_vecs = task_vecs - global_mean  # (3, n_embd)
    
    return final_task_vecs


def projection_removal_test_next_token_linear_padded_only(
    config,
    model: torch.nn.Module,
    train_task,
    final_task_vecs: torch.Tensor,
    *,
    layer_idx: int,
    p_idx: int,
    task_idx: int,
    B: int = 64,
    step: int = 1008600,
    hook_attr: str = "attn_block",
    return_logits: bool = False,
    verbose: bool = False,
):
    """
    Remove projection onto majority task subspace at padded token positions for linear regression.
    
    Args:
        config: Configuration object
        model: The model to intervene on
        train_task: Linear regression task object
        final_task_vecs: Tensor of shape (3, n_embd) defining the majority task subspace
        layer_idx: Layer index to apply intervention
        p_idx: Padded position index (maps to token_pos = 3 * p_idx + 1)
        task_idx: Task index to evaluate
        B: Batch size
        step: Step for sampling data
        hook_attr: Attribute name of the module to hook (default: "attn_block")
        return_logits: If True, return predictions and other detailed outputs
        verbose: If True, print sanity check information
        
    Returns:
        Dictionary with metrics and results (same format as injection_test_next_token_linear_padded_only)
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
    
    # Validate final_task_vecs shape
    if final_task_vecs.dim() != 2 or final_task_vecs.size(0) != 3:
        raise ValueError(f"final_task_vecs must have shape (3, n_embd), got {final_task_vecs.shape}")
    
    # Ensure final_task_vecs is on the correct device and dtype
    final_task_vecs = final_task_vecs.to(device=device, dtype=torch.float32)
    
    # Compute orthonormal basis Q for the span of final_task_vecs
    Q = _compute_orthonormal_basis(final_task_vecs)  # (D, rank)
    
    # Padded token position (mapsto format: positions 1, 4, 7, ...)
    token_pos = 3 * p_idx + 1
    
    # Get task from task_pool
    if train_task.task_pool is None:
        raise ValueError("train_task.task_pool is None. Cannot sample from task.")
    if task_idx >= train_task.task_pool.shape[0]:
        raise ValueError(f"task_idx {task_idx} >= n_tasks {train_task.task_pool.shape[0]}")
    
    task_vector = train_task.task_pool[task_idx:task_idx+1]  # (1, n_dims, 1)
    
    # Generate batch
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
    
    def _remove_projection(hid: torch.Tensor, cache: dict):
        """
        Remove projection onto majority task subspace from hidden states at padded position.
        
        Supports hid shaped (B, L, d).
        Only modifies the hidden vector at the *padded* position token_pos.
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
        
        cache["orig_h"] = h.detach()  # (B, d)
        
        # Ensure Q has same dtype as h for matmul
        Q_dtype = Q.to(dtype=h.dtype)
        
        # Compute projection: proj = Q @ (Q.T @ h)
        QTh = Q_dtype.T @ h.T  # (rank, B)
        proj = (Q_dtype @ QTh).T  # (B, D)
        
        # Remove projection: h_new = h - proj
        h_new = h - proj
        
        cache["proj_h"] = proj.detach()
        cache["h_new"] = h_new.detach()
        
        # Sanity check: verify Q.T @ h_new is approximately zero
        QTh_new = Q_dtype.T @ h_new.T  # (rank, B)
        QTh_new_norm = QTh_new.norm(dim=0)  # (B,)
        cache["QTh_new_norm"] = QTh_new_norm.detach()
        cache["QTh_new_norm_mean"] = float(QTh_new_norm.mean().item())
        cache["QTh_new_norm_max"] = float(QTh_new_norm.max().item())
        
        if verbose:
            print(f"  [p_idx={p_idx}] Q.T @ h_new norm: mean={cache['QTh_new_norm_mean']:.6f}, "
                  f"max={cache['QTh_new_norm_max']:.6f}")
        
        setter(out2, h_new)
        return out2
    
    @torch.no_grad()
    def _run_intervened():
        cache = {}
        block = model.transformer.blocks[layer_idx]
        if not hasattr(block, hook_attr):
            raise AttributeError(f"Block {layer_idx} has no attribute {hook_attr!r}.")
        mod = getattr(block, hook_attr)
        
        def hook_fn(module, inp, out):
            # out can be Tensor or tuple(Tensor, ...)
            if torch.is_tensor(out):
                return _remove_projection(out, cache)
            if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                new0 = _remove_projection(out[0], cache)
                return (new0,) + out[1:]
            raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = mod.register_forward_hook(hook_fn)
        try:
            out = model(demo_data, demo_target)
        finally:
            handle.remove()
        
        return out, cache
    
    # Run both passes
    base_preds = _run_baseline()
    int_preds, cache = _run_intervened()
    
    # Extract prediction at position p_idx
    base_pred_at_pos = base_preds[:, p_idx]  # (B,)
    int_pred_at_pos = int_preds[:, p_idx]  # (B,)
    target = demo_target[:, p_idx]  # (B,)
    
    # Compute MSE differences
    base_mse = ((base_pred_at_pos - target) ** 2).mean().item()
    int_mse = ((int_pred_at_pos - target) ** 2).mean().item()
    delta_mse = int_mse - base_mse
    
    # For consistency with discrete tasks API, we use -delta_mse as delta_target_logprob_mean
    delta_target_logprob_mean = -delta_mse
    
    # Also compute log probability differences for reference
    noise_scale = getattr(config.task, "noise_scale", 1.0)
    base_logp = -0.5 * ((base_pred_at_pos - target) / noise_scale) ** 2
    int_logp = -0.5 * ((int_pred_at_pos - target) / noise_scale) ** 2
    delta_logp = (int_logp - base_logp).mean().item()
    
    # Count padded tokens modified (should be B, one per batch item)
    n_padded_modified = B
    
    results = {
        "layer_idx": int(layer_idx),
        "p_idx": int(p_idx),
        "token_pos": int(token_pos),
        "task_idx": int(task_idx),
        "B": int(B),
        "hook_attr": hook_attr,
        "hidden_layout": cache.get("layout", None),
        "n_padded_modified": int(n_padded_modified),
        
        "base_mse": float(base_mse),
        "int_mse": float(int_mse),
        "delta_mse": float(delta_mse),
        "delta_target_logprob_mean": float(delta_target_logprob_mean),  # -delta_mse for consistency
        "delta_logp_gaussian": float(delta_logp),  # Gaussian log-likelihood for reference
        
        # Sanity check metrics
        "QTh_new_norm_mean": cache.get("QTh_new_norm_mean", 0.0),
        "QTh_new_norm_max": cache.get("QTh_new_norm_max", 0.0),
    }
    
    if return_logits:
        results.update({
            "base_pred_at_pos": base_pred_at_pos.detach(),
            "int_pred_at_pos": int_pred_at_pos.detach(),
            "target": target.detach(),
            "orig_hidden": cache.get("orig_h", None),
            "proj_hidden": cache.get("proj_h", None),
            "h_new": cache.get("h_new", None),
            "QTh_new_norm": cache.get("QTh_new_norm", None),
        })
    
    return results


def projection_removal_linear_padded_only(
    exp_name: str,
    n_major: int | None = None,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    step: int | None = None,
    layer_idx: int = 5,
    hook_attr: str = "attn_block",
    agg: str = "mean",           # {"mean","median"}
    verbose: bool = False,
    final_task_vecs: torch.Tensor | None = None,
    compute_final_task_vecs: bool = True,
    radius: float = 2.0,  # radius for OOD task sampling
):
    """
    Apply projection removal intervention at all padded positions for linear regression task.
    
    Args:
        exp_name: Experiment name
        n_major: Number of major tasks to use (default: all available, typically 3)
        n_minor: Number of minor tasks to include
        n_ood: Number of OOD tasks to generate
        B: Batch size
        step: Training step (default: final step)
        layer_idx: Layer index for intervention
        hook_attr: Hook attribute name (default: "attn_block")
        agg: Aggregation method ("mean" or "median")
        verbose: Print progress and sanity checks
        final_task_vecs: Pre-computed final_task_vecs of shape (3, n_embd).
                        If None, will be computed from the model.
        compute_final_task_vecs: If True and final_task_vecs is None, compute them.
        radius: Radius for OOD task sampling
        
    Returns:
        out: dict with keys "major", "minor", "ood", "all" - each is list[float] length n_points
        info: dict with task lists and per-task values for each group
    """
    # Load config/model/task
    model, train_task, config = load_model_task_config(exp_name)
    
    if step is None:
        step = getattr(config.training, "num_epochs", 1008600)
    
    # Check if model uses "mapsto" padding
    if not hasattr(model, "pad") or model.pad != "mapsto":
        raise ValueError(
            "projection_removal_linear_padded_only requires model with pad='mapsto' "
            f"(got pad={getattr(model, 'pad', None)})."
        )
    
    if train_task.task_pool is None:
        raise ValueError("train_task.task_pool is None. Cannot iterate over tasks.")
    
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Compute or use provided final_task_vecs
    # IMPORTANT: final_task_vecs are ALWAYS computed from the first 3 major tasks,
    # regardless of the n_major parameter. The n_major parameter only controls
    # which tasks we evaluate, not which tasks define the majority subspace.
    if final_task_vecs is None:
        if compute_final_task_vecs:
            if verbose:
                print("Computing final_task_vecs from model (using first 3 major tasks)...")
            final_task_vecs = compute_final_task_vecs_linear(
                config=config,
                model=model,
                train_task=train_task,
                layer_idx=layer_idx,
                B=B,
                step=step,
            )
            if verbose:
                print(f"Computed final_task_vecs: shape {final_task_vecs.shape}")
        else:
            raise ValueError("final_task_vecs must be provided or compute_final_task_vecs must be True")
    else:
        final_task_vecs = final_task_vecs.to(device=config.device, dtype=torch.float32)
        if verbose:
            print(f"Using provided final_task_vecs: shape {final_task_vecs.shape}")
    
    # Create expanded task pool: major + minor + OOD (similar to random_injection_linear_padded_only)
    # Note: n_major parameter only affects which tasks we evaluate, not final_task_vecs computation
    from icl.linear.sampling import sample_points_from_balls
    
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)  # (n_major_available, n_dims)
    n_major_available = anchor_pool.shape[0]
    
    # Determine how many major tasks to use
    if n_major is None:
        n_major_use = n_major_available
    else:
        n_major_use = min(n_major, n_major_available)
    
    # Start with major tasks (use first n_major_use)
    major_pool = anchor_pool[:n_major_use]
    eval_task_pool_list = [major_pool]
    n_major_tasks = n_major_use
    
    # Add minor tasks if available
    n_minor_sampled = 0
    if train_task.minor_pool is not None and train_task.n_minor_tasks > 0:
        minor_pool = train_task.minor_pool.squeeze(-1).to(device)  # (n_minor_total, n_dims)
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
        M = n_major_use  # Use number of major tasks we're actually using
        base = n_ood // M if M > 0 else n_ood
        rem = n_ood % M if M > 0 else 0
        n_per_ball = torch.full((M,), base, dtype=torch.long, device=device)
        if rem > 0:
            n_per_ball[:rem] += 1
        
        ood_pool, _ = sample_points_from_balls(
            major_pool,  # Sample around the major tasks we're using
            r=radius,
            n_per_ball=n_per_ball,
        )
        eval_task_pool_list.append(ood_pool)
    
    # Combine all task pools
    eval_task_pool = torch.cat(eval_task_pool_list, dim=0)  # (n_total, n_dims)
    n_tasks_total = eval_task_pool.shape[0]
    
    # Define task ID ranges
    major_task_ids = list(range(n_major_tasks))
    minor_start = n_major_tasks
    minor_end = minor_start + n_minor_sampled
    minor_task_ids = list(range(minor_start, minor_end)) if n_minor_sampled > 0 else []
    ood_start = minor_end
    ood_end = ood_start + n_ood
    ood_task_ids = list(range(ood_start, ood_end)) if n_ood > 0 else []
    all_task_ids = major_task_ids + minor_task_ids + ood_task_ids
    
    n_points = int(getattr(config.task, "n_points", train_task.n_points))
    P = n_points  # number of padded locations
    
    # Create a temporary task object with the expanded task pool for sampling
    original_task_pool = train_task.task_pool
    train_task.task_pool = eval_task_pool.unsqueeze(-1)  # (n_total, n_dims, 1)
    
    try:
        out_major = []
        out_minor = []
        out_ood = []
        out_all = []
        per_task_major = []
        per_task_minor = []
        per_task_ood = []
        per_task_all = []
        
        for p_idx in range(P):
            vals_major = []
            vals_minor = []
            vals_ood = []
            vals_all = []
            task_sanity_checks = [] if (verbose and p_idx == 0) else None
            
            # Process major tasks
            for tid in major_task_ids:
                r = projection_removal_test_next_token_linear_padded_only(
                    config=config,
                    model=model,
                    train_task=train_task,
                    final_task_vecs=final_task_vecs,
                    layer_idx=layer_idx,
                    p_idx=p_idx,
                    task_idx=tid,
                    B=B,
                    step=step,
                    hook_attr=hook_attr,
                    verbose=False,
                )
                vals_major.append(r["delta_target_logprob_mean"])
                vals_all.append(r["delta_target_logprob_mean"])
                
                if task_sanity_checks is not None:
                    task_sanity_checks.append({
                        "task": tid,
                        "group": "major",
                        "mean": r["QTh_new_norm_mean"],
                        "max": r["QTh_new_norm_max"],
                    })
            
            # Process minor tasks
            for tid in minor_task_ids:
                r = projection_removal_test_next_token_linear_padded_only(
                    config=config,
                    model=model,
                    train_task=train_task,
                    final_task_vecs=final_task_vecs,
                    layer_idx=layer_idx,
                    p_idx=p_idx,
                    task_idx=tid,
                    B=B,
                    step=step,
                    hook_attr=hook_attr,
                    verbose=False,
                )
                vals_minor.append(r["delta_target_logprob_mean"])
                vals_all.append(r["delta_target_logprob_mean"])
                
                if task_sanity_checks is not None:
                    task_sanity_checks.append({
                        "task": tid,
                        "group": "minor",
                        "mean": r["QTh_new_norm_mean"],
                        "max": r["QTh_new_norm_max"],
                    })
            
            # Process OOD tasks
            for tid in ood_task_ids:
                r = projection_removal_test_next_token_linear_padded_only(
                    config=config,
                    model=model,
                    train_task=train_task,
                    final_task_vecs=final_task_vecs,
                    layer_idx=layer_idx,
                    p_idx=p_idx,
                    task_idx=tid,
                    B=B,
                    step=step,
                    hook_attr=hook_attr,
                    verbose=False,
                )
                vals_ood.append(r["delta_target_logprob_mean"])
                vals_all.append(r["delta_target_logprob_mean"])
                
                if task_sanity_checks is not None:
                    task_sanity_checks.append({
                        "task": tid,
                        "group": "ood",
                        "mean": r["QTh_new_norm_mean"],
                        "max": r["QTh_new_norm_max"],
                    })
            
            # Aggregate for each group
            def aggregate_vals(vals):
                if len(vals) == 0:
                    return None
                vals_t = torch.tensor(vals, dtype=torch.float32)
                if agg == "mean":
                    return float(vals_t.mean().item())
                elif agg == "median":
                    return float(vals_t.median().item())
                else:
                    raise ValueError(f"Unknown agg={agg!r}. Expected 'mean' or 'median'.")
            
            agg_major = aggregate_vals(vals_major)
            agg_minor = aggregate_vals(vals_minor)
            agg_ood = aggregate_vals(vals_ood)
            agg_all = aggregate_vals(vals_all)
            
            out_major.append(agg_major)
            out_minor.append(agg_minor)
            out_ood.append(agg_ood)
            out_all.append(agg_all)
            per_task_major.append(vals_major)
            per_task_minor.append(vals_minor)
            per_task_ood.append(vals_ood)
            per_task_all.append(vals_all)
            
            # Print aggregated sanity check info for first position
            if task_sanity_checks is not None:
                all_means = [sc["mean"] for sc in task_sanity_checks]
                all_maxs = [sc["max"] for sc in task_sanity_checks]
                print(
                    f"  [p_idx={p_idx}] Q.T @ h_new norm (over {len(task_sanity_checks)} tasks): "
                    f"mean={sum(all_means)/len(all_means):.6f} (range: {min(all_means):.6f}-{max(all_means):.6f}), "
                    f"max={max(all_maxs):.6f}"
                )
            
            if verbose and (p_idx % 5 == 0 or p_idx == P - 1):
                major_str = f"major={-agg_major:+.4f}" if agg_major is not None else "major=N/A"
                minor_str = f"minor={-agg_minor:+.4f}" if agg_minor is not None else "minor=N/A"
                ood_str = f"ood={-agg_ood:+.4f}" if agg_ood is not None else "ood=N/A"
                print(
                    f"[linear] p_idx={p_idx:02d}/{P-1:02d} "
                    f"(padded token_pos={3*p_idx+1:03d}) "
                    f"ΔMSE({agg}): {major_str}, {minor_str}, {ood_str}, all={-agg_all:+.4f} "
                    f"(positive=worse, negative=better)"
                )
        
        out = {
            "major": out_major,
            "minor": out_minor,
            "ood": out_ood,
            "all": out_all,
        }
        
        info = {
            "major_task_ids": major_task_ids,
            "minor_task_ids": minor_task_ids,
            "ood_task_ids": ood_task_ids,
            "all_task_ids": all_task_ids,
            "per_task_major": per_task_major,
            "per_task_minor": per_task_minor,
            "per_task_ood": per_task_ood,
            "per_task_all": per_task_all,
            "final_task_vecs_shape": tuple(final_task_vecs.shape),
            "n_major": n_major_tasks,
            "n_minor": n_minor_sampled,
            "n_ood": n_ood,
        }
        return out, info
    finally:
        # Restore original task pool
        train_task.task_pool = original_task_pool


# ============================================================
# Latent Markov Task Implementation
# ============================================================

def compute_final_task_vecs_latent(
    config,
    model: torch.nn.Module,
    sampler,
    layer_idx: int,
    B: int = 64,
):
    """
    Compute final_task_vecs for latent Markov task from hidden states at the final timestep.
    
    Args:
        config: Configuration object
        model: The model to extract hidden states from
        sampler: Latent Markov task sampler
        layer_idx: Layer index to extract hidden states from
        B: Batch size for computing task vectors
        
    Returns:
        final_task_vecs: Tensor of shape (3, n_embd) containing the final task vectors
                        for the first 3 (major) tasks at the last timestep.
    """
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(device)
    
    n_major_tasks = int(getattr(sampler, "n_major_tasks", 3))
    if n_major_tasks < 3:
        raise ValueError(f"Need at least 3 major tasks, got {n_major_tasks}")
    
    seq_len = int(getattr(sampler, "seq_len"))
    n_embd = int(config.model.emb_dim)
    
    # Extract hidden states at the final padded position for all major tasks
    # Final position: p_idx = seq_len - 2, token_pos = 2 * (seq_len - 2) + 1 = 2 * seq_len - 3
    final_p_idx = seq_len - 2
    final_token_pos = 2 * final_p_idx + 1
    
    # Collect hidden states for first 3 major tasks at final position
    hiddens_final = []  # Will be list of (B, n_embd) tensors
    
    for task_id in range(3):
        batch_data = _generate_batch(sampler, mode="testing", task=task_id, B=B).to(device)
        
        # Hook to extract hidden state at final padded position
        cache = {}
        layer = model.layers[layer_idx]
        
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                # out: (B, L, d)
                if final_token_pos >= out.size(1):
                    raise RuntimeError(f"final_token_pos={final_token_pos} >= L={out.size(1)}")
                cache["hidden"] = out[:, final_token_pos, :].detach()  # (B, d)
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                if final_token_pos >= out[0].size(1):
                    raise RuntimeError(f"final_token_pos={final_token_pos} >= L={out[0].size(1)}")
                cache["hidden"] = out[0][:, final_token_pos, :].detach()  # (B, d)
            else:
                raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = layer.attn_block.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _ = model(batch_data)
            hiddens_final.append(cache["hidden"])  # (B, n_embd)
        finally:
            handle.remove()
    
    # Stack: (3, B, n_embd)
    hiddens_final = torch.stack(hiddens_final, dim=0)
    
    # Compute mean over batch dimension: (3, n_embd)
    task_vecs = hiddens_final.mean(dim=1)
    
    # Center by subtracting global mean (mean over all 3 tasks)
    global_mean = task_vecs.mean(dim=0, keepdim=True)  # (1, n_embd)
    final_task_vecs = task_vecs - global_mean  # (3, n_embd)
    
    return final_task_vecs


def projection_removal_latent_padded_only(
    exp_name: str,
    n_major: int | None = None,
    n_minor: int = 256,
    n_ood: int = 40,
    B: int = 96,
    step: int | None = None,
    layer_idx: int = 5,
    hook_attr: str = "attn_block",
    agg: str = "mean",           # {"mean","median"}
    mode: str = "testing",
    verbose: bool = False,
    final_task_vecs: torch.Tensor | None = None,
    compute_final_task_vecs: bool = True,
):
    """
    Apply projection removal intervention at all padded positions for latent Markov task.
    
    For each padded position p_idx, removes the projection of hidden states onto
    the majority task subspace (defined by final_task_vecs) and measures the effect
    on next-token prediction.
    
    Args:
        exp_name: Experiment name
        n_major: Number of major tasks to use (default: all available, typically 3)
        n_minor: Number of minor tasks
        n_ood: Number of OOD tasks
        B: Batch size
        step: Training step (default: final step)
        layer_idx: Layer index for intervention
        hook_attr: Hook attribute name (default: "attn_block")
        agg: Aggregation method ("mean" or "median")
        mode: Sampling mode
        verbose: Print progress and sanity checks
        final_task_vecs: Pre-computed final_task_vecs of shape (3, n_embd).
                        If None, will be computed from the model.
        compute_final_task_vecs: If True and final_task_vecs is None, compute them.
        
    Returns:
        out: dict with keys "major", "minor", "ood", "all" - each is list[float] length (seq_len-1)
        info: dict with task lists and per-task values for each group
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    
    # Get latent sampler + task count
    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood)
    
    if not getattr(sampler, "pad", False):
        raise ValueError(
            "projection_removal_latent_padded_only requires padded latent sequences (sampler.pad must be True)."
        )
    
    # Compute or use provided final_task_vecs
    # IMPORTANT: final_task_vecs are ALWAYS computed from the first 3 major tasks,
    # regardless of the n_major parameter. The n_major parameter only controls
    # which tasks we evaluate, not which tasks define the majority subspace.
    if final_task_vecs is None:
        if compute_final_task_vecs:
            if verbose:
                print("Computing final_task_vecs from model (using first 3 major tasks)...")
            final_task_vecs = compute_final_task_vecs_latent(
                config=config,
                model=model,
                sampler=sampler,
                layer_idx=layer_idx,
                B=B,
            )
            if verbose:
                print(f"Computed final_task_vecs: shape {final_task_vecs.shape}")
        else:
            raise ValueError("final_task_vecs must be provided or compute_final_task_vecs must be True")
    else:
        final_task_vecs = final_task_vecs.to(device=config.device, dtype=torch.float32)
        if verbose:
            print(f"Using provided final_task_vecs: shape {final_task_vecs.shape}")
    
    # Separate tasks into major, minor, and OOD
    # Note: n_major parameter only affects which tasks we evaluate, not final_task_vecs computation
    # get_latent_sampler structure: minor_pool = [OOD tasks (first n_ood), original minor tasks (last k_minor)]
    n_major_available = int(getattr(sampler, "n_major_tasks", 0))
    n_minor_total = int(getattr(sampler, "n_minor_tasks", 0))  # Total in minor pool (OOD + original minor)
    
    if n_major is None:
        n_major_use = n_major_available
    else:
        n_major_use = min(n_major, n_major_available)
    
    # Task ID ranges
    # get_latent_sampler structure: minor_pool = [OOD tasks (first n_ood), original minor tasks (last k_minor)]
    # So task IDs are:
    #   - Major: 0 to n_major_use-1
    #   - OOD: n_major_available to n_major_available + n_ood - 1 (first n_ood in minor pool)
    #   - Minor: n_major_available + n_ood to n_major_available + n_minor_total - 1 (remaining k_minor in minor pool)
    major_task_ids = list(range(n_major_use))
    ood_start = n_major_available
    ood_end = ood_start + n_ood
    ood_task_ids = list(range(ood_start, ood_end)) if n_ood > 0 else []
    minor_start = ood_end
    minor_end = minor_start + k_minor
    minor_task_ids = list(range(minor_start, minor_end)) if k_minor > 0 else []
    all_task_ids = major_task_ids + ood_task_ids + minor_task_ids
    
    P = int(sampler.seq_len) - 1  # number of padded locations
    
    out_major = []
    out_minor = []
    out_ood = []
    out_all = []
    per_task_major = []
    per_task_minor = []
    per_task_ood = []
    per_task_all = []
    
    for p_idx in range(P):
        vals_major = []
        vals_minor = []
        vals_ood = []
        vals_all = []
        task_sanity_checks = [] if (verbose and p_idx == 0) else None
        
        # Process major tasks
        for tid in major_task_ids:
            r = projection_removal_test_next_token_padded_only(
                config=config,
                model=model,
                sampler=sampler,
                final_task_vecs=final_task_vecs,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                hook_attr=hook_attr,
                verbose=False,
            )
            vals_major.append(r["delta_target_logprob_mean"])
            vals_all.append(r["delta_target_logprob_mean"])
            
            if task_sanity_checks is not None:
                task_sanity_checks.append({
                    "task": tid,
                    "group": "major",
                    "mean": r["QTh_new_norm_mean"],
                    "max": r["QTh_new_norm_max"],
                })
        
        # Process minor tasks
        for tid in minor_task_ids:
            r = projection_removal_test_next_token_padded_only(
                config=config,
                model=model,
                sampler=sampler,
                final_task_vecs=final_task_vecs,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                hook_attr=hook_attr,
                verbose=False,
            )
            vals_minor.append(r["delta_target_logprob_mean"])
            vals_all.append(r["delta_target_logprob_mean"])
            
            if task_sanity_checks is not None:
                task_sanity_checks.append({
                    "task": tid,
                    "group": "minor",
                    "mean": r["QTh_new_norm_mean"],
                    "max": r["QTh_new_norm_max"],
                })
        
        # Process OOD tasks
        for tid in ood_task_ids:
            r = projection_removal_test_next_token_padded_only(
                config=config,
                model=model,
                sampler=sampler,
                final_task_vecs=final_task_vecs,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                hook_attr=hook_attr,
                verbose=False,
            )
            vals_ood.append(r["delta_target_logprob_mean"])
            vals_all.append(r["delta_target_logprob_mean"])
            
            if task_sanity_checks is not None:
                task_sanity_checks.append({
                    "task": tid,
                    "group": "ood",
                    "mean": r["QTh_new_norm_mean"],
                    "max": r["QTh_new_norm_max"],
                })
        
        # Aggregate for each group
        def aggregate_vals(vals):
            if len(vals) == 0:
                return None
            vals_t = torch.tensor(vals, dtype=torch.float32)
            if agg == "mean":
                return float(vals_t.mean().item())
            elif agg == "median":
                return float(vals_t.median().item())
            else:
                raise ValueError(f"Unknown agg={agg!r}. Expected 'mean' or 'median'.")
        
        agg_major = aggregate_vals(vals_major)
        agg_minor = aggregate_vals(vals_minor)
        agg_ood = aggregate_vals(vals_ood)
        agg_all = aggregate_vals(vals_all)
        
        out_major.append(agg_major)
        out_minor.append(agg_minor)
        out_ood.append(agg_ood)
        out_all.append(agg_all)
        per_task_major.append(vals_major)
        per_task_minor.append(vals_minor)
        per_task_ood.append(vals_ood)
        per_task_all.append(vals_all)
        
        # Print aggregated sanity check info for first position
        if task_sanity_checks is not None:
            all_means = [sc["mean"] for sc in task_sanity_checks]
            all_maxs = [sc["max"] for sc in task_sanity_checks]
            print(
                f"  [p_idx={p_idx}] Q.T @ h_new norm (over {len(task_sanity_checks)} tasks): "
                f"mean={sum(all_means)/len(all_means):.6f} (range: {min(all_means):.6f}-{max(all_means):.6f}), "
                f"max={max(all_maxs):.6f}"
            )
        
        if verbose and (p_idx % 5 == 0 or p_idx == P - 1):
            major_str = f"major={agg_major:+.4f}" if agg_major is not None else "major=N/A"
            minor_str = f"minor={agg_minor:+.4f}" if agg_minor is not None else "minor=N/A"
            ood_str = f"ood={agg_ood:+.4f}" if agg_ood is not None else "ood=N/A"
            print(
                f"[latent] p_idx={p_idx:02d}/{P-1:02d} "
                f"(padded token_pos={2*p_idx+1:03d}) "
                f"Δlogp({agg}): {major_str}, {minor_str}, {ood_str}, all={agg_all:+.4f}"
            )
    
    out = {
        "major": out_major,
        "minor": out_minor,
        "ood": out_ood,
        "all": out_all,
    }
    
    info = {
        "major_task_ids": major_task_ids,
        "minor_task_ids": minor_task_ids,
        "ood_task_ids": ood_task_ids,
        "all_task_ids": all_task_ids,
        "per_task_major": per_task_major,
        "per_task_minor": per_task_minor,
        "per_task_ood": per_task_ood,
        "per_task_all": per_task_all,
        "final_task_vecs_shape": tuple(final_task_vecs.shape),
        "k_minor": int(k_minor),  # Number of original minor tasks kept
        "n_major": n_major_use,
        "n_minor": len(minor_task_ids),
        "n_ood": len(ood_task_ids),
    }
    return out, info


# ============================================================
# Plotting Functions
# ============================================================

import matplotlib.pyplot as plt

def plot_projection_removal_results(out, task_name: str = "coin", show_groups: bool = True):
    """
    Plot projection removal intervention results.
    
    Args:
        out: Dictionary with keys "major", "minor", "ood" (optional), "all"
             Each value is a list of floats (may contain None values)
        task_name: Task name for title (default: "coin")
        show_groups: If True, plot separate lines for major/minor/ood groups (and hide "all")
    """
    if isinstance(out, dict):
        # New format: dictionary with separate groups
        x_all = list(range(len(out["all"])))
        
        plt.figure(figsize=(10, 6))
        
        if show_groups:
            # Plot each group separately
            if out.get("major") and any(v is not None for v in out["major"]):
                y_major = [v if v is not None else float('nan') for v in out["major"]]
                plt.plot(x_all, y_major, marker="o", label="major", alpha=0.7, linewidth=2)
            
            if out.get("minor") and any(v is not None for v in out["minor"]):
                y_minor = [v if v is not None else float('nan') for v in out["minor"]]
                plt.plot(x_all, y_minor, marker="s", label="minor", alpha=0.7, linewidth=2)
            
            if out.get("ood") and any(v is not None for v in out["ood"]):
                y_ood = [v if v is not None else float('nan') for v in out["ood"]]
                plt.plot(x_all, y_ood, marker="^", label="ood", alpha=0.7, linewidth=2)
        else:
            # When show_groups is False, plot "all" only
            y_all = [v if v is not None else float('nan') for v in out["all"]]
            plt.plot(x_all, y_all, marker=".", label="all", alpha=0.5, linewidth=1, linestyle="--")
        
        plt.xlabel("p_idx (padded slot between token i and i+1)")
        plt.ylabel("mean Δ target log-prob (intervened - base)")
        plt.title(f"Projection removal at padded positions ({task_name})")
        plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        # Old format: simple list (for backward compatibility)
        x = list(range(len(out)))
        y = out
        
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("p_idx (padded slot between token i and i+1)")
        plt.ylabel("mean Δ target log-prob (intervened - base)")
        plt.title(f"Projection removal at padded positions ({task_name})")
        plt.axhline(0.0, linestyle="--")
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_projection_removal_results_linear(out, show_groups: bool = True):
    """
    Plot projection removal intervention results for linear regression.
    
    Args:
        out: Dictionary with keys "major", "minor", "ood", "all"
             Each value is a list of floats (may contain None values)
             Note: These are -delta_mse values, so positive = better, negative = worse
        show_groups: If True, plot separate lines for major/minor/ood groups (and hide "all")
    """
    if isinstance(out, dict):
        # New format: dictionary with separate groups
        x_all = list(range(len(out["all"])))
        
        plt.figure(figsize=(10, 6))
        
        if show_groups:
            # Plot each group separately
            if out.get("major") and any(v is not None for v in out["major"]):
                y_major = [v if v is not None else float('nan') for v in out["major"]]
                # Convert to -delta_mse for display (out already contains -delta_mse)
                plt.plot(x_all, [-v if v is not None else float('nan') for v in out["major"]], 
                        marker="o", label="major", alpha=0.7, linewidth=2)
            
            if out.get("minor") and any(v is not None for v in out["minor"]):
                plt.plot(x_all, [-v if v is not None else float('nan') for v in out["minor"]], 
                        marker="s", label="minor", alpha=0.7, linewidth=2)
            
            if out.get("ood") and any(v is not None for v in out["ood"]):
                plt.plot(x_all, [-v if v is not None else float('nan') for v in out["ood"]], 
                        marker="^", label="ood", alpha=0.7, linewidth=2)
        else:
            # When show_groups is False, plot "all" only
            plt.plot(x_all, [-v if v is not None else float('nan') for v in out["all"]], 
                    marker=".", label="all", alpha=0.5, linewidth=1, linestyle="--")
        
        plt.xlabel("p_idx (padded position index)")
        plt.ylabel("mean Δ MSE (positive = worse, negative = better)")
        plt.title("Projection removal at padded positions (linear regression)")
        plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        # Old format: simple list (for backward compatibility)
        x = list(range(len(out)))
        y = [-v for v in out]  # Convert -delta_mse to delta_mse for display
        
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("p_idx (padded position index)")
        plt.ylabel("mean Δ MSE (positive = worse, negative = better)")
        plt.title("Projection removal at padded positions (linear regression)")
        plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.show()

