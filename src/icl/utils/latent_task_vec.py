from typing import Tuple
import torch
from tqdm.notebook import trange
import numpy as np
import plotly.graph_objects as go
import os

# The following functions are used to compute hidden representations from the model at different granularity and different layers.

def compute_hiddens(config,
                    model: torch.nn.Module,
                    sampler,
                    layer_index: int = 1,
                    return_final: bool = False,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract hidden representations from the attention block of a specified layer.
    
    Args:
        config: Configuration object, must contain device and model.emb_dim
        model: Transformer model
        sampler: Data sampler
        layer_index: Index of the layer to extract from
        return_final: If True, only return the hidden at the final position; 
                     if False, return all positions
    
    Returns:
        Tensor shape depends on return_final:
        - return_final=True:  (n_tasks, B, n_embd) - final position only
        - return_final=False: (n_tasks, seq_len-1, B, n_embd) - all positions
    """
    device = config.device
    B = 256 
    max_tasks = 32
    n_tasks = min(max_tasks, sampler.n_major_tasks + sampler.n_minor_tasks)
    seq_len = sampler.seq_len
    n_embd = config.model.emb_dim

    if return_final:
        output_shape = (n_tasks, B, n_embd)
        task_pos = 2 * seq_len - 1
    else:
        output_shape = (n_tasks, seq_len-1, B, n_embd) 
        task_pos = 2 * torch.arange(seq_len-1, device=device) + 1  # (seq_len-1,)

    all_hiddens = torch.empty(output_shape, device=device)

    def run_and_extract(batch_data: torch.Tensor, pos):
        cache = {}

        def hook_fn(module, inp, out):
            # out: (B, L, d_model)
            if isinstance(pos, torch.Tensor):
                # (B, P, d)
                cache['vecs'] = out.index_select(dim=1, index=pos).detach()
            else:
                # (B, d)
                cache['vecs'] = out[:, pos, :].detach()

        handle = model.layers[layer_index].attn_block.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(batch_data)
        handle.remove()
        return cache['vecs']

    for i in range(n_tasks):
        demo_data, _ = sampler.generate(mode="testing", task=i, num_samples=B)  # (B, L_in)

        vecs = run_and_extract(demo_data, task_pos)
        #   return_final=True  -> (B, n_embd)
        #   return_final=False -> (B, P, n_embd)

        if return_final:
            all_hiddens[i, :, :] = vecs  # (B, d)
        else:
            all_hiddens[i, :, :, :] = vecs.permute(1, 0, 2)  # (P, B, d)
    
    return all_hiddens


def compute_hiddens_tokenwise(
    config,
    model: torch.nn.Module,
    sampler,
    layer_index: int = 1,
    B_pool: int = 512,    
    Bmasked: int = 32,     
    max_tasks=16,
) -> torch.Tensor:
    """
    Compute hidden representations by token type. For each task, position, and token type,
    collect sufficient samples and extract the attention block output from the specified layer.
    
    Args:
        config: Configuration object, must contain device, vocab_size, and model.emb_dim
        model: Transformer model
        sampler: Data sampler, must contain seq_len, num_states, n_major_tasks, n_minor_tasks
        layer_index: Index of the layer to extract from
        B_pool: Number of samples to fetch from the sampler each time
        Bmasked: Number of samples needed per token type
        max_tasks: Maximum number of tasks to process
    
    Returns:
        hiddens: Tensor of shape (n_tasks, vocab_size_eff, seq_len-1, Bmasked, n_emb)
                where n_tasks is the actual number of tasks processed, and vocab_size_eff
                is the effective vocabulary size (excluding padding token)
    """
    device = config.device
    model_device = next(model.parameters()).device

    n_tasks = min(max_tasks, sampler.n_major_tasks + sampler.n_minor_tasks)
    seq_len = sampler.seq_len
    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
    assert n_emb is not None, "config.model.emb_dim / n_embd not found"

    vocab_size = config.vocab_size
    if getattr(sampler, "pad", False):
        vocab_size_eff = vocab_size - 1
    else:
        vocab_size_eff = vocab_size

    hiddens = torch.empty(
        (n_tasks, vocab_size_eff, seq_len - 1, Bmasked, n_emb),
        device=device,
        dtype=torch.float32,
    )

    def extract_at_pos(batch_data: torch.Tensor, pos: int) -> torch.Tensor:
        cache = {}

        def hook_fn(module, inp, out):
            # out: (B, L, d_model)
            L = out.size(1)
            if not (0 <= pos < L):
                raise RuntimeError(f"hook pos={pos} out of range L={L}")
            cache["vec"] = out[:, pos, :].detach()

        handle = model.layers[layer_index].attn_block.register_forward_hook(hook_fn)
        _ = model(batch_data)
        handle.remove()
        return cache["vec"]  # (Bcur, n_emb)

    for k in trange(n_tasks, desc="tasks"):
        for t in trange(0, seq_len - 1, leave=False, desc="positions"):
            input_pos = 2 * t
            task_pos = 2 * t + 1
            samples_pool, *_ = sampler.generate(mode="test", num_samples=B_pool, task=k)
            samples_pool = samples_pool.to(device)

            for tok in range(vocab_size_eff):
                collected = []
                while True:
                    curr = samples_pool[:, : (input_pos + 1)].clone()
                    mask = curr[:, input_pos] == tok
                    matched = samples_pool[mask]  
                    if matched.size(0) > 0:
                        collected.append(matched)

                    total = sum(x.size(0) for x in collected)
                    if total >= Bmasked:
                        break
                    else:
                        samples_pool, *_ = sampler.generate(mode="test", num_samples=B_pool, task=k)
                        samples_pool = samples_pool.to(device)

                batch = torch.cat(collected, dim=0)[:Bmasked].to(device)
                vec = extract_at_pos(batch, task_pos)
                hiddens[k, tok, t, :, :] = vec

    return hiddens



def compute_hiddens_tokenwise_mlp(
    config,
    model: torch.nn.Module,
    sampler,
    layer_index: int = 1,
    B_pool: int = 512,    
    Bmasked: int = 32,     
    max_tasks=16,
) -> torch.Tensor:
    """
    Compute hidden representations from MLP layer by token type. Similar to 
    compute_hiddens_tokenwise, but extracts MLP output instead of attention block output.
    
    Args:
        config: Configuration object, must contain device, vocab_size, and model.emb_dim
        model: Transformer model
        sampler: Data sampler, must contain seq_len, num_states, n_major_tasks, n_minor_tasks
        layer_index: Index of the layer to extract from
        B_pool: Number of samples to fetch from the sampler each time
        Bmasked: Number of samples needed per token type
        max_tasks: Maximum number of tasks to process
    
    Returns:
        hiddens: Tensor of shape (n_tasks, vocab_size_eff, seq_len-1, Bmasked, n_emb)
                where n_tasks is the actual number of tasks processed, and vocab_size_eff
                is the effective vocabulary size (excluding padding token)
    """
    device = config.device
    model_device = next(model.parameters()).device

    n_tasks = min(max_tasks, sampler.n_major_tasks + sampler.n_minor_tasks)
    seq_len = sampler.seq_len
    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
    assert n_emb is not None, "config.model.emb_dim / n_embd not found"

    vocab_size = config.vocab_size
    if getattr(sampler, "pad", False):
        vocab_size_eff = vocab_size - 1
    else:
        vocab_size_eff = vocab_size

    hiddens = torch.empty(
        (n_tasks, vocab_size_eff, seq_len - 1, Bmasked, n_emb),
        device=device,
        dtype=torch.float32,
    )

    def extract_at_pos(batch_data: torch.Tensor, pos: int) -> torch.Tensor:
        cache = {}

        def hook_fn(module, inp, out):
            # out: (B, L, d_model)
            L = out.size(1)
            if not (0 <= pos < L):
                raise RuntimeError(f"hook pos={pos} out of range L={L}")
            cache["vec"] = out[:, pos, :].detach()

        handle = model.layers[layer_index].mlp.register_forward_hook(hook_fn)
        _ = model(batch_data)
        handle.remove()
        return cache["vec"]  # (Bcur, n_emb)

    for k in trange(n_tasks, desc="tasks"):
        for t in trange(0, seq_len - 1, leave=False, desc="positions"):
            input_pos = 2 * t
            task_pos = 2 * t + 1
            samples_pool, *_ = sampler.generate(mode="test", num_samples=B_pool, task=k)
            samples_pool = samples_pool.to(device)

            for tok in range(vocab_size_eff):
                collected = []
                while True:
                    curr = samples_pool[:, : (input_pos + 1)].clone()
                    mask = curr[:, input_pos] == tok
                    matched = samples_pool[mask]  
                    if matched.size(0) > 0:
                        collected.append(matched)

                    total = sum(x.size(0) for x in collected)
                    if total >= Bmasked:
                        break
                    else:
                        samples_pool, *_ = sampler.generate(mode="test", num_samples=B_pool, task=k)
                        samples_pool = samples_pool.to(device)

                batch = torch.cat(collected, dim=0)[:Bmasked].to(device)
                vec = extract_at_pos(batch, task_pos)
                hiddens[k, tok, t, :, :] = vec

    return hiddens


def compute_hiddens_from_existing_samples(
    config,
    model: torch.nn.Module,
    sampler,
    samples: torch.Tensor,
    layer_index: int = 1,
    activation: str = "attn_block",
) -> torch.Tensor:
    """
    Compute hidden vectors from pre-generated samples for a specified layer. 
    This method avoids redundant sampling.
    
    Args:
        config: Configuration object, must contain device, and model.emb_dim or model.n_embd
        model: Transformer model
        sampler: Data sampler, used to validate seq_len
        samples: Pre-generated samples of shape (n_tasks, seq_len-1, vocab_size_eff, Bmasked, 2*seq_len-1)
        layer_index: Index of the layer to extract from
        activation: Activation layer to extract, either "attn_block" or "mlp"
    
    Returns:
        hiddens: Tensor of shape (n_tasks, vocab_size_eff, seq_len-1, Bmasked, n_emb)
    """
    device = config.device
    model_device = next(model.parameters()).device
    model.eval()

    if samples.dtype != torch.long:
        samples = samples.long()

    n_tasks = len(samples)
    seq_len = sampler.seq_len

    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
    assert n_emb is not None, "config.model.emb_dim / n_embd not found"

    vocab_size_eff = sampler.num_states
    Bmasked = samples.shape[3]

    hiddens = torch.empty(
        (n_tasks, vocab_size_eff, seq_len - 1, Bmasked, n_emb),
        device=device,
        dtype=torch.float32,
    )

    cache = {}
    def hook_fn(module, inp, out):
        pos = cache["pos"]
        cache["vec"] = out[:, pos, :].detach()

    if activation == "attn_block":
        handle = model.layers[layer_index].attn_block.register_forward_hook(hook_fn)
    elif activation == "mlp":
        handle = model.layers[layer_index].mlp.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Invalid activation: {activation}")

    for k in trange(n_tasks, desc="tasks"): 
        for t in trange(seq_len - 1, leave=False, desc="positions"):
            task_pos = 2 * t + 1
            for tok in range(vocab_size_eff):
                batch = samples[k, t, tok, :, :]
                batch = batch.to(model_device, non_blocking=True)
                cache["pos"] = task_pos
                _ = model(batch)
                vec = cache["vec"].to(device)
                hiddens[k, tok, t, :, :] = vec

    handle.remove()
    return hiddens

# Although it has `fast` in the name, it is not fast. It is still slow and need to be speed up.

from contextlib import nullcontext

@torch.inference_mode()
def compute_hiddens_from_existing_samples_fast(
    config, model, sampler, samples,
    layer_index: int = 1,
    activation: str = "attn_block",
    dtype_out: torch.dtype = torch.float32,
    v_step: int | None = None,   # Chunk size along vocab dimension
    k_step: int | None = 4,      # Chunk size along n_tasks dimension to avoid OOM
):
    """
    Extract hidden representations at specified layer positions from existing tokenwise samples.
    - Output is on CPU; model runs on its own device
    - Additional support: segment processing along the first dimension (n_tasks) to reduce peak memory usage
    """
    # ---- Device and mode setup ----
    out_device = torch.device("cpu")   # Output always goes to CPU
    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    # ---- Sample tensor normalization ----
    assert samples.device.type == "cpu", "samples must be on CPU"
    if samples.dtype != torch.long:
        samples = samples.long()

    need_gpu = (model_device.type == "cuda")
    if need_gpu:
        samples = samples.pin_memory()

    # ---- Dimension inference/validation ----
    n_tasks, Tm1, V, Bm, Sfull = samples.shape
    seq_len_from_samples = (Sfull + 1) // 2
    assert 2 * seq_len_from_samples - 1 == Sfull, "Last dimension of samples should be 2*seq_len-1"

    if sampler is not None and hasattr(sampler, "seq_len"):
        assert sampler.seq_len == seq_len_from_samples, \
            f"sampler.seq_len={getattr(sampler, 'seq_len', None)} does not match inferred {seq_len_from_samples} from samples"
    seq_len = seq_len_from_samples

    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
    assert n_emb is not None, "config.model must contain emb_dim or n_embd"

    # ---- Pre-allocate output (CPU) ----
    hiddens = torch.empty((n_tasks, V, Tm1, Bm, n_emb),
                          device=out_device, dtype=dtype_out)

    # ---- Forward hook: capture hidden vectors at target position ----
    cache = {"pos": None, "vec": None}
    def hook_fn(module, inp, out):
        pos = cache["pos"]
        cache["vec"] = out[:, pos, :].detach()

    if activation == "attn_block":
        handle = model.layers[layer_index].attn_block.register_forward_hook(hook_fn)
    elif activation == "mlp":
        handle = model.layers[layer_index].mlp.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Invalid activation: {activation}")

    # ---- Temporarily disable output_layer to avoid large logits ----
    has_output_layer = hasattr(model, "output_layer")
    if has_output_layer:
        old_head = model.output_layer
        model.output_layer = torch.nn.Identity()

    # ---- Autocast (CUDA only) ----
    if need_gpu:
        major_cc = torch.cuda.get_device_capability(model_device.index or 0)[0]
        amp_dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    try:
        # Chunk along vocab dimension
        if v_step is None:
            v_step = max(1, min(V, 128))

        # Chunk along tasks dimension (first dimension)
        if k_step is None or k_step <= 0:
            k_step = 1

        for k0 in trange(0, n_tasks, k_step):
            k1 = min(k0 + k_step, n_tasks)
            nK = k1 - k0

            # For current segment of tasks, iterate over t and chunk along vocab
            for t in range(Tm1):
                task_pos = 2 * t + 1
                cache["pos"] = task_pos
                L = task_pos + 1  # Truncate to current position

                for v0 in range(0, V, v_step):
                    v1 = min(v0 + v_step, V)

                    # 1) Extract sub-block: (nK, v_step, Bm, L)
                    block = samples[k0:k1, t, v0:v1, :, :L].contiguous()
                    # 2) Flatten batch dimension: (nK*v_step*Bm, L)
                    batch = block.view(-1, L)

                    # 3) Move to model device
                    if need_gpu:
                        batch = batch.to(model_device, non_blocking=True)

                    # 4) Forward pass, hook captures vector
                    with autocast_ctx:
                        _ = model(batch)

                    # 5) Retrieve vector from cache and write back to CPU output
                    vec = cache["vec"]  # (nK*v_step*Bm, n_emb)
                    if need_gpu:
                        vec = vec.to(out_device, dtype=dtype_out, non_blocking=True)
                    else:
                        vec = vec.to(dtype=dtype_out)

                    vec = vec.view(nK, v1 - v0, Bm, n_emb)
                    hiddens[k0:k1, v0:v1, t, :, :].copy_(vec)

                    # 6) Release temporaries
                    del vec, batch, block

                if need_gpu:
                    torch.cuda.empty_cache()

    finally:
        handle.remove()
        if has_output_layer:
            model.output_layer = old_head
        if was_training:
            model.train()

    return hiddens



















# A core visualization function. 

def project_with_r2_size(task_vecs_over_all_time, final_task_vecs, r2_scores, lambdas,
                         task_labels=None, final_labels=None, n_minors=0,
                         size_min=6, size_max=20):
    """
    Project high-dimensional task vectors to 2D plane and adjust scatter point sizes by R² scores.
    Uses final_task_vecs to construct the projection plane (via SVD to get first two principal components).
    Supports time slider for dynamically viewing projections at different timesteps.
    
    Args:
        task_vecs_over_all_time: Tensor of shape (K, T, D), task vectors for K tasks at T timesteps.
                                Last n_minors tasks are minor tasks
        final_task_vecs: Tensor of shape (3, D), 3 final reference task vectors
        r2_scores: Tensor of shape (K, T), goodness-of-fit for each point
        lambdas: Tensor of shape (K, T, 3), weights of each point on the 3 final vectors
        task_labels: List of task labels, length should be K
        final_labels: List of labels for final vectors, length should be 3
        n_minors: Number of minor tasks, default 0
        size_min: Minimum scatter point size, corresponds to minimum R²
        size_max: Maximum scatter point size, corresponds to maximum R²
    
    Returns:
        fig: Plotly Figure object containing interactive projection plot with time slider
    """

    def to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    X = to_np(task_vecs_over_all_time)
    F = to_np(final_task_vecs)
    R2 = to_np(r2_scores)
    K, T, D = X.shape

    # --- build projection plane from final_task_vecs ---
    F_center = F.mean(axis=0, keepdims=True)
    F0 = F - F_center
    U, S, Vt = np.linalg.svd(F0, full_matrices=False)
    basis = Vt[:2].T
    F_proj = (F - F_center) @ basis # (3,2)

    # if n_minors > 0:
    #    F_proj = np.concatenate([F_proj, np.zeros((n_minors, 2))], axis=0) 

    # Separate major, OOD, and minor tasks
    if n_minors > 0:
        n_non_minor = K - n_minors
        n_major = min(3, n_non_minor)  # First 3 are major tasks
        n_ood = max(0, n_non_minor - 3)  # Rest (3 to -n_minors) are OOD tasks
        
        X_major = X[:n_major]  # (n_major, T, D)
        X_ood = X[n_major:n_major+n_ood] if n_ood > 0 else None  # (n_ood, T, D)
        X_minor = X[n_major+n_ood:]  # (n_minors, T, D)
        
        R2_major = R2[:n_major]  # (n_major, T)
        R2_ood = R2[n_major:n_major+n_ood] if n_ood > 0 else None  # (n_ood, T)
        R2_minor = R2[n_major+n_ood:]  # (n_minors, T)
        
        lambdas_major = lambdas[:n_major]  # (n_major, T, 3)
        lambdas_ood = lambdas[n_major:n_major+n_ood] if n_ood > 0 else None  # (n_ood, T, 3)
        lambdas_minor = lambdas[n_major+n_ood:]  # (n_minors, T, 3)
        
        # Precompute projections for major, OOD, and minor tasks separately
        X_major_proj_list = [(X_major[:, t, :] - F_center) @ basis for t in range(T)]
        X_ood_proj_list = [(X_ood[:, t, :] - F_center) @ basis for t in range(T)] if n_ood > 0 else None
        X_minor_proj_list = [(X_minor[:, t, :] - F_center) @ basis for t in range(T)]
    else:
        # No minor tasks, separate major and OOD
        n_major = min(3, K)
        n_ood = max(0, K - 3)
        
        X_major = X[:n_major]
        X_ood = X[n_major:] if n_ood > 0 else None
        X_minor = None
        
        R2_major = R2[:n_major]
        R2_ood = R2[n_major:] if n_ood > 0 else None
        R2_minor = None
        
        lambdas_major = lambdas[:n_major]
        lambdas_ood = lambdas[n_major:] if n_ood > 0 else None
        lambdas_minor = None
        
        X_major_proj_list = [(X_major[:, t, :] - F_center) @ basis for t in range(T)]
        X_ood_proj_list = [(X_ood[:, t, :] - F_center) @ basis for t in range(T)] if n_ood > 0 else None
        X_minor_proj_list = None

    # Generate labels
    if n_minors > 0:
        if task_labels is None:
            # First 3 are major tasks, rest (3 to -n_minors) are OOD tasks
            major_labels = [f"major_{i}" for i in range(3)]
            ood_labels = [f"ood_{i}" for i in range(3, K-n_minors)]
            minor_labels = [f"minor_{i}" for i in range(n_minors)]
        else:
            major_labels = task_labels[:3]
            ood_labels = task_labels[3:K-n_minors] if K-n_minors > 3 else []
            minor_labels = [f"minor_{i}" for i in range(n_minors)]
    else:
        if task_labels is None:
            major_labels = [f"major_{i}" for i in range(3)]
            ood_labels = [f"ood_{i}" for i in range(3, K)] if K > 3 else []
        else:
            major_labels = task_labels[:3]
            ood_labels = task_labels[3:K] if K > 3 else []
        minor_labels = None
    if final_labels is None:
        final_labels = [f"final_{i}" for i in range(3)]

    # Map R2 -> size (linear scaling)
    def scale_sizes(r2_values):
        r2_min, r2_max = R2.min(), R2.max()
        if r2_max == r2_min:
            return np.full_like(r2_values, (size_min+size_max)/2.0)
        return size_min + (r2_values - r2_min) / (r2_max - r2_min) * (size_max - size_min)

    # Initial scatter at t=0
    t0 = 0
    fig = go.Figure()

    # Add major tasks trace
    X_major0 = X_major_proj_list[t0]
    lambda_major0 = lambdas_major[:, t0, :]
    sizes_major0 = scale_sizes(R2_major[:, t0])
    
    fig.add_trace(go.Scatter(
        x=X_major0[:, 0], y=X_major0[:, 1],
        mode="markers",
        name="major tasks",
        marker=dict(size=sizes_major0, opacity=0.8, sizemode="diameter", color="blue"),
        text=major_labels,
        customdata=np.concatenate([R2_major[:, t0:t0+1], lambda_major0], axis=1),
        hovertemplate=(
            "task=%{text}"
            "<br>R²=%{customdata[0]:.3f}"
            "<br>λ₁=%{customdata[1]:.3f}"
            "<br>λ₂=%{customdata[2]:.3f}"
            "<br>λ₃=%{customdata[3]:.3f}"
            "<extra></extra>"
        )
    ))

    # Add OOD tasks trace if they exist
    if X_ood_proj_list is not None and len(ood_labels) > 0:
        X_ood0 = X_ood_proj_list[t0]
        lambda_ood0 = lambdas_ood[:, t0, :]
        sizes_ood0 = scale_sizes(R2_ood[:, t0])
        
        fig.add_trace(go.Scatter(
            x=X_ood0[:, 0], y=X_ood0[:, 1],
            mode="markers",
            name="OOD tasks",
            marker=dict(size=sizes_ood0, opacity=0.8, sizemode="diameter", color="green", symbol="square"),
            text=ood_labels,
            customdata=np.concatenate([R2_ood[:, t0:t0+1], lambda_ood0], axis=1),
            hovertemplate=(
                "task=%{text}"
                "<br>R²=%{customdata[0]:.3f}"
                "<br>λ₁=%{customdata[1]:.3f}"
                "<br>λ₂=%{customdata[2]:.3f}"
                "<br>λ₃=%{customdata[3]:.3f}"
                "<extra></extra>"
            )
        ))

    # Add minor tasks trace if they exist
    if n_minors > 0:
        X_minor0 = X_minor_proj_list[t0]
        lambda_minor0 = lambdas_minor[:, t0, :]
        sizes_minor0 = scale_sizes(R2_minor[:, t0])
        
        fig.add_trace(go.Scatter(
            x=X_minor0[:, 0], y=X_minor0[:, 1],
            mode="markers",
            name="minor tasks",
            marker=dict(size=sizes_minor0, opacity=0.8, sizemode="diameter", color="red", symbol="diamond"),
            text=minor_labels,
            customdata=np.concatenate([R2_minor[:, t0:t0+1], lambda_minor0], axis=1),
            hovertemplate=(
                "task=%{text}"
                "<br>R²=%{customdata[0]:.3f}"
                "<br>λ₁=%{customdata[1]:.3f}"
                "<br>λ₂=%{customdata[2]:.3f}"
                "<br>λ₃=%{customdata[3]:.3f}"
                "<extra></extra>"
            )
        ))

    # final refs
    fig.add_trace(go.Scatter(
        x=F_proj[:,0], y=F_proj[:,1],
        mode="markers+text",
        name="final refs",
        marker=dict(size=12, symbol="star", line=dict(width=1)),
        text=final_labels,
        textposition="top center"
    ))

    # Slider steps
    steps = []
    for t in range(T):
        X_major_t = X_major_proj_list[t]
        sizes_major_t = scale_sizes(R2_major[:, t])
        custom_major_t = np.concatenate([R2_major[:, t:t+1], lambdas_major[:, t, :]], axis=1)
        
        # Build args for slider step
        x_coords = [X_major_t[:, 0]]
        y_coords = [X_major_t[:, 1]]
        custom_data = [custom_major_t]
        markers = [dict(size=sizes_major_t, opacity=0.8, sizemode="diameter", color="blue")]
        texts = [major_labels]
        
        # Add OOD tasks if they exist
        if X_ood_proj_list is not None and len(ood_labels) > 0:
            X_ood_t = X_ood_proj_list[t]
            sizes_ood_t = scale_sizes(R2_ood[:, t])
            custom_ood_t = np.concatenate([R2_ood[:, t:t+1], lambdas_ood[:, t, :]], axis=1)
            
            x_coords.append(X_ood_t[:, 0])
            y_coords.append(X_ood_t[:, 1])
            custom_data.append(custom_ood_t)
            markers.append(dict(size=sizes_ood_t, opacity=0.8, sizemode="diameter", color="green", symbol="square"))
            texts.append(ood_labels)
        
        # Add minor tasks if they exist
        if n_minors > 0:
            X_minor_t = X_minor_proj_list[t]
            sizes_minor_t = scale_sizes(R2_minor[:, t])
            custom_minor_t = np.concatenate([R2_minor[:, t:t+1], lambdas_minor[:, t, :]], axis=1)
            
            x_coords.append(X_minor_t[:, 0])
            y_coords.append(X_minor_t[:, 1])
            custom_data.append(custom_minor_t)
            markers.append(dict(size=sizes_minor_t, opacity=0.8, sizemode="diameter", color="red", symbol="diamond"))
            texts.append(minor_labels)
        
        # Add final refs
        x_coords.append(F_proj[:, 0])
        y_coords.append(F_proj[:, 1])
        custom_data.append(None)
        markers.append(None)
        texts.append(final_labels)
        
        steps.append(dict(
            method="update",
            args=[
                {"x": x_coords,
                "y": y_coords,
                "customdata": custom_data,
                "marker": markers,
                "text": texts},
                {"title": f"Projection at t={t}"}
            ],
            label=str(t)
        ))


    fig.update_layout(
        title="Projection with R² (hover & size)" + (f" - Major/OOD/Minor tasks" if n_minors > 0 or len(ood_labels) > 0 else ""),
        xaxis_title="axis 1",
        yaxis_title="axis 2",
        sliders=[dict(
            active=0,
            steps=steps,
            currentvalue=dict(prefix="t = "),
            pad=dict(t=10)
        )],
        width=800, height=650
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig




import torch
from torch import Tensor
from typing import Dict, Any, Iterable, Optional, Tuple

# ===================== 缓存构建 =====================

def _build_cache(sampler, task_k: int, device: torch.device, eps: float = 1e-12) -> Dict[str, Any]:
    b = int(sampler.num_states)
    order = int(sampler.order)

    trans_mat: Tensor = sampler.get_task_matrix(task_k).to(device)
    S = trans_mat.shape[0]
    assert S == b ** order, "trans_mat first dim must equal num_states ** order"

    # 归一化
    trans_mat = trans_mat / trans_mat.sum(dim=1, keepdim=True).clamp_min(eps)

    # next_state_idx_table
    s_idx = torch.arange(S, device=device, dtype=torch.long).unsqueeze(1)
    v_idx = torch.arange(b, device=device, dtype=torch.long).unsqueeze(0)
    base_pow_order_1 = (b ** (order - 1)) if order > 1 else 1
    tail = s_idx % base_pow_order_1
    next_state_idx_table = tail * b + v_idx

    # 解码 tokens_s / last_token_of_state
    tokens_s = torch.empty((S, order), device=device, dtype=torch.long)
    tmp = s_idx.squeeze(1).clone()
    for pos in reversed(range(order)):
        tokens_s[:, pos] = tmp % b
        tmp = tmp // b
    last_token_of_state = tokens_s[:, -1]

    cache = dict(
        device=device,
        trans_mat=trans_mat,
        next_state_idx_table=next_state_idx_table,
        tokens_s=tokens_s,
        last_token_of_state=last_token_of_state,
    )

    # 小 S 用稠密矩阵，否则用父索引结构
    S_dense_max = 4096
    if S <= S_dense_max:
        A_dense = torch.zeros((S, S), device=device)
        src = (trans_mat).reshape(-1)
        dst = next_state_idx_table.reshape(-1)
        row = torch.arange(S, device=device).unsqueeze(1).expand(S, b).reshape(-1)
        A_dense.index_put_((row, dst), src, accumulate=True)
        cache["A_dense"] = A_dense
        cache["use_dense"] = True
    else:
        child = next_state_idx_table.reshape(-1)
        parent = torch.arange(S, device=device).unsqueeze(1).expand(S, b).reshape(-1)
        w = (trans_mat).reshape(-1)
        sort_idx = torch.argsort(child)
        child_sorted = child[sort_idx]
        parent_sorted = parent[sort_idx]
        w_sorted = w[sort_idx]
        unique_child, counts = torch.unique_consecutive(child_sorted, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.tensor([0], device=device), counts[:-1]]), dim=0)
        cache.update(dict(
            parents_sorted_index=parent_sorted,
            parents_sorted_weight=w_sorted,
            child_unique=unique_child,
            child_starts=starts,
            child_counts=counts,
            use_dense=False,
        ))
    return cache


def _get_cache(sampler, task_k: int, device: torch.device, eps: float = 1e-12) -> Dict[str, Any]:
    if not hasattr(sampler, "_ffbs_cache"):
        sampler._ffbs_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}
    key = (task_k, str(device))
    if key not in sampler._ffbs_cache:
        sampler._ffbs_cache[key] = _build_cache(sampler, task_k, device, eps=eps)
    return sampler._ffbs_cache[key]


# ===================== 前向 =====================

def _forward_alphas_uniform(L: Tensor, cache: Dict[str, Any], eps: float = 1e-12) -> Tensor:
    device = L.device
    trans_mat = cache["trans_mat"]
    next_state_idx_table = cache["next_state_idx_table"]
    S, b = trans_mat.shape
    T = L.shape[0]

    alphas = torch.empty((T, S), device=device)
    alpha = (torch.full((S,), 1.0 / S, device=device) * L[0]).clamp_min(eps)
    alpha = alpha / alpha.sum().clamp_min(eps)
    alphas[0] = alpha

    flat_dst = next_state_idx_table.reshape(-1)
    for u in range(1, T):
        src = (alpha.unsqueeze(1) * trans_mat).reshape(-1)
        buf = torch.zeros(S, device=device)
        buf.index_add_(0, flat_dst, src)
        alpha = (buf * L[u]).clamp_min(eps)
        alpha = alpha / alpha.sum().clamp_min(eps)
        alphas[u] = alpha
    return alphas


# ===================== 后向 =====================

def _backward_sample_dense(alphas: Tensor, K: int, cache: Dict[str, Any], eps: float = 1e-12) -> Tensor:
    device = alphas.device
    A = cache["A_dense"]
    T, S = alphas.shape
    z_path = torch.empty((K, T), device=device, dtype=torch.long)
    z_path[:, -1] = torch.multinomial(alphas[-1], K, replacement=True)
    for u in reversed(range(T - 1)):
        alpha_u = alphas[u]
        cols = A.index_select(1, z_path[:, u + 1])
        probs = (alpha_u.unsqueeze(1) * cols).clamp_min(eps)
        probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(eps)
        z_path[:, u] = torch.multinomial(probs.t(), 1).squeeze(1)
    return z_path


def _backward_sample_grouped(alphas: Tensor, K: int, cache: Dict[str, Any], eps: float = 1e-12) -> Tensor:
    device = alphas.device
    T, S = alphas.shape
    p_idx = cache["parents_sorted_index"]
    p_w = cache["parents_sorted_weight"]
    c_u = cache["child_unique"]
    c_s = cache["child_starts"]
    c_c = cache["child_counts"]

    z_path = torch.empty((K, T), device=device, dtype=torch.long)
    z_path[:, -1] = torch.multinomial(alphas[-1], K, replacement=True)
    for u in reversed(range(T - 1)):
        alpha_u = alphas[u]
        zu1 = z_path[:, u + 1]
        uniq, inv = torch.unique(zu1, return_inverse=True)
        for g, s_next in enumerate(uniq):
            k_idx = (inv == g).nonzero(as_tuple=False).squeeze(1)
            pos = (c_u == s_next).nonzero(as_tuple=False)
            if pos.numel() == 0:
                z_path[k_idx, u] = torch.randint(0, S, (k_idx.numel(),), device=device)
                continue
            pos = pos.item()
            start = c_s[pos].item()
            length = c_c[pos].item()
            parents = p_idx[start:start + length]
            weights = p_w[start:start + length]
            probs = (alpha_u[parents] * weights).clamp_min(eps)
            probs = probs / probs.sum().clamp_min(eps)
            z_u_g = torch.multinomial(probs.expand(k_idx.numel(), -1), 1).squeeze(1)
            z_path[k_idx, u] = parents[z_u_g]
    return z_path


# ===================== 公开 API =====================

@torch.no_grad()
def generate_conditional_sample_ffbs_uniform_prior(
    sampler,
    task_k: int,
    input_pos: int,         # 偶数：token 位置
    target_token: int,
    num_samples: int,
    *,
    eps: float = 1e-12,
    device_override: Optional[str] = None,
):
    device = torch.device(device_override) if device_override is not None else sampler.device

    # === 长度定义：总长为 seq_len_padded，偶数位是 token ===
    seq_len_padded = int(sampler.seq_len)            # 例如 257
    seq_len_tokens  = (seq_len_padded + 1) // 2      # 例如 129
    order = int(sampler.order)
    b = int(sampler.num_states)
    assert 0 <= target_token < b

    # input_pos 为偶数位，对应 token_idx
    token_idx = input_pos // 2
    assert 0 <= token_idx < seq_len_tokens, f"token_idx {token_idx} out of [0,{seq_len_tokens})"

    cache = _get_cache(sampler, task_k, device, eps=eps)
    S = cache["trans_mat"].shape[0]
    last_token = cache["last_token_of_state"]

    # 窗口时间轴：z_u 以 x_{order-1+u} 结尾
    T = seq_len_tokens - order + 1
    assert T >= 1, "seq_len_tokens must be >= order"
    u_star = token_idx - (order - 1)

    # 观测似然：仅在 u_star 约束“窗口最后一位 = target_token”
    L = torch.ones((T, S), device=device)
    if 0 <= u_star < T:
        mask = (last_token == int(target_token))
        L[u_star, ~mask] = 0.0

    # 前向
    alphas = _forward_alphas_uniform(L, cache, eps=eps)

    # 后向
    if cache.get("use_dense", False):
        z_path = _backward_sample_dense(alphas, num_samples, cache, eps=eps)
    else:
        z_path = _backward_sample_grouped(alphas, num_samples, cache, eps=eps)

    # 还原 token 序列
    tokens_s = cache["tokens_s"]
    samples_tokens = torch.empty((num_samples, seq_len_tokens), device=device, dtype=torch.long)
    samples_tokens[:, :order] = tokens_s[z_path[:, 0]]
    for u in range(1, T):
        samples_tokens[:, order - 1 + u] = cache["last_token_of_state"][z_path[:, u]]

    assert (samples_tokens[:, token_idx] == target_token).all()

    # === 按照“偶数位 token，奇数位 pad”的约定拼回总长 ===
    out = torch.full((num_samples, seq_len_padded), fill_value=b, device=device, dtype=torch.long)
    out[:, ::2] = samples_tokens
    return out


from tqdm.notebook import tqdm
import torch
from typing import Iterable, Optional

@torch.no_grad()
def sample_all_positions_all_targets_multi(
    sampler,
    task_ids: Iterable[int],
    num_samples_per_condition: int,
    *,
    eps: float = 1e-12,
    device_override: Optional[str] = None,
):
    """
    多 task，全位置、全 target 的条件采样（pad=True，seq_len 表示 padding 后长度）。
    ✅ 在 Jupyter Notebook 中带漂亮的 tqdm.notebook 进度条。
    返回形状: [n_tasks, seq_len//2, b, num_samples, seq_len]
    """
    device = torch.device(device_override) if device_override is not None else sampler.device
    seq_len_padded = int(sampler.seq_len)
    seq_len_tokens = (seq_len_padded + 1) // 2
    b = int(sampler.num_states)

    task_ids = list(task_ids)
    n_tasks = len(task_ids)

    out = torch.empty(
        (n_tasks, seq_len_tokens, b, num_samples_per_condition, seq_len_padded),
        device=device,
        dtype=torch.long,
    )

    # 外层 task 进度条
    for t_idx, task_k in enumerate(tqdm(task_ids, desc="Sampling tasks", leave=True)):
        _ = _get_cache(sampler, task_k, device, eps=eps)
        # 每个 task 内部位置进度条（子条）
        for pos in tqdm(range(seq_len_tokens), desc=f"Task {task_k} positions", leave=False):
            input_pos = 2 * pos
            for v in range(b):
                out[t_idx, pos, v] = generate_conditional_sample_ffbs_uniform_prior(
                    sampler, task_k, input_pos, v, num_samples_per_condition,
                    eps=eps, device_override=str(device)
                )
    return out





def _generate_token_samples_for_vocab(args):
    """Helper function for parallel processing of vocab tokens."""
    sampler, task_k, input_pos, tok, Bmasked, B_pool = args
    batch_size = max(1, min(Bmasked, B_pool))
    collected = []
    remaining = Bmasked
    
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        samples = generate_conditional_sample(
            sampler, task_k, input_pos, tok, current_batch
        )
        collected.append(samples.cpu())
        remaining -= current_batch
    
    # Concatenate all batches
    result = torch.cat(collected, dim=0)[:Bmasked]
    return tok, result


def collect_tokenwise_batches(
    sampler,
    task_k: int,
    input_pos: int,
    B_pool: int=1024,
    Bmasked: int=32,
    parallel_vocab: bool=True,
    num_workers: int=None,
) -> torch.Tensor:
    """
    Collect tokenwise batches for a specified task and position using conditional sampling.
    This improved version uses conditional probability instead of rejection sampling.
    
    Performance improvements:
    - No rejection sampling: directly generates samples with correct conditional probability
    - Much faster for rare tokens (old method could waste many samples)
    - Optional parallel processing across vocab tokens
    
    Args:
        sampler: Data sampler
        task_k: Task index
        input_pos: Input position index (in padded space if pad=True)
        B_pool: Batch size for generating samples (kept for compatibility)
        Bmasked: Number of samples needed per token type
        parallel_vocab: If True, process different vocab tokens in parallel (faster)
        num_workers: Number of worker processes for parallel vocab processing 
                    (defaults to min(4, vocab_size) if parallel_vocab=True)
    
    Returns:
        batches: Tensor of shape (vocab_size_eff, Bmasked, 2*seq_len-1) if pad=True,
                 or (vocab_size_eff, Bmasked, seq_len) if pad=False
                Bmasked complete sequences for each token type
    """
    seq_len = sampler.seq_len
    vocab_size_eff = sampler.num_states
    
    # Determine output shape based on padding
    if sampler.pad:
        output_seq_len = 2 * seq_len - 1
    else:
        output_seq_len = seq_len
    
    # Pre-allocate output
    out = torch.empty((vocab_size_eff, Bmasked, output_seq_len), dtype=torch.long, device=sampler.device)
    
    if parallel_vocab and vocab_size_eff > 1:
        # Parallel processing across vocab tokens
        if num_workers is None:
            num_workers = min(4, vocab_size_eff, os.cpu_count() or 1)
        
        if num_workers > 1:
            # Use multiprocessing for vocab tokens
            from torch.multiprocessing import Pool, set_start_method
            try:
                set_start_method("spawn", force=True)
            except RuntimeError:
                pass
            
            args_list = [(sampler, task_k, input_pos, tok, Bmasked, B_pool) 
                        for tok in range(vocab_size_eff)]
            
            with Pool(processes=num_workers) as pool:
                results = pool.map(_generate_token_samples_for_vocab, args_list)
            
            # Assemble results
            for tok, result in results:
                out[tok] = result.to(sampler.device)
        else:
            # Fall through to sequential processing
            parallel_vocab = False
    
    if not parallel_vocab or vocab_size_eff == 1:
        # Sequential processing (or fallback)
        batch_size = max(1, min(Bmasked, B_pool))
        
        for tok in range(vocab_size_eff):
            collected = []
            remaining = Bmasked
            
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                samples = generate_conditional_sample(
                    sampler, task_k, input_pos, tok, current_batch
                )
                collected.append(samples)
                remaining -= current_batch
            
            # Concatenate all batches
            out[tok] = torch.cat(collected, dim=0)[:Bmasked]
    
    return out

import os
import torch
from multiprocessing.pool import ThreadPool  # 外层用线程池，避免嵌套进程池
from multiprocessing import get_context      # 若后续改回进程池可用
from .basic import get_hash

def worker_job(k, t, sampler, B_pool, Bmasked):
    torch.manual_seed(1337 + 1000 * k + t)
    batches = collect_tokenwise_batches(
        sampler, k, input_pos=2 * t, B_pool=B_pool, Bmasked=Bmasked
    )
    return k, t, batches.cpu()

def generate_tokenwise_samples_mp(
    sampler,
    n_tasks,
    config,
    B_pool=4096,
    Bmasked=32,
    num_workers=None,
):
    """
    Generate tokenwise samples using multi-*threading* outside (to avoid nested process pools),
    with caching to disk.
    pad=True，seq_len 表示 padding 后长度（偶数 token，奇数 pad）
    返回形状: (n_tasks, (L+1)//2, vocab, Bmasked, L)
    """
    exp_name = f"train_{get_hash(config)}"

    cur_dir = os.getcwd()
    base_path = os.path.join("..", "results", "latent") if cur_dir.endswith("notebooks") \
                else os.path.join("results", "latent")
    save_dir = os.path.join(base_path, exp_name)
    save_path = os.path.join(save_dir, "samples.pt")

    vocab = int(sampler.num_states)
    L = int(sampler.seq_len)             # 总长度（含 pad）
    L_tokens = (L + 1) // 2              # token 位置数

    # 如果已有缓存
    if os.path.exists(save_path):
        print(f"Loading cached samples from {save_path}")
        samples = torch.load(save_path, map_location="cpu")
        expected_shape = (n_tasks, L_tokens, vocab, Bmasked, L)
        if samples.shape != expected_shape:
            print(f"Warning: cached shape {samples.shape} != expected {expected_shape}. Regenerating...")
        else:
            return samples

    print(f"Generating samples and saving to {save_path}")
    samples = torch.empty((n_tasks, L_tokens, vocab, Bmasked, L), dtype=torch.long, device="cpu")

    jobs = [(k, t) for k in range(n_tasks) for t in range(L_tokens)]
    num_workers = num_workers or max(1, (os.cpu_count() or 2) // 2)
    print("Number of Workers (threads):", num_workers)

    # 外层用线程池，避免在 daemon 进程里再开进程池
    with ThreadPool(processes=num_workers) as pool:
        total_jobs = len(jobs)
        print(f"Starting threaded mapping over {total_jobs} jobs ...")
        processed = 0

        for k, t, batches in pool.starmap(
            worker_job,
            ((k, t, sampler, B_pool, Bmasked) for k, t in jobs),
            chunksize=1,
        ):
            samples[k, t] = batches
            processed += 1
            if processed % max(1, total_jobs // 10) == 0 or processed == total_jobs:
                print(f"Progress: {processed}/{total_jobs} ({processed/total_jobs:.0%})")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(samples, save_path)
    print(f"Saved samples to {save_path}")
    return samples





def generate_tokenwise_samples(
    sampler,
    n_tasks,
    B_pool: int=1024,
    Bmasked: int=32,
):
    """
    Generate tokenwise samples for all tasks and all positions. Main preprocessing function
    for generating data needed for subsequent hidden representation computation.
    
    Args:
        sampler: Data sampler
        n_tasks: Number of tasks to process
        B_pool: Number of samples to fetch from the sampler each time
        Bmasked: Number of samples needed per token type
    
    Returns:
        samples: Tensor of shape (n_tasks, seq_len-1, vocab_size_eff, Bmasked, 2*seq_len-1)
                Complete sequences for all tasks, positions, and token types
    """
    vocab_size_eff = sampler.num_states
    seq_len = sampler.seq_len
    device = sampler.device
    samples = torch.empty(
        (n_tasks, seq_len-1, vocab_size_eff, Bmasked, 2*seq_len - 1),
        device=device,
        dtype=torch.long,
    )
    
    # Use single progress bar over all iterations to reduce overhead
    total_iters = n_tasks * (seq_len - 1)
    pbar = trange(total_iters, desc="generating samples")
    
    for k in range(n_tasks):
        for t in range(seq_len-1):
            input_pos = 2 * t
    
            batches = collect_tokenwise_batches(
                        sampler,
                        k,
                        input_pos,
                        B_pool,
                        Bmasked,
                    )
    
            samples[k, t] = batches
            pbar.update(1)
    
    pbar.close()
    return samples



# Currently not used, not working as expected

@torch.no_grad()
def induction_head_score_with_pad_i_minus_1(attn: torch.Tensor,
                                            tokens: torch.Tensor,
                                            pad_id: int,
                                            reduce: str = "mean_over_batch") -> torch.Tensor:
    """
    Compute induction head (IH) score considering special matching conditions for padding tokens.
    Matching condition: x[j-2] == x[i-1] and x[j-1] is a padding token.
    This function is used to analyze whether the model has learned the induction head mechanism.
    
    Args:
        attn: Attention weights of shape (B, H, T, T2)
        tokens: Token sequences of shape (B, T)
        pad_id: ID of the padding token
        reduce: Reduction mode, "mean_over_batch" averages over batch dimension, "none" does no reduction
    
    Returns:
        scores: Shape (B, H) if reduce="none", otherwise shape (H,)
    """
    assert attn.dim() == 4 and tokens.dim() == 2
    B, H, T, T2 = attn.shape
    assert T == T2 and tokens.shape[1] == T
    dev = attn.device
    
    idx = torch.arange(T, device=dev)
    I = idx.view(T, 1).expand(T, T)          
    J = idx.view(1, T).expand(T, T)          
    j_lt_i = (J < I)                          

    is_pad_q = (tokens == pad_id)        

    prev1_is_pad = torch.zeros((B, T), dtype=torch.bool, device=dev)
    prev1_is_pad[:, 1:] = (tokens[:, :-1] == pad_id)   

    prev2 = torch.full((B, T), fill_value=-1, dtype=tokens.dtype, device=dev)
    prev2[:, 2:] = tokens[:, :-2]                       

    i_minus1 = torch.full((B, T), fill_value=-1, dtype=tokens.dtype, device=dev)
    i_minus1[:, 1:] = tokens[:, :-1]                    

    cond_q_pad   = is_pad_q[:, :, None]                         
    cond_j_lt_i  = j_lt_i[None, :, :]                           
    cond_prev1   = prev1_is_pad[:, None, :]                     
    cond_match   = (prev2[:, None, :] == i_minus1[:, :, None])  # (B,T,T)

    M = cond_q_pad & cond_j_lt_i & cond_prev1 & cond_match     

    M_f = M.float()
    num = (attn * M_f[:, None, :, :]).sum(dim=(-1, -2))     
    den = M_f.sum(dim=(-1, -2)).clamp_min(1e-8)             
    scores_bh = num / den[:, None]                           

    if reduce == "none":
        return scores_bh
    return scores_bh.mean(dim=0) 



from typing import Optional, Sequence

# Not used currently

@torch.no_grad()
def steer_at_pos_and_get_next_attn(
    model,
    batch: torch.Tensor,
    layer: int,                   # steering at layer
    pos:   int,
    final_task_vectors: torch.Tensor,   # (K, D) or (D,)
    batch_index: Optional[Sequence[int]] = None,
    eps: float = 1e-6,
    use_pinv: bool = False,
):
    """
    Perform vector steering at a specified position and capture the next layer's attention.
    Steering operation projects the feature vector at the specified position onto the orthogonal complement
    of the subspace spanned by final_task_vectors. Used to analyze the effect of removing task vector components
    on model behavior.
    
    Args:
        model: Transformer model
        batch: Input batch of shape (B, T)
        layer: Index of the layer to perform steering (must be valid and next layer must exist)
        pos: Position index to steer
        final_task_vectors: Task vectors of shape (K, D) or (D,), where K is number of tasks, D is hidden dim
        batch_index: Optional, specifies which batch samples to steer (None means all)
        eps: Numerical stability constant for matrix inversion
        use_pinv: Whether to use pseudo-inverse instead of regular inverse
    
    Returns:
        cache: Dictionary containing:
            - vec_before: Feature vector before steering, (B, D)
            - vec_after: Feature vector after steering (projected onto orthogonal complement), (B, D)
            - attn_next: Next layer's attention weights (if available), depends on model implementation
    """
    cache = {"vec_before": None, "vec_after": None, "attn_next": None}

    # ---------- basic checks ----------
    if layer < 0 or layer + 1 >= len(model.layers):
        raise IndexError(f"`layer={layer}` invalid or no layer+1 for attention")

    steer_module = model.layers[layer].attn_block
    attn_module_next = getattr(model.layers[layer+1].attn_block, "MHA", model.layers[layer+1].attn_block)

    dev  = batch.device
    dt = torch.float32

    V = final_task_vectors
    if V.dim() == 1:    # (D,) -> (1, D)
        V = V.unsqueeze(0)
    # V: (K, D)
    V = V.to(dev)
    K, Dv = V.shape
    if K == 0:
        raise ValueError("`final_task_vectors` is empty（K=0）, cannot do projection")

    A = V.t().contiguous()        # (D, K)
    G = A.t() @ A                 # (K, K)
    if use_pinv:
        G_inv = torch.linalg.pinv(G)          
    else:
        G_inv = torch.linalg.inv(G + eps * torch.eye(K, device=dev, dtype=dt))

    # ---------- steering hook ----------
    def steering_hook(module, inp, out):
        # out: (B, T, D)
        if out.dim() != 3:
            raise RuntimeError(f"Expect out to be (B,T,D), but get {tuple(out.shape)}")
        B, T, D = out.shape
        if pos < 0 or pos >= T:
            raise IndexError(f"`pos={pos}` exceeds T={T}")

        if Dv != D:
            raise ValueError(f"`final_task_vectors` dim D={Dv} inconsistent with D={D}")

        y = out.clone()

        y_pos = y[:, pos, :]               # (B, D)
        cache["vec_before"] = y_pos.detach().clone()

        Yp = y_pos @ A                     # (B, K)
        Yp = Yp @ G_inv                    # (B, K)
        Yp = Yp @ A.t()                    # (B, D)
        y_pos_perp = y_pos - Yp            # (B, D)

        if batch_index is None:
            y[:, pos, :] = y_pos_perp
            cache["vec_after"] = y_pos_perp.detach().clone()
        else:
            idx = torch.as_tensor(batch_index, device=y.device)
            y[idx, pos, :] = y_pos_perp[idx]
            cache["vec_after"] = y_pos_perp.detach().clone()

        return y

    # ---------- capture hook ----------
    def capture_hook(module, inp, out):
        attn = getattr(module, "attn", None)
        if attn is not None:
            cache["attn_next"] = attn.detach().cpu().clone()
        return out

    prev_flash = getattr(attn_module_next, "flash", None)
    if prev_flash is not None:
        attn_module_next.flash = False

    h1 = steer_module.register_forward_hook(steering_hook)
    h2 = attn_module_next.register_forward_hook(capture_hook)

    try:
        _ = model(batch)
    finally:
        h1.remove()
        h2.remove()
        if prev_flash is not None:
            attn_module_next.flash = prev_flash

    return cache