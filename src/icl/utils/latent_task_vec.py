from typing import Tuple
import torch
from tqdm.notebook import trange
import numpy as np
import plotly.graph_objects as go

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

@torch.inference_mode()
def compute_hiddens_from_existing_samples_fast(
    config, model, sampler, samples,
    layer_index=1, activation="attn_block",
    dtype_out=torch.float32,
    v_step: int | None = None,   # 代替 chunk_size，按 vocab 维切
):
    device = "cpu"                               # 关键：输出落 CPU
    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    if samples.dtype != torch.long:
        samples = samples.long()
    if samples.device.type == "cpu":
        samples = samples.pin_memory()

    n_tasks, Tm1, V, Bm, Sfull = samples.shape
    seq_len = sampler.seq_len
    assert Sfull == 2 * seq_len - 1

    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
    assert n_emb is not None

    hiddens = torch.empty((n_tasks, V, Tm1, Bm, n_emb), device=device, dtype=dtype_out)

    cache = {"pos": None, "vec": None}
    def hook_fn(module, inp, out):
        cache["vec"] = out[:, cache["pos"], :].detach()

    if activation == "attn_block":
        handle = model.layers[layer_index].attn_block.register_forward_hook(hook_fn)
    elif activation == "mlp":
        handle = model.layers[layer_index].mlp.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Invalid activation: {activation}")

    # 临时禁掉 lm_head，避免巨型 logits
    has_lm_head = hasattr(model, "lm_head")
    if has_lm_head:
        old_head = model.lm_head
        model.lm_head = torch.nn.Identity()

    try:
        # 经验：优先按 vocab 切
        if v_step is None:
            # 给个保守缺省值（可按显存调整，比如 128 或 256）
            v_step = max(1, min(V, 128))

        # bf16/FP16 自动混精
        use_bf16 = (torch.cuda.is_available() and
                    torch.cuda.get_device_capability(model_device.index or 0)[0] >= 8)
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        for t in range(Tm1):
            task_pos = 2 * t + 1
            cache["pos"] = task_pos

            L = task_pos + 1   # 截断长度

            for v0 in range(0, V, v_step):
                v1 = min(v0 + v_step, V)

                # 取 (n_tasks, v_step, Bm, L)
                block = samples[:, t, v0:v1, :, :L]     # CPU view
                block = block.reshape(-1, L).clone()    # 紧凑小块
                batch = block.to(model_device, non_blocking=True)
                del block

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _ = model(batch)  # logits 被 Identity 截掉

                vec = cache["vec"].to("cpu", dtype=dtype_out)  # (n_tasks*v_step*Bm, n_emb)
                vec = vec.view(n_tasks, v1 - v0, Bm, n_emb)

                # 写回 CPU 缓冲
                hiddens[:, v0:v1, t, :, :].copy_(vec)
                del vec, batch
            # 清理到 t 的粒度即可
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        handle.remove()
        if has_lm_head:
            model.lm_head = old_head
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




# These two functions are very slow and need to be speed up. 

def collect_tokenwise_batches(
    sampler,
    task_k: int,
    input_pos: int,
    B_pool: int=1024,
    Bmasked: int=32,
) -> torch.Tensor:
    """
    Collect tokenwise batches for a specified task and position. Efficiently filters
    the sampling pool to get the required number of samples per token type, avoiding redundant computation.
    
    Args:
        sampler: Data sampler
        task_k: Task index
        input_pos: Input position index
        B_pool: Number of samples to fetch from the sampler each time
        Bmasked: Number of samples needed per token type
    
    Returns:
        batches: Tensor of shape (vocab_size_eff, Bmasked, 2*seq_len-1)
                Bmasked complete sequences for each token type
    """
    seq_len = sampler.seq_len
    vocab_size_eff = sampler.num_states
    
    # Pre-allocate output to avoid concatenation overhead
    out = torch.empty((vocab_size_eff, Bmasked, 2*seq_len-1), dtype=torch.long, device=sampler.device)
    write_ptr = torch.zeros(vocab_size_eff, dtype=torch.int64, device=sampler.device)

    while (write_ptr < Bmasked).any():
        pool, *_ = sampler.generate(mode="test", num_samples=B_pool, task=task_k)  # (B_pool, seq_len)

        toks = pool[:, input_pos]                          # (B_pool,)
        toks_sorted, idx_sorted = torch.sort(toks)
        pool_sorted = pool[idx_sorted]                     # (B_pool, seq_len)

        uniq, counts = torch.unique_consecutive(toks_sorted, return_counts=True)
        starts = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)

        for i in range(uniq.numel()):
            u = uniq[i].item()
            if u >= vocab_size_eff:
                continue
            have = write_ptr[u].item()
            need = Bmasked - have
            if need <= 0:
                continue
            st = starts[i].item()
            ct = counts[i].item()
            take = min(need, ct)
            if take > 0:
                # Write directly into pre-allocated output tensor
                out[u, have:have+take] = pool_sorted[st:st+take]
                write_ptr[u] += take

    return out

import os
from torch.multiprocessing import Pool, set_start_method
from tqdm.notebook import trange
from .basic import get_hash


def worker_job(k, t, sampler, B_pool, Bmasked):
    torch.manual_seed(1337 + 1000 * k + t)
    batches = collect_tokenwise_batches(
        sampler, k, input_pos=2*t, B_pool=B_pool, Bmasked=Bmasked
    )
    return k, t, batches.cpu()


def generate_tokenwise_samples_mp(sampler, n_tasks, config, B_pool=4096, Bmasked=32, num_workers=None):
    """
    Generate tokenwise samples using multiprocessing, with caching to disk.
    
    Checks if samples are already saved in results/latent/{exp_name}/samples.pt.
    If found, loads and returns them. Otherwise, generates samples, saves them, and returns.
    
    Args:
        sampler: Data sampler
        n_tasks: Number of tasks to process
        config: Configuration object used to determine experiment name
        B_pool: Number of samples to fetch from the sampler each time
        Bmasked: Number of samples needed per token type
        num_workers: Number of worker processes (defaults to cpu_count() // 2)
    
    Returns:
        samples: Tensor of shape (n_tasks, seq_len-1, vocab_size_eff, Bmasked, 2*seq_len-1)
                Complete sequences for all tasks, positions, and token types
    """
    exp_name = f"train_{get_hash(config)}"
    
    # Construct save path: results/latent/{exp_name}/samples.pt
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        base_path = os.path.join("..", "results", "latent")
    else:
        base_path = os.path.join("results", "latent")
    
    save_dir = os.path.join(base_path, exp_name)
    save_path = os.path.join(save_dir, "samples.pt")
    
    # Check if samples already exist
    if os.path.exists(save_path):
        print(f"Loading cached samples from {save_path}")
        samples = torch.load(save_path, map_location="cpu")
        # Verify shape matches expected dimensions
        vocab = sampler.num_states
        L = sampler.seq_len
        expected_shape = (n_tasks, L-1, vocab, Bmasked, 2*L-1)
        if samples.shape != expected_shape:
            print(f"Warning: Cached samples shape {samples.shape} != expected {expected_shape}. Regenerating...")
        else:
            return samples
    
    # Generate samples if not cached
    print(f"Generating samples and saving to {save_path}")
    vocab = sampler.num_states
    L = sampler.seq_len
    samples = torch.empty((n_tasks, L-1, vocab, Bmasked, 2*L-1), dtype=torch.long, device="cpu")

    jobs = [(k, t) for k in range(n_tasks) for t in range(L-1)]
    num_workers = num_workers or os.cpu_count() // 2
    print("Number of Workers: ", num_workers)

    set_start_method("spawn", force=True)
    with Pool(processes=num_workers) as pool:
        with trange(len(jobs), desc="generating samples (mp)") as pbar:
            for k, t, batches in pool.starmap(
                worker_job,
                ((k, t, sampler, B_pool, Bmasked) for k, t in jobs),
                chunksize=1,
            ):
                samples[k, t] = batches
                pbar.update(1)
    
    # Save generated samples
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