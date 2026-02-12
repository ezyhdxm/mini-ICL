"""
Hidden representation extraction for Coin and Latent tasks.

These tasks differ from linear regression in that:
1. Each task requires sampling a new batch of data (via sampler.generate(task=i))
2. Padding positions are at odd indices (1, 3, 5, ...) instead of 3*i+1
3. The sampler structure is different (no task_pool, uses generate() method)
"""

import torch
from typing import Sequence, Tuple, Optional
from torch import nn


@torch.no_grad()
def extract_hidden_multi_coin_latent(
    model: nn.Module,
    batch_data: torch.Tensor,
    layers: Sequence[int],
    task_pos: torch.Tensor,
) -> torch.Tensor:
    """
    Extract hidden vectors from several layers in ONE forward pass.
    
    Parameters:
    -----------
    model : nn.Module
        The model (should have model.layers[l].attn_block)
    batch_data : torch.Tensor
        Input batch data: (batch_size, seq_len)
    layers : Sequence[int]
        Which layers to extract from
    task_pos : torch.Tensor
        Positions to extract: (P,) tensor of position indices
    
    Returns:
    --------
    torch.Tensor
        Shape: (L, B, P, D) where L=len(layers), B=batch_size, P=len(task_pos), D=emb_dim
    """
    layers = list(layers)
    L = len(layers)
    out_by_layer = {}
    handles = []

    def make_hook(l):
        def hook_fn(module, inputs, output):
            # output: (batch, seq_len, d_model)
            pos = task_pos.to(output.device)
            out_by_layer[l] = output.index_select(dim=1, index=pos).detach()  # (B, P, D)
        return hook_fn

    # Register hooks
    for l in layers:
        h = model.layers[l].attn_block.register_forward_hook(make_hook(l))
        handles.append(h)

    try:
        _ = model(batch_data)
    finally:
        for h in handles:
            h.remove()

    # Stack in the requested layer order
    stacked = torch.stack([out_by_layer[l] for l in layers], dim=0)  # (L, B, P, D)
    return stacked


def compute_hiddens_multi_coin_latent(
    config,
    model: nn.Module,
    sampler,
    layers: Sequence[int] = None,
    batch_size: int = 64,
    positions_of_interest: Sequence[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute hidden representations for Coin or Latent tasks.
    
    For each task, samples a new batch of data and extracts hiddens at specified positions.
    
    Parameters:
    -----------
    config : ConfigDict
        Configuration object
    model : nn.Module
        The model (should have model.layers[l].attn_block)
    sampler : Coins or LatentMarkov
        The sampler (should have generate() method and pad attribute)
    layers : Sequence[int], optional
        Which layers to extract. If None, uses all layers.
    batch_size : int, default=64
        Batch size for sampling
    positions_of_interest : Sequence[int], optional
        Position indices (0 to seq_len-2) to analyze. If None, uses all positions.
        These are indices into the padded sequence positions [1, 3, 5, ...]
        So position 0 means the first padded token (at index 1), position 1 means second (at index 3), etc.
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_tasks, n_positions, batch_size, n_embd)
    position_info : dict
        Information about positions:
        - 'positions': list of position indices analyzed
        - 'padded_positions': list of actual sequence positions (odd indices)
        - 'seq_len': sequence length
    """
    device = config.device
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    n_embd = config.model.emb_dim
    
    if layers is None:
        layers = list(range(len(model.layers)))
    layers = list(layers)
    L = len(layers)
    
    # Check padding
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded sequences (sampler.pad must be True)")
    
    # Determine positions of interest
    if positions_of_interest is None:
        positions_of_interest = list(range(seq_len - 1))
    else:
        positions_of_interest = list(positions_of_interest)
        if not all(0 <= p < seq_len - 1 for p in positions_of_interest):
            raise ValueError(f"All positions must be in [0, {seq_len-2}]")
    
    n_positions = len(positions_of_interest)
    
    # Map position indices to actual sequence positions (odd indices: 1, 3, 5, ...)
    # position i -> sequence position 2*i + 1
    padded_positions = [2 * p + 1 for p in positions_of_interest]
    task_pos = torch.tensor(padded_positions, device=device, dtype=torch.long)
    
    # Initialize output tensor
    output_shape = (L, n_tasks, n_positions, batch_size, n_embd)
    all_hiddens = torch.empty(output_shape, dtype=torch.float32, device=device)
    
    # For each task, sample new data and extract hiddens
    for task_idx in range(n_tasks):
        # Sample new batch for this task
        demo_data, _ = sampler.generate(
            mode="testing", task=task_idx, num_samples=batch_size
        )
        demo_data = demo_data.to(device)
        
        # Extract hiddens from all specified layers
        # Returns: (L, B, P, D)
        chunk_hiddens = extract_hidden_multi_coin_latent(
            model=model,
            batch_data=demo_data,
            layers=layers,
            task_pos=task_pos,
        )
        
        # Reshape: (L, B, P, D) -> (L, P, B, D)
        chunk_hiddens = chunk_hiddens.permute(0, 2, 1, 3)
        
        # Store: (L, n_positions, batch_size, n_embd)
        all_hiddens[:, task_idx] = chunk_hiddens
    
    position_info = {
        'positions': positions_of_interest,
        'padded_positions': padded_positions,
        'seq_len': seq_len,
    }
    
    return all_hiddens.detach().cpu(), position_info


@torch.no_grad()
def compute_hiddens_token_conditioned_coin_latent(
    config,
    model: nn.Module,
    sampler,
    layers: Sequence[int] = None,
    batch_size: int = 64,
    positions_of_interest: Sequence[int] = None,
    max_unique_tokens: int = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute hidden representations conditioned on fixed tokens at specific positions.
    
    For each position of interest, this function:
    1. Samples batches for all tasks to collect tokens at that position
    2. Finds unique tokens at that position
    3. For each unique token, creates modified batches where all sequences share the same token
    4. Extracts hidden representations at the padding token immediately following that position
    5. Repeats for different token values at the same position
    
    Note: For coin/latent tasks with padding:
    - Real tokens are at even positions: 0, 2, 4, 6, ...
    - PAD tokens are at odd positions: 1, 3, 5, 7, ...
    - For point index i: fixes token at sequence position 2*i (real token),
      extracts hidden at sequence position 2*i+1 (PAD token)
    
    Parameters:
    -----------
    config : ConfigDict
        Configuration object
    model : nn.Module
        The model (should have model.layers[l].attn_block)
    sampler : Coins or LatentMarkov
        The sampler (should have generate() method and pad attribute)
    layers : Sequence[int], optional
        Which layers to extract. If None, uses all layers.
    batch_size : int, default=64
        Batch size for sampling
    positions_of_interest : Sequence[int], optional
        Position indices (0 to seq_len-2) to analyze. If None, uses all positions.
        These are indices into the real token positions [0, 2, 4, ...]
        So position 0 means the first real token (at index 0), position 1 means second (at index 2), etc.
    max_unique_tokens : int, optional
        Maximum number of unique tokens to consider per position. If None, uses all unique tokens.
        Helps limit memory usage by sampling a subset.
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)
        Hidden representations for each layer, position, unique token value, task, and batch.
    token_info : dict
        Information about the tokens used:
        - 'positions': list of position indices analyzed
        - 'unique_tokens': dict mapping position -> list of unique token IDs
        - 'token_type': always "data"
        - 'n_unique_tokens': dict mapping position -> number of unique tokens
    """
    device = config.device
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    n_embd = config.model.emb_dim
    
    if layers is None:
        layers = list(range(len(model.layers)))
    layers = list(layers)
    L = len(layers)
    
    # Check padding
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded sequences (sampler.pad must be True)")
    
    # Determine positions of interest
    if positions_of_interest is None:
        positions_of_interest = list(range(seq_len - 1))
    else:
        positions_of_interest = list(positions_of_interest)
        if not all(0 <= p < seq_len - 1 for p in positions_of_interest):
            raise ValueError(f"All positions must be in [0, {seq_len-2}]")
    
    n_positions = len(positions_of_interest)
    
    # Step 1: Collect tokens at positions of interest across all tasks
    # For each task, sample a batch and collect tokens
    all_tokens_by_position = {}  # position -> list of token tensors from all tasks
    for pos_idx in positions_of_interest:
        all_tokens_by_position[pos_idx] = []
    
    for task_idx in range(n_tasks):
        # Sample batch for this task
        demo_data, _ = sampler.generate(
            mode="testing", task=task_idx, num_samples=batch_size
        )
        demo_data = demo_data.to(device)
        
        # Extract tokens at positions of interest
        # Real tokens are at even positions: 0, 2, 4, ...
        # Position i -> sequence position 2*i
        for pos_idx in positions_of_interest:
            seq_pos = 2 * pos_idx  # Real token position
            tokens = demo_data[:, seq_pos]  # (batch_size,)
            all_tokens_by_position[pos_idx].append(tokens)
    
    # Find unique tokens for each position
    unique_tokens_by_position = {}
    for pos_idx in positions_of_interest:
        tokens_list = all_tokens_by_position[pos_idx]
        all_tokens = torch.cat(tokens_list, dim=0)  # (n_tasks * batch_size,)
        
        # Find unique tokens
        unique_tokens = torch.unique(all_tokens, sorted=True)
        
        # Sample subset if max_unique_tokens is specified
        if max_unique_tokens is not None and len(unique_tokens) > max_unique_tokens:
            indices = torch.randperm(len(unique_tokens), device=unique_tokens.device)[:max_unique_tokens]
            unique_tokens = unique_tokens[indices]
        
        unique_tokens_by_position[pos_idx] = unique_tokens
    
    # Determine output shapes
    max_unique_tokens_actual = max(len(ut) for ut in unique_tokens_by_position.values())
    
    # Initialize storage for results
    results_by_layer = {}
    for l in layers:
        results_by_layer[l] = {}
        for pos_idx in positions_of_interest:
            results_by_layer[l][pos_idx] = {}
    
    # Step 2-4: For each position, create modified batches and extract hiddens
    for pos_idx in positions_of_interest:
        unique_tokens = unique_tokens_by_position[pos_idx]
        n_unique = len(unique_tokens)
        
        # Determine sequence positions
        # Fix token at position 2*pos_idx (real token), extract hidden at position 2*pos_idx+1 (PAD token)
        fix_seq_pos = 2 * pos_idx
        extract_seq_pos = 2 * pos_idx + 1
        
        # Check bounds
        if extract_seq_pos >= 2 * seq_len - 1:
            continue
        
        # For each unique token value
        for token_idx, fixed_token_value in enumerate(unique_tokens):
            # For each task
            for task_idx in range(n_tasks):
                # Sample original batch for this task
                demo_data, _ = sampler.generate(
                    mode="testing", task=task_idx, num_samples=batch_size
                )
                demo_data = demo_data.to(device)
                
                # Fix token at position fix_seq_pos
                modified_demo_data = demo_data.clone()
                modified_demo_data[:, fix_seq_pos] = fixed_token_value
                
                # Extract hidden representations at extract_seq_pos
                extract_pos_tensor = torch.tensor([extract_seq_pos], device=device, dtype=torch.long)
                chunk_hiddens = extract_hidden_multi_coin_latent(
                    model=model,
                    batch_data=modified_demo_data,
                    layers=layers,
                    task_pos=extract_pos_tensor,
                )  # (L, batch_size, 1, n_embd)
                
                # Store results
                for l_idx, l in enumerate(layers):
                    if pos_idx not in results_by_layer[l]:
                        results_by_layer[l][pos_idx] = {}
                    if token_idx not in results_by_layer[l][pos_idx]:
                        results_by_layer[l][pos_idx][token_idx] = []
                    
                    # Reshape: (batch_size, 1, n_embd) -> (batch_size, n_embd)
                    hiddens_reshaped = chunk_hiddens[l_idx, :, 0, :]  # (batch_size, n_embd)
                    results_by_layer[l][pos_idx][token_idx].append(hiddens_reshaped)
    
    # Convert results to a structured tensor format
    # Initialize output tensor
    output_shape = (L, n_positions, max_unique_tokens_actual, n_tasks, batch_size, n_embd)
    all_hiddens = torch.zeros(output_shape, dtype=torch.float32, device=device)
    
    # Fill in the results
    for l_idx, l in enumerate(layers):
        for pos_idx_idx, pos_idx in enumerate(positions_of_interest):
            n_unique = len(unique_tokens_by_position[pos_idx])
            for token_idx in range(min(n_unique, max_unique_tokens_actual)):
                # Concatenate results from all tasks
                task_results = results_by_layer[l][pos_idx][token_idx]
                if task_results:
                    combined = torch.stack(task_results, dim=0)  # (n_tasks, batch_size, n_embd)
                    # Pad or truncate to n_tasks
                    if combined.shape[0] < n_tasks:
                        padding = torch.zeros(
                            (n_tasks - combined.shape[0], batch_size, n_embd),
                            dtype=combined.dtype,
                            device=combined.device
                        )
                        combined = torch.cat([combined, padding], dim=0)
                    else:
                        combined = combined[:n_tasks]
                    
                    all_hiddens[l_idx, pos_idx_idx, token_idx] = combined
    
    # Prepare token_info
    token_info = {
        'positions': positions_of_interest,
        'unique_tokens': {pos: tokens.cpu().numpy().tolist()
                         for pos, tokens in unique_tokens_by_position.items()},
        'token_type': 'data',
        'n_unique_tokens': {pos: len(tokens) for pos, tokens in unique_tokens_by_position.items()},
    }
    
    return all_hiddens.detach().cpu(), token_info

