import torch
from typing import Union, Tuple
from torch import nn
from ml_collections import config_flags, ConfigDict

def extract_hidden(
        model, demo_data, demo_target, l=0, task_pos: Union[int, torch.Tensor] = 1
    ):
    extracted_vector = {}

    def hook_fn(module, input, output):
        # output: (batch, seq_len, d_model)
        extracted_vector['vector'] = output[:, task_pos, :].detach()

    hook_handle = model.transformer.blocks[l].attn_block.register_forward_hook(hook_fn)
    with torch.no_grad(): _ = model(demo_data, demo_target)
    hook_handle.remove()
    return extracted_vector['vector']



def compute_hiddens(config,
                    model: torch.nn.Module,
                    train_task,
                    layer_index: int = 1,
                    chunk_size: int = 16,
                    return_final=False,
                    step=1008600) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts task vectors for all tasks using a given model and data generator.

    Parameters:
    -----------
    config : ConfigDict
        Configuration object containing task/model setup.
    model : torch.nn.Module
        The model used to extract task vectors.
    train_task : object
        Object that supports `.sample_from_task()` and has a `.task_pool` list.
    layer_index : int, default=1
        The layer at which task vectors are extracted.

    Returns:
    --------
    hiddens : torch.Tensor
        Tensor of shape (n_tasks, n_points, batch_size, n_embd)
    """
    n_tasks = train_task.task_pool.shape[0]
    n_points = config.task.n_points
    batch_size = train_task.batch_size
    n_embd = config.model.n_embd

    # model = model.to(config.device)

    demo_data = train_task.sample_data(step=step)  # shape (batch_size, n_points, n_dims)

    if return_final:
        output_shape = (n_tasks, batch_size, n_embd)
        task_pos = 3 * n_points - 2
    else:
        output_shape = (n_tasks, n_points, batch_size, n_embd)
        task_pos = 3 * torch.arange(n_points, device=config.device) + 1
    
    all_hiddens = torch.empty(output_shape, dtype=demo_data.dtype, device=config.device)

    # chunk to prevent out-of-memory errors
    for i in range(0, n_tasks, chunk_size):
        # Get a chunk of tasks
        chunk_end = min(i + chunk_size, n_tasks)
        chunk_size_actual = chunk_end - i  # calculate the actual chunk size
        
        # Repeat the demo data for the tasks in this chunk
        demo_data_repeated = demo_data.unsqueeze(0).expand(chunk_size_actual, batch_size, n_points, -1)
        demo_data_repeated = demo_data_repeated.reshape(-1, n_points, demo_data.size(-1))  # shape (chunk_size*batch_size, n_points, n_dims)
        
        # demo_data_repeated = demo_data.repeat(n_tasks, 1, 1) # shape (batch_size * n_tasks, n_points, n_dims)
        # demo_data_repeated = demo_data.unsqueeze(0).expand(n_tasks, batch_size, n_points, -1).reshape(-1, n_points, demo_data.size(-1)) # shape (n_tasks*batch_size, n_points, n_dims)
    
        # train_task.task_pool: shape (n_tasks, n_dims, 1)

        # Get the task-specific targets for this chunk
        demo_target = train_task.evaluate(demo_data, 
                                          train_task.task_pool[i:chunk_end].squeeze(-1).T, 
                                          step=step)  # (batch_size, n_points, chunk_size) or (batch_size, n_points)
        if demo_target.ndim == 3:
            demo_target = demo_target.permute(2, 0, 1).reshape(-1, n_points)  # shape (chunk_size*batch_size, n_points)
        
        chunk_hiddens = extract_hidden(
            model=model,
            demo_data=demo_data_repeated,
            demo_target=demo_target,
            l=layer_index,
            task_pos=task_pos
        )  # shape (chunk_size*batch_size, n_points, n_embd) or (chunk_size*batch_size, n_dims)

        if not return_final:
            chunk_hiddens = chunk_hiddens.reshape(chunk_size_actual, batch_size, n_points, n_embd).transpose(1, 2)
        else:
            chunk_hiddens = chunk_hiddens.reshape(chunk_size_actual, batch_size, n_embd) # (chunk_size, batch_size, n_embd)
        
        # Single CPU transfer per chunk
        all_hiddens[i:chunk_end] = chunk_hiddens
        
        # Clear GPU memory
        # del chunk_hiddens
        # torch.cuda.empty_cache()

    return all_hiddens, demo_data
    
    
def get_task_vector_from_hidden(config, model, task, layer_index=3, 
                                compute_mean=True, return_final=False):
    hiddens, _ = compute_hiddens(config,
                                 model,
                                 task,
                                 layer_index=layer_index,
                                 return_final=return_final) # (n_tasks, n_points, batch_size, n_embd) or (n_tasks, batch_size, n_embd)
    if return_final:
        global_mean = hiddens.mean(dim=(0,1), keepdim=True) # (1, 1, n_embd)
    else:
        global_mean = hiddens.mean(dim=(0,2), keepdim=True) # (1, n_points, 1, n_embd)
    task_vectors = hiddens - global_mean # (n_tasks, n_points, batch_size, n_embd) or (n_tasks, batch_size, n_embd)
    if compute_mean:
        task_vectors = task_vectors.mean(dim=-2) # (n_tasks, n_points, n_embd) or (n_tasks, n_embd)

    return hiddens, task_vectors






from typing import Union, Sequence, Tuple
import torch

@torch.no_grad()
def extract_hidden_multi(
    model,
    demo_data,
    demo_target,
    layers: Sequence[int],
    task_pos: Union[int, torch.Tensor] = 1,
    *,
    module_getter=None,   # optional override
) -> torch.Tensor:
    """
    Extract hidden vectors from several layers in ONE forward pass.

    Returns:
      - if task_pos is int:      (L, B, D)
      - if task_pos is tensor:   (L, B, P, D) where P=len(task_pos)
    """
    if module_getter is None:
        # default for your model layout
        module_getter = lambda l: model.transformer.blocks[l].attn_block

    layers = list(layers)
    out_by_layer = {}
    handles = []

    def make_hook(l):
        def hook_fn(module, inputs, output):
            # output: (batch, seq_len, d_model)
            if isinstance(task_pos, int):
                out_by_layer[l] = output[:, task_pos, :].detach()
            else:
                # task_pos: (P,) positions -> gather along seq dim
                pos = task_pos.to(output.device)
                out_by_layer[l] = output.index_select(dim=1, index=pos).detach()  # (B, P, D)
        return hook_fn

    # Register hooks
    for l in layers:
        h = module_getter(l).register_forward_hook(make_hook(l))
        handles.append(h)

    try:
        _ = model(demo_data, demo_target)
    finally:
        for h in handles:
            h.remove()

    # Stack in the requested layer order
    stacked = torch.stack([out_by_layer[l] for l in layers], dim=0)
    return stacked


def compute_hiddens_multi(
    config,
    model: torch.nn.Module,
    train_task,
    layers: Sequence[int] = list(range(16)),
    chunk_size: int = 16,
    return_final: bool = False,
    step: int = 1008600,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      - all_hiddens:
          return_final=False: (L, n_tasks, n_points, batch_size, n_embd)
          return_final=True:  (L, n_tasks, batch_size, n_embd)
      - demo_data: (batch_size, n_points, n_dims)  (same as before)
    """
    n_tasks = train_task.task_pool.shape[0]
    n_points = config.task.n_points
    batch_size = train_task.batch_size
    n_embd = config.model.n_embd
    layers = list(layers)
    L = len(layers)

    demo_data = train_task.sample_data(step=step)  # (batch_size, n_points, n_dims)

    if return_final:
        task_pos = 3 * n_points - 2
        output_shape = (L, n_tasks, batch_size, n_embd)
    else:
        task_pos = 3 * torch.arange(n_points, device=config.device) + 1
        output_shape = (L, n_tasks, n_points, batch_size, n_embd)

    # Allocate on CPU to avoid OOM for large numbers of tasks
    # We'll move chunks to GPU for processing, then move back to CPU
    all_hiddens = torch.empty(output_shape, dtype=demo_data.dtype, device='cpu')

    for i in range(0, n_tasks, chunk_size):
        chunk_end = min(i + chunk_size, n_tasks)
        chunk_size_actual = chunk_end - i

        demo_data_repeated = demo_data.unsqueeze(0).expand(
            chunk_size_actual, batch_size, n_points, -1
        ).reshape(-1, n_points, demo_data.size(-1))  # (chunk*B, n_points, n_dims)

        demo_target = train_task.evaluate(
            demo_data,
            train_task.task_pool[i:chunk_end].squeeze(-1).T,
            step=step
        )
        if demo_target.ndim == 3:
            demo_target = demo_target.permute(2, 0, 1).reshape(-1, n_points)  # (chunk*B, n_points)

        chunk_hiddens = extract_hidden_multi(
            model=model,
            demo_data=demo_data_repeated,
            demo_target=demo_target,
            layers=layers,
            task_pos=task_pos,
        )
        # chunk_hiddens:
        #   return_final=False: (L, chunk*B, P, D)
        #   return_final=True:  (L, chunk*B, D)

        if return_final:
            chunk_hiddens = chunk_hiddens.reshape(L, chunk_size_actual, batch_size, n_embd)
            # Move to CPU before assignment
            all_hiddens[:, i:chunk_end] = chunk_hiddens.cpu()
        else:
            # (L, chunk*B, P, D) -> (L, chunk, B, P, D) -> (L, chunk, P, B, D)
            chunk_hiddens = chunk_hiddens.reshape(L, chunk_size_actual, batch_size, n_points, n_embd)
            chunk_hiddens = chunk_hiddens.permute(0, 1, 3, 2, 4)
            # Move to CPU before assignment
            all_hiddens[:, i:chunk_end] = chunk_hiddens.cpu()
        
        # Clear GPU memory after each chunk
        del chunk_hiddens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_hiddens.detach(), demo_data


@torch.no_grad()
def compute_hiddens_token_conditioned(
    config,
    model: torch.nn.Module,
    train_task,
    layers: Sequence[int] = list(range(16)),
    chunk_size: int = 16,
    step: int = 1008600,
    positions_of_interest: Sequence[int] = None,
    max_unique_tokens: int = None,  # Maximum number of unique tokens to consider per position
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compute hidden representations conditioned on fixed data tokens at specific positions.
    
    For each position of interest, this function:
    1. Collects a batch of demo_data (same as compute_hiddens_multi)
    2. Creates modified batches where all sequences share the same data token at that position
    3. Extracts hidden representations at the padding token immediately following that position
    4. Repeats for different token values at the same position
    
    Note: This function always fixes DATA tokens (not target tokens).
    For point index i: fixes token at sequence position 3*i (data token),
    extracts hidden at sequence position 3*i+1 (PAD token).
    
    Parameters:
    -----------
    config : ConfigDict
        Configuration object containing task/model setup.
    model : torch.nn.Module
        The model used to extract task vectors.
    train_task : object
        Object that supports `.sample_data()`, `.evaluate()`, and has a `.task_pool` list.
    layers : Sequence[int], default=list(range(16))
        Layers from which to extract hidden representations.
    chunk_size : int, default=16
        Number of tasks to process in each chunk.
    step : int, default=1008600
        Step for data generation.
    positions_of_interest : Sequence[int], optional
        Point indices (0 to n_points-1) to analyze. If None, uses all points.
    max_unique_tokens : int, optional
        Maximum number of unique tokens to consider per position. If None, uses all unique tokens.
        For data tokens (vectors), this helps limit memory usage by sampling a subset.
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)
        Hidden representations for each layer, position, unique token value, task, and batch.
    demo_data : torch.Tensor
        Original demo_data batch: (batch_size, n_points, n_dims)
    token_info : dict
        Information about the tokens used:
        - 'positions': list of point indices analyzed
        - 'unique_tokens': dict mapping position -> list of unique token values
        - 'token_type': always "data"
    """
    device = config.device
    n_tasks = train_task.task_pool.shape[0]
    n_points = config.task.n_points
    batch_size = train_task.batch_size
    n_embd = config.model.n_embd
    n_dims = config.task.n_dims
    layers = list(layers)
    L = len(layers)
    
    # Check model padding format
    if not hasattr(model, "pad"):
        raise ValueError("Model must have a 'pad' attribute ('bos' or 'mapsto')")
    if model.pad != "mapsto":
        raise ValueError(f"This function currently only supports 'mapsto' padding format, got '{model.pad}'")
    
    # Determine positions of interest
    if positions_of_interest is None:
        positions_of_interest = list(range(n_points))
    else:
        positions_of_interest = list(positions_of_interest)
        if not all(0 <= p < n_points for p in positions_of_interest):
            raise ValueError(f"All positions must be in [0, {n_points-1}]")
    
    n_positions = len(positions_of_interest)
    
    # Step 1: Collect demo_data batch (same as compute_hiddens_multi)
    demo_data = train_task.sample_data(step=step)  # (batch_size, n_points, n_dims)
    demo_data = demo_data.to(device)
    
    # We'll need to evaluate targets for all tasks to understand token distributions
    # But we'll process in chunks like compute_hiddens_multi
    
    # For each position, we need to:
    # 1. Find unique tokens at that position across the batch
    # 2. For each unique token, create modified batches
    # 3. Extract hiddens at the padding token following that position
    
    # Collect data tokens at positions of interest
    # Data tokens are the same across all tasks since we use the same demo_data
    all_tokens_by_position = {}  # position -> list of tokens
    for pos_idx in positions_of_interest:
        all_tokens_by_position[pos_idx] = [demo_data[:, pos_idx, :]]  # (batch_size, n_dims)
    
    # Find unique tokens for each position
    unique_tokens_by_position = {}
    for pos_idx in positions_of_interest:
        tokens_list = all_tokens_by_position[pos_idx]
        all_tokens = torch.cat(tokens_list, dim=0)  # (batch_size, n_dims)
        
        # For data tokens (vectors), we can't easily find "unique" vectors
        # Instead, we'll sample a subset if max_unique_tokens is specified
        # Otherwise use all tokens (may be memory intensive)
        if max_unique_tokens is not None and len(all_tokens) > max_unique_tokens:
            # Randomly sample max_unique_tokens
            indices = torch.randperm(len(all_tokens), device=all_tokens.device)[:max_unique_tokens]
            unique_tokens = all_tokens[indices]  # (max_unique_tokens, n_dims)
        else:
            unique_tokens = all_tokens  # (batch_size, n_dims)
        
        unique_tokens_by_position[pos_idx] = unique_tokens
    
    # Determine output shapes
    max_unique = max(len(ut) for ut in unique_tokens_by_position.values())
    # We'll use variable-length lists, but for now let's use the max
    # Actually, let's use lists of tensors for flexibility
    
    # Initialize storage for results
    # We'll use a list of lists: results[layer][position][token_idx] = tensor
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
        # Fix data token at position 3*pos_idx, extract hidden at position 3*pos_idx + 1 (PAD token)
        fix_seq_pos = 3 * pos_idx
        extract_seq_pos = 3 * pos_idx + 1
        
        # Check bounds
        if extract_seq_pos >= 3 * n_points:
            # Skip if extraction position is out of bounds
            continue
        
        # For each unique token value
        for token_idx, fixed_token_value in enumerate(unique_tokens):
            # Process tasks in chunks
            for i in range(0, n_tasks, chunk_size):
                chunk_end = min(i + chunk_size, n_tasks)
                chunk_size_actual = chunk_end - i
                
                # Fix data token at position pos_idx
                # Start with original demo_data
                modified_demo_data = demo_data.clone()  # (batch_size, n_points, n_dims)
                # Set all sequences to have the same data token at pos_idx
                # fixed_token_value: (n_dims,)
                modified_demo_data[:, pos_idx, :] = fixed_token_value.unsqueeze(0).expand(batch_size, -1)
                
                # Re-evaluate targets with fixed data (targets depend on data)
                modified_demo_target = train_task.evaluate(
                    modified_demo_data,
                    train_task.task_pool[i:chunk_end].squeeze(-1).T,
                    step=step
                )  # (batch_size, n_points, chunk_size) or (batch_size, n_points)
                
                # Repeat data for chunk
                modified_demo_data_repeated = modified_demo_data.unsqueeze(0).expand(
                    chunk_size_actual, batch_size, n_points, -1
                ).reshape(-1, n_points, n_dims)
                
                # Flatten targets for model input
                if modified_demo_target.ndim == 3:
                    modified_demo_target = modified_demo_target.permute(2, 0, 1).reshape(-1, n_points)
                else:
                    modified_demo_target = modified_demo_target.unsqueeze(0).expand(
                        chunk_size_actual, -1, -1
                    ).reshape(-1, n_points)
                
                # Extract hidden representations at extract_seq_pos
                chunk_hiddens = extract_hidden_multi(
                    model=model,
                    demo_data=modified_demo_data_repeated,
                    demo_target=modified_demo_target,
                    layers=layers,
                    task_pos=extract_seq_pos,
                )  # (L, chunk*batch_size, n_embd)
                
                # Store results
                for l_idx, l in enumerate(layers):
                    if pos_idx not in results_by_layer[l]:
                        results_by_layer[l][pos_idx] = {}
                    if token_idx not in results_by_layer[l][pos_idx]:
                        results_by_layer[l][pos_idx][token_idx] = []
                    
                    # Reshape: (chunk*batch_size, n_embd) -> (chunk, batch_size, n_embd)
                    hiddens_reshaped = chunk_hiddens[l_idx].reshape(chunk_size_actual, batch_size, n_embd)
                    results_by_layer[l][pos_idx][token_idx].append(hiddens_reshaped)
    
    # Convert results to a structured tensor format
    # We need to handle variable numbers of unique tokens per position
    # Use the maximum number of unique tokens across all positions and pad/truncate
    max_unique_tokens_actual = max(len(unique_tokens_by_position[p]) for p in positions_of_interest)
    
    # Initialize output tensor
    output_shape = (L, n_positions, max_unique_tokens_actual, n_tasks, batch_size, n_embd)
    all_hiddens = torch.zeros(output_shape, dtype=demo_data.dtype, device=device)
    
    # Fill in the results
    for l_idx, l in enumerate(layers):
        for pos_idx_idx, pos_idx in enumerate(positions_of_interest):
            n_unique = len(unique_tokens_by_position[pos_idx])
            for token_idx in range(min(n_unique, max_unique_tokens_actual)):
                # Concatenate results from all chunks
                chunk_results = results_by_layer[l][pos_idx][token_idx]
                if chunk_results:
                    combined = torch.cat(chunk_results, dim=0)  # (n_tasks_processed, batch_size, n_embd)
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
    
    return all_hiddens.detach().cpu(), demo_data, token_info




