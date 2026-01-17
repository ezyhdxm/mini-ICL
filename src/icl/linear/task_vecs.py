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

    all_hiddens = torch.empty(output_shape, dtype=demo_data.dtype, device=config.device)

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
            all_hiddens[:, i:chunk_end] = chunk_hiddens
        else:
            # (L, chunk*B, P, D) -> (L, chunk, B, P, D) -> (L, chunk, P, B, D)
            chunk_hiddens = chunk_hiddens.reshape(L, chunk_size_actual, batch_size, n_points, n_embd)
            chunk_hiddens = chunk_hiddens.permute(0, 1, 3, 2, 4)
            all_hiddens[:, i:chunk_end] = chunk_hiddens

    return all_hiddens.detach().cpu(), demo_data




