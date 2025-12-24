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










