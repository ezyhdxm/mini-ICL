import torch
import torch.nn.functional as F

from icl.figures.attn_plots_beta import visualize_attention
from icl.models import apply_rotary_emb


#######################
# Attention Extraction Functions #
#######################



def get_attn(model, data, target):
    """
    Extract attention weights from all layers of the model.
    
    Registers forward hooks to capture attention matrices from each layer
    during forward pass. Returns attention for first sample in batch.
    
    Args:
        model: Transformer model with layers
        data: Input data, shape (B, T)
        target: Target data for computing context
    
    Returns:
        attns: Dictionary mapping layer indices to attention weights, shape varies by layer
    """
    attns = {}
    
    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            # Get the input to the attention module
            x = input[0]
            batch_size, seq_len, _ = x.size()
            
            # Compute Q, K, V
            Q = module.query(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
            K = module.key(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
            
            # Apply rotary embeddings
            Q, K = apply_rotary_emb(Q.transpose(1, 2), K.transpose(1, 2), freqs_cis=module.freqs_cis[:seq_len])
            Q, K = Q.transpose(1, 2), K.transpose(1, 2)
            
            # Compute attention weights
            scale = 1.0 / (module.head_dim ** 0.5)
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
            
            # Apply causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_weights.masked_fill_(mask, float('-inf'))
            
            # Softmax to get attention probabilities
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            attns[layer_idx] = attn_weights.detach().squeeze(0)
        
        return hook_fn
    
    hook_handles = []
    num_layers = len(model.transformer.blocks)
    
    for l in range(num_layers):
        handle = model.transformer.blocks[l].attn_block.attn.register_forward_hook(create_hook_fn(l))
        hook_handles.append(handle)
    
    with torch.no_grad():
        _ = model(data, target)
    
    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
    
    return attns

def get_filtered_attn_output_at_layer(model, data, target, l, task_pos=-1):
    """
    Extract filtered attention output from a specific layer.
    
    Computes attention output with zeroed attention from task position to itself
    and previous position. Useful for ablation studies.
    
    Args:
        model: Transformer model
        data: Input data, shape (B, T)
        target: Target data
        l: Layer index
        task_pos: Task position index to zero out
    
    Returns:
        filtered_output: Dictionary with filtered attention output for the layer
    """
    filtered_output = {}
    
    def hook_fn(module, input, output):
        # Get the input to the attention module
        x = input[0]
        batch_size, seq_len, _ = x.size()
        
        # Compute Q, K, V
        Q = module.query(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
        K = module.key(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
        V = module.value(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
        
        # Apply rotary embeddings
        Q, K = apply_rotary_emb(Q.transpose(1, 2), K.transpose(1, 2), freqs_cis=module.freqs_cis[:seq_len])
        Q, K = Q.transpose(1, 2), K.transpose(1, 2)
        
        # Compute attention weights
        scale = 1.0 / (module.head_dim ** 0.5)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights.masked_fill_(mask, float('-inf'))
        
        # Softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_weights[:, :, task_pos, task_pos] = 0  # Zero out attention to the task position
        attn_weights[:, :, task_pos, task_pos-1] = 0  # Zero out attention to the task position
        out = attn_weights @ V  # (B, H, T, D)
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
        out = module.out(out)
        filtered_output[l] = out.detach().squeeze(0)
    
    handle = model.transformer.blocks[l].attn_block.attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(data, target)
    
    # Clean up hooks
    handle.remove()
    
    return filtered_output

# View attention map
def view_attn(train_task, model):
    """
    Create an interactive widget to visualize attention maps.
    
    Samples data from first task and displays attention patterns
    in a Jupyter widget for interactive exploration.
    
    Args:
        train_task: Training task object
        model: Transformer model
    
    Returns:
        Interactive widget showing attention heatmaps
    """
    train_task.batch_size = 1
    demo_data0, demo_target = train_task.sample_from_task(train_task.task_pool[0], step=2)
    attns = get_attn(model, demo_data0, demo_target)
    cap = 100
    attns_capped = {layer_key: tensor[:, :cap, :cap] for layer_key, tensor in attns.items()}
    
    widget = visualize_attention(attns_capped, mode='widget')
    return widget

def get_attn_mean_var(train_task, model):
    """
    Compute mean and variance statistics of attention weights.
    
    Analyzes attention patterns by computing norms and variances
    to understand attention concentration across layers.
    
    Args:
        train_task: Training task object
        model: Transformer model
    
    Returns:
        attn_vars: Dictionary mapping layer indices to variance statistics
        attn_means: Dictionary mapping layer indices to mean statistics
    """
    train_task.batch_size = 256
    demo_data, demo_target = train_task.sample_from_task(train_task.task_pool[1], step=2)
    attns = get_attn(model, demo_data, demo_target)
    attn_means = {layer_key: tensor[:, :, 1::3].mean(dim=0).norm(dim=(-1,-2)).square().cpu().item() for layer_key, tensor in attns.items()}
    attn_vars = {layer_key: tensor[:, :, 1::3].var(dim=0).sum(dim=(-1,-2)).cpu().item() for layer_key, tensor in attns.items()}
    return attn_vars, attn_means