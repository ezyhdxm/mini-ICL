import torch
from icl.latent_markov import *
from torchinfo import summary


def get_bayes_loss(bayes_prob, prob):
    """
    Compute cross-entropy loss between Bayesian predictions and target probabilities.
    
    This measures how well the Bayesian probability distribution matches the target
    probability distribution using the KL divergence formulation.
    
    Args:
        bayes_prob: Bayesian probability predictions, shape (B, N) where N is vocab size
        prob: Target probability distributions, shape (B, N)
    
    Returns:
        Scalar loss value averaged over batch
    """
    return -torch.sum(prob * torch.log(bayes_prob), dim=-1).mean()


def last_token_loss(logits, probs):
    """
    Compute cross-entropy loss on the last token predictions.
    
    Args:
        logits: Model predictions, shape (B, N) where N is vocab size
        probs: Target probability distributions, shape (B, N)
    
    Returns:
        Scalar loss value averaged over batch
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1).mean()



def get_train_result(**kwargs):
    """
    Simple wrapper to collect training results as a dictionary.
    
    This is a convenience function to create result dictionaries from keyword arguments.
    
    Args:
        **kwargs: Any keyword arguments to include in the results
    
    Returns:
        Dictionary of the keyword arguments
    """
    return kwargs


def tabulate_model(model: torch.nn.Module, seq_len: int, batch_size: int, device: str) -> str:
    """
    Generate a summary table of the model architecture and parameters.
    
    Uses torchinfo to create a detailed summary including input/output sizes
    and parameter counts for each layer up to depth 3.
    
    Args:
        model: PyTorch model to summarize
        seq_len: Sequence length for the dummy input
        batch_size: Batch size for the dummy input
        device: Device to create dummy data on
    
    Returns:
        String representation of the model summary, or error message if summarization fails
    """
    dummy_data = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    try:
        info = summary(model, 
                       input_data=dummy_data, 
                       depth=3, 
                       col_names=["input_size", "output_size", "num_params"])
        return str(info)
    except Exception as e:
        return f"Could not tabulate model: {e}"


def compute_cross_entropy(preds, samples, eps=1e-12):
    """
    Compute cross-entropy loss given predictions and target samples.
    
    For each position, computes the negative log probability of the actual token
    that occurred in the samples.
    
    Args:
        preds: Predicted probability distributions, shape (B, T, N) where N is vocab size
        samples: Target token indices, shape (B, T)
        eps: Small epsilon for numerical stability in log computation
    
    Returns:
        cross_entropy: Loss per position, shape (B, T)
    """
    log_preds = torch.log(preds + eps)  # (B, T, N)
    indices = samples.unsqueeze(-1)  # (B, T, 1)
    log_probs_true = torch.gather(log_preds, dim=2, index=indices).squeeze(-1)  # (B, T)
    cross_entropy = -log_probs_true  # (B, T)

    return cross_entropy


#######################
# Attention utilities #
#######################

def get_attn_base(model, batch):
    """
    Extract attention maps from all layers of the model.
    
    Registers forward hooks on all attention layers to capture attention weights,
    temporarily disables flash attention for compatibility, then restores it.
    
    Args:
        model: Transformer model with layers and attention blocks
        batch: Input batch, shape (B, T)
    
    Returns:
        attn_maps: Dictionary mapping layer indices to attention tensors.
                  Each attention tensor has shape (B, H, T, T) where H is num heads
    """
    attn_maps = {}

    num_layers = len(model.layers)

    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            attn_maps[layer_idx] = module.attn.detach().cpu()
    
        return hook_fn

    handles = []

    for l in range(num_layers):
        model.layers[l].attn_block.MHA.flash = False
        handles.append(model.layers[l].attn_block.MHA.register_forward_hook(create_hook_fn(l)))
    
    with torch.no_grad():
        _ = model(batch)
    
    for l in range(num_layers):
        handles[l].remove()
        model.layers[l].attn_block.MHA.flash = True
    
    return attn_maps

def get_attn_at_layer_base(model, batch, layer):
    """
    Extract attention map from a specific layer of the model.
    
    Similar to get_attn_base but only captures attention from one layer.
    Temporarily disables flash attention for compatibility.
    
    Args:
        model: Transformer model with layers and attention blocks
        batch: Input batch, shape (B, T)
        layer: Index of the layer to extract attention from
    
    Returns:
        attn: Attention tensor from the specified layer, shape (B, H, T, T)
    """
    model.layers[layer].attn_block.MHA.flash = False
    attn_maps = {}

    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            attn_maps[layer_idx] = module.attn.detach().cpu()
    
        return hook_fn

    handle = model.layers[layer].attn_block.MHA.register_forward_hook(create_hook_fn(layer))
    _ = model(batch)
    handle.remove()
    model.layers[layer].attn_block.MHA.flash = True
    return attn_maps[layer]

def pth_score(model, batch, layer=0):
    """
    Compute the PTH (Previous Token Head) score for a specific layer.
    
    The PTH score measures how much attention each position pays to the previous token.
    This is computed as the average attention weight along the diagonal that's one step before.
    
    Args:
        model: Transformer model
        batch: Input batch, shape (B, T)
        layer: Layer index to compute score for
    
    Returns:
        Scalar PTH score
    """
    attn = get_attn_at_layer_base(model, batch, layer)
    attn = attn.squeeze(1)
    return attn.mean(dim=0).diagonal(offset=-1).mean().item()

def ih_score(model, batch, device, layer=1):
    """
    Compute the IH (Induction Head) score for a specific layer.
    
    The IH score measures how much attention each position pays to previous positions
    that match the pattern (i.e., attending to earlier tokens with the same value).
    This captures the model's ability to recognize and repeat patterns.
    
    Args:
        model: Transformer model
        batch: Input batch, shape (B, T)
        device: Device to perform computations on
        layer: Layer index to compute score for
    
    Returns:
        Scalar IH score
    """
    attns = get_attn_at_layer_base(model, batch, layer)
    attns = attns.squeeze(1) # (B, H, T, T)
    B, T = batch.shape
    
    # Compare all pairs: (B, T, T), batch[b, i] == batch[b, t]
    matches = (batch.unsqueeze(2) == batch.unsqueeze(1))  # (B, T, T)
    
    # Mask out positions where i >= t (i.e., keep only i < t)
    tril_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)  # (T, T)
    valid_matches = matches & tril_mask  # broadcasted to (B, T, T)
    b_indices, t_indices, i_indices = valid_matches.nonzero(as_tuple=True)
    grouped_sums = torch.zeros((B, T), device=device)  # (B, T)

    # Accumulate values at corresponding (b, t) positions
    grouped_sums.index_put_((b_indices, t_indices), attns[b_indices, t_indices, i_indices+1], accumulate=True)
    return grouped_sums.mean(dim=0)[1:].mean().item()




    


























