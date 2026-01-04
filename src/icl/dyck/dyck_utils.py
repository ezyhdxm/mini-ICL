import torch

def sample_binary_mask(config, device=None):
    """
    Generate a 1D binary mask:
      1. Sample from Bernoulli(p) of length ceil(seq_len / 2)
      2. Keep first (or random) `keep_n` ones
      3. Insert 0 after each element → final length = seq_len

    Args:
        keep_n (int): number of 1s to keep
        random_keep (bool): if True, randomly keep `keep_n` ones
        device (str or torch.device): target device
    """
    device = device or torch.device("cpu")
    seq_len = config.seq_len
    p = config.task.repeat_prob
    keep_n = 2*config.task.dyck_length
    
    short_len = (seq_len + 1) // 2  # number of "active" positions before expansion

    # Step 1: Bernoulli sampling
    x = torch.bernoulli(torch.full((short_len,), p, device=device))

    # Step 2: Keep up to `keep_n` ones
    out_short = torch.zeros_like(x)
    idx = torch.nonzero(x, as_tuple=False).squeeze()
    if idx.numel() > 0:
        idx = idx[:keep_n]
        out_short[idx] = 1

    return out_short.long()