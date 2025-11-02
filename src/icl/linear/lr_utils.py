import torch
from torchinfo import summary 
from ml_collections import ConfigDict


def to_seq(data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # print(data.shape, targets.shape)
    if data.ndim == 4:
        data = data.squeeze(0)
    if targets.ndim == 3:
        targets = targets.squeeze(0)
    
    batch_size, seq_len, n_dims = data.shape
    dtype = data.dtype

    # data: prepend 0 to feature vector → (B, T, 1 + n_dims)
    padded_data = torch.cat([torch.zeros((batch_size, seq_len, 1), dtype=dtype, device=data.device), data], dim=2)

    # targets: append zeros to match shape → (B, T, 1 + n_dims)
    padded_targets = torch.cat([targets.unsqueeze(-1), torch.zeros((batch_size, seq_len, n_dims), dtype=dtype, device=data.device)], dim=2)

    # Interleave data and target representations → (B, 2T, 1 + n_dims)
    seq = torch.stack([padded_data, padded_targets], dim=2)  # shape (B, T, 2, D+1)
    seq = seq.reshape(batch_size, 2 * seq_len, n_dims + 1)

    return seq

def seq_to_targets(seq: torch.Tensor) -> torch.Tensor:
    # Extract targets from the even-indexed positions of the sequence (i.e., where data was padded)
    return seq[:, ::2, 0]  # (B, T)


def tabulate_model(model: torch.nn.Module, n_dims: int, n_points: int, batch_size: int) -> str:
    dummy_data = torch.ones((batch_size, n_points, n_dims), dtype=model.dtype)
    dummy_targets = torch.ones((batch_size, n_points), dtype=model.dtype)

    try:
        info = summary(model, input_data=(dummy_data, dummy_targets), depth=3, col_names=["input_size", "output_size", "num_params"])
        return str(info)
    except Exception as e:
        return f"Could not tabulate model: {e}"
    

def filter_config(config: ConfigDict) -> ConfigDict:
    with config.unlocked():
        for k, v in config.items():
            if v is None:
                del config[k]
            elif isinstance(v, ConfigDict):
                config[k] = filter_config(v)
    return config