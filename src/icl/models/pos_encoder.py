import torch
import torch.nn as nn
# import torch.nn.functional as F


# Relative Positional Encoding, https://arxiv.org/pdf/1803.02155
# from https://github.com/JiajunSong629/ood-generalization-via-composition/blob/main/synthetic-experiments/model.py#L71

class RelativePositionalEncoding(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_length = max_seq_len
        self.pe = nn.Parameter(torch.randn(2*self.max_length+1, head_dim) / head_dim ** 0.5)
        
    def forward(self, seq_len):
        range_vec = torch.arange(seq_len)
        distances = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        distances = distances.clamp(-self.max_length, self.max_length) + self.max_length
        return self.pe[distances]



# https://arxiv.org/pdf/2108.12409
# See https://github.com/jaketae/alibi/blob/main/alibi/attention.py
class AliBiPositionalEncoding(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        base = (2**8) ** (1 / num_heads)
        slopes = 1. / base ** torch.arange(1,num_heads+1)
        slopes = slopes.unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("slopes", slopes, persistent=False)
        
    def forward(self, seq_len):
        device = self.slopes.device
        range_vec = torch.arange(seq_len).to(device)
        distances = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        alibi_emb = self.slopes * distances.unsqueeze(0)
        return alibi_emb.unsqueeze(0)
    


# Rotary Positional Encoding

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Rotary embedding helper function"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (freqs_cis.shape, x.shape)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)