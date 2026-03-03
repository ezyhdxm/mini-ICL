# from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
# import nvtx
# from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .pos_encoder import *
from .kv_cache import KVCache


# TODO: Add MLA

# causal mask for flex_attention, not in use yet. 
# flex_attention is a fast implementation of multihead attention. 
# Yet it has not support positional encodings with training parameters.  

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

######################
# MultiHeadAttention #
######################

class MultiHeadAttention(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.emb_dim = config.model.emb_dim
        self.n_head = config.model.num_heads[layer]
        self.head_dim = self.emb_dim // self.n_head
        self.attn_type = "MHA"

        assert self.emb_dim % self.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        
        self.query = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        self.key = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        self.value = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        
        self.out = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        
        self.pos_enc = "rotary"
        self.scale = self.head_dim ** 0.5
        self.flash = config.model.flash
        self.dropout = config.model.dropout if config.model.dropout else 0.
        
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, config.model.pos_max_len * 2))
    
    def forward(
        self, 
        x: Tensor, *, 
        kv_cache: Optional[KVCache] = None, cache_pos: int = 0, update_cache: bool = True
    ) -> Tensor: # x: (B,Tnew,C)
        batch_size, seq_len, _ = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        K = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        V = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
                        
        # expected shape for apply_rotary_emb: (batch_size, max_seq_len, num_head, d_head)
        if cache_pos + seq_len > self.freqs_cis.size(0):
            raise RuntimeError("rotary freqs overflow ...")

        freqs = self.freqs_cis[cache_pos:cache_pos+seq_len].to(x.device)
        Qr, Kr = apply_rotary_emb(Q.transpose(1, 2), K.transpose(1, 2), freqs_cis=freqs)
        Q = Qr.transpose(1, 2).contiguous()
        K = Kr.transpose(1, 2).contiguous()
        
        if kv_cache is not None:
            end = cache_pos + seq_len
            if end > kv_cache.k.size(2):
                raise RuntimeError(f"KV cache overflow: end={end} > max_len={kv_cache.k.size(2)}")

            kv_cache.k[:, :, cache_pos:end, :] = K
            kv_cache.v[:, :, cache_pos:end, :] = V

            Ktot = kv_cache.k[:, :, :end, :]
            Vtot = kv_cache.v[:, :, :end, :]

            if update_cache:
                kv_cache.cur_len = end
        else:
            Ktot, Vtot = K, V
        
         # SDPA expects same head dim; expand MQA (Hk=1) to match Q heads if needed (view, no copy)
        if Ktot.size(1) != Q.size(1):
            Ktot = Ktot.expand(-1, Q.size(1), -1, -1)
            Vtot = Vtot.expand(-1, Q.size(1), -1, -1)
            
        if self.flash:
            drop = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(Q, Ktot, Vtot, dropout_p=drop, is_causal=True)
            out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
            out = self.out(out)
            return out
        else:
            attn_score = Q @ Ktot.transpose(-1,-2) / self.scale # (B,H,T,T)

            q_len = seq_len
            k_len = Ktot.size(-2)
            past_len = k_len - q_len
            causal_mask = torch.ones(q_len, k_len, device=x.device, dtype=torch.bool).tril(diagonal=past_len)
            attn_score = attn_score.masked_fill(~causal_mask, float("-inf"))

            self.attn = F.softmax(attn_score, dim=-1)
            out = self.attn @ Vtot  # (B,H,Tnew,D)
            
            out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
            out = self.out(out)
            
            return out