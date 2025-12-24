import torch
from torch import nn
from typing import List, Optional

from .kv_cache import KVCache
from .attention import MultiHeadAttention

class TFAttnBlock(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.MHA = MultiHeadAttention(config, layer)
        self.ln1 = nn.LayerNorm(config.model.emb_dim) if config.model.layer_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None
    
    def forward(self, x, kv_cache: Optional[KVCache] = None, cache_pos: Optional[int] = None, update_cache: bool = True):
        y = self.MHA(self.ln1(x), kv_cache=kv_cache, cache_pos=cache_pos, update_cache=update_cache)
        return x + (self.attn_dropout(y) if self.attn_dropout is not None else y)

class TFMLPBlock(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.mlp = None
        if config.model.mlp[layer]:
            if config.model.activation[layer]:
                assert config.model.ff_dim is not None
                self.mlp = nn.Sequential(
                    nn.Linear(config.model.emb_dim, config.model.ff_dim, bias=config.model.mlp_bias),
                    nn.SiLU(), # seems to be the most popular activation function nowadays
                    nn.Linear(config.model.ff_dim, config.model.emb_dim, bias=config.model.mlp_bias)
                )
            else:
                self.mlp = nn.Linear(config.model.emb_dim, config.model.emb_dim)
            self.ln2 = nn.LayerNorm(config.model.emb_dim) if config.model.layer_norm else nn.Identity()
            self.mlp_dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None
    
    def forward(self, x):
        if self.mlp is not None:
            mlp_out = self.mlp(self.ln2(x))
            x = x + self.mlp_dropout(mlp_out) if self.mlp_dropout is not None else x + mlp_out
        return x


class TFBlock(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.attn_block = TFAttnBlock(config, layer)
        self.mlp = TFMLPBlock(config, layer)

    def forward(self, x, *, kv_cache: Optional[KVCache] = None, cache_pos: Optional[int] = None, update_cache: bool = True):
        x = self.attn_block(x, kv_cache=kv_cache, cache_pos=cache_pos, update_cache=update_cache)
        return self.mlp(x)
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.model.emb_dim)
        self.layers = nn.ModuleList([TFBlock(config, layer) for layer in range(config.model.num_layers)])
        self.output_layer = nn.Linear(config.model.emb_dim, config.vocab_size)
        self.to(config.device)

    def forward(self, x, kv_caches: Optional[List[KVCache]] = None, cache_pos: Optional[int] = None, update_cache: bool = True):
        x = self.embed(x)
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x = layer(x, kv_cache=cache_i, cache_pos=cache_pos, update_cache=update_cache)
        return self.output_layer(x)