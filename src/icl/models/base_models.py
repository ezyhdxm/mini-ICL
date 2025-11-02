from dataclasses import dataclass
from collections import namedtuple
import torch

from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
from .pos_encoder import *
from .attention import *

class TFAttnBlock(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.MHA = MultiHeadAttention(config, layer)
        self.ln1 = nn.LayerNorm(config.model.emb_dim) if config.model.layer_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None
    
    def forward(self, x):
        atten_out = self.MHA(self.ln1(x))
        x = x + self.attn_dropout(atten_out) if self.attn_dropout is not None else x + atten_out
        return x

class TFMLPBlock(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.mlp = None
        if config.model.mlp[layer]:
            if config.model.activation[layer]:
                assert config.model.ff_dim is not None, "FeedForward dimension cannot be empty."
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

    def forward(self, x):
        x = self.attn_block(x)
        x = self.mlp(x)
        return x
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.model.emb_dim).to(config.device)
        self.pos_enc = config.model.pos_enc
        if config.model.pos_enc == "abs":
            self.positional_encoding = nn.Embedding(config.seq_len, config.model.emb_dim)
        self.layers = nn.ModuleList([TFBlock(config, layer) for layer in range(config.model.num_layers)])
        self.output_layer = nn.Linear(config.model.emb_dim, config.vocab_size)

    def to(self, device):
        self.embed = self.embed.to(device)
        if self.pos_enc == "abs":
            self.positional_encoding = self.positional_encoding.to(device)
        for layer in self.layers:
            layer = layer.to(device)
        self.output_layer = self.output_layer.to(device)
        return self

    def forward(self, x):
        if self.pos_enc == "abs":
            x = self.embed(x) + self.positional_encoding(torch.arange(x.size(1), device=x.device).view(1, x.size(1)))
        else:
            x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
            
        logits = self.output_layer(x) # (batch_size, seq_len, vocab_size)
        return logits