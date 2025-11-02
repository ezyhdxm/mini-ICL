# from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
# import nvtx
# from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .pos_encoder import *

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
        self.attn_type = config.model.attention_type if hasattr(config.model, 'attention_type') else "MHA"
        if self.attn_type == "GQA":
            self.g = config.model.attn_groups if hasattr(config.model, 'attn_groups') else 2

        
        assert self.emb_dim % self.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        
        if config.training.identity_query:
            self.query = nn.Identity()
        else:
            self.query = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        
        if self.attn_type == "MQA":
            self.key = nn.Linear(self.emb_dim, self.head_dim, bias=config.model.bias)
            self.value = nn.Linear(self.emb_dim, self.head_dim, bias=config.model.bias)
        elif self.attn_type == "MHA":
            self.key = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
            self.value = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        elif self.attn_type == "GQA":
            self.key = nn.Linear(self.emb_dim, self.g * self.head_dim, bias=config.model.bias)
            self.value = nn.Linear(self.emb_dim, self.g * self.head_dim, bias=config.model.bias)
            self.register_buffer("head_to_group", torch.arange(self.n_head) // (self.n_head // self.g))
        
        if config.training.freeze_value: self.value.weight.requires_grad_(False)
        
        self.out = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        if config.training.freeze_out: self.out.weight.requires_grad_(False)
        
        self.pos_enc = config.model.pos_enc
        self.scale = self.head_dim ** 0.5
        self.flash = config.model.flash
        self.dropout = config.model.dropout if config.model.dropout else 0.
        assert not (self.flash and self.pos_enc == "rpe"), "Flash Attention does not support RPE currently."  
        
        if self.pos_enc == "rpe":
            if not self.flash:
                self.PEV = RelativePositionalEncoding(self.head_dim, config.model.pos_max_len) # (T,T,D)
                self.PEK = RelativePositionalEncoding(self.head_dim, config.model.pos_max_len) # (T,T,D)
            elif config.device == "cuda":
                self.rpe = torch.randn((2*config.model.pos_max_len+1, self.head_dim), device=config.device) / (self.head_dim ** 0.5)
                
            else:
                raise ValueError("Flash Attention with RPE is currently only supported on CUDA devices.") # TODO: pay a closer look to flex_attention
        
        elif self.pos_enc == "rotary":
            self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, config.model.pos_max_len * 2))

        
        elif self.pos_enc == "alibi":
            self.alibi_emb = AliBiPositionalEncoding(self.n_head)
    
    def forward(self, x): # x: (B,T,C)
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        if self.attn_type == "MQA":
            # Multi-query attention: only one key and value projection
            K = self.key(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1,2)
            V = self.value(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1,2)
        elif self.attn_type == "MHA":
            K = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
            V = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        elif self.attn_type == "GQA":
            K = self.key(x).view(batch_size, seq_len, self.g, self.head_dim).transpose(1,2) # (B,G,T,D)
            V = self.value(x).view(batch_size, seq_len, self.g, self.head_dim).transpose(1,2) # (B,G,T,D)
            K = K[:, self.head_to_group]  # (B, H, T, D)
            V = V[:, self.head_to_group]  # (B, H, T, D)
        
        if self.pos_enc == "rotary":                                
            # expected shape for apply_rotary_emb: (batch_size, max_seq_len, num_head, d_head)
            Q, K = apply_rotary_emb(
                Q.transpose(1, 2), K.transpose(1, 2), 
                freqs_cis=self.freqs_cis[:seq_len].to(x.device)
                )  # (B,H,T,D)
            Q, K = Q.transpose(1, 2), K.transpose(1, 2)
            
        if self.flash:
            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.dropout, is_causal=True)
            out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
            out = self.out(out)
            return out
        else:
            attn_score = Q @ K.transpose(-1,-2) / self.scale # (B,H,T,T)
            if self.pos_enc == "rpe":
                Q2 = Q.transpose(0,2) # (T,B,H,D)
                Q2 = Q2.contiguous().view(seq_len, batch_size*self.n_head, self.head_dim) # (T,BH,D)
                attn_score2 = torch.matmul(Q2, self.PEK(seq_len).transpose(1,2)) # (T,BH,D) @ (T,D,T) -> (T,BH,T)
                attn_score2 = attn_score2.view(seq_len, self.n_head, batch_size, seq_len).transpose(0,2).contiguous() # (B,H,T,T)
                attn_score += attn_score2 / self.scale
            elif self.pos_enc=="alibi":
                attn_score += self.alibi_emb(seq_len)

            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            attn_score = attn_score.masked_fill(~causal_mask, float('-inf'))

            self.attn = F.softmax(attn_score, dim=-1) # (B,H,T,T)
            out = self.attn @ V # (B,H,T,D)
            
            if self.pos_enc == "rpe":
                attn2 = self.attn.transpose(0,2).contiguous().view(seq_len, -1, seq_len) # (T,BH,T)
                out2 = torch.matmul(attn2, self.PEV(seq_len)) # (T,BH,T) @ (T,T,D) -> (T,BH,D)
                out2 = out2.view(seq_len, -1, batch_size, self.head_dim).transpose(0,2) # (B,H,T,D)
                out += out2
            
            out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
            out = self.out(out)
            
            return out