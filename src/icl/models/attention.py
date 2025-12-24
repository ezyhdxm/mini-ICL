import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .pos_encoder import precompute_freqs_cis, apply_rotary_emb
from .kv_cache import KVCache


class MultiHeadAttention(nn.Module):
    def __init__(self, config, layer: int = 0):
        super().__init__()
        self.emb_dim = int(config.model.emb_dim)
        self.n_head = int(config.model.num_heads[layer])
        if self.emb_dim % self.n_head != 0:
            raise ValueError("emb_dim must be divisible by n_head")
        self.head_dim = self.emb_dim // self.n_head
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

        self.query = nn.Linear(self.emb_dim, self.emb_dim, bias=bool(config.model.bias))
        self.key   = nn.Linear(self.emb_dim, self.emb_dim, bias=bool(config.model.bias))
        self.value = nn.Linear(self.emb_dim, self.emb_dim, bias=bool(config.model.bias))
        self.out   = nn.Linear(self.emb_dim, self.emb_dim, bias=bool(config.model.bias))

        self.flash = bool(getattr(config.model, "flash", False))
        self.causal = bool(getattr(config.model, "mask", True))
        self.dropout = float(config.model.dropout) if getattr(config.model, "dropout", None) else 0.0
        self.inv_sqrt = self.head_dim ** -0.5

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, int(config.model.pos_max_len) * 2),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        *,
        kv_cache: Optional[KVCache] = None,
        cache_pos: Optional[int] = None,
        update_cache: bool = True,
    ) -> Tensor:
        B, Tnew, _ = x.size()

        # ---- cache position ----
        if kv_cache is None:
            cache_pos_i = 0 if cache_pos is None else int(cache_pos)
            cur = 0
        else:
            cur = int(kv_cache.cur_len)
            cache_pos_i = cur if cache_pos is None else int(cache_pos)
            if cache_pos_i > cur:
                raise RuntimeError("cache_pos cannot be > kv_cache.cur_len")

        end = cache_pos_i + Tnew

        # ---- project ----
        q = self.query(x).view(B, Tnew, self.n_head, self.head_dim)  # (B,T,H,D)
        k = self.key(x).view(B, Tnew, self.n_head, self.head_dim)
        v = self.value(x).view(B, Tnew, self.n_head, self.head_dim)

        # ---- rotary ----
        if end > self.freqs_cis.size(0):
            raise RuntimeError("rotary freqs overflow")
        freqs = self.freqs_cis[cache_pos_i:end]
        if freqs.device != x.device:
            freqs = freqs.to(device=x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs)  # (B,T,H,D)

        Q = q.transpose(1, 2)  # (B,H,T,D)
        K = k.transpose(1, 2)
        V = v.transpose(1, 2)

        # ---- kv cache combine ----
        if kv_cache is not None:
            if end > kv_cache.k.size(2):
                raise RuntimeError("KV cache overflow")

            if update_cache:
                kv_cache.k[:, :, cache_pos_i:end, :].copy_(K)
                kv_cache.v[:, :, cache_pos_i:end, :].copy_(V)
                kv_cache.cur_len = end
                Ktot = kv_cache.k[:, :, :end, :]
                Vtot = kv_cache.v[:, :, :end, :]
            else:
                # Fast "probe at end": write into unused tail but do not advance cur_len.
                if cache_pos_i == cur:
                    kv_cache.k[:, :, cache_pos_i:end, :].copy_(K)
                    kv_cache.v[:, :, cache_pos_i:end, :].copy_(V)
                    Ktot = kv_cache.k[:, :, :end, :]
                    Vtot = kv_cache.v[:, :, :end, :]
                else:
                    # Rewind probe without mutating cache contents.
                    Ktot = torch.cat((kv_cache.k[:, :, :cache_pos_i, :], K), dim=2)
                    Vtot = torch.cat((kv_cache.v[:, :, :cache_pos_i, :], V), dim=2)
        else:
            Ktot, Vtot = K, V

        # ---- attention ----
        k_len = Ktot.size(-2)
        past_len = k_len - Tnew  # 0 if no prefix

        if self.flash:
            drop = self.dropout if self.training else 0.0
            if not self.causal:
                out = F.scaled_dot_product_attention(Q, Ktot, Vtot, dropout_p=drop, is_causal=False)
            else:
                if past_len == 0:
                    out = F.scaled_dot_product_attention(Q, Ktot, Vtot, dropout_p=drop, is_causal=True)
                elif Tnew == 1:
                    out = F.scaled_dot_product_attention(Q, Ktot, Vtot, dropout_p=drop, is_causal=False)
                else:
                    # offset causal mask: allow key positions <= past_len + i for query row i
                    attn_mask = torch.ones((Tnew, k_len), device=Q.device, dtype=torch.bool).tril(diagonal=past_len)
                    out = F.scaled_dot_product_attention(
                        Q, Ktot, Vtot, attn_mask=attn_mask, dropout_p=drop, is_causal=False
                    )
        else:
            attn_score = (Q @ Ktot.transpose(-1, -2)) * self.inv_sqrt  # (B,H,Tnew,K)
            if self.causal:
                causal_mask = torch.ones((Tnew, k_len), device=attn_score.device, dtype=torch.bool).tril(diagonal=past_len)
                attn_score = attn_score.masked_fill(~causal_mask, torch.finfo(attn_score.dtype).min)
            self.attn = F.softmax(attn_score, dim=-1)
            if self.dropout and self.training:
                self.attn = F.dropout(self.attn, p=self.dropout)
            out = self.attn @ Vtot

        out = out.transpose(1, 2).contiguous().view(B, Tnew, self.emb_dim)
        return self.out(out)
