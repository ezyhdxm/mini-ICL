"""
Nonpadded task-vector computation for latent Markov models.

KV-cache backend:
  - compute_hiddens_onepos_all_layers_kvcache_beta: KV-cache approach (no V chunking)
"""
from __future__ import annotations
from typing import List, Optional, Literal, Dict

import torch
from icl.models.kv_cache import KVCache

ActivationKind = Literal["attn_block", "mlp"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_n_emb(config) -> int:
    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
    if n_emb is None:
        raise ValueError("config.model must contain emb_dim or n_embd")
    return int(n_emb)


def _infer_n_head_head_dim(model) -> tuple[int, int]:
    mha = model.layers[0].attn_block.MHA
    return int(mha.n_head), int(mha.head_dim)


def _alloc_kv_caches(*, model: torch.nn.Module, batch_size: int, max_len: int, device: torch.device) -> List[KVCache]:
    n_layers = len(model.layers)
    n_head, head_dim = _infer_n_head_head_dim(model)
    cache_dtype = next(model.parameters()).dtype
    caches: List[KVCache] = []
    for _ in range(n_layers):
        k = torch.empty((batch_size, n_head, max_len, head_dim), device=device, dtype=cache_dtype)
        v = torch.empty((batch_size, n_head, max_len, head_dim), device=device, dtype=cache_dtype)
        caches.append(KVCache(k=k, v=v, cur_len=0))
    return caches


# ---------------------------------------------------------------------------
# KV-cache backend
# ---------------------------------------------------------------------------

class _HookLastTokenAllLayersToBuf:
    """
    When enabled, writes last-token activations into buf[i] for layer i:
      buf[i] <- out[:, -1, :]
    """
    def __init__(self, modules: List[torch.nn.Module]):
        self.modules = modules
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.enabled: bool = True
        self.buf: Optional[torch.Tensor] = None

    def set_buf(self, buf: torch.Tensor) -> None:
        self.buf = buf

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def __enter__(self):
        for i, m in enumerate(self.modules):
            def hook(_module, _inp, out, i=i):
                if not self.enabled:
                    return
                if self.buf is None:
                    raise RuntimeError("Hook buffer not set.")
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if out.ndim != 3:
                    raise RuntimeError(f"Expected (B,T,D), got {tuple(out.shape)}")
                self.buf[i].copy_(out[:, -1, :])
            self.handles.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.inference_mode()
def compute_hiddens_onepos_all_layers_kvcache_beta(
    config,
    model: torch.nn.Module,
    samples: torch.Tensor,
    *,
    activation: ActivationKind = "attn_block",
    dtype_out: torch.dtype = torch.float16,
    k_step: int = 4,
    b_step: int = 16,
    t_step: int = 16,
    pin_samples: bool = True,
    pin_output: bool = False,
) -> torch.Tensor:
    """
    Clean fast KV-cache version (no V chunking).

    samples: (n_tasks, B, Sfull) on CPU
    Returns: out: (n_layers, n_tasks, Tm1, V, B, D) on CPU
    """
    device = next(model.parameters()).device
    on_cuda = (device.type == "cuda")
    model.eval()

    if samples.device.type != "cpu":
        raise ValueError("samples must be on CPU.")
    if samples.dtype != torch.long:
        samples = samples.long()
    if on_cuda and pin_samples:
        samples = samples.pin_memory()
    if samples.ndim != 3:
        raise ValueError(f"samples must be (n_tasks,B,Sfull), got {tuple(samples.shape)}")

    n_tasks, B, Sfull = samples.shape
    seq_len = (Sfull + 1) // 2
    if 2 * seq_len - 1 != Sfull:
        raise ValueError(f"Expected Sfull = 2*seq_len-1, got {Sfull}")
    Tm1 = seq_len - 1

    D = _get_n_emb(config)
    n_layers = len(model.layers)

    sep_id = int(samples[..., 1].max().item())
    V = sep_id

    max_cache_len = 2 * Tm1

    if activation == "attn_block":
        modules = [layer.attn_block for layer in model.layers]
    elif activation == "mlp":
        modules = [layer.mlp for layer in model.layers]
    else:
        raise ValueError(f"activation must be 'attn_block' or 'mlp', got {activation!r}")

    out_pin = bool(on_cuda and pin_output)
    out = torch.empty((n_layers, n_tasks, Tm1, V, B, D), device="cpu", dtype=dtype_out, pin_memory=out_pin)
    async_d2h = out_pin

    old_head = model.output_layer
    model.output_layer = torch.nn.Identity()

    kv_pool: Dict[int, List[KVCache]] = {}
    act_pool: Dict[int, torch.Tensor] = {}
    upd_pool: Dict[int, torch.Tensor] = {}

    def get_kvs(Bflat: int) -> List[KVCache]:
        kvs = kv_pool.get(Bflat)
        if kvs is None:
            kvs = _alloc_kv_caches(model=model, batch_size=Bflat, max_len=max_cache_len, device=device)
            kv_pool[Bflat] = kvs
        return kvs

    def get_act(Bflat: int) -> torch.Tensor:
        buf = act_pool.get(Bflat)
        if buf is None:
            buf = torch.empty((n_layers, Bflat, D), device=device, dtype=dtype_out)
            act_pool[Bflat] = buf
        return buf

    def get_upd(Bflat: int) -> torch.Tensor:
        buf = upd_pool.get(Bflat)
        if buf is None:
            buf = torch.empty((Bflat, 2), device=device, dtype=torch.long)
            upd_pool[Bflat] = buf
        return buf

    try:
        with _HookLastTokenAllLayersToBuf(modules) as bank:
            for k0 in range(0, n_tasks, max(1, k_step)):
                k1 = min(k0 + max(1, k_step), n_tasks)
                K = k1 - k0

                for b0 in range(0, B, max(1, b_step)):
                    b1 = min(b0 + max(1, b_step), B)
                    bchunk = b1 - b0

                    base_cpu = samples[k0:k1, b0:b1, :max_cache_len]
                    base_dev = base_cpu.to(device, non_blocking=on_cuda)
                    base_pairs = base_dev.view(K, bchunk, Tm1, 2)

                    v_ids_task = torch.arange(V, device=device, dtype=torch.long).repeat_interleave(bchunk)
                    v_ids = v_ids_task.repeat(K)
                    Bflat = K * V * bchunk

                    probe = torch.empty((Bflat, 2), device=device, dtype=torch.long)
                    probe[:, 0] = v_ids
                    probe[:, 1] = sep_id

                    kvs = get_kvs(Bflat)
                    act = get_act(Bflat)
                    upd = get_upd(Bflat)
                    bank.set_buf(act)

                    for c in kvs:
                        c.cur_len = 0

                    for t00 in range(0, Tm1, max(1, t_step)):
                        t11 = min(t00 + max(1, t_step), Tm1)
                        tchunk = t11 - t00

                        stage = torch.empty((n_layers, K, tchunk, V, bchunk, D), device=device, dtype=dtype_out)

                        for ti, t in enumerate(range(t00, t11)):
                            bank.set_enabled(True)
                            _ = model(probe, kv_caches=kvs, cache_pos=None, update_cache=False)
                            bank.set_enabled(False)

                            stage[:, :, ti].copy_(act.view(n_layers, K, V, bchunk, D))

                            base_pair_t = base_pairs[:, :, t, :]

                            upd_view = upd.view(K, V, bchunk, 2)
                            upd_view.copy_(base_pair_t[:, None, :, :].expand(K, V, bchunk, 2))

                            _ = model(upd, kv_caches=kvs, cache_pos=None, update_cache=True)

                        dst = out[:, k0:k1, t00:t11, :, b0:b1, :]
                        dst.copy_(stage, non_blocking=async_d2h)

            if on_cuda and async_d2h:
                torch.cuda.synchronize(device)

    finally:
        model.output_layer = old_head

    return out


# Legacy ultra task vec extraction (moved to icl.latent_markov.legacy.ultra_task_vecs)
from icl.latent_markov.legacy.ultra_task_vecs import (  # noqa: F401, E402
    compute_hiddens_onepos_all_layers_ultra,
)
