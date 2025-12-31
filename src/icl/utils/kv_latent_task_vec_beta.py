from __future__ import annotations
from typing import List, Optional, Literal, Dict, Tuple

import torch
from icl.models.kv_cache import KVCache

ActivationKind = Literal["attn_block", "mlp"]


def _get_n_emb(config) -> int:
    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
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


class _HookLastTokenAllLayersToBuf:
    """
    When enabled, writes last-token activations into buf[i] for layer i:
      buf[i] <- out[:, -1, :]
    """
    def __init__(self, modules: List[torch.nn.Module]):
        self.modules = modules
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.enabled: bool = True
        self.buf: Optional[torch.Tensor] = None  # (n_layers, Bflat, D)

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
                # probe tokens are [v, sep], so -1 is the sep position
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
    samples: torch.Tensor,   # (n_tasks, B, Sfull) on CPU
    *,
    activation: ActivationKind = "attn_block",
    dtype_out: torch.dtype = torch.float16,

    k_step: int = 4,
    b_step: int = 16,
    t_step: int = 16,          # staging over t to reduce D2H calls

    pin_samples: bool = True,
    pin_output: bool = False,
) -> torch.Tensor:
    """
    Clean fast KV-cache version (no V chunking).

    For each (task chunk K, batch chunk bchunk):
      - Build a single probe batch containing ALL v (0..V-1):
          probe = [[0, sep], [0, sep],..., [1, sep],..., [V-1, sep]] with repeats over b and k
      - Maintain KV cache across t:
          probe(update_cache=False) -> capture sep hidden
          update with true [x_t, sep] (update_cache=True) -> advance cache

    Returns:
      out: (n_layers, n_tasks, Tm1, V, B, D) on CPU
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
    V = sep_id  # values are 0..V-1, sep is V

    max_cache_len = 2 * Tm1  # we only ever cache up to 2*Tm1

    # hook modules
    if activation == "attn_block":
        modules = [layer.attn_block for layer in model.layers]
    elif activation == "mlp":
        modules = [layer.mlp for layer in model.layers]
    else:
        raise ValueError(f"activation must be 'attn_block' or 'mlp', got {activation!r}")

    out_pin = bool(on_cuda and pin_output)
    out = torch.empty((n_layers, n_tasks, Tm1, V, B, D), device="cpu", dtype=dtype_out, pin_memory=out_pin)
    async_d2h = out_pin

    # disable head
    old_head = model.output_layer
    model.output_layer = torch.nn.Identity()

    # simple pools keyed by Bflat to avoid reallocations
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

                    # Move only needed prefix tokens to GPU (up to 2*Tm1)
                    base_cpu = samples[k0:k1, b0:b1, :max_cache_len]  # (K,bchunk,2*Tm1)
                    base_dev = base_cpu.to(device, non_blocking=on_cuda)
                    base_pairs = base_dev.view(K, bchunk, Tm1, 2)      # (K,bchunk,t,2)

                    # Build one PROBE batch for all v at once.
                    # Batch order: task-major, then v-major, then b-major.
                    # So batch dims correspond to view(..., K, V, bchunk, ...)
                    v_ids_task = torch.arange(V, device=device, dtype=torch.long).repeat_interleave(bchunk)  # (V*bchunk,)
                    v_ids = v_ids_task.repeat(K)  # (K*V*bchunk,)
                    Bflat = K * V * bchunk

                    probe = torch.empty((Bflat, 2), device=device, dtype=torch.long)
                    probe[:, 0] = v_ids
                    probe[:, 1] = sep_id

                    kvs = get_kvs(Bflat)
                    act = get_act(Bflat)
                    upd = get_upd(Bflat)
                    bank.set_buf(act)

                    # reset caches
                    for c in kvs:
                        c.cur_len = 0

                    # Process time in chunks to reduce D2H calls
                    for t00 in range(0, Tm1, max(1, t_step)):
                        t11 = min(t00 + max(1, t_step), Tm1)
                        tchunk = t11 - t00

                        # stage on GPU: (n_layers, K, tchunk, V, bchunk, D)
                        stage = torch.empty((n_layers, K, tchunk, V, bchunk, D), device=device, dtype=dtype_out)

                        for ti, t in enumerate(range(t00, t11)):
                            # 1) probe: capture sep hidden
                            bank.set_enabled(True)
                            _ = model(probe, kv_caches=kvs, cache_pos=None, update_cache=False)
                            bank.set_enabled(False)

                            # act: (n_layers, Bflat, D) -> view as (n_layers, K, V, bchunk, D)
                            stage[:, :, ti].copy_(act.view(n_layers, K, V, bchunk, D))

                            # 2) update: advance cache with true [x_t, sep], broadcast across V
                            base_pair_t = base_pairs[:, :, t, :]  # (K,bchunk,2)

                            upd_view = upd.view(K, V, bchunk, 2)
                            upd_view.copy_(base_pair_t[:, None, :, :].expand(K, V, bchunk, 2))

                            _ = model(upd, kv_caches=kvs, cache_pos=None, update_cache=True)

                        # copy this t-chunk to CPU output
                        dst = out[:, k0:k1, t00:t11, :, b0:b1, :]  # v dimension is full
                        dst.copy_(stage, non_blocking=async_d2h)

            if on_cuda and async_d2h:
                torch.cuda.synchronize(device)

    finally:
        model.output_layer = old_head

    return out
