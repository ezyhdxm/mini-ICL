from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Literal

import torch
from icl.models.kv_cache import KVCache

ActivationKind = Literal["attn_block", "mlp"]


def _get_n_emb(config) -> int:
    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
    return int(n_emb)


def _infer_n_head_head_dim(model) -> tuple[int, int]:
    mha = model.layers[0].attn_block.MHA
    return int(mha.n_head), int(mha.head_dim)


def _alloc_kv_caches(
    *,
    model: torch.nn.Module,
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> List[KVCache]:
    """Allocate KV caches (one per layer) for a fixed batch_size and max_len."""
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
    Hooks modules once.
    When enabled, each forward writes last-token activations into a pre-set buffer:
      buf[i] <- out[:, -1, :] for module i

    This lets us do ONE D2H copy per probe (for all layers), rather than one per layer.
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
                    raise RuntimeError("Hook buffer not set (call set_buf before forward).")

                if isinstance(out, (tuple, list)):
                    out = out[0]
                if out.ndim != 3:
                    raise RuntimeError(f"Expected (B,T,D), got {tuple(out.shape)}")

                # out[:, -1, :] is the LAST token in this forward.
                # In probe calls, tokens are [v, sep] so -1 corresponds to sep.
                self.buf[i].copy_(out[:, -1, :])

            self.handles.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.inference_mode()
def compute_hiddens_onepos_all_layers_kvcache(
    config,
    model: torch.nn.Module,
    samples: torch.Tensor,   # (n_tasks, B, Sfull) on CPU
    *,
    activation: ActivationKind = "attn_block",
    dtype_out: torch.dtype = torch.float16,

    # chunking (these matter a lot for speed; bigger => fewer loops)
    k_step: int = 4,
    b_step: int = 16,
    v_step: int = 8,

    # staging copies over time dimension (reduces number of D2H copies)
    t_step: int = 8,

    # transfers
    pin_samples: bool = True,
    pin_output: bool = False,
) -> torch.Tensor:
    """
    FAST KV-cache version.

    Maintains the KV cache across t:
      - probe with [v, sep] using update_cache=False
      - then advance prefix by writing base [x_t, sep] using update_cache=True
        (overwrites probe tokens and advances cur_len)

    Returns: out (n_layers, n_tasks, Tm1, V, B, D) on CPU.
    """
    model_device = next(model.parameters()).device
    on_cuda = (model_device.type == "cuda")
    model.eval()

    # ---- validate inputs ----
    if samples.device.type != "cpu":
        raise ValueError("Provide samples on CPU for pinned-transfer path.")
    if samples.dtype != torch.long:
        samples = samples.long()
    if on_cuda and pin_samples:
        samples = samples.pin_memory()

    if samples.ndim != 3:
        raise ValueError(f"samples must be (n_tasks,B,Sfull), got {tuple(samples.shape)}")

    n_tasks, B, Sfull = samples.shape
    seq_len = (Sfull + 1) // 2
    if 2 * seq_len - 1 != Sfull:
        raise ValueError(f"Expected Sfull = 2*seq_len-1, got Sfull={Sfull}")
    Tm1 = seq_len - 1  # number of sep positions we care about

    D = _get_n_emb(config)
    n_layers = len(model.layers)

    # sep is stored at odd positions by convention
    sep_id = int(samples[..., 1].max().item())
    V = sep_id

    # We only need cache positions up to end=2*Tm1 (probe/update max end)
    max_cache_len = 2 * Tm1  # for Sfull=257, this is 256

    # hook targets
    if activation == "attn_block":
        modules = [layer.attn_block for layer in model.layers]
    elif activation == "mlp":
        modules = [layer.mlp for layer in model.layers]
    else:
        raise ValueError(f"activation must be 'attn_block' or 'mlp', got {activation!r}")

    # output allocation
    out_pin = bool(on_cuda and pin_output)
    out = torch.empty(
        (n_layers, n_tasks, Tm1, V, B, D),
        device="cpu",
        dtype=dtype_out,
        pin_memory=out_pin,
    )
    async_d2h = out_pin  # only reliable when destination is pinned

    # disable output head to avoid logits
    old_head = model.output_layer
    model.output_layer = torch.nn.Identity()

    # ---- pools to avoid allocations in hot loop ----
    kv_pool: Dict[int, List[KVCache]] = {}                # key: Bflat
    act_buf_pool: Dict[int, torch.Tensor] = {}            # (n_layers, Bflat, D) on GPU
    upd_buf_pool: Dict[int, torch.Tensor] = {}            # (Bflat, 2) on GPU long
    probe_pool: Dict[Tuple[int, int, int, int, int], torch.Tensor] = {}  # (K,bchunk,v0,vchunk,sep) -> (Bflat,2)
    stage_pool: Dict[Tuple[int, int, int, int, int], torch.Tensor] = {}  # (K,bchunk,vchunk,tchunk,Bflat) -> stage

    def get_kvs(Bflat: int) -> List[KVCache]:
        kvs = kv_pool.get(Bflat)
        if kvs is None:
            kvs = _alloc_kv_caches(model=model, batch_size=Bflat, max_len=max_cache_len, device=model_device)
            kv_pool[Bflat] = kvs
        return kvs

    def get_act_buf(Bflat: int) -> torch.Tensor:
        buf = act_buf_pool.get(Bflat)
        if buf is None:
            buf = torch.empty((n_layers, Bflat, D), device=model_device, dtype=dtype_out)
            act_buf_pool[Bflat] = buf
        return buf

    def get_upd_buf(Bflat: int) -> torch.Tensor:
        buf = upd_buf_pool.get(Bflat)
        if buf is None:
            buf = torch.empty((Bflat, 2), device=model_device, dtype=torch.long)
            upd_buf_pool[Bflat] = buf
        return buf

    def get_probe(K: int, bchunk: int, v0: int, vchunk: int) -> torch.Tensor:
        key = (K, bchunk, v0, vchunk, sep_id)
        p = probe_pool.get(key)
        if p is None:
            # Batch order: task-major, then v-major, then b-major.
            # For one task:
            #   v_ids_task = [v0 repeated bchunk, v0+1 repeated bchunk, ...]
            v_ids_task = torch.arange(v0, v0 + vchunk, device=model_device, dtype=torch.long).repeat_interleave(bchunk)
            v_ids = v_ids_task.repeat(K)  # task-major blocks

            p = torch.empty((K * vchunk * bchunk, 2), device=model_device, dtype=torch.long)
            p[:, 0] = v_ids
            p[:, 1] = sep_id
            probe_pool[key] = p
        return p

    def get_stage(K: int, bchunk: int, vchunk: int, tchunk: int, Bflat: int) -> torch.Tensor:
        key = (K, bchunk, vchunk, tchunk, Bflat)
        st = stage_pool.get(key)
        if st is None:
            # stage holds outputs for a block of t values to reduce D2H calls
            st = torch.empty((n_layers, K, tchunk, vchunk, bchunk, D), device=model_device, dtype=dtype_out)
            stage_pool[key] = st
        return st

    try:
        with _HookLastTokenAllLayersToBuf(modules) as bank:
            for k0 in range(0, n_tasks, max(1, k_step)):
                k1 = min(k0 + max(1, k_step), n_tasks)
                K = k1 - k0

                for b0 in range(0, B, max(1, b_step)):
                    b1 = min(b0 + max(1, b_step), B)
                    bchunk = b1 - b0

                    # Move only the tokens needed for updates: positions [0 : 2*Tm1)
                    # shape: (K, bchunk, max_cache_len)
                    base_cpu = samples[k0:k1, b0:b1, :max_cache_len]  # CPU (possibly pinned)
                    base_dev = base_cpu.to(model_device, non_blocking=on_cuda)

                    # Group into pairs (x_t, sep) for each t
                    # shape: (K, bchunk, Tm1, 2)
                    base_pairs = base_dev.view(K, bchunk, Tm1, 2)

                    for v0 in range(0, V, max(1, v_step)):
                        v1 = min(v0 + max(1, v_step), V)
                        vchunk = v1 - v0
                        Bflat = K * vchunk * bchunk

                        kvs = get_kvs(Bflat)
                        act_buf = get_act_buf(Bflat)
                        upd_buf = get_upd_buf(Bflat)
                        probe = get_probe(K, bchunk, v0, vchunk)

                        # reset caches
                        for c in kvs:
                            c.cur_len = 0

                        bank.set_buf(act_buf)

                        # Process t sequentially (cache evolves), but copy results in chunks of t_step
                        for t00 in range(0, Tm1, max(1, t_step)):
                            t11 = min(t00 + max(1, t_step), Tm1)
                            tchunk = t11 - t00

                            stage = get_stage(K, bchunk, vchunk, tchunk, Bflat)

                            for ti, t in enumerate(range(t00, t11)):
                                # ---- PROBE: [v, sep] without advancing cache ----
                                bank.set_enabled(True)
                                _ = model(probe, kv_caches=kvs, cache_pos=None, update_cache=False)
                                bank.set_enabled(False)

                                # act_buf: (n_layers, Bflat, D)
                                # reshape batch dim into (K, vchunk, bchunk) (task-major, v-major, b-major)
                                act_view = act_buf.view(n_layers, K, vchunk, bchunk, D)
                                stage[:, :, ti].copy_(act_view)

                                # ---- UPDATE: write the true base [x_t, sep] advancing cache ----
                                # base_pair_t: (K, bchunk, 2)
                                base_pair_t = base_pairs[:, :, t, :]

                                # Fill upd_buf in the same batch order: (K, vchunk, bchunk, 2)
                                upd_view = upd_buf.view(K, vchunk, bchunk, 2)
                                upd_view.copy_(base_pair_t[:, None, :, :].expand(K, vchunk, bchunk, 2))

                                _ = model(upd_buf, kv_caches=kvs, cache_pos=None, update_cache=True)

                            # One D2H copy for the whole tchunk block
                            dst = out[:, k0:k1, t00:t11, v0:v1, b0:b1, :]
                            dst.copy_(stage, non_blocking=async_d2h)

            if on_cuda and async_d2h:
                torch.cuda.synchronize(model_device)

    finally:
        model.output_layer = old_head

    return out
