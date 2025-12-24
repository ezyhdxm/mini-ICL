from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Optional

import torch

ActivationKind = Literal["attn_block", "mlp"]


def _get_n_emb(config) -> int:
    n_emb = getattr(config.model, "emb_dim", getattr(config.model, "n_embd", None))
    if n_emb is None:
        raise ValueError("config.model must contain emb_dim or n_embd")
    return int(n_emb)


class _MultiLayerHookBankOnce:
    """
    Register forward hooks once and reuse them.
    For each forward, set `pos` then run model; hooks will populate `store[i]` with (Bflat, D).
    """
    def __init__(self, modules: List[torch.nn.Module]):
        self.modules = modules
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.store: List[Optional[torch.Tensor]] = [None] * len(modules)
        self.pos: Optional[torch.Tensor] = None
        self._row: Optional[torch.Tensor] = None  # cached arange(Bflat) on the right device

    def set_pos(self, pos: torch.Tensor) -> None:
        if pos.dtype != torch.long or pos.ndim != 1:
            raise ValueError(f"pos must be 1D long, got {pos.dtype} shape={tuple(pos.shape)}")
        self.pos = pos
        bflat = pos.numel()
        if self._row is None or self._row.numel() != bflat or self._row.device != pos.device:
            self._row = torch.arange(bflat, device=pos.device, dtype=torch.long)

    def clear(self) -> None:
        # Clear previous forward's results to avoid stale reuse if something goes wrong.
        for i in range(len(self.store)):
            self.store[i] = None

    def __enter__(self):
        for i, m in enumerate(self.modules):
            # Bind i as default arg so each hook gets its own index.
            def hook(_module, _inp, out, i=i):
                # Some modules might return tuples; we only want the main tensor.
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if not torch.is_tensor(out):
                    raise RuntimeError(f"Hook expected Tensor output, got {type(out)}")

                pos = self.pos
                row = self._row
                if pos is None or row is None:
                    raise RuntimeError("Hook bank pos/row not set before forward")

                if out.ndim != 3:
                    raise RuntimeError(f"Expected 3D output, got shape={tuple(out.shape)}")

                bflat = pos.numel()

                # Exact layout resolution:
                # - (B,S,D): out.shape[0] == Bflat
                # - (S,B,D): out.shape[1] == Bflat
                if out.shape[0] == bflat:
                    # (B,S,D) -> select per-row position
                    vec = out[row, pos]  # (Bflat, D)
                elif out.shape[1] == bflat:
                    # (S,B,D) -> select per-row position without transposing
                    vec = out[pos, row]  # (Bflat, D)
                else:
                    raise RuntimeError(
                        f"Cannot infer layout from out.shape={tuple(out.shape)} with Bflat={bflat}"
                    )

                self.store[i] = vec.detach()

            self.handles.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.inference_mode()
def compute_hiddens_onepos_all_layers_ultra(
    config,
    model: torch.nn.Module,
    sampler,
    samples: torch.Tensor,
    *,
    activation: ActivationKind = "attn_block",
    dtype_out: torch.dtype = torch.float16,
    # chunk sizes (OOM control)
    k_step: int = 2,
    t_step: int = 16,
    v_step: int = 32,
    b_step: int = 8,
    # transfer / memory
    pin_samples: bool = True,
    pin_output: bool = False,   # keep False unless outputs are small enough to pin safely
) -> torch.Tensor:
    """
    samples: (n_tasks, Tm1, V, B, Sfull) on CPU
    Returns:
      hiddens: (n_layers, n_tasks, Tm1, V, B, D) on CPU

    For each (k,t,v,b), we feed sequence samples[k,t,v,b,:] and extract activation at pos(t)=2*t+1 (0-based t).
    """
    model_device = next(model.parameters()).device
    on_cuda = (model_device.type == "cuda")
    model.eval()

    # ---- validate/normalize ----
    if samples.device.type != "cpu":
        raise ValueError("Provide samples on CPU (chunked transfer to GPU avoids OOM).")
    if samples.dtype != torch.long:
        samples = samples.long()
    if on_cuda and pin_samples:
        samples = samples.pin_memory()

    if samples.ndim != 5:
        raise ValueError(f"samples must be 5D, got shape {tuple(samples.shape)}")

    n_tasks, Tm1, V, B, Sfull = samples.shape
    seq_len_from_samples = (Sfull + 1) // 2
    if 2 * seq_len_from_samples - 1 != Sfull:
        raise ValueError(f"Expected Sfull = 2*seq_len-1, got Sfull={Sfull}")
    if seq_len_from_samples != Tm1 + 1:
        raise ValueError(f"Inconsistent: inferred seq_len={seq_len_from_samples}, but Tm1+1={Tm1+1}")
    if sampler is not None and hasattr(sampler, "seq_len"):
        if int(sampler.seq_len) != int(seq_len_from_samples):
            raise ValueError("sampler.seq_len mismatch vs samples")

    D = _get_n_emb(config)
    n_layers = len(model.layers)

    # ---- pick modules for all layers ----
    if activation == "attn_block":
        modules = [layer.attn_block for layer in model.layers]
    elif activation == "mlp":
        modules = [layer.mlp for layer in model.layers]
    else:
        raise ValueError(f"activation must be 'attn_block' or 'mlp', got {activation!r}")

    # ---- output ----
    out_pin = bool(on_cuda and pin_output)
    out = torch.empty(
        (n_layers, n_tasks, Tm1, V, B, D),
        device="cpu",
        dtype=dtype_out,
        pin_memory=out_pin,
    )
    async_d2h = out_pin  # non_blocking D2H only reliable when destination is pinned

    # Disable output head if present (avoid logits materialization)
    has_output_layer = hasattr(model, "output_layer")
    if has_output_layer:
        old_head = model.output_layer
        model.output_layer = torch.nn.Identity()

    try:
        # Register hooks ONCE
        with _MultiLayerHookBankOnce(modules) as bank:
            for k0 in range(0, n_tasks, max(1, k_step)):
                k1 = min(k0 + max(1, k_step), n_tasks)
                nK = k1 - k0

                for t0 in range(0, Tm1, max(1, t_step)):
                    t1 = min(t0 + max(1, t_step), Tm1)
                    tchunk = t1 - t0

                    # max pos in this t-chunk: 2*(t1-1)+1
                    pos_max = 2 * (t1 - 1) + 1
                    if pos_max >= Sfull:
                        raise RuntimeError(f"pos exceeds Sfull: max pos {pos_max} vs Sfull {Sfull}")

                    for v0 in range(0, V, max(1, v_step)):
                        v1 = min(v0 + max(1, v_step), V)
                        vchunk = v1 - v0

                        for b0 in range(0, B, max(1, b_step)):
                            b1 = min(b0 + max(1, b_step), B)
                            bchunk = b1 - b0

                            # View into CPU samples (still pinned if samples is pinned)
                            block = samples[k0:k1, t0:t1, v0:v1, b0:b1, :]  # (nK,tchunk,vchunk,bchunk,Sfull)

                            # Build pos on the SAME device we’ll index with
                            # Flatten order is (k, t, v, b)
                            if on_cuda:
                                pos_by_t = (2 * torch.arange(t0, t1, device=model_device, dtype=torch.long) + 1)
                                pos = pos_by_t.repeat_interleave(vchunk * bchunk).repeat(nK)  # (Bflat,)
                            else:
                                pos_by_t = (2 * torch.arange(t0, t1, dtype=torch.long) + 1)
                                pos = pos_by_t.repeat_interleave(vchunk * bchunk).repeat(nK)

                            # Move to GPU FIRST, then flatten there (avoids CPU contiguous() and preserves pinned H2D)
                            if on_cuda:
                                block = block.to(model_device, non_blocking=True)  # contiguous on GPU
                            batch = block.reshape(-1, Sfull)  # (Bflat,Sfull) on CPU or GPU

                            bank.set_pos(pos)
                            bank.clear()

                            _ = model(batch)

                            # Write out layer by layer (no per-layer CPU temp allocations)
                            for li, vec in enumerate(bank.store):
                                if vec is None:
                                    raise RuntimeError(f"Hook missing for layer {li}")

                                vec = vec.reshape(nK, tchunk, vchunk, bchunk, D)

                                dst = out[li, k0:k1, t0:t1, v0:v1, b0:b1]  # CPU view

                                # Cast on source device (GPU) then copy directly into dst
                                if vec.dtype != dtype_out:
                                    vec = vec.to(dtype_out)

                                dst.copy_(vec, non_blocking=async_d2h)

                                bank.store[li] = None  # release ASAP

                            del block, batch, pos

            if on_cuda and async_d2h:
                torch.cuda.synchronize(model_device)

    finally:
        if has_output_layer:
            model.output_layer = old_head

    return out
