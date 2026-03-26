"""General-purpose hidden-state extraction for linear regression models.

The functions here are pad-mode agnostic: callers supply ``task_pos``
explicitly so the same code works for ``pad="none"``, ``"bos"``, and
``"mapsto"`` layouts.

Padded-sequence specific wrappers (``compute_hiddens``,
``compute_hiddens_multi``, ``compute_hiddens_token_conditioned``) that
hardcode mapsto positions have been moved to
``icl.linear.legacy.task_vecs_padded``.
"""

import torch
from typing import Union, Sequence, Tuple


def extract_hidden(
        model, demo_data, demo_target, l=0, task_pos: Union[int, torch.Tensor] = 1
    ):
    """Extract hidden vectors from a single layer via forward hook."""
    extracted_vector = {}

    def hook_fn(module, input, output):
        extracted_vector['vector'] = output[:, task_pos, :].detach()

    hook_handle = model.transformer.blocks[l].attn_block.register_forward_hook(hook_fn)
    with torch.no_grad(): _ = model(demo_data, demo_target)
    hook_handle.remove()
    return extracted_vector['vector']


@torch.no_grad()
def extract_hidden_multi(
    model,
    demo_data,
    demo_target,
    layers: Sequence[int],
    task_pos: Union[int, torch.Tensor] = 1,
    *,
    module_getter=None,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
) -> torch.Tensor:
    """
    Extract hidden vectors from several layers in ONE forward pass.

    Parameters
    ----------
    post_layernorm : bool
        If True, apply the LayerNorm that immediately follows the
        extraction point.  For ``"post_attn"`` this is LN2 (before MLP);
        for ``"post_mlp"`` this is LN1 of the next block (or ``ln_f``
        for the last block).
    extraction_point : str
        ``"post_attn"`` — after attention block (default).
        ``"post_mlp"``  — after the full block (attention + MLP).

    Returns:
      - if task_pos is int:      (L, B, D)
      - if task_pos is tensor:   (L, B, P, D) where P=len(task_pos)
    """
    n_blocks = len(model.transformer.blocks)

    if module_getter is None:
        if extraction_point == "post_mlp":
            module_getter = lambda l: model.transformer.blocks[l]
        else:
            module_getter = lambda l: model.transformer.blocks[l].attn_block

    layers = list(layers)
    out_by_layer = {}
    handles = []

    def _apply_next_ln(h, l):
        """Apply the LayerNorm that immediately follows extraction_point at layer l."""
        if extraction_point == "post_attn":
            blk = model.transformer.blocks[l]
            if hasattr(blk, "mlp_block") and hasattr(blk.mlp_block, "ln_2"):
                return blk.mlp_block.ln_2(h)
        elif extraction_point == "post_mlp":
            if l + 1 < n_blocks:
                next_blk = model.transformer.blocks[l + 1]
                if hasattr(next_blk, "attn_block") and hasattr(next_blk.attn_block, "ln_1"):
                    return next_blk.attn_block.ln_1(h)
            elif hasattr(model.transformer, "ln_f"):
                return model.transformer.ln_f(h)
        return h

    def make_hook(l):
        def hook_fn(module, inputs, output):
            if isinstance(task_pos, int):
                h = output[:, task_pos, :].detach()
            else:
                pos = task_pos.to(output.device)
                h = output.index_select(dim=1, index=pos).detach()
            if post_layernorm:
                h = _apply_next_ln(h, l)
            out_by_layer[l] = h
        return hook_fn

    for l in layers:
        h = module_getter(l).register_forward_hook(make_hook(l))
        handles.append(h)

    try:
        _ = model(demo_data, demo_target)
    finally:
        for h in handles:
            h.remove()

    stacked = torch.stack([out_by_layer[l] for l in layers], dim=0)
    return stacked
