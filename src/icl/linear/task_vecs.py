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
) -> torch.Tensor:
    """
    Extract hidden vectors from several layers in ONE forward pass.

    Returns:
      - if task_pos is int:      (L, B, D)
      - if task_pos is tensor:   (L, B, P, D) where P=len(task_pos)
    """
    if module_getter is None:
        module_getter = lambda l: model.transformer.blocks[l].attn_block

    layers = list(layers)
    out_by_layer = {}
    handles = []

    def make_hook(l):
        def hook_fn(module, inputs, output):
            if isinstance(task_pos, int):
                out_by_layer[l] = output[:, task_pos, :].detach()
            else:
                pos = task_pos.to(output.device)
                out_by_layer[l] = output.index_select(dim=1, index=pos).detach()
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
