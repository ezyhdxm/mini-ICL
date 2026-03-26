"""Minimal device helper to avoid circular imports. Only depends on torch."""

import torch


def get_default_device() -> str:
    """Return the default compute device string.
    Uses 'cuda:0' when CUDA is available so multi-GPU setups can use cuda:1, etc. consistently.
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"
