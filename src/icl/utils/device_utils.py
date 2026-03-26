"""Shared device utilities for consistent CUDA/CPU handling across the codebase.
Re-exports from icl.device_utils to avoid circular imports when configs load.
"""

from icl.device_utils import get_default_device

__all__ = ["get_default_device"]
