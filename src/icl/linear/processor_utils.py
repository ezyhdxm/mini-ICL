"""Utility functions for linear processor operations."""

from typing import Optional

from icl.utils.device_utils import get_default_device

__all__ = ["get_default_device", "setup_device"]


def setup_device(device: Optional[str] = None) -> str:
    """Setup and return the appropriate device. Prefer config.device in callers; this is for when no config is available."""
    return device if device is not None else get_default_device()
