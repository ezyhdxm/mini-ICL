"""Utility functions for linear processor operations."""

from typing import Optional

import torch


def setup_device(device: Optional[str] = None) -> str:
    """Setup and return the appropriate device."""
    return device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
