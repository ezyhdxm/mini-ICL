"""Backward-compat shim — estimate_lambda_with_r2 now lives in icl.utils.linear_algebra_utils."""

from icl.utils.linear_algebra_utils import estimate_lambda_with_r2  # noqa: F401

import numpy as np


def compute_circumcenter(p1, p2, p3):
    """Compute the circumcenter (equidistant point) of three 2D points."""
    A = np.stack([p2 - p1, p3 - p1])
    b = np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1)
    ]) / 2
    x = np.linalg.lstsq(A.T, b, rcond=None)[0]
    return x
