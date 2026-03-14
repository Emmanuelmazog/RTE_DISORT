"""Gauss-Legendre and Double-Gauss quadrature schemes."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def gauleg(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature nodes and weights on [-1, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def double_gauss(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Double-Gauss quadrature: GL applied separately to each hemisphere.

    Returns *n* nodes ordered as [negative (downward), positive (upward)]
    with paired symmetry  mu_up[i] = -mu_down[i].
    """
    if n % 2 != 0:
        raise ValueError("n must be even for double-Gauss quadrature.")
    n_half = n // 2
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_half)
    # Map GL nodes from [-1,1] to [0,1]:  mu_j = (1 + x_j) / 2
    mu_pos = (1.0 + x_gl) / 2.0
    w_half = w_gl / 2.0
    # Downward (negative) first, upward (positive) second
    mus = np.concatenate([-mu_pos, mu_pos])
    weights = np.concatenate([w_half, w_half])
    return mus, weights
