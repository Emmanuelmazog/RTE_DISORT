"""Henyey-Greenstein phase function expansion and related matrices."""

from __future__ import annotations

import numpy as np


def legendre_expansion_hg(g: float, n_terms: int) -> np.ndarray:
    """Henyey-Greenstein expansion coefficients:  χ_l = (2l+1) g^l."""
    return np.array([(2 * l + 1) * (g ** l) for l in range(n_terms)])


def _legendre_polynomials(mu: np.ndarray, l_max: int) -> np.ndarray:
    """Evaluate Legendre polynomials P_l(mu) for l = 0 … l_max-1.

    Returns array of shape ``(l_max, len(mu))``.
    """
    n = len(mu)
    P = np.zeros((l_max, n))
    P[0, :] = 1.0
    if l_max > 1:
        P[1, :] = mu
    for l in range(1, l_max - 1):
        P[l + 1, :] = ((2 * l + 1) * mu * P[l, :] - l * P[l - 1, :]) / (l + 1)
    return P


def get_phase_matrix(mu_array: np.ndarray, g: float, n_streams: int) -> np.ndarray:
    """Phase matrix P[i, j] = Σ_l  χ_l P_l(μ_i) P_l(μ_j)."""
    l_max = n_streams
    chi = legendre_expansion_hg(g, l_max)
    P_leg = _legendre_polynomials(mu_array, l_max)           # (l_max, n)
    # Vectorised:  P = (chi · P_leg)^T  @  P_leg
    weighted = chi[:, None] * P_leg                           # (l_max, n)
    return weighted.T @ P_leg                                 # (n, n)


def get_beam_source(mu_array: np.ndarray, mu0: float, g: float,
                    omega: float, f0: float) -> np.ndarray:
    """Beam source vector  S(μ_i) = (ω F₀ / 4π) P(μ_i, −μ₀)."""
    n = len(mu_array)
    l_max = n
    chi = legendre_expansion_hg(g, l_max)

    # P_l(−μ₀)
    Pl_mu0 = np.zeros(l_max)
    Pl_mu0[0] = 1.0
    if l_max > 1:
        Pl_mu0[1] = -mu0
    for l in range(1, l_max - 1):
        Pl_mu0[l + 1] = ((2 * l + 1) * (-mu0) * Pl_mu0[l] - l * Pl_mu0[l - 1]) / (l + 1)

    P_leg = _legendre_polynomials(mu_array, l_max)           # (l_max, n)

    # P(μ_i, −μ₀) = Σ_l  χ_l P_l(μ_i) P_l(−μ₀)
    P_val = (chi * Pl_mu0) @ P_leg                           # (n,)
    # Note: chi*Pl_mu0 is element-wise product → shape (l_max,), then
    #       dot with P_leg (l_max, n) → (n,)

    return (omega * f0 / (4.0 * np.pi)) * P_val


def phase_function_at_angle(mu_user: float, mu_quad: np.ndarray,
                            g: float, n_terms: int) -> np.ndarray:
    """Phase function p(mu_user, mu_j) for one arbitrary angle vs all quadrature nodes.

    Parameters
    ----------
    mu_user : float
        Cosine of the arbitrary viewing angle (can be positive or negative).
    mu_quad : ndarray, shape (N,)
        Quadrature cosines (the μ_j nodes).
    g : float
        Henyey-Greenstein asymmetry parameter.
    n_terms : int
        Number of Legendre terms in the expansion.

    Returns
    -------
    ndarray, shape (N,)
        p(mu_user, mu_j) for each quadrature node j.
    """
    chi = legendre_expansion_hg(g, n_terms)

    # P_l(mu_user) via recurrence
    Pl_user = np.zeros(n_terms)
    Pl_user[0] = 1.0
    if n_terms > 1:
        Pl_user[1] = mu_user
    for l in range(1, n_terms - 1):
        Pl_user[l + 1] = ((2 * l + 1) * mu_user * Pl_user[l]
                          - l * Pl_user[l - 1]) / (l + 1)

    # P_l(mu_quad) for all nodes
    P_leg = _legendre_polynomials(mu_quad, n_terms)       # (n_terms, N)

    # p(mu_user, mu_j) = sum_l chi_l P_l(mu_user) P_l(mu_j)
    return (chi * Pl_user) @ P_leg                        # (N,)


def phase_function_scalar(mu1: float, mu2: float,
                          g: float, n_terms: int) -> float:
    """Phase function p(mu1, mu2) for two scalar angles.

    Useful for the beam term: p(mu, -mu0).

    Parameters
    ----------
    mu1, mu2 : float
        Cosines of the two directions.
    g : float
        Henyey-Greenstein asymmetry parameter.
    n_terms : int
        Number of Legendre terms in the expansion.

    Returns
    -------
    float
        p(mu1, mu2).
    """
    chi = legendre_expansion_hg(g, n_terms)

    Pl1 = np.zeros(n_terms)
    Pl1[0] = 1.0
    if n_terms > 1:
        Pl1[1] = mu1
    for l in range(1, n_terms - 1):
        Pl1[l + 1] = ((2 * l + 1) * mu1 * Pl1[l] - l * Pl1[l - 1]) / (l + 1)

    Pl2 = np.zeros(n_terms)
    Pl2[0] = 1.0
    if n_terms > 1:
        Pl2[1] = mu2
    for l in range(1, n_terms - 1):
        Pl2[l + 1] = ((2 * l + 1) * mu2 * Pl2[l] - l * Pl2[l - 1]) / (l + 1)

    return float(np.dot(chi * Pl1, Pl2))
