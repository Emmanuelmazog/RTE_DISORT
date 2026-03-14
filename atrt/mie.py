"""Mie theory — aerosol optical properties from lognormal size distributions."""

from __future__ import annotations

import numpy as np
from typing import Tuple


# Aerosol type presets: (r_g_um, sigma_g, n_real, n_imag)
AEROSOL_TYPES = {
    "continental":    (0.03, 2.24, 1.53, 0.008),
    "urban":          (0.03, 2.24, 1.55, 0.025),
    "maritime":       (0.40, 2.51, 1.38, 0.001),
    "desert_dust":    (0.50, 2.20, 1.53, 0.005),
    "biomass_burning":(0.07, 1.80, 1.52, 0.020),
}


def _mie_q(m: complex, x: float, n_max: int = 0) -> Tuple[float, float, float]:
    """Compute Mie efficiencies Qext, Qsca, and asymmetry parameter g.

    Parameters
    ----------
    m : complex
        Complex refractive index of the sphere (n + ik).
    x : float
        Size parameter  2 pi r / lambda.
    n_max : int
        Number of terms; if 0 use Wiscombe criterion.
    """
    if x < 1e-6:
        return 0.0, 0.0, 0.0

    if n_max == 0:
        n_max = int(x + 4.0 * x ** (1.0 / 3.0) + 2)

    # Logarithmic derivative D_n(mx) by downward recurrence
    mx = m * x
    nmx = int(max(n_max, abs(mx)) + 16)
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)

    # Riccati-Bessel functions psi_n, chi_n
    psi = np.zeros(n_max + 1)
    chi = np.zeros(n_max + 1)
    psi[0] = np.sin(x)
    psi[1] = np.sin(x) / x - np.cos(x)
    chi[0] = np.cos(x)
    chi[1] = np.cos(x) / x + np.sin(x)

    for n in range(1, n_max):
        psi[n + 1] = (2 * n + 1) / x * psi[n] - psi[n - 1]
        chi[n + 1] = (2 * n + 1) / x * chi[n] - chi[n - 1]

    xi = psi - 1j * chi

    Qext = 0.0
    Qsca = 0.0
    g_num = 0.0

    a_prev = 0.0 + 0j
    b_prev = 0.0 + 0j

    for n in range(1, n_max + 1):
        a_n = ((D[n] / m + n / x) * psi[n] - psi[n - 1]) / \
              ((D[n] / m + n / x) * xi[n] - xi[n - 1])
        b_n = ((D[n] * m + n / x) * psi[n] - psi[n - 1]) / \
              ((D[n] * m + n / x) * xi[n] - xi[n - 1])

        Qext += (2 * n + 1) * np.real(a_n + b_n)
        Qsca += (2 * n + 1) * (abs(a_n) ** 2 + abs(b_n) ** 2)

        if n > 1:
            g_num += ((n - 1) * (n + 1) / n) * np.real(
                a_prev * np.conj(a_n) + b_prev * np.conj(b_n)
            )
            g_num += ((2 * (n - 1) + 1) / ((n - 1) * n)) * np.real(
                a_prev * np.conj(b_prev)
            )

        a_prev = a_n
        b_prev = b_n

    Qext *= 2.0 / x ** 2
    Qsca *= 2.0 / x ** 2
    g_val = (4.0 / (Qsca * x ** 2)) * g_num if Qsca > 0 else 0.0

    return Qext, Qsca, g_val


def compute_aerosol_optics(
    wavelength_nm: float,
    r_g_um: float = 0.1,
    sigma_g: float = 1.5,
    m_real: float = 1.53,
    m_imag: float = 0.008,
    n_radii: int = 200,
) -> dict:
    """Compute bulk aerosol optical properties from a lognormal size distribution.

    Uses built-in Mie calculations (Bohren & Huffman).

    Returns dict with keys: Bext, Bsca, SSA, g.
    """
    lam_um = wavelength_nm * 1e-3

    # Logarithmic radius grid (um)
    r_min = r_g_um / 10.0
    r_max = r_g_um * 10.0
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    dr = np.diff(radii)

    # Lognormal number distribution (per um)
    ln_sg = np.log(sigma_g)
    n_r = (1.0 / (radii * ln_sg * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((np.log(radii) - np.log(r_g_um)) / ln_sg) ** 2
    )

    m = complex(m_real, m_imag)
    Qext_arr = np.zeros(n_radii)
    Qsca_arr = np.zeros(n_radii)
    g_arr    = np.zeros(n_radii)

    for i, r in enumerate(radii):
        x = 2.0 * np.pi * r / lam_um
        Qext_arr[i], Qsca_arr[i], g_arr[i] = _mie_q(m, x)

    # Trapezoidal integration
    geo_mid = np.pi * (0.5 * (radii[:-1] + radii[1:])) ** 2
    n_mid = 0.5 * (n_r[:-1] + n_r[1:])
    Qext_mid = 0.5 * (Qext_arr[:-1] + Qext_arr[1:])
    Qsca_mid = 0.5 * (Qsca_arr[:-1] + Qsca_arr[1:])
    g_mid = 0.5 * (g_arr[:-1] + g_arr[1:])

    Bext = np.sum(Qext_mid * geo_mid * n_mid * dr)
    Bsca = np.sum(Qsca_mid * geo_mid * n_mid * dr)
    SSA = Bsca / Bext if Bext > 0 else 1.0
    g_bulk = np.sum(g_mid * Qsca_mid * geo_mid * n_mid * dr) / Bsca if Bsca > 0 else 0.0

    return {"Bext": Bext, "Bsca": Bsca, "SSA": SSA, "g": g_bulk}


def get_aerosol_preset(aerosol_type: str, wavelength_nm: float = 550.0) -> dict:
    """Get aerosol optical properties for a named preset type."""
    if aerosol_type not in AEROSOL_TYPES:
        raise ValueError(f"Unknown aerosol type '{aerosol_type}'. "
                         f"Choose from: {list(AEROSOL_TYPES.keys())}")
    r_g, sig_g, nr, ni = AEROSOL_TYPES[aerosol_type]
    return compute_aerosol_optics(wavelength_nm, r_g, sig_g, nr, ni)
