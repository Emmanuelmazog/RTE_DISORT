"""
atrt — Atmospheric Radiative Transfer package
==============================================

Discrete Ordinates (DISORT) forward & inverse solver for
plane-parallel atmospheres with aerosol retrieval capabilities.
"""

from .quadrature import gauleg, double_gauss
from .phase import (legendre_expansion_hg, get_phase_matrix, get_beam_source,
                     phase_function_at_angle)
from .profile import AtmosphereProfile
from .atmosphere import rayleigh_cross_section, StandardAtmosphere
from .mie import AEROSOL_TYPES, compute_aerosol_optics, get_aerosol_preset
from .solver import DisortSolver
from .inverse import InverseOptimizer, OptimalEstimation, PhillipsTwomey, TotalVariation

__all__ = [
    "gauleg", "double_gauss",
    "legendre_expansion_hg", "get_phase_matrix", "get_beam_source", "phase_function_at_angle",
    "AtmosphereProfile",
    "rayleigh_cross_section", "StandardAtmosphere",
    "AEROSOL_TYPES", "compute_aerosol_optics", "get_aerosol_preset",
    "DisortSolver",
    "InverseOptimizer", "OptimalEstimation", "PhillipsTwomey", "TotalVariation",
]
