"""AtmosphereProfile — optical properties of a plane-parallel atmosphere."""

from __future__ import annotations

import numpy as np


class AtmosphereProfile:
    """Defines the optical properties of an N-layer plane-parallel atmosphere.

    Accepts scalars (single homogeneous layer) or numpy arrays (multi-layer).
    Scalars are internally converted to length-1 arrays so the solver always
    iterates over arrays.

    Parameters
    ----------
    aod : float | array_like
        Aerosol Optical Depth **per layer**.  For a single-layer atmosphere
        this equals the total column AOD.
    ssa : float | array_like
        Single-Scattering Albedo (ω₀) per layer.
    g : float | array_like
        Henyey-Greenstein asymmetry parameter per layer.
    albedo : float
        Lambertian surface albedo (scalar).
    theta0 : float
        Solar zenith angle in **radians**.
    f0 : float
        Extra-terrestrial solar irradiance (default π  → convenient normalisation).
    """

    def __init__(
        self,
        aod: float | np.ndarray = 0.5,
        ssa: float | np.ndarray = 0.9,
        g: float | np.ndarray = 0.7,
        albedo: float = 0.1,
        theta0: float = np.deg2rad(30),
        f0: float = np.pi,
    ) -> None:
        # --- normalise to 1-D arrays ---
        self.tau: np.ndarray = np.atleast_1d(np.asarray(aod, dtype=float))
        self.ssa: np.ndarray = np.atleast_1d(np.asarray(ssa, dtype=float))
        self.g: np.ndarray   = np.atleast_1d(np.asarray(g, dtype=float))

        # Validate same length
        n = self.n_layers
        if self.ssa.shape[0] != n or self.g.shape[0] != n:
            raise ValueError(
                f"aod ({n}), ssa ({self.ssa.shape[0]}), and g ({self.g.shape[0]}) "
                "must have the same length."
            )

        # Scalar surface / illumination parameters
        self.albedo: float = float(albedo)
        self.theta0: float = float(theta0)
        self.mu0: float    = float(np.cos(theta0))
        self.f0: float     = float(f0)

    # ----- properties -----
    @property
    def n_layers(self) -> int:
        """Number of atmospheric layers."""
        return self.tau.shape[0]

    @property
    def tau_total(self) -> float:
        """Total column optical depth (sum of all layers)."""
        return float(np.sum(self.tau))

    @property
    def cumulative_tau(self) -> np.ndarray:
        """Cumulative optical depth at each layer interface.

        Returns array of length ``n_layers + 1``.
        ``cumulative_tau[0] = 0``  (TOA),
        ``cumulative_tau[-1] = tau_total``  (BOA).
        """
        return np.concatenate(([0.0], np.cumsum(self.tau)))

    def copy(self) -> "AtmosphereProfile":
        """Return a deep copy."""
        return AtmosphereProfile(
            aod=self.tau.copy(),
            ssa=self.ssa.copy(),
            g=self.g.copy(),
            albedo=self.albedo,
            theta0=self.theta0,
            f0=self.f0,
        )

    def __repr__(self) -> str:
        return (
            f"AtmosphereProfile(n_layers={self.n_layers}, "
            f"tau={self.tau}, ssa={self.ssa}, g={self.g}, "
            f"albedo={self.albedo:.3f}, θ₀={np.rad2deg(self.theta0):.1f}°)"
        )
