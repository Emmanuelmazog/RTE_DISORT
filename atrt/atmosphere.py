"""US Standard Atmosphere 1976 with Rayleigh scattering and aerosol mixing."""

from __future__ import annotations

import numpy as np

from .profile import AtmosphereProfile


def rayleigh_cross_section(wavelength_nm: float) -> float:
    """Rayleigh scattering cross section per molecule (cm^2).

    Simplified from Bodhaine et al. (1999).
    """
    lam_um = wavelength_nm * 1e-3
    return 4.02e-28 / lam_um ** 4.04


class StandardAtmosphere:
    """US Standard Atmosphere 1976 simplified for 20 layers (0-100 km).

    Provides temperature, pressure, and air number density profiles,
    and builds an :class:`AtmosphereProfile` by combining molecular
    Rayleigh scattering with a user-specified aerosol loading.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometres (default 550).
    """

    # 21 boundaries → 20 layers  (km)
    Z_BOUNDS = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12,
                         15, 20, 25, 30, 35, 40, 50, 60, 70, 85, 100],
                        dtype=float)

    # US-76 lapse-rate segments (z_base km, z_top km, L K/km)
    _LAPSE = [
        (0,   11,  -6.5),
        (11,  20,   0.0),
        (20,  32,   1.0),
        (32,  47,   2.8),
        (47,  51,   0.0),
        (51,  71,  -2.8),
        (71, 100,  -2.0),
    ]

    # Physical constants
    _g0 = 9.80665            # m/s^2
    _M  = 0.0289644          # kg/mol  (dry air)
    _R  = 8.31447            # J/(mol·K)
    _k  = 1.380649e-23       # J/K
    _N_A = 6.02214076e23     # mol^-1

    def __init__(self, wavelength_nm: float = 550.0) -> None:
        self.wavelength_nm = wavelength_nm
        n_bounds = len(self.Z_BOUNDS)
        n_layers = n_bounds - 1

        # Compute T, P, n_air at each boundary
        self.T = np.zeros(n_bounds)
        self.P = np.zeros(n_bounds)
        self.T[0] = 288.15     # K
        self.P[0] = 101325.0   # Pa

        for i in range(1, n_bounds):
            z_km = self.Z_BOUNDS[i]
            z_prev = self.Z_BOUNDS[i - 1]
            # Find lapse rate for z_prev
            L = self._lapse_at(z_prev)
            dz = (z_km - z_prev) * 1e3  # m

            if abs(L) < 1e-10:  # isothermal
                self.T[i] = self.T[i - 1]
                self.P[i] = self.P[i - 1] * np.exp(
                    -self._g0 * self._M * dz / (self._R * self.T[i - 1]))
            else:
                L_per_m = L * 1e-3  # K/m
                self.T[i] = self.T[i - 1] + L_per_m * dz
                self.P[i] = self.P[i - 1] * (
                    self.T[i] / self.T[i - 1]
                ) ** (-self._g0 * self._M / (self._R * L_per_m))

        # Number density n = P / (k T)  [m^-3] → convert to cm^-3
        self.n_air = (self.P / (self._k * self.T)) * 1e-6  # cm^-3

        # Layer-mean values
        self.z_mid = 0.5 * (self.Z_BOUNDS[:-1] + self.Z_BOUNDS[1:])  # km
        self.dz = (self.Z_BOUNDS[1:] - self.Z_BOUNDS[:-1]) * 1e5     # cm
        self.n_air_mid = 0.5 * (self.n_air[:-1] + self.n_air[1:])

        # Rayleigh optical depth per layer
        sigma_ray = rayleigh_cross_section(wavelength_nm)
        self.tau_ray = sigma_ray * self.n_air_mid * self.dz

    def _lapse_at(self, z_km: float) -> float:
        """Return lapse rate (K/km) at altitude z_km."""
        for z_base, z_top, L in self._LAPSE:
            if z_km < z_top or np.isclose(z_km, z_top):
                return L
        return self._LAPSE[-1][2]

    def to_profile(
        self,
        aod_total: float = 0.3,
        ssa_aer: float = 0.92,
        g_aer: float = 0.70,
        H_aer_km: float = 2.0,
        albedo: float = 0.10,
        theta0: float = np.deg2rad(30),
        f0: float = np.pi,
    ) -> AtmosphereProfile:
        """Build an AtmosphereProfile combining Rayleigh + aerosol.

        Aerosol vertical distribution is exponential:
            tau_aer(z) ~ exp(-z / H_aer_km),  normalised to *aod_total*.

        Parameters
        ----------
        aod_total : float
            Total column aerosol optical depth.
        ssa_aer : float
            Aerosol single-scattering albedo.
        g_aer : float
            Aerosol asymmetry parameter.
        H_aer_km : float
            Aerosol scale height (km).
        albedo : float
            Surface albedo.
        theta0 : float
            Solar zenith angle (radians).
        """
        n_layers = len(self.z_mid)

        # Aerosol tau per layer (exponential profile)
        weights = np.exp(-self.z_mid / H_aer_km)
        weights /= weights.sum()
        tau_aer = aod_total * weights

        # Combined per-layer optical properties
        tau_total = tau_aer + self.tau_ray
        scat_aer = tau_aer * ssa_aer
        scat_ray = self.tau_ray * 1.0  # Rayleigh SSA = 1
        scat_total = scat_aer + scat_ray

        omega = np.where(tau_total > 0, scat_total / tau_total, 0.0)
        omega = np.clip(omega, 0.0, 0.999999)

        g_eff = np.where(scat_total > 0,
                         (scat_aer * g_aer + scat_ray * 0.0) / scat_total,
                         0.0)

        return AtmosphereProfile(
            aod=tau_total, ssa=omega, g=g_eff,
            albedo=albedo, theta0=theta0, f0=f0,
        )
