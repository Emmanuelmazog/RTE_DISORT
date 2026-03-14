"""DisortSolver — Discrete-Ordinates eigenvalue solver for plane-parallel RT."""

from __future__ import annotations

import numpy as np
import scipy.linalg
from typing import List, Tuple

from .quadrature import double_gauss
from .phase import (get_phase_matrix, get_beam_source,
                     phase_function_at_angle, phase_function_scalar)
from .profile import AtmosphereProfile


class DisortSolver:
    """Discrete-Ordinates (eigenvalue) solver for the plane-parallel RTE.

    Supports single-layer and multi-layer atmospheres through a global
    boundary-condition matrix that enforces:
      • TOA  (τ = 0):        downward diffuse intensity = 0
      • Interfaces (τ_k):    continuity of all streams across layer boundaries
      • BOA  (τ_total):      Lambertian surface reflection

    Parameters
    ----------
    n_streams : int
        Number of Gauss-Legendre quadrature streams (total, including ±μ).
        Must be even.  Default 16.
    """

    def __init__(self, n_streams: int = 16) -> None:
        if n_streams % 2 != 0:
            raise ValueError("n_streams must be even.")
        self.n_streams: int = n_streams
        self.mus: np.ndarray
        self.weights: np.ndarray
        self.mus, self.weights = double_gauss(n_streams)

        # Index masks (set once)
        self.down_idx: np.ndarray = np.where(self.mus < 0)[0]
        self.up_idx: np.ndarray   = np.where(self.mus > 0)[0]
        self.n_half: int = len(self.down_idx)

    # ------------------------------------------------------------------
    # Delta-M scaling
    # ------------------------------------------------------------------
    def _apply_delta_m(self, tau: float, omega: float, g: float
                       ) -> Tuple[float, float, float]:
        """Apply delta-M scaling to truncate the forward scattering peak.

        The truncated fraction is  f = g^{N_streams}.
        Returns scaled (tau', omega', g').
        """
        f = g ** self.n_streams
        tau_p = tau * (1.0 - omega * f)
        denom = 1.0 - omega * f
        omega_p = omega * (1.0 - f) / denom if denom > 0 else omega
        g_p = (g - f) / (1.0 - f) if abs(1.0 - f) > 1e-15 else 0.0
        return tau_p, omega_p, g_p

    # ------------------------------------------------------------------
    # Per-layer eigen-decomposition
    # ------------------------------------------------------------------
    def _layer_eigen(self, omega: float, g: float) -> dict:
        """Reduced eigenvalue problem for one layer (DISORT formulation).

        Exploits the block symmetry of double-Gauss quadrature to solve
        an (N/2)×(N/2) system instead of N×N, guaranteeing real eigenvalues.

        Eigenvalues are ordered [+k_1,...,+k_{N/2}, -k_1,...,-k_{N/2}]
        (positive first = referenced to layer top, negative = to bottom).

        Returns a dict with keys: A, evals, E, n_pos.
        """
        N = self.n_streams
        n_half = self.n_half
        mus = self.mus
        weights = self.weights

        # Clamp omega below 1 to avoid singular k=0 eigenvalue in the
        # conservative-scattering limit.  Standard practice (cf. CDISORT).
        omega = min(omega, 1.0 - 1e-12)

        M_inv = np.diag(1.0 / mus)
        W_diag = np.diag(weights)
        P_mat = get_phase_matrix(mus, g, N)

        # Full A matrix (still needed for _particular_solution)
        A = M_inv @ (-np.eye(N) + (omega / 2.0) * (P_mat @ W_diag))

        # Extract blocks using hemisphere indices
        ui = self.up_idx
        di = self.down_idx
        A_uu = A[np.ix_(ui, ui)]
        A_ud = A[np.ix_(ui, di)]

        # Reduced matrix: M_red = (A_uu - A_ud)(A_uu + A_ud)
        apb = A_uu + A_ud    # "alpha plus beta"
        amb = A_uu - A_ud    # "alpha minus beta"
        M_red = amb @ apb

        # Solve reduced eigenproblem: M_red v = k² v
        k2_vals, V = scipy.linalg.eig(M_red)
        k2_vals = np.real(k2_vals)   # guaranteed real by construction
        V = np.real(V)

        # k_j = sqrt(k²_j), guard against near-zero
        k_vals = np.sqrt(np.maximum(k2_vals, 0.0))
        k_vals = np.maximum(k_vals, 1e-30)   # avoid division by zero

        # Full eigenvalues: [+k, -k]
        evals = np.concatenate([k_vals, -k_vals])

        # Recover full eigenvectors
        E = np.zeros((N, N))
        for j in range(n_half):
            apb_v = apb @ V[:, j]
            g_plus  = (V[:, j] + apb_v / k_vals[j]) / 2.0
            g_minus = (V[:, j] - apb_v / k_vals[j]) / 2.0
            # +k_j mode: E[up] = g_plus, E[down] = g_minus
            E[ui, j] = g_plus
            E[di, j] = g_minus
            # -k_j mode: swap
            E[ui, n_half + j] = g_minus
            E[di, n_half + j] = g_plus

        return {"A": A, "evals": evals, "E": E, "n_pos": n_half}

    @staticmethod
    def _stable_exp(evals: np.ndarray, n_pos: int, dtau: float,
                    tau_prime: float) -> np.ndarray:
        """Compute stabilized exponential factors at local coordinate tau_prime.

        For eigenvalues with Re(k) >= 0 (indices 0..n_pos-1):
            factor = exp(-k_j * tau_prime)             [ref to layer top]
        For eigenvalues with Re(k) < 0  (indices n_pos..N-1):
            factor = exp(-k_j * (tau_prime - dtau))    [ref to layer bottom]

        Both cases produce non-growing exponentials within 0 <= tau' <= dtau.
        """
        N = len(evals)
        out = np.zeros(N, dtype=float)
        out[:n_pos] = np.exp(-evals[:n_pos] * tau_prime)
        out[n_pos:] = np.exp(-evals[n_pos:] * (tau_prime - dtau))
        return out

    def _particular_solution(self, A: np.ndarray, mu0: float, g: float,
                             omega: float, f0: float) -> np.ndarray:
        """Particular solution Z_p for the direct-beam source."""
        S_beam = get_beam_source(self.mus, mu0, g, omega, f0)
        M_inv = np.diag(1.0 / self.mus)
        Q = M_inv @ S_beam
        LHS = A - (1.0 / mu0) * np.eye(self.n_streams)
        Z_p = scipy.linalg.solve(LHS, -Q)
        return Z_p

    # ------------------------------------------------------------------
    # Public solve entry point
    # ------------------------------------------------------------------
    def solve(self, profile: AtmosphereProfile,
              output: str = "toa",
              delta_m: bool = True,
              store_state: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the RTE and return radiances.

        Parameters
        ----------
        profile : AtmosphereProfile
            Atmospheric profile to solve.
        output : str
            ``'toa'``  — upwelling radiances at top-of-atmosphere (satellite view).
            ``'boa'``  — downwelling radiances at bottom-of-atmosphere.
            ``'both'`` — tuple (mu_up, I_toa, mu_down, I_boa).
        store_state : bool
            If True, store internal solver state in ``self._state`` so that
            ``interpolate_intensity()`` can compute radiances at arbitrary angles.

        Returns
        -------
        mu_view : ndarray, shape (n_half,)
            Absolute cosines of the viewing angles.
        I : ndarray, shape (n_half,)
            Radiances at the requested level.
            If output='both', returns (mu_up, I_toa, mu_down, I_boa).
        """
        N = self.n_streams
        n_layers = profile.n_layers
        mu0 = profile.mu0
        f0 = profile.f0

        # Guard: perturb mu0 if it nearly coincides with a quadrature node.
        # When mu0 ≈ mu_i the particular-solution matrix (A + I/mu0) is singular.
        _MU0_EPS = 1e-3
        for mu_node in np.abs(self.mus):
            if abs(mu0 - mu_node) < _MU0_EPS:
                mu0 = mu_node + _MU0_EPS
                break

        # ---- 1. Per-layer eigen-decomposition (with optional delta-M) ----
        layers: List[dict] = []
        for k in range(n_layers):
            tau_k = profile.tau[k]
            omega_k = profile.ssa[k]
            g_k = profile.g[k]

            if delta_m:
                tau_k, omega_k, g_k = self._apply_delta_m(tau_k, omega_k, g_k)

            eig = self._layer_eigen(omega_k, g_k)
            Z_p = self._particular_solution(eig["A"], mu0, g_k, omega_k, f0)
            eig["Z_p"] = Z_p
            eig["dtau"] = tau_k
            eig["omega"] = min(omega_k, 1.0 - 1e-12)  # post-clamp value
            eig["g"] = g_k
            layers.append(eig)

        cum_tau = profile.cumulative_tau          # length n_layers+1

        # ---- 2. Build global boundary system (stabilized) ----
        #   Unknowns are C̃_j per layer: for positive eigenvalues C̃=C (ref top),
        #   for negative eigenvalues C̃ = C*exp(-k_j*dtau) (ref bottom).
        n_unknowns = N * n_layers
        BC = np.zeros((n_unknowns, n_unknowns))
        rhs = np.zeros(n_unknowns)

        eq = 0   # equation counter

        # ── 2a.  TOA boundary (τ' = 0 of layer 0): I_down = 0 ──
        E0 = layers[0]["E"]
        Z_p0 = layers[0]["Z_p"]
        n_pos0 = layers[0]["n_pos"]
        dtau_0 = layers[0]["dtau"]
        exp_top0 = self._stable_exp(layers[0]["evals"], n_pos0, dtau_0, 0.0)

        for str_i in self.down_idx:
            for j in range(N):
                BC[eq, j] = E0[str_i, j] * exp_top0[j]
            rhs[eq] = -Z_p0[str_i]
            eq += 1

        # ── 2b.  Interface continuity (between layers k and k+1) ──
        for k in range(n_layers - 1):
            Ek = layers[k]["E"]
            lam_k = layers[k]["evals"]
            n_pos_k = layers[k]["n_pos"]
            Zp_k = layers[k]["Z_p"]
            dtau_k = layers[k]["dtau"]

            Ek1 = layers[k + 1]["E"]
            lam_k1 = layers[k + 1]["evals"]
            n_pos_k1 = layers[k + 1]["n_pos"]
            Zp_k1 = layers[k + 1]["Z_p"]
            dtau_k1 = layers[k + 1]["dtau"]

            tau_interface = cum_tau[k + 1]
            exp_beam_if = np.exp(-tau_interface / mu0)

            # Stabilized exp at bottom of layer k (tau' = dtau_k)
            exp_bot_k = self._stable_exp(lam_k, n_pos_k, dtau_k, dtau_k)
            # Stabilized exp at top of layer k+1 (tau' = 0)
            exp_top_k1 = self._stable_exp(lam_k1, n_pos_k1, dtau_k1, 0.0)

            col_k  = k * N
            col_k1 = (k + 1) * N

            for str_i in range(N):
                for j in range(N):
                    BC[eq, col_k + j]  =  Ek[str_i, j] * exp_bot_k[j]
                    BC[eq, col_k1 + j] = -Ek1[str_i, j] * exp_top_k1[j]
                rhs[eq] = -Zp_k[str_i] * exp_beam_if + Zp_k1[str_i] * exp_beam_if
                eq += 1

        # ── 2c.  BOA boundary (τ = τ_total): Lambertian surface ──
        last = n_layers - 1
        E_L = layers[last]["E"]
        lam_L = layers[last]["evals"]
        n_pos_L = layers[last]["n_pos"]
        Zp_L = layers[last]["Z_p"]
        dtau_L = layers[last]["dtau"]

        # Stabilized exp at bottom of last layer
        exp_bot_L = self._stable_exp(lam_L, n_pos_L, dtau_L, dtau_L)
        exp_beam_boa = np.exp(-cum_tau[-1] / mu0)

        alb_fac = profile.albedo / np.pi

        # Downward diffuse flux modal coefficients at BOA (stabilized)
        flux_mode = np.zeros(N)
        for j in range(N):
            s = 0.0
            for ki in self.down_idx:
                s += self.weights[ki] * np.abs(self.mus[ki]) * E_L[ki, j]
            flux_mode[j] = 2.0 * np.pi * s

        # Flux from particular solution at BOA
        flux_zp = 0.0
        for ki in self.down_idx:
            flux_zp += self.weights[ki] * np.abs(self.mus[ki]) * Zp_L[ki]
        flux_zp_down = 2.0 * np.pi * flux_zp

        flux_dir_down = mu0 * f0 * exp_beam_boa

        col_L = last * N

        for str_i in self.up_idx:
            for j in range(N):
                BC[eq, col_L + j] = (
                    E_L[str_i, j] * exp_bot_L[j]
                    - alb_fac * flux_mode[j] * exp_bot_L[j]
                )
            term_zp = Zp_L[str_i] * exp_beam_boa - alb_fac * flux_zp_down * exp_beam_boa
            rhs[eq] = alb_fac * flux_dir_down - term_zp
            eq += 1

        # ---- 3. Solve global system ----
        C_all = scipy.linalg.solve(BC[:eq, :], rhs[:eq])

        # ---- 3b. Optionally store internal state for interpolate_intensity ----
        if store_state:
            # Scaled cumulative tau for diffuse transport (from delta-M dtau)
            scaled_dtaus = np.array([lay["dtau"] for lay in layers])
            cum_tau_scaled = np.concatenate(([0.0], np.cumsum(scaled_dtaus)))
            self._state = {
                "layers": layers,
                "C_all": C_all.copy(),
                "cum_tau": cum_tau.copy(),          # original (for beam)
                "cum_tau_scaled": cum_tau_scaled,    # delta-M scaled (for diffuse)
                "profile": profile,
                "mu0": mu0,
                "n_layers": n_layers,
            }

        # ---- 4a. Extract TOA upwelling radiances ----
        #   Evaluate at the top of the first layer (τ' = 0)
        result_toa = None
        if output in ("toa", "both"):
            C_first = C_all[0:N]
            E_first = layers[0]["E"]
            Z_p_first = layers[0]["Z_p"]
            exp_toa = self._stable_exp(layers[0]["evals"], layers[0]["n_pos"],
                                       layers[0]["dtau"], 0.0)

            I_toa = np.zeros(self.n_half)
            for k_out, str_i in enumerate(self.up_idx):
                I_toa[k_out] = (np.dot(C_first * exp_toa, E_first[str_i, :])
                                + Z_p_first[str_i])

            result_toa = (self.mus[self.up_idx], I_toa)

        # ---- 4b. Extract BOA downwelling radiances ----
        result_boa = None
        if output in ("boa", "both"):
            C_last = C_all[last * N : (last + 1) * N]
            E_last = layers[last]["E"]
            Z_p_last = layers[last]["Z_p"]
            exp_boa = self._stable_exp(layers[last]["evals"], layers[last]["n_pos"],
                                        layers[last]["dtau"], layers[last]["dtau"])
            exp_beam_last = np.exp(-cum_tau[-1] / mu0)

            I_boa = np.zeros(self.n_half)
            for k_out, str_i in enumerate(self.down_idx):
                I_boa[k_out] = (
                    np.dot(C_last * exp_boa, E_last[str_i, :])
                    + Z_p_last[str_i] * exp_beam_last
                )

            result_boa = (np.abs(self.mus[self.down_idx]), I_boa)

        # ---- 5. Return requested output ----
        if output == "toa":
            return result_toa
        elif output == "boa":
            return result_boa
        else:  # both
            return (*result_toa, *result_boa)

    # ------------------------------------------------------------------
    # Eqs 35a/35b: Intensity at arbitrary angles via source-function
    # integration (DISORT Report Section 3.3)
    # ------------------------------------------------------------------

    def _surface_upwelling(self, mu: float) -> float:
        """Lambertian surface contribution I_sup(mu).

        I_sup = (A_s/pi) * [F_dir_down + F_dif_down]

        Uses the same beam convention as solve(): original cum_tau for beam,
        eigenexpansion for diffuse.
        """
        st = self._state
        layers = st["layers"]
        C_all = st["C_all"]
        cum_tau = st["cum_tau"]             # original (matches solve())
        mu0 = st["mu0"]
        profile = st["profile"]
        n_layers = st["n_layers"]
        N = self.n_streams

        last = n_layers - 1
        E_L = layers[last]["E"]
        evals_L = layers[last]["evals"]
        n_pos_L = layers[last]["n_pos"]
        Zp_L = layers[last]["Z_p"]
        dtau_L = layers[last]["dtau"]

        C_last = C_all[last * N : (last + 1) * N]
        exp_boa = self._stable_exp(evals_L, n_pos_L, dtau_L, dtau_L)
        exp_beam_boa = np.exp(-cum_tau[-1] / mu0)

        # Downward diffuse flux at BOA (from eigenexpansion, same as solve())
        F_dif_down = 0.0
        for ki in self.down_idx:
            I_ki = (np.dot(C_last * exp_boa, E_L[ki, :])
                    + Zp_L[ki] * exp_beam_boa)
            F_dif_down += self.weights[ki] * np.abs(self.mus[ki]) * I_ki
        F_dif_down *= 2.0 * np.pi

        F_dir_down = mu0 * profile.f0 * exp_beam_boa

        return (profile.albedo / np.pi) * (F_dir_down + F_dif_down)

    def _upwelling_at_toa(self, user_mus: np.ndarray) -> np.ndarray:
        """Eq 35a: upwelling intensity at TOA for arbitrary mu > 0.

        I(0, +mu) = I_sup(mu)*exp(-tau_L_s/mu)
                   + (1/mu) * sum_n integral_n S_n(tau', mu) exp(-tau'/mu) dtau'

        All optical depths are delta-M scaled (the particular solution Z_p
        already encodes beam decay as exp(-tau_scaled/mu0)).
        """
        st = self._state
        layers = st["layers"]
        C_all = st["C_all"]
        cum_tau_s = st["cum_tau_scaled"]    # scaled cumulative tau
        mu0 = st["mu0"]
        profile = st["profile"]
        n_layers = st["n_layers"]
        N = self.n_streams
        tau_L_s = cum_tau_s[-1]
        f0 = profile.f0

        I_out = np.zeros(len(user_mus))

        for m_idx, mu in enumerate(user_mus):
            # Surface term attenuated through atmosphere
            I_surf = self._surface_upwelling(mu) * np.exp(-tau_L_s / mu)

            I_layers = 0.0
            for n in range(n_layers):
                lay = layers[n]
                evals = lay["evals"]
                E = lay["E"]
                n_pos = lay["n_pos"]
                Z_p = lay["Z_p"]
                dtau = lay["dtau"]
                omega = lay["omega"]
                g_n = lay["g"]
                C_n = C_all[n * N : (n + 1) * N]
                tau_top = cum_tau_s[n]

                # Psi_j(+mu) = (omega/2) * sum_i w_i * p(+mu, mu_i) * E[i,j]
                p_mu = phase_function_at_angle(mu, self.mus, g_n, N)
                psi_j = np.zeros(N)
                for j in range(N):
                    s = 0.0
                    for i in range(N):
                        s += self.weights[i] * p_mu[i] * E[i, j]
                    psi_j[j] = (omega / 2.0) * s

                # Eigenmode contributions (numerically stable formulation)
                # Both cases rewritten via alpha = k_j + 1/mu so that
                #   denom = 1 + k_j*mu = alpha*mu
                # and the integral uses expm1-type Taylor near alpha*dtau ~ 0.
                for j in range(N):
                    k_j = evals[j]
                    alpha = k_j + 1.0 / mu
                    x = alpha * dtau       # dimensionless

                    # Numerically stable (exp(x)-1)/x  (→ 1 as x → 0)
                    if abs(x) < 1e-2:
                        em1_over_x = 1.0 + x * (0.5 + x / 6.0)
                    else:
                        em1_over_x = np.expm1(x) / x

                    if j < n_pos:
                        # (1-exp(-x))/x = exp(-x) * (exp(x)-1)/x
                        contrib = (C_n[j] * psi_j[j]
                                   * np.exp(-tau_top / mu - x)
                                   * (dtau / mu) * em1_over_x)
                    else:
                        # exp(-dtau/mu) * (exp(x)-1)/x  * dtau/mu
                        contrib = (C_n[j] * psi_j[j]
                                   * np.exp(-(tau_top + dtau) / mu)
                                   * (dtau / mu) * em1_over_x)

                    I_layers += contrib

                # Beam + Z_p scattering contribution (decay as exp(-τ_s/μ₀))
                psi_zp = 0.0
                for i in range(N):
                    psi_zp += self.weights[i] * p_mu[i] * Z_p[i]
                psi_zp *= omega / 2.0
                psi_direct = (omega * f0 / (4.0 * np.pi)) * phase_function_scalar(mu, -mu0, g_n, N)
                psi_beam = psi_zp + psi_direct

                denom_beam = 1.0 / mu + 1.0 / mu0
                x_beam = denom_beam * dtau

                if abs(x_beam) < 1e-2:
                    em1_beam = 1.0 + x_beam * (0.5 + x_beam / 6.0)
                else:
                    em1_beam = np.expm1(x_beam) / x_beam

                # (1 - exp(-x))/x = exp(-x)*(exp(x)-1)/x
                beam_contrib = (psi_beam * np.exp(-tau_top * denom_beam - x_beam)
                                * (dtau / mu) * em1_beam)

                I_layers += beam_contrib

            I_out[m_idx] = I_surf + I_layers

        return I_out

    def _downwelling_at_boa(self, user_mus: np.ndarray) -> np.ndarray:
        """Eq 35b: downwelling intensity at BOA for arbitrary mu > 0.

        I(tau_L, -mu) = (1/mu) * sum_n integral_n S_n(tau', -mu) exp(-(tau_L-tau')/mu) dtau'

        Uses original cum_tau (matches solve() beam convention at BOA).
        """
        st = self._state
        layers = st["layers"]
        C_all = st["C_all"]
        cum_tau = st["cum_tau"]             # original (matches solve() BOA)
        mu0 = st["mu0"]
        profile = st["profile"]
        n_layers = st["n_layers"]
        N = self.n_streams
        tau_L = cum_tau[-1]
        f0 = profile.f0

        I_out = np.zeros(len(user_mus))

        for m_idx, mu in enumerate(user_mus):
            I_layers = 0.0
            for n in range(n_layers):
                lay = layers[n]
                evals = lay["evals"]
                E = lay["E"]
                n_pos = lay["n_pos"]
                Z_p = lay["Z_p"]
                dtau = lay["dtau"]
                omega = lay["omega"]
                g_n = lay["g"]
                C_n = C_all[n * N : (n + 1) * N]
                tau_top = cum_tau[n]
                tau_bot = cum_tau[n + 1]

                # Psi_j(-mu) = (omega/2) * sum_i w_i * p(-mu, mu_i) * E[i,j]
                p_mu = phase_function_at_angle(-mu, self.mus, g_n, N)
                psi_j = np.zeros(N)
                for j in range(N):
                    s = 0.0
                    for i in range(N):
                        s += self.weights[i] * p_mu[i] * E[i, j]
                    psi_j[j] = (omega / 2.0) * s

                # Eigenmode contributions (numerically stable formulation)
                # For downwelling: denom = 1 - k_j*mu = alpha*mu where alpha = 1/mu - k_j.
                # Singularity when k_j = 1/mu (j < n_pos only).
                # Integral rewritten using (exp(x)-1)/x with x = alpha*dtau.
                for j in range(N):
                    k_j = evals[j]
                    alpha = 1.0 / mu - k_j
                    x = alpha * dtau       # dimensionless

                    # Numerically stable (exp(x)-1)/x  (→ 1 as x → 0)
                    if abs(x) < 1e-2:
                        em1_over_x = 1.0 + x * (0.5 + x / 6.0)
                    else:
                        em1_over_x = np.expm1(x) / x

                    if j < n_pos:
                        # Positive eigenvalue: singularity possible when k_j ≈ 1/mu.
                        # contrib = C̃ Ψ exp(-(τ_L-τ_top)/μ) (Δτ/μ) (exp(x)-1)/x
                        contrib = (C_n[j] * psi_j[j]
                                   * np.exp(-(tau_L - tau_top) / mu)
                                   * (dtau / mu) * em1_over_x)
                    else:
                        # Negative eigenvalue: denom > 0 always, no singularity.
                        # contrib = C̃ Ψ exp(k_j Δτ) exp(-(τ_L-τ_top)/μ) (Δτ/μ) (exp(x)-1)/x
                        contrib = (C_n[j] * psi_j[j]
                                   * np.exp(k_j * dtau - (tau_L - tau_top) / mu)
                                   * (dtau / mu) * em1_over_x)

                    I_layers += contrib

                # Beam + Z_p scattering contribution
                psi_zp = 0.0
                for i in range(N):
                    psi_zp += self.weights[i] * p_mu[i] * Z_p[i]
                psi_zp *= omega / 2.0
                psi_direct = (omega * f0 / (4.0 * np.pi)) * phase_function_scalar(-mu, -mu0, g_n, N)
                psi_beam = psi_zp + psi_direct

                denom_beam = 1.0 / mu0 - 1.0 / mu
                x_beam = denom_beam * dtau

                if abs(x_beam) < 1e-2:
                    em1_beam = 1.0 + x_beam * (0.5 + x_beam / 6.0)
                else:
                    em1_beam = np.expm1(x_beam) / x_beam

                beam_contrib = (psi_beam
                                * np.exp(-tau_bot / mu0 - (tau_L - tau_bot) / mu)
                                * (dtau / mu) * em1_beam)

                I_layers += beam_contrib

            I_out[m_idx] = I_layers

        return I_out

    def interpolate_intensity(self, user_mus: np.ndarray,
                              output: str = "toa") -> Tuple[np.ndarray, np.ndarray]:
        """Intensity at arbitrary angles via Eqs 35a/35b (source-function integration).

        Requires ``solve(store_state=True)`` to have been called first.

        Parameters
        ----------
        user_mus : array_like
            Cosines of viewing angles, 0 < mu <= 1.
        output : str
            ``'toa'``  — upwelling at top of atmosphere.
            ``'boa'``  — downwelling at bottom of atmosphere.
            ``'both'`` — tuple (user_mus, I_toa, user_mus, I_boa).

        Returns
        -------
        user_mus : ndarray
            The input angles (for consistency with solve() interface).
        I : ndarray
            Radiances at the requested level.
        """
        if not hasattr(self, "_state") or self._state is None:
            raise RuntimeError("Must call solve(store_state=True) before "
                               "interpolate_intensity().")

        user_mus = np.atleast_1d(np.asarray(user_mus, dtype=float))
        if np.any(user_mus <= 0) or np.any(user_mus > 1):
            raise ValueError("user_mus must satisfy 0 < mu <= 1.")

        result_toa = None
        result_boa = None

        if output in ("toa", "both"):
            I_toa = self._upwelling_at_toa(user_mus)
            result_toa = (user_mus, I_toa)

        if output in ("boa", "both"):
            I_boa = self._downwelling_at_boa(user_mus)
            result_boa = (user_mus, I_boa)

        if output == "toa":
            return result_toa
        elif output == "boa":
            return result_boa
        else:
            return (*result_toa, *result_boa)
