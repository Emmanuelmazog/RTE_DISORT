"""Inverse methods: LM+Tikhonov, Optimal Estimation, Phillips-Twomey, Total Variation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Optional

from .profile import AtmosphereProfile
from .solver import DisortSolver


class InverseOptimizer:
    """Tikhonov-regularised Levenberg-Marquardt retrieval of aerosol properties.

    The augmented residual vector passed to ``least_squares`` is:

    .. math::

        r(x) = \\begin{bmatrix}
                    y_{obs} - F(x) \\\\
                    \\sqrt{\\gamma} \\, (x - x_a)
               \\end{bmatrix}

    so the implicit cost is  J = ‖y − F(x)‖² + γ ‖x − x_a‖².

    Parameters
    ----------
    solver : DisortSolver
        Pre-configured forward solver.
    profile_template : AtmosphereProfile
        Template profile whose per-layer *ratios* are preserved when the
        state vector is mapped back to physical parameters.
    x_a : ndarray
        A priori state vector.
    gamma : float
        Tikhonov regularisation strength.
    user_mus : ndarray, optional
        Cosines of user-defined viewing angles.  When provided, the forward
        model uses ``interpolate_intensity`` (Eqs 35a/35b, continuous
        source-function integration) instead of returning intensities at
        the discrete quadrature nodes.
    """

    def __init__(
        self,
        solver: DisortSolver,
        profile_template: AtmosphereProfile,
        x_a: np.ndarray,
        gamma: float = 1.0,
        user_mus: Optional[np.ndarray] = None,
    ) -> None:
        self.solver = solver
        self.template = profile_template
        self.x_a = np.asarray(x_a, dtype=float)
        self.gamma = gamma
        self.user_mus = user_mus

    # ------------------------------------------------------------------
    def _state_to_profile(self, x: np.ndarray,
                          param_names: List[str]) -> AtmosphereProfile:
        """Map a state vector *x* back to an AtmosphereProfile.

        For multi-layer templates the per-layer ratios are preserved:
        e.g. if the state contains a total AOD, each layer's τ is scaled
        so that the individual fractions remain unchanged.
        """
        prof = self.template.copy()

        for i, name in enumerate(param_names):
            if name == "aod":
                # Scale per-layer tau so they sum to x[i]
                total = prof.tau.sum()
                if total > 0:
                    prof.tau = prof.tau * (x[i] / total)
                else:
                    prof.tau = np.full_like(prof.tau, x[i] / prof.n_layers)
            elif name == "ssa":
                # Shift all layers so that the mean SSA matches x[i]
                mean_ssa = prof.ssa.mean()
                if mean_ssa > 0:
                    prof.ssa = prof.ssa * (x[i] / mean_ssa)
                else:
                    prof.ssa = np.full_like(prof.ssa, x[i])
                # Clip to physical range
                prof.ssa = np.clip(prof.ssa, 0.0, 0.999999)
            elif name == "g":
                mean_g = prof.g.mean()
                if mean_g > 0:
                    prof.g = prof.g * (x[i] / mean_g)
                else:
                    prof.g = np.full_like(prof.g, x[i])
                prof.g = np.clip(prof.g, -0.999, 0.999)

        return prof

    # ------------------------------------------------------------------
    def _residual(self, x: np.ndarray, y_obs: np.ndarray,
                  param_names: List[str]) -> np.ndarray:
        """Augmented residual vector (data-fit + Tikhonov penalty)."""
        prof = self._state_to_profile(x, param_names)
        if self.user_mus is not None:
            self.solver.solve(prof, output="toa", store_state=True)
            _, I_toa = self.solver.interpolate_intensity(
                self.user_mus, output="toa")
        else:
            _, I_toa = self.solver.solve(prof, output="toa")
        I_toa = np.abs(I_toa)  # ensure physical positivity

        r_data = y_obs - I_toa
        r_reg  = np.sqrt(self.gamma) * (x - self.x_a)
        return np.concatenate([r_data, r_reg])

    # ------------------------------------------------------------------
    def retrieve(
        self,
        y_obs: np.ndarray,
        x0: np.ndarray,
        param_names: List[str],
        bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
        verbose: int = 0,
    ):
        """Run the retrieval.

        Parameters
        ----------
        y_obs : ndarray
            Observed TOA radiances.
        x0 : ndarray
            Initial guess for the state vector.
        param_names : list of str
            Names of retrieved parameters (``'aod'``, ``'ssa'``, ``'g'``).
        bounds : tuple of (lower, upper)
            Box constraints for each element of x.
        verbose : int
            Verbosity level (0 = silent, 1 = summary, 2 = per-iteration).

        Returns
        -------
        result : scipy.optimize.OptimizeResult
        """
        x0_arr = np.asarray(x0, dtype=float)
        use_bounds = not (np.all(np.isneginf(bounds[0]))
                         and np.all(np.isinf(bounds[1])))
        result = least_squares(
            fun=self._residual,
            x0=x0_arr,
            args=(y_obs, param_names),
            method="trf" if use_bounds else "lm",
            bounds=bounds,
            verbose=verbose,
        )
        return result


class OptimalEstimation:
    """Full Optimal Estimation retrieval following Rodgers (2000).

    Gauss-Newton iteration:
        x_{n+1} = x_a + G_n [y - F(x_n) + K_n (x_n - x_a)]
    where
        G_n = (K^T S_eps^{-1} K + S_a^{-1})^{-1} K^T S_eps^{-1}

    Parameters
    ----------
    forward_model : callable
        F(x) -> y   (state vector to observation vector).
    x_a : ndarray, shape (n,)
        A priori state vector.
    S_a : ndarray, shape (n, n)
        A priori covariance matrix.
    S_eps : ndarray, shape (m, m)
        Measurement error covariance matrix.
    """

    def __init__(self, forward_model, x_a: np.ndarray,
                 S_a: np.ndarray, S_eps: np.ndarray) -> None:
        self.F = forward_model
        self.x_a = np.asarray(x_a, dtype=float)
        self.S_a = np.asarray(S_a, dtype=float)
        self.S_eps = np.asarray(S_eps, dtype=float)
        self.S_a_inv = np.linalg.inv(self.S_a)
        self.S_eps_inv = np.linalg.inv(self.S_eps)

    def jacobian(self, x: np.ndarray, dx_frac: float = 0.01) -> np.ndarray:
        """Jacobian K = dF/dx by finite differences."""
        y0 = self.F(x)
        n = len(x)
        m = len(y0)
        K = np.zeros((m, n))
        for j in range(n):
            dx = max(abs(x[j]) * dx_frac, 1e-8)
            x_pert = x.copy()
            x_pert[j] += dx
            K[:, j] = (self.F(x_pert) - y0) / dx
        return K

    def _cost(self, x: np.ndarray, y_obs: np.ndarray,
              y_model: np.ndarray) -> float:
        """OE cost function J(x)."""
        dy = y_obs - y_model
        dx = x - self.x_a
        return float(dy @ self.S_eps_inv @ dy + dx @ self.S_a_inv @ dx)

    def retrieve(self, y_obs: np.ndarray, x0: np.ndarray,
                 max_iter: int = 20, conv_thresh: float = 0.1,
                 gamma_lm: float = 0.0) -> dict:
        """Run Gauss-Newton OE iteration.

        Parameters
        ----------
        y_obs : ndarray
            Observed radiances.
        x0 : ndarray
            Initial guess.
        max_iter : int
            Maximum iterations.
        conv_thresh : float
            Convergence factor (d^2 < n * conv_thresh).
        gamma_lm : float
            Levenberg-Marquardt damping (0 = pure Gauss-Newton).

        Returns
        -------
        dict with keys: x, S_hat, A, DFS, K, G, n_iter, cost,
                        cost_history, converged.
        """
        x = np.asarray(x0, dtype=float).copy()
        y_obs = np.asarray(y_obs, dtype=float)
        n = len(x)
        cost_history = []
        converged = False

        for it in range(max_iter):
            y_model = self.F(x)
            K = self.jacobian(x)
            J = self._cost(x, y_obs, y_model)
            cost_history.append(J)

            S_hat_inv = K.T @ self.S_eps_inv @ K + self.S_a_inv
            if gamma_lm > 0:
                S_hat_inv += gamma_lm * np.diag(np.diag(S_hat_inv))
            S_hat = np.linalg.inv(S_hat_inv)
            G = S_hat @ K.T @ self.S_eps_inv

            x_new = self.x_a + G @ (y_obs - y_model + K @ (x - self.x_a))

            # Convergence check (Rodgers Eq. 5.29)
            dx = x_new - x
            d2 = float(dx @ S_hat_inv @ dx)
            x = x_new

            if d2 < n * conv_thresh:
                y_model = self.F(x)
                K = self.jacobian(x)
                J = self._cost(x, y_obs, y_model)
                cost_history.append(J)
                converged = True
                break

        # Final diagnostics
        S_hat_inv = K.T @ self.S_eps_inv @ K + self.S_a_inv
        S_hat = np.linalg.inv(S_hat_inv)
        G = S_hat @ K.T @ self.S_eps_inv
        A = G @ K  # Averaging kernel

        return {
            "x": x,
            "S_hat": S_hat,
            "A": A,
            "DFS": float(np.trace(A)),
            "K": K,
            "G": G,
            "n_iter": it + 1,
            "cost": cost_history[-1],
            "cost_history": cost_history,
            "converged": converged,
        }


class PhillipsTwomey:
    """Phillips-Twomey constrained linear inversion.

    Minimises  ||y - Kx||^2 + gamma * ||Hx||^2
    Solution:  x_hat = (K^T K + gamma H^T H)^{-1} K^T y

    Parameters
    ----------
    order : int
        Constraint order (0 = identity, 1 = first differences, 2 = second differences).
    """

    def __init__(self, order: int = 2) -> None:
        self.order = order

    @staticmethod
    def _diff_matrix(n: int, order: int) -> np.ndarray:
        """Build finite-difference constraint matrix H."""
        if order == 0:
            return np.eye(n)
        elif order == 1:
            H = np.zeros((n - 1, n))
            for i in range(n - 1):
                H[i, i] = -1.0
                H[i, i + 1] = 1.0
            return H
        else:  # order 2
            H = np.zeros((n - 2, n))
            for i in range(n - 2):
                H[i, i] = 1.0
                H[i, i + 1] = -2.0
                H[i, i + 2] = 1.0
            return H

    def solve(self, K: np.ndarray, y: np.ndarray,
              gamma: float, x_a: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear inversion for a single gamma value.

        If x_a is given, minimises ||y - K(x - x_a)||^2 + gamma ||H x||^2
        effectively shifting the solution towards x_a.
        """
        n = K.shape[1]
        H = self._diff_matrix(n, self.order)
        HtH = H.T @ H
        KtK = K.T @ K

        if x_a is not None:
            return np.linalg.solve(KtK + gamma * HtH,
                                   K.T @ y + gamma * HtH @ x_a)
        return np.linalg.solve(KtK + gamma * HtH, K.T @ y)

    def l_curve(self, K: np.ndarray, y: np.ndarray,
                gammas: np.ndarray,
                x_a: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute L-curve data for a range of gamma values.

        Returns (gammas, residual_norms, solution_norms).
        """
        n = K.shape[1]
        H = self._diff_matrix(n, self.order)

        res_norms = np.zeros(len(gammas))
        sol_norms = np.zeros(len(gammas))

        for i, g in enumerate(gammas):
            x_hat = self.solve(K, y, g, x_a)
            res_norms[i] = np.linalg.norm(y - K @ x_hat)
            sol_norms[i] = np.linalg.norm(H @ x_hat)

        return gammas, res_norms, sol_norms


class TotalVariation:
    """Total Variation regularisation via IRLS.

    Minimises  ||y - F(x)||^2 + gamma * TV(x)
    where TV(x) = sum_i |x_{i+1} - x_i|

    Uses Iteratively Reweighted Least Squares (IRLS).
    """

    def __init__(self, gamma: float = 1.0, max_irls: int = 20,
                 epsilon: float = 1e-6) -> None:
        self.gamma = gamma
        self.max_irls = max_irls
        self.epsilon = epsilon

    def solve(self, K: np.ndarray, y: np.ndarray,
              x0: np.ndarray) -> np.ndarray:
        """Linear TV-regularised inversion via IRLS."""
        n = K.shape[1]
        x = x0.copy()

        # D1 matrix
        D1 = np.zeros((n - 1, n))
        for i in range(n - 1):
            D1[i, i] = -1.0
            D1[i, i + 1] = 1.0

        KtK = K.T @ K
        Kty = K.T @ y

        for _ in range(self.max_irls):
            # Weights
            diffs = D1 @ x
            w = 1.0 / (np.abs(diffs) + self.epsilon)
            W = np.diag(w)

            H = np.sqrt(W) @ D1
            x_new = np.linalg.solve(KtK + self.gamma * H.T @ H, Kty)

            if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-15) < 1e-6:
                break
            x = x_new

        return x
