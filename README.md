# DISORT Atmospheric Radiative Transfer Model

A 1D plane-parallel atmospheric radiative transfer model using the **Discrete Ordinates method (DISORT)**, with forward and inverse solvers for aerosol optical property retrieval.

Developed as part of the *Optical Methods in the Atmosphere* graduate course at **Universidad EAFIT** (Maestria en Fisica Aplicada).

---

## Overview

This project implements a complete forward-inverse modelling chain:

1. **Forward model** — Solves the Radiative Transfer Equation (RTE) via eigenvalue decomposition, supporting multi-layer atmospheres, delta-M scaling, and source-function interpolation for arbitrary viewing angles.
2. **Inverse methods** — Four retrieval algorithms recover aerosol optical depth (AOD), single-scattering albedo (SSA), and asymmetry parameter (g) from simulated or real radiance measurements.
3. **Validation & analysis** — Energy conservation tests, extreme-case verification, parameter sensitivity studies, and real-data comparison against AERONET/SURFRAD observations.

---

## Project Structure

```
Atmospheric/
├── atrt/                          # Core library package
│   ├── __init__.py                # Public API exports
│   ├── quadrature.py              # Gauss-Legendre & Double-Gauss quadrature
│   ├── phase.py                   # Henyey-Greenstein phase function (Legendre expansion)
│   ├── profile.py                 # AtmosphereProfile container (per-layer optical properties)
│   ├── atmosphere.py              # US Standard Atmosphere 1976 + Rayleigh scattering
│   ├── mie.py                     # Built-in Mie theory (lognormal size distributions)
│   ├── solver.py                  # DisortSolver (eigenvalue BVP + source-function interpolation)
│   └── inverse.py                 # LM+Tikhonov, Optimal Estimation, Phillips-Twomey, Total Variation
│
├── main.py                        # Orchestration script (validation, retrievals, studies, plots)
├── test_g095.py                   # Stream convergence test for g=0.95
├── test_g095_v2.py                # Extended convergence diagnostics
│
├── aeronet_bondville_*.csv        # AERONET Level 2.0 data (Bondville, IL)
├── surfrad_bondville/             # SURFRAD surface radiation measurements
│
├── Images_final_report/           # Generated figures (publication-quality PNGs)
│
├── Final_Report.tex               # Course final report (LaTeX)
├── 1D_Atmosphere_Model_Doc.tex    # Full model documentation
├── DISORT_Intensity_Derivation.tex # Source-function interpolation derivation
└── CLAUDE.md                      # AI assistant instructions
```

---

## The `atrt` Package

### `quadrature.py`
- **`gauleg(n)`** — Standard Gauss-Legendre nodes and weights on [-1, 1].
- **`double_gauss(n)`** — Double-Gauss scheme: GL applied separately per hemisphere, ensuring symmetric pairing `mu_up[i] = -mu_down[i]`. Required for the reduced eigenvalue formulation.

### `phase.py`
- **`legendre_expansion_hg(g, n_terms)`** — HG expansion coefficients: `chi_l = (2l+1) * g^l`.
- **`get_phase_matrix(mu, g, N)`** — Full N x N phase matrix via Legendre polynomial recurrence.
- **`get_beam_source(mu, mu0, g, omega, f0)`** — Solar beam source vector `S(mu_i) = (omega * F0 / 4pi) * P(mu_i, -mu0)`.
- **`phase_function_at_angle(mu_user, mu_quad, g, n_terms)`** — Phase function for one arbitrary angle vs all quadrature nodes (used by source-function interpolation).
- **`phase_function_scalar(mu1, mu2, g, n_terms)`** — Phase function between two scalar angles.

### `profile.py`
- **`AtmosphereProfile`** — Container for per-layer optical properties (tau, SSA, g) plus surface albedo and solar geometry. Accepts scalars (single layer) or arrays (multi-layer). Provides `cumulative_tau`, `tau_total`, `n_layers`, and `copy()`.

### `atmosphere.py`
- **`rayleigh_cross_section(wavelength_nm)`** — Rayleigh cross section per molecule (Bodhaine et al., 1999).
- **`StandardAtmosphere`** — US Standard 1976 with 20 layers (0-100 km). Computes T, P, and air density profiles from lapse-rate segments. `to_profile()` combines molecular Rayleigh scattering with an exponentially-distributed aerosol loading to produce an `AtmosphereProfile`.

### `mie.py`
- **`_mie_q(m, x)`** — Single-sphere Mie efficiencies (Qext, Qsca, g) using Bohren & Huffman algorithm with logarithmic derivative downward recurrence and Wiscombe's termination criterion.
- **`compute_aerosol_optics(wavelength_nm, r_g, sigma_g, m_real, m_imag)`** — Bulk optical properties from a lognormal size distribution via trapezoidal integration over 200 radii.
- **`get_aerosol_preset(type)`** — Five built-in presets:

| Preset | r_g (um) | sigma_g | n_real | n_imag |
|--------|----------|---------|--------|--------|
| `continental` | 0.03 | 2.24 | 1.53 | 0.008 |
| `urban` | 0.03 | 2.24 | 1.55 | 0.025 |
| `maritime` | 0.40 | 2.51 | 1.38 | 0.001 |
| `desert_dust` | 0.50 | 2.20 | 1.53 | 0.005 |
| `biomass_burning` | 0.07 | 1.80 | 1.52 | 0.020 |

### `solver.py`
The `DisortSolver` implements the full DISORT algorithm:

- **Delta-M scaling** — Truncates the forward-scattering peak: `f = g^N`, rescales tau, omega, and g.
- **Reduced eigenvalue problem** — Exploits Double-Gauss block symmetry to solve an `(N/2) x (N/2)` system instead of `N x N`, guaranteeing real eigenvalues.
- **Eigenvector recovery** — `G_up = (V + apb*V/k)/2`, `G_down = (V - apb*V/k)/2`.
- **Particular solution** — `Z_p = -(A - I/mu0)^(-1) * Q` (the minus sign on `1/mu0` is critical).
- **Exponential stabilisation** — Reference-point scaling: positive eigenvalues referenced to layer top, negative to layer bottom, ensuring only non-growing exponentials.
- **Conservative scattering** — SSA clamped to `1 - 1e-12` to avoid singular `k=0` eigenvalue.
- **Multi-layer BVP** — Global boundary system enforcing TOA (no downwelling), interface continuity, and Lambertian BOA reflection.
- **Source-function interpolation** — `interpolate_intensity()` implements Eqs 35a/35b from the DISORT report for continuous angular radiance fields at arbitrary viewing angles.

### `inverse.py`
Four retrieval methods:

| Method | Class | Description |
|--------|-------|-------------|
| **LM + Tikhonov** | `InverseOptimizer` | Augmented residual `[y - F(x); sqrt(gamma)*(x - x_a)]` passed to `scipy.optimize.least_squares`. Supports box constraints. |
| **Optimal Estimation** | `OptimalEstimation` | Gauss-Newton iteration following Rodgers (2000). Returns posterior covariance, averaging kernels, degrees of freedom. |
| **Phillips-Twomey** | `PhillipsTwomey` | Constrained linear inversion `(K'K + gamma*H'H)^(-1) K'y` with 0th/1st/2nd order smoothing and L-curve selection. |
| **Total Variation** | `TotalVariation` | TV regularisation via IRLS, preserving sharp edges in the retrieved profile. |

---

## Running

### Requirements
- Python 3.12+
- NumPy, SciPy, Matplotlib

```bash
pip install numpy scipy matplotlib
```

### Execution

```bash
# Run the full pipeline
python3 main.py

# Run individual diagnostic scripts
python3 test_g095.py
python3 test_g095_v2.py
```

`main.py` is controlled by boolean flags at the top of the file:

| Flag | Section |
|------|---------|
| `RUN_VALIDATION` | Forward model validation (7 tests) |
| `RUN_LEVEL1` | Single-layer idealised AOD retrieval |
| `RUN_LEVEL2` | 20-layer joint retrieval (LM, OE, Phillips-Twomey, TV) |
| `RUN_LEVEL3` | Real-data retrieval (AERONET + SURFRAD) |
| `RUN_STUDY_A` | Parameter sweeps (SSA, AOD, g, albedo, SZA, scenarios) |
| `RUN_STUDY_B` | Extreme physical cases |
| `RUN_STUDY_C` | Energy balance analysis |
| `RUN_STUDY_D` | Discrete vs continuous solution comparison |

Set individual flags to `True`/`False` to enable/disable sections.

### Configuration

All tuneable parameters are centralised at the top of `main.py`:

```python
N_STREAMS   = 36          # quadrature streams
ALBEDO      = 0.1         # surface albedo
THETA0_DEG  = 30.0        # solar zenith angle
L1_AOD_TRUE = 0.3         # Level 1 true AOD
L2_AEROSOL_TYPE = "continental"
L2_WAVELENGTH   = 550.0   # nm
```

---

## Sample Results

### Forward Model Validation
- Energy conservation verified to machine precision across all SSA and albedo combinations
- Extreme cases (pure absorption, conservative scattering, black/reflecting surfaces) reproduce analytical predictions
- Source-function interpolation matches discrete solution at quadrature nodes

### Retrieval Performance
- **Level 1** (single layer): OE recovers AOD with negligible bias across `tau in [0, 1]`
- **Level 2** (20-layer): OE achieves <5% error in all three parameters (tau: 4.3%, SSA: 1.4%, g: 3.1%); LM+Tikhonov exceeds 50% error due to tau-g degeneracy
- **Level 3** (real data): Unbiased retrieval (bias = +0.006) for AOD < 0.3; saturation of diffuse fraction limits performance at higher AOD

### Parameter Sensitivities
- SSA and surface albedo have the strongest effect on TOA radiance
- Asymmetry parameter primarily affects the BOA forward-scattering peak
- Delta-M scaling achieves convergence with N=64 streams even at g=0.95

---

## Documentation

| Document | Description |
|----------|-------------|
| `Final_Report.tex` | Course final report with full results and analysis |
| `1D_Atmosphere_Model_Doc.tex` | Comprehensive model documentation |
| `DISORT_Intensity_Derivation.tex` | Derivation of source-function interpolation (Eqs 35a/35b) |

Compile with `pdflatex`.

---

## References

- Stamnes, K. et al. (1988). *Numerically stable algorithm for discrete-ordinate-method radiative transfer*. Appl. Opt., 27(12), 2502-2509.
- Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding*. World Scientific.
- Wiscombe, W. J. (1977). *The delta-M method*. J. Atmos. Sci., 34, 1408-1422.
- Bohren, C. F. & Huffman, D. R. (1983). *Absorption and Scattering of Light by Small Particles*. Wiley.
- NOAA/NASA/USAF (1976). *U.S. Standard Atmosphere, 1976*.

---

## License

Academic project. Not licensed for redistribution.
