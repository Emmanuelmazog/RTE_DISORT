# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

1D plane-parallel atmospheric radiative transfer model using the Discrete Ordinates method (DISORT), with forward and inverse solvers for aerosol retrieval. Academic project for an Optical Methods course (graduate level).

## Running

```bash
# Run the full pipeline (validation, retrievals, parameter studies, plots)
python3 main.py

# Run individual test scripts
python3 test_g095.py
python3 test_g095_v2.py
```

`main.py` is controlled by boolean flags at the top (`RUN_VALIDATION`, `RUN_LEVEL1`, `RUN_LEVEL2`, `RUN_LEVEL3`, `RUN_STUDY_A`–`RUN_STUDY_D`). Set individual flags to `False` to skip sections.

No build step, no linter, no test framework — scripts are run directly with system python3.

## Dependencies

Python 3.12+ (system python on WSL2), numpy, scipy, matplotlib. Install with `pip install --break-system-packages numpy scipy matplotlib`. PyMieScatt is installed but **not used** (incompatible with scipy>=1.14 — `trapz` was removed).

## Architecture

The core library is the `atrt/` package. `main.py` is the orchestration/analysis script that imports from `atrt` and runs everything. `1D_Atmosphere_Model.py` is an older monolithic version (kept for reference).

### `atrt/` package modules

| Module | Key exports | Role |
|---|---|---|
| `quadrature.py` | `gauleg`, `double_gauss` | Gauss-Legendre quadrature; Double-Gauss applies GL per hemisphere |
| `phase.py` | `legendre_expansion_hg`, `get_phase_matrix`, `get_beam_source`, `phase_function_at_angle` | Henyey-Greenstein phase function via Legendre expansion |
| `profile.py` | `AtmosphereProfile` | Per-layer optical properties container (tau, ssa, g, albedo, mu0, f0) |
| `atmosphere.py` | `StandardAtmosphere`, `rayleigh_cross_section` | US Standard 1976 (20 layers) with Rayleigh scattering |
| `mie.py` | `compute_aerosol_optics`, `get_aerosol_preset`, `AEROSOL_TYPES` | Built-in Mie theory (replaces PyMieScatt); 5 presets: continental, urban, maritime, desert_dust, biomass_burning |
| `solver.py` | `DisortSolver` | Eigenvalue DISORT solver with delta-M scaling, multi-layer BVP, and source-function interpolation (Eqs 35a/35b) |
| `inverse.py` | `InverseOptimizer`, `OptimalEstimation`, `PhillipsTwomey`, `TotalVariation` | Four retrieval methods: LM+Tikhonov, Rodgers OE, Phillips-Twomey with L-curve, TV via IRLS |

### Critical numerical details

- **Eigenvalue formulation**: reduced (N/2)x(N/2) guaranteeing real eigenvalues. Eigenvector recovery: `G_u = (V + apb·V/k)/2`, `G_d = (V - apb·V/k)/2`.
- **Particular solution**: `Z_p = -(A - I/mu0)^-1 Q` — the minus sign on `1/mu0` is critical.
- **Stabilization**: reference-point scaling (positive eigenvalues referenced to layer top, negative to bottom) to avoid overflow.
- **Conservative scattering** (SSA=1.0): omega clamped to `1-1e-12` to avoid singular `k=0`.
- **Delta-M scaling**: `solve()` uses scaled `cum_tau` for TOA upwelling, original `cum_tau` for BOA downwelling in `interpolate_intensity()`.
- **All arrays are real** — no complex dtype needed.

### `main.py` structure

Organized in sequential sections, each behind a `RUN_*` flag:
- **Validation**: 7 forward-model tests (isotropic, conservative, Rayleigh, HG, multi-layer, etc.)
- **Level 1**: Single-layer idealized AOD retrieval
- **Level 2**: 20-layer joint retrieval (LM, OE, Phillips-Twomey, TV) with multi-wavelength
- **Level 3**: Real data comparison (AERONET + SURFRAD CSVs)
- **Studies A–D**: Parameter sweeps (SSA, AOD, g, albedo, SZA), extreme cases, energy balance, discrete vs continuous

## LaTeX Documentation

- `1D_Atmosphere_Model_Doc.tex` — Full model documentation
- `DISORT_Intensity_Derivation.tex` — Derivation of source-function interpolation (Eqs 35a/35b)
- `Final_Report.tex` — Course final report
- Compile with `pdflatex`
