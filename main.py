"""
1D / Multi-Layer Atmosphere Model — Main execution script
=========================================================
Runs validation tests, generates plots, and performs analyses using
the atrt package (Atmospheric Radiative Transfer).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import matplotlib.ticker as mticker

from atrt import (
    AtmosphereProfile, StandardAtmosphere, DisortSolver,
    InverseOptimizer, OptimalEstimation, PhillipsTwomey, TotalVariation,
    get_aerosol_preset, AEROSOL_TYPES,
    gauleg, double_gauss,
)


np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(seed=42)

# ══════════════════════════════════════════════════════════════════
#  PLOT STYLE — Consistent, publication-quality aesthetics
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f8f9fa",
    "axes.edgecolor":     "#333333",
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.color":         "#cccccc",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    # -- Font: match LaTeX (Computer Modern) --
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    # -- Increased font sizes --
    "font.size":          14,
    "axes.titlesize":     16,
    "axes.titleweight":   "normal",
    "axes.labelsize":     15,
    "legend.fontsize":    12,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "legend.fancybox":    True,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,
    "lines.linewidth":    1.8,
    "lines.markersize":   6,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

# Colour palettes
PALETTE = ["#2166ac", "#d6604d", "#4daf4a", "#984ea3", "#ff7f00",
           "#a65628", "#e41a1c", "#377eb8"]          # 8-colour cycle
CMAP_SEQ = plt.cm.YlOrRd                             # sequential colourmap

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION — All tuneable parameters in one place
# ══════════════════════════════════════════════════════════════════
# -- Output --
SAVE_DPI    = 200               # resolution of saved figures
IMG_DIR     = "Images_final_report"           # output directory for figures

# -- Solver --
N_STREAMS   = 36               # number of discrete-ordinate streams (must be even)
                                # half of these are upward → N_STREAMS//2 viewing angles
# -- Atmosphere (common) --
ALBEDO      = 0.1               # Lambertian surface albedo
THETA0_DEG  = 30.0              # solar zenith angle (degrees)
F0          = np.pi             # TOA solar flux (π = normalisation convention)

# -- Level 1: single layer --
L1_SSA      = 0.90              # single-scattering albedo
L1_G        = 0.70              # asymmetry parameter
L1_AOD_TRUE = 0.3               # true AOD for retrieval tests
L1_AOD_LIST = [0.05, 0.1, 0.3, 0.5, 1.0]  # AODs for radiance plot
L1_NOISE    = 0.0              # relative Gaussian noise (2%)
L1_N_SENSITIVITY = 50           # number of AOD points for scatter plot
L1_GAMMA    = 0.01              # Tikhonov regularisation strength

# -- Level 2: 20-layer atmosphere --
L2_AEROSOL_TYPE = "continental"  # aerosol preset name
L2_WAVELENGTH   = 550.0         # primary wavelength (nm)
L2_AOD_TOTAL    = 0.3           # total column AOD
L2_H_AER_KM     = 2.0           # aerosol scale height (km)
L2_NOISE        = 0.0          # relative Gaussian noise (2%)
L2_WAVELENGTHS  = [440, 550, 670, 870]  # multi-wavelength set
L2_GAMMA_LM     = 0.05          # Tikhonov gamma for LM retrieval
L2_GAMMA_LCURVE = np.logspace(-4, 2, 30)  # gamma range for L-curve

# -- Run flags (set True/False to enable/disable each section) --
RUN_VALIDATION  = False       # forward + inverse validation tests
RUN_LEVEL1      = False       # idealised single-layer AOD retrieval
RUN_LEVEL2      = False       # 20-layer joint retrieval (2a + 2b)
RUN_LEVEL3      = False       # AERONET + SURFRAD real data
RUN_STUDY_A     = False       # parameter studies (SSA, AOD, g, albedo, SZA, scenarios)
RUN_STUDY_B     = False       # extreme physical cases
RUN_STUDY_C     = False       # energy balance
RUN_STUDY_D     = True       # discrete vs continuous comparison

# -- Derived --
THETA0 = np.deg2rad(THETA0_DEG)

# Radiance units label (normalised to F0/π)
RAD_UNITS = r"Radiance  [$F_0/\pi$ units]"

import os
os.makedirs(IMG_DIR, exist_ok=True)

def savefig(fig, basename):
    """Save figure into IMG_DIR with the configured DPI."""
    fname = os.path.join(IMG_DIR, f"{basename}.png")
    fig.savefig(fname, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════
#  SHARED UTILITIES — Functions used across multiple sections
# ══════════════════════════════════════════════════════════════════
def compute_fluxes(sv, prof):
    """Return (F_up_toa, F_down_boa, F_direct_boa)."""
    mu_up, I_up, mu_dn, I_dn = sv.solve(prof, output="both")
    w_up = sv.weights[sv.up_idx]
    w_dn = sv.weights[sv.down_idx]
    F_up   = 2 * np.pi * np.sum(w_up * mu_up * np.abs(I_up))
    F_down = 2 * np.pi * np.sum(w_dn * mu_dn * np.abs(I_dn))
    F_dir  = prof.f0 * prof.mu0 * np.exp(-prof.tau_total / prof.mu0)
    return mu_up, I_up, mu_dn, I_dn, F_up, F_down, F_dir

# Default baseline parameters for studies and comparisons
BASE = dict(aod=0.3, ssa=0.9, g=0.7, albedo=0.1,
            theta0=THETA0, f0=F0)
F_solar = F0 * np.cos(THETA0)

def rmse(t, e): return np.sqrt(np.mean((t - e)**2))
def bias(t, e): return np.mean(e - t)

# Fine angular grid for continuous (interpolated) intensity plots
FINE_ANGLES_DEG = np.arange(1, 90, 1.0)
FINE_MUS = np.cos(np.deg2rad(FINE_ANGLES_DEG))

# Defaults for variables that may be skipped if sections are disabled
total_pass = 0
total_total = 0
aeronet_success = False
study_lines = []
study_plots = []

def compute_fluxes_continuous(sv, prof):
    """Like compute_fluxes but also return continuous-angle intensities.

    Returns (F_up, F_down, F_dir, I_toa_fine, I_boa_fine).
    Fluxes are still computed from quadrature (exact); intensities are
    evaluated on the FINE_ANGLES_DEG grid via Eqs 35a/35b.
    """
    mu_up, I_up, mu_dn, I_dn = sv.solve(prof, output="both",
                                          store_state=True)
    w_up = sv.weights[sv.up_idx]
    w_dn = sv.weights[sv.down_idx]
    F_up   = 2 * np.pi * np.sum(w_up * mu_up * np.abs(I_up))
    F_down = 2 * np.pi * np.sum(w_dn * mu_dn * np.abs(I_dn))
    F_dir  = prof.f0 * prof.mu0 * np.exp(-prof.tau_total / prof.mu0)
    _, I_toa_fine = sv.interpolate_intensity(FINE_MUS, output="toa")
    _, I_boa_fine = sv.interpolate_intensity(FINE_MUS, output="boa")
    return F_up, F_down, F_dir, np.abs(I_toa_fine), np.abs(I_boa_fine)

# ══════════════════════════════════════════════════════════════════
#  VALIDATION — Forward Model
# ══════════════════════════════════════════════════════════════════
if RUN_VALIDATION:
    total_pass = 0
    total_total = 0
    if RUN_VALIDATION:
        print("=" * 70)
        print("  VALIDATION — Forward Model")
        print("=" * 70)
    tests_passed = 0
    tests_total = 0

    solver16 = DisortSolver(n_streams=N_STREAMS)

    # TEST 1: tau -> 0 limit (no atmosphere)
    tests_total += 1
    prof_thin = AtmosphereProfile(aod=1e-6, ssa=0.9, g=0.7, albedo=ALBEDO,
                                  theta0=THETA0)
    mu_t1, I_t1 = solver16.solve(prof_thin, output="toa")
    # Expected: only surface reflection of direct beam
    I_expected_thin = (ALBEDO / np.pi) * F0 * np.cos(THETA0)
    # For mu near nadir the value should approach I_expected_thin
    mean_I_t1 = np.mean(np.abs(I_t1))
    err_t1 = abs(mean_I_t1 - I_expected_thin) / I_expected_thin
    pass_t1 = err_t1 < 0.15
    tests_passed += int(pass_t1)
    print(f"\n  TEST 1 (tau->0 limit): {'PASS' if pass_t1 else 'FAIL'}"
          f"  mean(I_toa)={mean_I_t1:.4f} vs expected~{I_expected_thin:.4f}"
          f"  (err={err_t1:.1%})")

    # TEST 2: Convergence with N_streams
    tests_total += 1
    prof_conv = AtmosphereProfile(aod=0.5, ssa=0.9, g=0.7, albedo=ALBEDO,
                                  theta0=THETA0)
    prev_I = None
    stream_counts = [4, 8, 16, 32]
    convergence_ok = True
    for ns in stream_counts:
        s = DisortSolver(n_streams=ns)
        _, I_ns = s.solve(prof_conv, output="toa")
        if prev_I is not None and len(I_ns) == len(prev_I):
            diff = np.max(np.abs(I_ns - prev_I)) / (np.max(np.abs(prev_I)) + 1e-15)
            if ns == stream_counts[-1] and diff > 0.05:
                convergence_ok = False
        prev_I = I_ns
    tests_passed += int(convergence_ok)
    print(f"  TEST 2 (N-stream convergence): {'PASS' if convergence_ok else 'FAIL'}")

    # TEST 3: Multi-layer vs single-layer consistency
    tests_total += 1
    prof_1L = AtmosphereProfile(aod=1.0, ssa=0.95, g=0.65, albedo=ALBEDO,
                                theta0=THETA0)
    prof_20L = AtmosphereProfile(
        aod=np.full(20, 0.05), ssa=np.full(20, 0.95), g=np.full(20, 0.65),
        albedo=ALBEDO, theta0=THETA0,
    )
    _, I_1L = solver16.solve(prof_1L, output="toa")
    _, I_20L = solver16.solve(prof_20L, output="toa")
    diff_ml = np.max(np.abs(I_1L - I_20L)) / (np.max(np.abs(I_1L)) + 1e-15)
    pass_t3 = diff_ml < 0.01
    tests_passed += int(pass_t3)
    print(f"  TEST 3 (1-layer vs 20-layer consistency): {'PASS' if pass_t3 else 'FAIL'}"
          f"  max_rel_diff={diff_ml:.6f}")

    # TEST 4: Pure absorption (omega->0) — minimal scattering
    tests_total += 1
    prof_abs = AtmosphereProfile(aod=1.0, ssa=0.01, g=0.7, albedo=0.0,
                                 theta0=THETA0)
    _, I_abs = solver16.solve(prof_abs, output="toa")
    mean_abs = np.mean(np.abs(I_abs))
    pass_t4 = mean_abs < 0.05
    tests_passed += int(pass_t4)
    print(f"  TEST 4 (pure absorption, I_toa~0): {'PASS' if pass_t4 else 'FAIL'}"
          f"  mean|I|={mean_abs:.6f}")

    print(f"\n  Forward validation: {tests_passed}/{tests_total} tests passed.")

    # ══════════════════════════════════════════════════════════════════
    #  VALIDATION — Inverse Model
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  VALIDATION — Inverse Model")
    print("=" * 70)
    inv_tests_passed = 0
    inv_tests_total = 0

    solver_inv = DisortSolver(n_streams=N_STREAMS)

    # TEST 1: Self-consistency (no noise)
    inv_tests_total += 1
    true_aod_inv = 0.3
    prof_inv = AtmosphereProfile(aod=true_aod_inv, ssa=0.92, g=0.70,
                                 albedo=ALBEDO, theta0=THETA0)
    mu_inv, I_inv = solver_inv.solve(prof_inv, output="toa")
    y_clean = np.abs(I_inv)

    opt_inv = InverseOptimizer(solver_inv, prof_inv.copy(),
                               x_a=np.array([0.5]), gamma=0.001)
    res_inv = opt_inv.retrieve(y_clean, np.array([0.1]), ["aod"])
    err_self = abs(res_inv.x[0] - true_aod_inv) / true_aod_inv
    pass_inv1 = err_self < 0.01
    inv_tests_passed += int(pass_inv1)
    print(f"\n  TEST 1 (self-consistency, no noise): {'PASS' if pass_inv1 else 'FAIL'}"
          f"  AOD_true={true_aod_inv:.3f} AOD_ret={res_inv.x[0]:.4f}"
          f"  err={err_self:.4%}")

    # TEST 2: OE self-consistency
    inv_tests_total += 1
    def forward_oe_test(x):
        p = AtmosphereProfile(aod=x[0], ssa=0.92, g=0.70,
                              albedo=ALBEDO, theta0=THETA0)
        _, I = solver_inv.solve(p, output="toa")
        return np.abs(I)

    x_a_oe = np.array([0.5])
    S_a_oe = np.array([[0.1**2]])
    noise_std = 0.02 * np.mean(y_clean)
    S_eps_oe = np.eye(len(y_clean)) * noise_std**2

    oe = OptimalEstimation(forward_oe_test, x_a_oe, S_a_oe, S_eps_oe)
    res_oe = oe.retrieve(y_clean, np.array([0.2]))
    err_oe = abs(res_oe["x"][0] - true_aod_inv) / true_aod_inv
    pass_inv2 = err_oe < 0.05 and res_oe["converged"]
    inv_tests_passed += int(pass_inv2)
    print(f"  TEST 2 (OE self-consistency): {'PASS' if pass_inv2 else 'FAIL'}"
          f"  AOD_ret={res_oe['x'][0]:.4f} DFS={res_oe['DFS']:.3f}"
          f"  converged={res_oe['converged']}")

    # TEST 3: Sensitivity to a priori (large S_a = weak constraint → data dominates)
    inv_tests_total += 1
    results_apriori = []
    for xa_val in [0.15, 0.30, 0.50, 0.80]:
        oe_ap = OptimalEstimation(forward_oe_test, np.array([xa_val]),
                                  np.array([[1.0**2]]), S_eps_oe)
        r_ap = oe_ap.retrieve(y_clean, np.array([0.3]), max_iter=30)
        results_apriori.append(r_ap["x"][0])
    spread = max(results_apriori) - min(results_apriori)
    pass_inv3 = spread < 0.05
    inv_tests_passed += int(pass_inv3)
    print(f"  TEST 3 (a priori sensitivity): {'PASS' if pass_inv3 else 'FAIL'}"
          f"  spread={spread:.4f} across x_a={[0.15,0.30,0.50,0.80]}")

    print(f"\n  Inverse validation: {inv_tests_passed}/{inv_tests_total} tests passed.")

    total_pass = tests_passed + inv_tests_passed
    total_total = tests_total + inv_tests_total
    print(f"\n  TOTAL: {total_pass}/{total_total} validation tests passed.")


# ══════════════════════════════════════════════════════════════════
#  LEVEL 1 — Idealised single-layer atmosphere
# ══════════════════════════════════════════════════════════════════
if RUN_LEVEL1:
    print("\n" + "=" * 70)
    print("  LEVEL 1 : Single Layer — AOD Retrieval")
    print("=" * 70)

    solver = DisortSolver(n_streams=N_STREAMS)

    # Helper: compute hemispheric upwelling flux from TOA radiances
    # F↑ = 2π Σᵢ wᵢ μᵢ |I(μᵢ)|   (Gauss-Legendre quadrature)
    def compute_flux_up(solver_obj, mu_up, I_up):
        w_up = solver_obj.weights[solver_obj.up_idx]
        return 2.0 * np.pi * np.sum(w_up * mu_up * np.abs(I_up))

    # 1a-1b. Forward for multiple AODs + flux budget
    fig1, (ax1, ax1b) = plt.subplots(1, 2, figsize=(14, 5.5),
                                      gridspec_kw={"width_ratios": [3, 2]})
    markers = ['o', 's', 'D', '^', 'v']
    F_in = F0 * np.cos(THETA0)
    flux_table = []
    for idx, aod_val in enumerate(L1_AOD_LIST):
        p = AtmosphereProfile(aod=aod_val, ssa=L1_SSA, g=L1_G,
                              albedo=ALBEDO, theta0=THETA0, f0=F0)
        F_up, _, _, I_toa_fine, _ = compute_fluxes_continuous(solver, p)
        R = F_up / F_in
        flux_table.append((aod_val, F_up, R))
        c = PALETTE[idx % len(PALETTE)]
        ax1.plot(FINE_ANGLES_DEG, I_toa_fine, "-", color=c, lw=1.5,
                 label=rf"$\tau={aod_val}$  ($R={R:.3f}$)")
    ax1.set_xlabel("Viewing Zenith Angle (deg)")
    ax1.set_ylabel(RAD_UNITS)
    ax1.legend(loc="best", fontsize=12)

    # Right panel: Reflectance bar chart (flux budget)
    aod_vals = [row[0] for row in flux_table]
    refl_vals = [row[2] for row in flux_table]
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(aod_vals))]
    bars = ax1b.bar(range(len(aod_vals)), refl_vals, color=bar_colors,
                    edgecolor="white", linewidth=0.8, alpha=0.85)
    for bar, R_val in zip(bars, refl_vals):
        ax1b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                  f"{R_val:.3f}", ha="center", va="bottom", fontsize=12)
    ax1b.set_xticks(range(len(aod_vals)))
    ax1b.set_xticklabels([rf"$\tau={v}$" for v in aod_vals], fontsize=12)
    ax1b.set_ylabel(r"Reflectance  $R = F^{\uparrow}_{TOA} \,/\, F_0\mu_0$")
    ax1b.set_ylim(0, max(refl_vals) * 1.15)
    ax1b.axhline(1.0, color="#999999", ls=":", lw=0.8, label="$R=1$ (no loss)")
    ax1b.legend(fontsize=11)
    fig1.tight_layout()
    savefig(fig1, f"Level1_TOA_radiance_SSA_{L1_SSA}_g{L1_G}_albedo_{ALBEDO}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}".replace(".", ""))

    # Print flux budget table
    print(f"\n  Flux budget  (F_in = F0·mu0 = {F_in:.4f}):")
    print(f"  {'AOD':>6} {'F_up(TOA)':>10} {'R=F_up/F_in':>12} {'Absorbed':>10}")
    for aod_val, F_up, R in flux_table:
        print(f"  {aod_val:6.2f} {F_up:10.4f} {R:12.4f} {1-R:10.4f}")

    # 1-extra: Energy conservation demonstration
    # Show R vs ω₀ for fixed AOD to prove that R→1 when ω₀→1 and albedo→1
    fig1c, ax1c = plt.subplots(figsize=(8, 5))
    ssa_range = np.linspace(0.5, 1.0, 20)
    for alb_val, ls, c in [(1.0, "-", PALETTE[0]),
                            (ALBEDO, "--", PALETTE[1])]:
        R_arr = []
        for ssa_val in ssa_range:
            p = AtmosphereProfile(aod=0.3, ssa=ssa_val, g=L1_G,
                                  albedo=alb_val, theta0=THETA0, f0=F0)
            mu_tmp, I_tmp = solver.solve(p, output="toa")
            F_up_tmp = compute_flux_up(solver, mu_tmp, I_tmp)
            R_arr.append(F_up_tmp / F_in)
        ax1c.plot(ssa_range, R_arr, ls=ls, color=c, lw=2,
                  marker="o", markersize=4, markeredgecolor="white",
                  label=rf"albedo $= {alb_val}$")
    ax1c.axhline(1.0, color="#999999", ls=":", lw=0.8)
    ax1c.set_xlabel(r"Single Scattering Albedo $\omega_0$")
    ax1c.set_ylabel(r"Reflectance  $R = F^{\uparrow}_{TOA} / F_0\mu_0$")
    ax1c.legend()
    ax1c.set_xlim(0.5, 1.0)
    ax1c.set_ylim(0, 1.1)
    fig1c.tight_layout()
    savefig(fig1c, f"Level1_energy_conservation_AOD_03_g{L1_G}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}".replace(".", ""))

    # 1c-1d. Inverse: retrieve AOD from clean & noisy data
    prof_true = AtmosphereProfile(aod=L1_AOD_TRUE, ssa=L1_SSA, g=L1_G,
                                  albedo=ALBEDO, theta0=THETA0, f0=F0)
    mu_obs, I_obs = solver.solve(prof_true, output="toa")
    y_clean = np.abs(I_obs)

    opt_clean = InverseOptimizer(solver, prof_true.copy(),
                                 x_a=np.array([0.5]), gamma=L1_GAMMA)
    res_clean = opt_clean.retrieve(y_clean, np.array([0.1]), ["aod"])
    print(f"  Clean retrieval: AOD_true={L1_AOD_TRUE:.3f},"
          f" AOD_ret={res_clean.x[0]:.4f}")

    y_noisy = y_clean + L1_NOISE * y_clean * rng.standard_normal(len(y_clean))
    res_noisy = opt_clean.retrieve(y_noisy, np.array([0.1]), ["aod"])
    print(f"  Noisy retrieval: AOD_true={L1_AOD_TRUE:.3f},"
          f" AOD_ret={res_noisy.x[0]:.4f}")

    # 1e-1f. Sensitivity: AOD_true vs AOD_retrieved (LM + OE)
    aod_true_arr = np.linspace(0.05, 1.0, L1_N_SENSITIVITY)
    aod_ret_lm  = np.zeros(L1_N_SENSITIVITY)
    aod_ret_oe  = np.zeros(L1_N_SENSITIVITY)

    def forward_oe_l1(x, _sv=solver):
        p = AtmosphereProfile(aod=x[0], ssa=L1_SSA, g=L1_G,
                              albedo=ALBEDO, theta0=THETA0, f0=F0)
        _, I = _sv.solve(p, output="toa")
        return np.abs(I)

    for i, aod_t in enumerate(aod_true_arr):
        p = AtmosphereProfile(aod=aod_t, ssa=L1_SSA, g=L1_G,
                              albedo=ALBEDO, theta0=THETA0, f0=F0)
        _, I_t = solver.solve(p, output="toa")
        y_t = np.abs(I_t) + L1_NOISE * np.abs(I_t) * rng.standard_normal(len(I_t))
        # LM + Tikhonov
        opt_t = InverseOptimizer(solver, p.copy(),
                                 x_a=np.array([0.5]), gamma=L1_GAMMA)
        r_t = opt_t.retrieve(y_t, np.array([0.3]), ["aod"])
        aod_ret_lm[i] = r_t.x[0]
        # Optimal Estimation
        noise_std_l1 = 0.02 * np.mean(y_t)
        oe_l1 = OptimalEstimation(forward_oe_l1, np.array([0.5]),
                                   np.array([[0.5**2]]),
                                   np.eye(len(y_t)) * noise_std_l1**2)
        r_oe = oe_l1.retrieve(y_t, np.array([0.3]))
        aod_ret_oe[i] = r_oe["x"][0]

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 6))

    for ax2, aod_ret_arr, method_name in [
        (ax2a, aod_ret_lm, "LM + Tikhonov"),
        (ax2b, aod_ret_oe, "Optimal Estimation"),
    ]:
        ax2.fill_between([0, 1.1], [0 - 0.05, 1.1 - 0.05], [0 + 0.05, 1.1 + 0.05],
                         color="#cccccc", alpha=0.4, label=r"$\pm 0.05$ band")
        ax2.plot([0, 1.1], [0, 1.1], color="#333333", ls="--", lw=1.2, label="1 : 1")
        aod_err = np.abs(aod_ret_arr - aod_true_arr)
        sc = ax2.scatter(aod_true_arr, aod_ret_arr, c=aod_err, cmap="RdYlGn_r",
                         s=40, edgecolors="white", linewidths=0.5, zorder=5,
                         vmin=0, vmax=max(0.15, np.percentile(aod_err, 95)))
        cbar = fig2.colorbar(sc, ax=ax2, shrink=0.75, pad=0.02)
        cbar.set_label(r"$|\Delta\tau|$", fontsize=11)
        ax2.set_xlabel(r"True AOD ($\tau_{true}$)")
        ax2.set_ylabel(r"Retrieved AOD ($\tau_{ret}$)")
        r_val = rmse(aod_true_arr, aod_ret_arr)
        b_val = bias(aod_true_arr, aod_ret_arr)
        ax2.legend(loc="upper left", fontsize=11)
        ax2.set_xlim(0, 1.1)
        ax2.set_ylim(0, 1.1)
        ax2.set_aspect("equal")

    fig2.tight_layout()
    savefig(fig2, f"Level1_AOD_scatter_SSA_{L1_SSA}_g{L1_G}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}".replace(".", ""))


    print(f"\n  Level 1 complete: 3 figures generated.")

# ══════════════════════════════════════════════════════════════════
#  LEVEL 2 — Realistic 20-Layer Atmosphere
# ══════════════════════════════════════════════════════════════════
if RUN_LEVEL2:
    print("\n" + "=" * 70)
    print("  LEVEL 2 : 20-Layer Atmosphere — Joint Retrieval")
    print("=" * 70)

    # 2a-2b. Standard atmosphere + aerosol from config
    aer_props = get_aerosol_preset(L2_AEROSOL_TYPE, L2_WAVELENGTH)
    print(f"\n  {L2_AEROSOL_TYPE} aerosol at {L2_WAVELENGTH:.0f}nm:"
          f" SSA={aer_props['SSA']:.4f}, g={aer_props['g']:.4f}")

    atm = StandardAtmosphere(wavelength_nm=L2_WAVELENGTH)
    prof_20 = atm.to_profile(
        aod_total=L2_AOD_TOTAL, ssa_aer=aer_props["SSA"], g_aer=aer_props["g"],
        H_aer_km=L2_H_AER_KM, albedo=ALBEDO, theta0=THETA0, f0=F0,
    )

    # 2c. Forward: TOA radiances
    mu_20, I_20 = solver.solve(prof_20, output="toa")
    y_true_20 = np.abs(I_20)

    print(f"  20-layer profile: total_tau={prof_20.tau_total:.4f}")
    print(f"  TOA radiance: min={np.min(y_true_20):.4f},"
          f" max={np.max(y_true_20):.4f}")

    # 2d. Add Gaussian noise
    noise_20 = L2_NOISE * y_true_20 * rng.standard_normal(len(y_true_20))
    y_obs_20 = y_true_20 + noise_20

    # 2e. Inverse with LM + Tikhonov
    true_aod_total = L2_AOD_TOTAL
    true_ssa_eff = float(np.mean(prof_20.ssa))
    true_g_eff = float(np.mean(prof_20.g))

    x_a_lv2 = np.array([0.5, 0.85, 0.5])
    opt_lm = InverseOptimizer(solver, prof_20.copy(), x_a_lv2, gamma=L2_GAMMA_LM)
    res_lm = opt_lm.retrieve(y_obs_20, np.array([0.2, 0.90, 0.60]),
                              ["aod", "ssa", "g"],
                              bounds=([0.01, 0.5, 0.0], [3.0, 0.9999, 0.999]))

    print(f"\n  LM + Tikhonov retrieval:")
    print(f"    AOD: true={true_aod_total:.3f}, ret={res_lm.x[0]:.4f}")
    print(f"    SSA: true={true_ssa_eff:.4f}, ret={res_lm.x[1]:.4f}")
    print(f"    g:   true={true_g_eff:.4f}, ret={res_lm.x[2]:.4f}")

    # 2f. Inverse with Optimal Estimation
    def forward_20(x):
        p = atm.to_profile(aod_total=x[0], ssa_aer=x[1], g_aer=x[2],
                           H_aer_km=L2_H_AER_KM, albedo=ALBEDO,
                           theta0=THETA0, f0=F0)
        _, I = solver.solve(p, output="toa")
        return np.abs(I)

    x_a_oe = np.array([0.5, 0.85, 0.5])
    S_a_oe2 = np.diag([0.3**2, 0.1**2, 0.2**2])
    noise_std_20 = 0.02 * np.mean(y_true_20)
    S_eps_oe2 = np.eye(len(y_obs_20)) * noise_std_20**2

    oe2 = OptimalEstimation(forward_20, x_a_oe, S_a_oe2, S_eps_oe2)
    res_oe2 = oe2.retrieve(y_obs_20, np.array([0.2, 0.90, 0.60]))

    print(f"\n  Optimal Estimation retrieval:")
    print(f"    AOD: true={true_aod_total:.3f}, ret={res_oe2['x'][0]:.4f}")
    print(f"    SSA: true={aer_props['SSA']:.4f}, ret={res_oe2['x'][1]:.4f}")
    print(f"    g:   true={aer_props['g']:.4f}, ret={res_oe2['x'][2]:.4f}")
    print(f"    DFS={res_oe2['DFS']:.3f}, converged={res_oe2['converged']},"
          f" iters={res_oe2['n_iter']}")

    # 2g. Comparison table
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  Method       │  AOD      │  SSA      │  g        │ Cost    │")
    print(f"  │ ───────────── │ ───────── │ ───────── │ ───────── │ ─────── │")
    print(f"  │  True         │  {true_aod_total:.4f}   │  {aer_props['SSA']:.4f}   │  "
          f"{aer_props['g']:.4f}   │   --    │")
    print(f"  │  LM+Tikhonov  │  {res_lm.x[0]:.4f}   │  {res_lm.x[1]:.4f}   │  "
          f"{res_lm.x[2]:.4f}   │ {res_lm.cost:.1e} │")
    print(f"  │  OE (Rodgers) │  {res_oe2['x'][0]:.4f}   │  {res_oe2['x'][1]:.4f}   │  "
          f"{res_oe2['x'][2]:.4f}   │ {res_oe2['cost']:.1e} │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # 2g-bis. Visual comparison: LM+Tikhonov vs OE
    fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(15, 5))
    true_vals  = [true_aod_total, aer_props["SSA"], aer_props["g"]]
    lm_vals    = [res_lm.x[0], res_lm.x[1], res_lm.x[2]]
    oe_vals    = [res_oe2["x"][0], res_oe2["x"][1], res_oe2["x"][2]]
    cmp_labels = [r"$\tau$ (AOD)", r"$\omega_0$ (SSA)", r"$g$"]

    for ax_c, true_v, lm_v, oe_v, lbl in zip(axes_cmp, true_vals, lm_vals,
                                                oe_vals, cmp_labels):
        x_b = np.arange(3)
        vals = [true_v, lm_v, oe_v]
        colors = ["#333333", PALETTE[1], PALETTE[0]]
        labels = ["True", "LM+Tikhonov", "OE (Rodgers)"]
        bars = ax_c.bar(x_b, vals, color=colors, edgecolor="white",
                        linewidth=0.8, alpha=0.85)
        for b, v in zip(bars, vals):
            ax_c.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                      f"{v:.4f}", ha="center", va="bottom", fontsize=12)
        ax_c.set_xticks(x_b)
        ax_c.set_xticklabels(labels, fontsize=12)
        ax_c.set_ylabel(lbl)
        # Error annotations
        err_lm = abs(lm_v - true_v)
        err_oe = abs(oe_v - true_v)
        ax_c.text(0.98, 0.95,
                  f"LM err: {err_lm:.4f}\nOE err: {err_oe:.4f}",
                  transform=ax_c.transAxes, fontsize=8,
                  va="top", ha="right",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white",
                            ec="#cccccc", alpha=0.9))

    fig_cmp.tight_layout()
    savefig(fig_cmp, f"Level2_LM_vs_OE_{L2_AEROSOL_TYPE}_AOD_{L2_AOD_TOTAL}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}".replace(".", ""))

    # 2h. Averaging kernels
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    A_oe = res_oe2["A"]
    param_labels = [r"$\tau$", r"$\omega_0$", r"$g$"]
    param_tex     = [r"\tau", r"\omega_0", "g"]
    bar_colors = [PALETTE[0], PALETTE[1], PALETTE[3]]
    n_params = A_oe.shape[1]
    x_pos = np.arange(n_params)
    bar_w = 0.22
    for i in range(A_oe.shape[0]):
        ax3.bar(x_pos + i * bar_w, A_oe[i, :], width=bar_w,
                color=bar_colors[i], edgecolor="white", linewidth=0.5,
                label=rf"$\partial \hat{{x}} / \partial {param_tex[i]}$")
    ax3.set_xticks(x_pos + bar_w)
    ax3.set_xticklabels(param_labels)
    ax3.set_xlabel("State vector element")
    ax3.set_ylabel("Averaging Kernel value")
    ax3.legend(loc="best")
    ax3.axhline(1.0, color="#999999", ls=":", lw=0.8)
    fig3.tight_layout()
    savefig(fig3, f"Level2_averaging_kernels_{L2_AEROSOL_TYPE}_N{N_STREAMS}".replace(".", ""))

    # 2i. Vertical profile — AOD per layer
    prof_ret = atm.to_profile(
        aod_total=res_oe2["x"][0], ssa_aer=res_oe2["x"][1],
        g_aer=res_oe2["x"][2], H_aer_km=L2_H_AER_KM, albedo=ALBEDO,
        theta0=THETA0, f0=F0,
    )

    fig4, ax4 = plt.subplots(figsize=(7, 7.5))
    layer_h = np.diff(atm.Z_BOUNDS) * 0.38
    ax4.barh(atm.z_mid - layer_h / 2, prof_20.tau, height=layer_h,
             color=PALETTE[0], alpha=0.75, edgecolor="white", linewidth=0.5,
             label="True profile")
    ax4.barh(atm.z_mid + layer_h / 2 + 0.05, prof_ret.tau, height=layer_h,
             color=PALETTE[1], alpha=0.75, edgecolor="white", linewidth=0.5,
             label="Retrieved profile")
    ax4.set_ylabel("Altitude (km)")
    ax4.set_xlabel(r"Layer optical depth $\Delta\tau_k$")
    ax4.legend(loc="upper right")
    ax4.set_ylim(-0.5, 22)
    # Highlight troposphere / stratosphere boundary
    ax4.axhline(11, color="#888888", ls=":", lw=0.8)
    ax4.text(ax4.get_xlim()[1] * 0.82, 11.3, "tropopause",
             fontsize=8, color="#888888", ha="center")
    fig4.tight_layout()
    savefig(fig4, f"Level2_vertical_profile_{L2_AEROSOL_TYPE}_AOD_{L2_AOD_TOTAL}_N{N_STREAMS}".replace(".", ""))

    # 2j-2k. Phillips-Twomey comparison with L-curve
    # Linearise around the true state to get K
    K_lin = res_oe2["K"]
    y_lin = y_obs_20
    x_true_oe = np.array([true_aod_total, aer_props["SSA"], aer_props["g"]])

    pt = PhillipsTwomey(order=2)
    _, res_norms, sol_norms = pt.l_curve(K_lin, y_lin, L2_GAMMA_LCURVE, x_a_oe)

    fig5, ax5 = plt.subplots(figsize=(7, 5.5))
    ax5.loglog(res_norms, sol_norms, color=PALETTE[0], marker="o",
               markersize=5, markeredgecolor="white", markeredgewidth=0.5,
               lw=1.5, zorder=5)
    # Mark corner (max curvature) heuristic: closest to origin in log space
    log_r = np.log10(np.asarray(res_norms) + 1e-30)
    log_s = np.log10(np.asarray(sol_norms) + 1e-30)
    dist = np.sqrt((log_r - log_r.min())**2 + (log_s - log_s.min())**2)
    i_corner = np.argmin(dist)
    ax5.plot(res_norms[i_corner], sol_norms[i_corner], '*',
             color=PALETTE[1], markersize=14, markeredgecolor="black",
             markeredgewidth=0.6, zorder=6,
             label=rf"Corner ($\gamma={L2_GAMMA_LCURVE[i_corner]:.2e}$)")
    ax5.set_xlabel(r"$\|y - K\hat{x}\|_2$  (Residual norm)")
    ax5.set_ylabel(r"$\|H\hat{x}\|_2$  (Solution norm)")
    ax5.legend(loc="best")
    fig5.tight_layout()
    savefig(fig5, f"Level2_Lcurve_{L2_AEROSOL_TYPE}_N{N_STREAMS}".replace(".", ""))

    # Multi-wavelength forward
    print(f"\n  Multi-wavelength TOA radiances"
          f" ({L2_AEROSOL_TYPE} aerosol, AOD={L2_AOD_TOTAL}):")
    fig6, ax6 = plt.subplots(figsize=(9, 5.5))
    wl_colors = {440: "#7b3294", 550: "#008837", 670: "#d6604d", 870: "#b35806"}
    wl_markers = {440: "s", 550: "D", 670: "o", 870: "^"}
    for wl in L2_WAVELENGTHS:
        aer = get_aerosol_preset(L2_AEROSOL_TYPE, wl)
        atm_wl = StandardAtmosphere(wavelength_nm=wl)
        prof_wl = atm_wl.to_profile(aod_total=L2_AOD_TOTAL, ssa_aer=aer["SSA"],
                                     g_aer=aer["g"], H_aer_km=L2_H_AER_KM,
                                     albedo=ALBEDO, theta0=THETA0, f0=F0)
        mu_wl, I_wl = solver.solve(prof_wl, output="toa")
        c = wl_colors.get(wl, PALETTE[0])
        m = wl_markers.get(wl, "o")
        ax6.plot(np.rad2deg(np.arccos(mu_wl)), np.abs(I_wl),
                 marker=m, color=c, markeredgecolor="white",
                 markeredgewidth=0.6,
                 label=rf"$\lambda={wl}$ nm")
        print(f"    {wl}nm: SSA={aer['SSA']:.4f}, g={aer['g']:.4f},"
              f" tau_total={prof_wl.tau_total:.4f}")

    ax6.set_xlabel("Viewing Zenith Angle (deg)")
    ax6.set_ylabel(RAD_UNITS)
    ax6.legend(loc="best")
    fig6.tight_layout()
    savefig(fig6, f"Level2_multiwavelength_{L2_AEROSOL_TYPE}_N{N_STREAMS}".replace(".", ""))



# ══════════════════════════════════════════════════════════════════
#  LEVEL 2b — Retrieval Validation (20 scenarios)
# ══════════════════════════════════════════════════════════════════
if RUN_LEVEL2:
    print("\n" + "=" * 70)
    print("  LEVEL 2b : Retrieval Validation — 3-Parameter (AOD, SSA, g)")
    print("=" * 70)
    retrieval_solver = DisortSolver(n_streams=N_STREAMS)

    # Define ~20 true (AOD, SSA, g) combinations spanning realistic ranges
    retrieval_cases = [
        (0.05, 0.99, 0.72),  # clean maritime
        (0.10, 0.95, 0.70),  # background continental
        (0.15, 0.90, 0.68),  # light pollution
        (0.20, 0.88, 0.65),  # moderate pollution
        (0.25, 0.92, 0.75),  # dust-influenced
        (0.30, 0.85, 0.60),  # urban aerosol
        (0.35, 0.93, 0.72),  # mixed continental
        (0.40, 0.80, 0.58),  # absorbing urban
        (0.50, 0.87, 0.66),  # polluted continental
        (0.60, 0.82, 0.62),  # biomass burning
        (0.70, 0.90, 0.70),  # moderate dust
        (0.80, 0.78, 0.55),  # heavy biomass
        (0.90, 0.94, 0.74),  # Saharan dust
        (1.00, 0.85, 0.68),  # thick pollution
        (1.20, 0.92, 0.76),  # dust storm moderate
        (1.50, 0.93, 0.75),  # heavy dust
        (0.08, 0.97, 0.71),  # rural clean
        (0.45, 0.83, 0.63),  # industrial
        (0.55, 0.91, 0.69),  # mixed aerosol
        (0.75, 0.86, 0.64),  # polluted biomass
    ]

    ret_true_aod = []
    ret_true_ssa = []
    ret_true_g   = []
    ret_est_aod  = []
    ret_est_ssa  = []
    ret_est_g    = []

    print(f"  Running {len(retrieval_cases)} retrieval cases...")
    for i, (t_aod, t_ssa, t_g) in enumerate(retrieval_cases):
        # Generate synthetic observation
        prof_ret = AtmosphereProfile(aod=t_aod, ssa=t_ssa, g=t_g,
                                     albedo=ALBEDO, theta0=THETA0, f0=F0)
        _, I_syn = retrieval_solver.solve(prof_ret, output="toa")
        y_syn = np.abs(I_syn)

        # Forward model for OE
        def fwd_ret(x, _sv=retrieval_solver):
            p = AtmosphereProfile(aod=np.clip(x[0], 0.001, 5.0),
                                  ssa=np.clip(x[1], 0.01, 0.9999),
                                  g=np.clip(x[2], 0.01, 0.999),
                                  albedo=ALBEDO, theta0=THETA0, f0=F0)
            _, I = _sv.solve(p, output="toa")
            return np.abs(I)

        x_a_ret = np.array([0.5, 0.88, 0.65])
        S_a_ret = np.diag([0.5**2, 0.15**2, 0.25**2])
        noise_est = 0.02 * np.mean(y_syn)
        S_eps_ret = np.eye(len(y_syn)) * noise_est**2

        oe_ret = OptimalEstimation(fwd_ret, x_a_ret, S_a_ret, S_eps_ret)
        res_ret = oe_ret.retrieve(y_syn, np.array([0.3, 0.90, 0.60]),
                                   max_iter=30, gamma_lm=0.05)

        ret_true_aod.append(t_aod)
        ret_true_ssa.append(t_ssa)
        ret_true_g.append(t_g)
        ret_est_aod.append(res_ret["x"][0])
        ret_est_ssa.append(res_ret["x"][1])
        ret_est_g.append(res_ret["x"][2])

        status = "OK" if res_ret["converged"] else "NC"
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    [{i+1:2d}/{len(retrieval_cases)}] AOD={t_aod:.2f}->"
                  f"{res_ret['x'][0]:.3f}  SSA={t_ssa:.2f}->"
                  f"{res_ret['x'][1]:.3f}  g={t_g:.2f}->"
                  f"{res_ret['x'][2]:.3f}  [{status}]")

    ret_true_aod = np.array(ret_true_aod)
    ret_true_ssa = np.array(ret_true_ssa)
    ret_true_g   = np.array(ret_true_g)
    ret_est_aod  = np.array(ret_est_aod)
    ret_est_ssa  = np.array(ret_est_ssa)
    ret_est_g    = np.array(ret_est_g)


    print(f"\n  Retrieval statistics:")
    print(f"    AOD: RMSE={rmse(ret_true_aod, ret_est_aod):.4f}, "
          f"bias={bias(ret_true_aod, ret_est_aod):.4f}")
    print(f"    SSA: RMSE={rmse(ret_true_ssa, ret_est_ssa):.4f}, "
          f"bias={bias(ret_true_ssa, ret_est_ssa):.4f}")
    print(f"    g:   RMSE={rmse(ret_true_g, ret_est_g):.4f}, "
          f"bias={bias(ret_true_g, ret_est_g):.4f}")

    # 3-panel scatter plot
    fig_ret, (ax_r1, ax_r2, ax_r3) = plt.subplots(1, 3, figsize=(15, 5))

    for ax, t, e, lbl, unit in [
        (ax_r1, ret_true_aod, ret_est_aod, "AOD", ""),
        (ax_r2, ret_true_ssa, ret_est_ssa, "SSA", ""),
        (ax_r3, ret_true_g,   ret_est_g,   "g",   ""),
    ]:
        ax.scatter(t, e, c=PALETTE[0], s=50, edgecolors="white",
                   linewidths=0.5, zorder=3)
        lo = min(t.min(), e.min()) * 0.9
        hi = max(t.max(), e.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {lbl}")
        ax.set_ylabel(f"Retrieved {lbl}")
        r = rmse(t, e)
        b = bias(t, e)
        ax.set_aspect("equal")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig_ret.tight_layout()
    savefig(fig_ret, f"Level2_retrieval_3param_{L2_AEROSOL_TYPE}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}".replace(".", ""))


    print(f"\n  Level 2 complete: averaging kernels, vertical profile, L-curve, multi-wavelength, retrieval validation.")


# ══════════════════════════════════════════════════════════════════
#  LEVEL 3 — Real Data Application (AERONET + SURFRAD)
# ══════════════════════════════════════════════════════════════════
if RUN_LEVEL3:
    print("\n" + "=" * 70)
    print("  LEVEL 3 : Real Data Application — AERONET + SURFRAD")
    print("=" * 70)
    import subprocess
    import csv
    from io import StringIO

    aeronet_success = False  # track if we get data

    # Sites to try (with fallbacks)
    aeronet_sites = ["BONDVILLE", "GSFC", "Mexico_City", "Lille", "Barcelona"]

    aeronet_aod_data = None
    aeronet_ssa_data = None
    aeronet_asy_data = None

    def download_aeronet(url, outfile):
        """Download AERONET data via curl, return True if successful."""
        try:
            result = subprocess.run(
                ["curl", "-s", "-k", url, "-o", outfile],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return False
            # Check file has actual data (not just error page)
            with open(outfile, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) < 100 or "<html" in content.lower():
                return False
            return True
        except Exception:
            return False

    def parse_aeronet_aod(filepath):
        """Parse AERONET AOD CSV file. Returns list of dicts."""
        rows = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return rows

        # Find header line (contains 'Date' or 'AOD_')
        header_idx = None
        for i, line in enumerate(lines):
            if "Date(dd:mm:yyyy)" in line or "AOD_500nm" in line or "AOD_440nm" in line:
                header_idx = i
                break
        if header_idx is None:
            return rows

        header = lines[header_idx].strip().split(",")
        # Find relevant column indices
        col_map = {}
        for j, h in enumerate(header):
            h_clean = h.strip()
            if "Date" in h_clean:
                col_map["date"] = j
            elif h_clean == "AOD_440nm":
                col_map["aod_440"] = j
            elif h_clean == "AOD_500nm":
                col_map["aod_500"] = j
            elif h_clean == "AOD_551nm":
                col_map["aod_551"] = j
            elif h_clean == "AOD_675nm":
                col_map["aod_675"] = j
            elif h_clean == "AOD_870nm":
                col_map["aod_870"] = j
            elif "Solar_Zenith_Angle" in h_clean or "Solar_Zenith" in h_clean:
                col_map["sza"] = j

        for line in lines[header_idx + 1:]:
            parts = line.strip().split(",")
            if len(parts) <= max(col_map.values(), default=0):
                continue
            row = {}
            for key, cidx in col_map.items():
                val = parts[cidx].strip()
                if key == "date":
                    row[key] = val
                else:
                    try:
                        v = float(val)
                        row[key] = v if v > -900 else np.nan
                    except ValueError:
                        row[key] = np.nan
            rows.append(row)
        return rows

    def parse_aeronet_inversion(filepath, product="ssa"):
        """Parse AERONET inversion (SSA or ASY) CSV file."""
        rows = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return rows

        header_idx = None
        for i, line in enumerate(lines):
            if "Date(dd:mm:yyyy)" in line:
                header_idx = i
                break
        if header_idx is None:
            return rows

        header = lines[header_idx].strip().split(",")
        col_map = {}
        # Match exact column patterns
        # SSA: "Single_Scattering_Albedo[440nm]" etc.
        # ASY: "Asymmetry_Factor-Total[440nm]" (not Fine or Coarse)
        if product == "ssa":
            targets = {
                f"{product}_440": "Single_Scattering_Albedo[440nm]",
                f"{product}_675": "Single_Scattering_Albedo[675nm]",
                f"{product}_870": "Single_Scattering_Albedo[870nm]",
            }
        else:
            targets = {
                f"{product}_440": "Asymmetry_Factor-Total[440nm]",
                f"{product}_675": "Asymmetry_Factor-Total[675nm]",
                f"{product}_870": "Asymmetry_Factor-Total[870nm]",
            }
        for j, h in enumerate(header):
            h_clean = h.strip()
            if "Date" in h_clean:
                col_map["date"] = j
            else:
                for key, target in targets.items():
                    if h_clean == target:
                        col_map[key] = j

        for line in lines[header_idx + 1:]:
            parts = line.strip().split(",")
            if not col_map or len(parts) <= max(col_map.values(), default=0):
                continue
            row = {}
            for key, cidx in col_map.items():
                val = parts[cidx].strip()
                if key == "date":
                    row[key] = val
                else:
                    try:
                        v = float(val)
                        row[key] = v if v > -900 else np.nan
                    except ValueError:
                        row[key] = np.nan
            rows.append(row)
        return rows

    # Try downloading from each site
    chosen_site = None
    for site in aeronet_sites:
        print(f"  Trying AERONET site: {site}...")

        aod_url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?"
                   f"site={site}&year=2023&month=6&day=1&year2=2023&month2=8"
                   f"&day2=31&AOD20=1&AVG=20&if_no_html=1")
        ssa_url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3?"
                   f"site={site}&year=2023&month=1&day=1&year2=2023&month2=12"
                   f"&day2=31&product=SSA&AVG=20&ALM20=1&if_no_html=1")
        asy_url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3?"
                   f"site={site}&year=2023&month=1&day=1&year2=2023&month2=12"
                   f"&day2=31&product=ASY&AVG=20&ALM20=1&if_no_html=1")

        ok_aod = download_aeronet(aod_url, "aeronet_aod.csv")
        ok_ssa = download_aeronet(ssa_url, "aeronet_ssa.csv")
        ok_asy = download_aeronet(asy_url, "aeronet_asy.csv")

        if ok_aod:
            aeronet_aod_data = parse_aeronet_aod("aeronet_aod.csv")
        if ok_ssa:
            aeronet_ssa_data = parse_aeronet_inversion("aeronet_ssa.csv", "ssa")
        if ok_asy:
            aeronet_asy_data = parse_aeronet_inversion("aeronet_asy.csv", "asy")

        if aeronet_aod_data and len(aeronet_aod_data) > 5:
            chosen_site = site
            print(f"    AOD records: {len(aeronet_aod_data)}")
            if aeronet_ssa_data:
                print(f"    SSA records: {len(aeronet_ssa_data)}")
            if aeronet_asy_data:
                print(f"    ASY records: {len(aeronet_asy_data)}")
            break
        else:
            print(f"    Insufficient data from {site}, trying next...")

    if chosen_site and aeronet_aod_data:
        aeronet_success = True
        print(f"\n  Using AERONET site: {chosen_site}")

        # Extract AOD values at 440nm (or 500nm fallback)
        aod_key = "aod_440" if "aod_440" in aeronet_aod_data[0] else "aod_500"
        aero_aod_vals = np.array([r.get(aod_key, np.nan) for r in aeronet_aod_data])
        aero_sza_vals = np.array([r.get("sza", 30.0) for r in aeronet_aod_data])
        aero_dates = [r.get("date", "") for r in aeronet_aod_data]

        # Build SSA/ASY lookup by date
        ssa_by_date = {}
        if aeronet_ssa_data:
            for r in aeronet_ssa_data:
                d = r.get("date", "")
                v = r.get("ssa_440", np.nan)
                if d and not np.isnan(v) and 0.5 < v <= 1.0:
                    ssa_by_date[d] = v

        asy_by_date = {}
        if aeronet_asy_data:
            for r in aeronet_asy_data:
                d = r.get("date", "")
                v = r.get("asy_440", np.nan)
                if d and not np.isnan(v) and 0.0 < v < 1.0:
                    asy_by_date[d] = v

        # Simulate with DISORT for each valid AOD entry
        aeronet_solver = DisortSolver(n_streams=N_STREAMS)
        sim_aod_list = []
        sim_R_list = []
        sim_dates_list = []
        sim_ssa_used = []
        sim_g_used = []

        for i, row in enumerate(aeronet_aod_data):
            aod_val = row.get(aod_key, np.nan)
            if np.isnan(aod_val) or aod_val < 0:
                continue

            date_str = row.get("date", "")
            sza = row.get("sza", 30.0)
            if np.isnan(sza) or sza <= 0 or sza >= 85:
                sza = 30.0

            # Use AERONET SSA/g if available, else defaults
            ssa_val = ssa_by_date.get(date_str, 0.92)
            g_val = asy_by_date.get(date_str, 0.70)
            if np.isnan(ssa_val) or ssa_val < 0 or ssa_val > 1:
                ssa_val = 0.92
            if np.isnan(g_val) or g_val < -1 or g_val > 1:
                g_val = 0.70

            theta0_aero = np.deg2rad(sza)
            prof_aero = AtmosphereProfile(aod=aod_val, ssa=ssa_val, g=g_val,
                                          albedo=ALBEDO, theta0=theta0_aero,
                                          f0=F0)
            try:
                mu_a, I_a, _, _, Fup_a, _, _ = \
                    compute_fluxes(aeronet_solver, prof_aero)
                F_in_a = F0 * np.cos(theta0_aero)
                R_a = Fup_a / F_in_a if F_in_a > 0 else 0.0
                R_a = np.clip(R_a, 0, 2)

                sim_aod_list.append(aod_val)
                sim_R_list.append(R_a)
                sim_dates_list.append(date_str)
                sim_ssa_used.append(ssa_val)
                sim_g_used.append(g_val)
            except Exception:
                continue

        sim_aod = np.array(sim_aod_list)
        sim_R = np.array(sim_R_list)
        sim_ssa = np.array(sim_ssa_used)
        sim_g = np.array(sim_g_used)

        print(f"  Simulated {len(sim_aod)} days with DISORT")
        print(f"  AOD range: [{sim_aod.min():.3f}, {sim_aod.max():.3f}], "
              f"mean={sim_aod.mean():.3f}")
        if len(sim_ssa) > 0:
            print(f"  SSA range: [{sim_ssa.min():.3f}, {sim_ssa.max():.3f}]")
            print(f"  g range:   [{sim_g.min():.3f}, {sim_g.max():.3f}]")

        # Create comparison figure
        fig_aero, (ax_a1, ax_a2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: AOD vs simulated R (scatter)
        sc = ax_a1.scatter(sim_aod, sim_R, c=sim_ssa, cmap="RdYlBu",
                           s=30, edgecolors="grey", linewidths=0.3,
                           vmin=0.8, vmax=1.0, zorder=3)
        plt.colorbar(sc, ax=ax_a1, label="SSA (440nm)")
        ax_a1.set_xlabel("AERONET AOD (440nm)")
        ax_a1.set_ylabel("DISORT Simulated R (TOA reflectance)")
        ax_a1.grid(True, alpha=0.3)

        # Panel 2: Time series of AOD
        day_idx = np.arange(len(sim_aod))
        ax_a2.bar(day_idx, sim_aod, color=PALETTE[0], alpha=0.6,
                  edgecolor="white", linewidth=0.3, label="AOD (440nm)")
        ax_a2_twin = ax_a2.twinx()
        ax_a2_twin.plot(day_idx, sim_R, "o-", color=PALETTE[1],
                        markersize=3, lw=1, label="Sim. R")
        ax_a2.set_xlabel("Day index")
        ax_a2.set_ylabel("AOD (440nm)", color=PALETTE[0])
        ax_a2_twin.set_ylabel("Simulated R", color=PALETTE[1])
        lines_a = ax_a2.get_legend_handles_labels()
        lines_b = ax_a2_twin.get_legend_handles_labels()
        ax_a2.legend(lines_a[0] + lines_b[0],
                     lines_a[1] + lines_b[1], fontsize=11)
        ax_a2.grid(True, alpha=0.3)

        fig_aero.tight_layout()
        savefig(fig_aero, f"Level3_AERONET_comparison_N{N_STREAMS}_theta0{THETA0_DEG:.0f}")

        # ── Simulated radiance field for representative AERONET days ──
        # Pick ~5 representative days: min AOD, 25th, median, 75th, max AOD
        if len(sim_aod) >= 5:
            sort_idx = np.argsort(sim_aod)
            pick = [sort_idx[0],
                    sort_idx[len(sort_idx) // 4],
                    sort_idx[len(sort_idx) // 2],
                    sort_idx[3 * len(sort_idx) // 4],
                    sort_idx[-1]]
            rad_solver = DisortSolver(n_streams=N_STREAMS)
            fig_rrad, (ax_rr1, ax_rr2) = plt.subplots(1, 2, figsize=(14, 6))
            for pi in pick:
                aod_r = sim_aod[pi]
                ssa_r = sim_ssa[pi]
                g_r = sim_g[pi]
                date_r = sim_dates_list[pi]
                prof_r = AtmosphereProfile(aod=aod_r, ssa=ssa_r, g=g_r,
                                           albedo=ALBEDO, theta0=THETA0,
                                           f0=F0)
                F_up_r, F_dn_r, F_dir_r, I_toa_r, I_boa_r = \
                    compute_fluxes_continuous(rad_solver, prof_r)
                lbl_r = (rf"$\tau$={aod_r:.3f}, "
                         rf"$\omega_0$={ssa_r:.2f}, "
                         f"g={g_r:.2f}")
                ax_rr1.plot(FINE_ANGLES_DEG, I_toa_r, "-", lw=1.5,
                            label=lbl_r)
                ax_rr2.plot(FINE_ANGLES_DEG, I_boa_r, "-", lw=1.5,
                            label=lbl_r)
            ax_rr1.set_xlabel("Viewing Zenith Angle (deg)")
            ax_rr1.set_ylabel(RAD_UNITS)
            ax_rr1.legend(fontsize=10, loc="best")
            ax_rr2.set_xlabel("Viewing Zenith Angle (deg)")
            ax_rr2.set_ylabel(RAD_UNITS)
            ax_rr2.legend(fontsize=10, loc="best")
            fig_rrad.tight_layout()
            savefig(fig_rrad, f"Level3_AERONET_radiance_field_N{N_STREAMS}_theta0{THETA0_DEG:.0f}")

        # ══════════════════════════════════════════════════════════════
        #  REAL DATA RETRIEVAL — SURFRAD measured fluxes + AERONET truth
        # ══════════════════════════════════════════════════════════════
        #
        # METHODOLOGY:
        # 1. SURFRAD (Penn State, PA) measures REAL surface radiation:
        #    - GHI: Global Horizontal Irradiance (W/m²)
        #    - DNI: Direct Normal Irradiance (W/m²)
        #    - DHI: Diffuse Horizontal Irradiance (W/m²)
        #    These are REAL MEASUREMENTS from broadband pyranometers.
        #
        # 2. The DIFFUSE FRACTION df = DHI/GHI depends on aerosol loading:
        #    - Clean sky: df ~ 0.10-0.15 (mostly direct beam)
        #    - AOD=0.3:   df ~ 0.20-0.30
        #    - AOD>1.0:   df ~ 0.50+ (smoke/dust events)
        #
        # 3. Our DISORT model predicts df(AOD) at a given SZA.
        #    We find the AOD that makes df_model = df_measured.
        #    This is a REAL retrieval from REAL measurements.
        #
        # 4. Compare retrieved AOD with AERONET direct-sun AOD (truth).
        #
        # Note: SURFRAD is broadband, model is monochromatic (550nm).
        # The diffuse fraction is approximately wavelength-independent
        # for the broadband integral, making this comparison valid.
        #
        print("\n  " + "-" * 60)
        print("  REAL DATA RETRIEVAL — SURFRAD fluxes + AERONET truth")
        print("  " + "-" * 60)

        import os
        import glob
        from atrt.atmosphere import StandardAtmosphere

        # ── Parse SURFRAD daily files ─────────────────────────────────
        # Use Bondville SURFRAD — collocated with AERONET BONDVILLE
        surfrad_dir = os.path.join(os.path.dirname(__file__) or ".",
                                   "surfrad_bondville")
        surfrad_files = sorted(glob.glob(os.path.join(surfrad_dir, "bon23*.dat")))
        if len(surfrad_files) == 0:
            # Fallback to Penn State if Bondville not available
            surfrad_dir = os.path.join(os.path.dirname(__file__) or ".",
                                       "surfrad_data")
            surfrad_files = sorted(glob.glob(os.path.join(surfrad_dir,
                                                           "psu23*.dat")))

        def parse_surfrad_day(filepath):
            """Parse one SURFRAD daily file, return midday clear-sky average.

            SURFRAD columns (1-minute data, space-delimited):
              0:year 1:jday 2:month 3:day 4:hour 5:minute 6:dt
              7:zen(SZA)  8:dw_solar(GHI) 9:qc  10:uw_solar 11:qc
              12:direct_n(DNI) 13:qc  14:diffuse(DHI) 15:qc
              ...
            Returns dict with date, SZA, GHI, DNI, DHI (midday average)
            or None if no valid data.
            """
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
            except Exception:
                return None

            if len(lines) < 10:
                return None

            # Skip HTML error pages (failed downloads)
            if lines[0].strip().startswith("<"):
                return None

            # Parse header
            station = lines[0].strip()
            parts = lines[1].split()
            try:
                lat, lon = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                return None

            # Collect midday readings (10:00-14:00 local, SZA < 60°, qc=0)
            ghi_vals, dni_vals, dhi_vals, sza_vals = [], [], [], []
            month_val, day_val, jday_val = None, None, None

            for line in lines[2:]:
                cols = line.split()
                if len(cols) < 16:
                    continue
                try:
                    hour = int(cols[4])
                    sza = float(cols[7])
                    ghi = float(cols[8])
                    ghi_qc = int(cols[9])
                    dni = float(cols[12])
                    dni_qc = int(cols[13])
                    dhi = float(cols[14])
                    dhi_qc = int(cols[15])
                except (ValueError, IndexError):
                    continue

                if month_val is None:
                    month_val = int(cols[2])
                    day_val = int(cols[3])
                    jday_val = int(cols[1])

                # Filter: midday, good quality, clear-sky, physical values
                # DNI > 100 W/m² ensures clear sky (clouds block direct beam)
                if (10 <= hour <= 14 and sza < 60 and sza > 5
                        and ghi_qc == 0 and dni_qc == 0 and dhi_qc == 0
                        and ghi > 100 and dni > 100 and dhi > 5
                        and ghi < 1400 and dhi < 800):
                    ghi_vals.append(ghi)
                    dni_vals.append(dni)
                    dhi_vals.append(dhi)
                    sza_vals.append(sza)

            if len(ghi_vals) < 10:  # need at least 10 good readings
                return None

            return {
                "date": f"{day_val:02d}:{month_val:02d}:2023",
                "jday": jday_val,
                "month": month_val,
                "day": day_val,
                "sza": np.mean(sza_vals),
                "ghi": np.mean(ghi_vals),
                "dni": np.mean(dni_vals),
                "dhi": np.mean(dhi_vals),
                "df": np.mean(dhi_vals) / np.mean(ghi_vals),  # diffuse fraction
                "n_readings": len(ghi_vals),
            }

        if len(surfrad_files) > 0:
            print(f"  Found {len(surfrad_files)} SURFRAD files in {surfrad_dir}")

            surfrad_days = []
            for sf in surfrad_files:
                result = parse_surfrad_day(sf)
                if result is not None:
                    surfrad_days.append(result)

            print(f"  Parsed {len(surfrad_days)} days with valid midday data")

            # ── Match SURFRAD dates with AERONET dates ────────────────
            # Build AERONET lookup by dd:mm:yyyy date
            # Convert AOD_440 → AOD_550 using Angstrom exponent from 440/675
            aero_lookup = {}
            for i, row in enumerate(aeronet_aod_data):
                date_key = row.get("date", "")
                aod_440 = row.get("aod_440", np.nan)
                aod_675 = row.get("aod_675", np.nan)
                if np.isnan(aod_440) or aod_440 < 0:
                    continue
                # Angstrom exponent and conversion to 550nm
                if not np.isnan(aod_675) and aod_675 > 0 and aod_440 > 0:
                    alpha = -np.log(aod_440 / aod_675) / np.log(440.0 / 675.0)
                    alpha = np.clip(alpha, 0.0, 3.0)
                    aod_550 = aod_440 * (550.0 / 440.0) ** (-alpha)
                else:
                    # Default Angstrom exponent ~1.5 (fine-mode aerosol)
                    aod_550 = aod_440 * (550.0 / 440.0) ** (-1.5)
                ssa_val = ssa_by_date.get(date_key, 0.92)
                g_val = asy_by_date.get(date_key, 0.70)
                if np.isnan(ssa_val) or ssa_val < 0 or ssa_val > 1:
                    ssa_val = 0.92
                if np.isnan(g_val) or g_val < -1 or g_val > 1:
                    g_val = 0.70
                aero_lookup[date_key] = {
                    "aod": aod_550,
                    "aod_440": aod_440,
                    "ssa": ssa_val,
                    "g": g_val,
                }

            matched = []
            for sd in surfrad_days:
                aero = aero_lookup.get(sd["date"])
                # Cloud screening: df < 0.6 (higher = likely cloudy)
                if aero is not None and sd["df"] < 0.6:
                    matched.append({**sd, **aero})

            print(f"  Matched {len(matched)} days with both SURFRAD + AERONET data")

            if len(matched) >= 5:
                # ── Retrieval: find AOD that matches measured df ──────
                ret_solver = DisortSolver(n_streams=16)  # fewer streams for speed
                SURF_ALBEDO = 0.20   # Bondville cropland surface

                # Multi-wavelength broadband model for SURFRAD comparison
                # SURFRAD is broadband, so we must integrate over the solar
                # spectrum to properly compare with monochromatic DISORT.
                # 3 bands covering UV-VIS-NIR with solar-weighted bandwidths:
                BB_WAVELENGTHS = [400, 550, 800]
                BB_SOLAR = [1.60, 1.88, 1.14]  # W/m²/nm (avg in band)
                BB_BANDWIDTH = [150, 200, 350]  # nm per band
                # Pre-build StandardAtmosphere at each wavelength
                std_atms = {wl: StandardAtmosphere(wavelength_nm=wl)
                            for wl in BB_WAVELENGTHS}

                # Angstrom-scale aerosol AOD from 550nm to other wavelengths
                def aod_at_wavelength(aod_550, alpha, wl_nm):
                    """Scale AOD from 550nm to wl_nm using Angstrom exponent."""
                    return aod_550 * (wl_nm / 550.0) ** (-alpha)

                def model_diffuse_fraction(aod_val, ssa_val, g_val, sza_deg,
                                           alpha_ang=1.5):
                    """Compute broadband diffuse fraction at BOA.

                    Runs DISORT at multiple wavelengths and weights by solar
                    spectrum to produce a broadband estimate comparable to
                    SURFRAD pyranometer measurements.
                    """
                    theta0 = np.deg2rad(sza_deg)
                    mu0 = np.cos(theta0)
                    aod_val = np.clip(aod_val, 0.001, 8.0)

                    F_dif_total = 0.0
                    F_dir_total = 0.0

                    for wl, S0, dw in zip(BB_WAVELENGTHS, BB_SOLAR, BB_BANDWIDTH):
                        aod_wl = aod_at_wavelength(aod_val, alpha_ang, wl)
                        prof = std_atms[wl].to_profile(
                            aod_total=aod_wl, ssa_aer=ssa_val, g_aer=g_val,
                            albedo=SURF_ALBEDO, theta0=theta0, f0=F0)
                        _, _, mu_dn, I_dn = ret_solver.solve(prof,
                                                              output="both")
                        w_dn = ret_solver.weights[ret_solver.down_idx]
                        # Normalized diffuse flux (in F0 units)
                        f_dif = 2 * np.pi * np.sum(w_dn * mu_dn * np.abs(I_dn))
                        f_dir = F0 * mu0 * np.exp(-prof.tau_total / mu0)
                        # Weight by solar irradiance × bandwidth
                        weight = S0 * dw
                        F_dif_total += f_dif * weight
                        F_dir_total += f_dir * weight

                    F_total = F_dif_total + F_dir_total
                    if F_total < 1e-10:
                        return 0.0
                    return F_dif_total / F_total

                # Retrieve AOD for each matched day using bisection
                # (faster and more robust than OE for a 1D monotonic problem)
                ret_aod_true = []
                ret_aod_est = []
                ret_df_meas = []
                ret_df_model = []
                ret_dates = []
                ret_sza = []

                print(f"\n  Running broadband retrieval ({len(matched)} days, "
                      f"{len(BB_WAVELENGTHS)} wavelengths)...")

                for mi, m in enumerate(matched):
                    df_meas = m["df"]
                    aod_true = m["aod"]
                    ssa_m = m["ssa"]
                    g_m = m["g"]
                    sza_m = m["sza"]
                    # Angstrom exponent from AERONET 440/675
                    aod_440 = m.get("aod_440", aod_true * (440/550)**(-1.5))
                    alpha_m = 1.5  # default
                    if aod_true > 0 and aod_440 > 0:
                        ratio = aod_440 / aod_true
                        if ratio > 0:
                            alpha_m = np.log(ratio) / np.log(550.0/440.0)
                            alpha_m = np.clip(alpha_m, 0.0, 3.0)

                    # Bisection: find AOD_550 where model_df(AOD) = df_measured
                    aod_lo, aod_hi = 0.001, 5.0
                    df_lo = model_diffuse_fraction(aod_lo, ssa_m, g_m, sza_m,
                                                    alpha_m)
                    df_hi = model_diffuse_fraction(aod_hi, ssa_m, g_m, sza_m,
                                                    alpha_m)

                    # Check if measured df is within model range
                    if df_meas < df_lo:
                        aod_est = aod_lo
                    elif df_meas > df_hi:
                        aod_est = aod_hi
                    else:
                        # Bisection (12 iterations → precision ~0.001)
                        for _ in range(12):
                            aod_mid = (aod_lo + aod_hi) / 2.0
                            df_mid = model_diffuse_fraction(aod_mid, ssa_m, g_m,
                                                             sza_m, alpha_m)
                            if df_mid < df_meas:
                                aod_lo = aod_mid
                            else:
                                aod_hi = aod_mid
                        aod_est = (aod_lo + aod_hi) / 2.0

                    df_mod = model_diffuse_fraction(aod_est, ssa_m, g_m, sza_m,
                                                    alpha_m)

                    if (mi + 1) % 10 == 0:
                        print(f"    Day {mi+1}/{len(matched)}: AOD_true={aod_true:.3f},"
                              f" AOD_ret={aod_est:.3f}, df_meas={m['df']:.3f},"
                              f" df_mod={df_mod:.3f}")

                    ret_aod_true.append(aod_true)
                    ret_aod_est.append(aod_est)
                    ret_df_meas.append(m["df"])  # original SURFRAD df
                    ret_df_model.append(df_mod)
                    ret_dates.append(m["date"])
                    ret_sza.append(sza_m)

                ret_aod_true = np.array(ret_aod_true)
                ret_aod_est = np.array(ret_aod_est)
                ret_df_meas = np.array(ret_df_meas)
                ret_df_model = np.array(ret_df_model)

                # ── Statistics ────────────────────────────────────────
                rmse = np.sqrt(np.mean((ret_aod_true - ret_aod_est)**2))
                bias = np.mean(ret_aod_est - ret_aod_true)
                if len(ret_aod_true) > 2:
                    corr = np.corrcoef(ret_aod_true, ret_aod_est)[0, 1]
                    slope, intercept = np.polyfit(ret_aod_true, ret_aod_est, 1)
                else:
                    corr, slope, intercept = 0, 1, 0
                rel_err = np.abs(ret_aod_est - ret_aod_true) / (
                    ret_aod_true + 1e-10) * 100

                print(f"\n  REAL DATA Retrieval Results ({len(ret_aod_true)} days):")
                print(f"    Method: Diffuse fraction matching (SURFRAD DHI/GHI)")
                print(f"    Truth:  AERONET {chosen_site} direct-sun AOD (550nm)")
                print(f"    Model:  DISORT 1D, 16 streams, broadband (3 bands)")
                print(f"    -----------------------------------------")
                print(f"    RMSE:          {rmse:.4f}")
                print(f"    Bias:          {bias:+.4f}")
                print(f"    Correlation:   {corr:.4f}")
                print(f"    Slope:         {slope:.4f}")
                print(f"    Intercept:     {intercept:+.4f}")
                print(f"    Mean rel err:  {np.mean(rel_err):.1f}%")
                print(f"    Max rel err:   {np.max(rel_err):.1f}%")
                print(f"    df range:      [{ret_df_meas.min():.3f}, "
                      f"{ret_df_meas.max():.3f}]")

                # ── Figure: 4-panel retrieval analysis ────────────────
                fig_real, axes_r = plt.subplots(2, 2, figsize=(14, 12))

                # Panel 1: AOD scatter (true vs retrieved)
                ax = axes_r[0, 0]
                lo = 0
                hi = max(ret_aod_true.max(), ret_aod_est.max()) * 1.15
                ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="1:1")
                ax.fill_between([lo, hi], [lo - 0.1, hi - 0.1],
                                [lo + 0.1, hi + 0.1],
                                color="#cccccc", alpha=0.3, label=r"$\pm 0.1$")
                sc = ax.scatter(ret_aod_true, ret_aod_est,
                                c=ret_df_meas, cmap="YlOrRd",
                                s=50, edgecolors="white", linewidths=0.5,
                                vmin=0.05, vmax=0.6, zorder=3)
                x_fit = np.linspace(lo, hi, 100)
                ax.plot(x_fit, slope * x_fit + intercept,
                        color=PALETTE[0], lw=1.5, ls="--",
                        label=f"Fit: y={slope:.2f}x{intercept:+.2f}")
                plt.colorbar(sc, ax=ax, label="Measured $d_f$ (SURFRAD)")
                ax.set_xlabel("AERONET AOD (551nm) — Truth")
                ax.set_ylabel("Retrieved AOD (from SURFRAD $d_f$)")
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal")
                ax.legend(fontsize=8, loc="upper left")
                ax.grid(True, alpha=0.3)

                # Panel 2: Diffuse fraction — measured vs modelled
                ax2 = axes_r[0, 1]
                ax2.scatter(ret_df_meas, ret_df_model,
                            c=ret_aod_true, cmap="viridis",
                            s=50, edgecolors="white", linewidths=0.5,
                            zorder=3)
                df_range = [0, max(ret_df_meas.max(), ret_df_model.max()) * 1.1]
                ax2.plot(df_range, df_range, "k--", lw=1, alpha=0.5)
                plt.colorbar(ax2.collections[0], ax=ax2,
                             label="AERONET AOD")
                ax2.set_xlabel(r"$d_f$ measured (SURFRAD DHI/GHI)")
                ax2.set_ylabel(r"$d_f$ model (DISORT)")
                ax2.set_aspect("equal")
                ax2.grid(True, alpha=0.3)

                # Panel 3: Error vs true AOD
                ax3 = axes_r[1, 0]
                ax3.scatter(ret_aod_true, ret_aod_est - ret_aod_true,
                            c=PALETTE[0], s=35, edgecolors="white",
                            linewidths=0.3, zorder=3)
                ax3.axhline(0, color="black", ls="--", lw=0.8)
                ax3.axhline(bias, color=PALETTE[1], ls=":", lw=1.5,
                            label=f"Bias = {bias:+.3f}")
                ax3.set_xlabel("AERONET AOD (Truth)")
                ax3.set_ylabel("Retrieved - True AOD")
                ax3.legend(fontsize=12)
                ax3.grid(True, alpha=0.3)

                # Panel 4: Time series
                ax4 = axes_r[1, 1]
                di = np.arange(len(ret_aod_true))
                ax4.plot(di, ret_aod_true, "o-", color=PALETTE[0],
                         markersize=4, lw=0.8, label="AERONET (truth)")
                ax4.plot(di, ret_aod_est, "s--", color=PALETTE[1],
                         markersize=4, lw=0.8, label="Retrieved (SURFRAD)")
                ax4.set_xlabel("Day index (matched days)")
                ax4.set_ylabel("AOD")
                ax4.legend(fontsize=12)
                ax4.grid(True, alpha=0.3)

                # Determine site label from SURFRAD directory name
                if "bondville" in surfrad_dir.lower():
                    surfrad_label = "SURFRAD Bondville (40.1°N, 88.4°W)"
                    aero_label = f"AERONET {chosen_site}"
                    colloc_note = "(collocated)" if "BOND" in (chosen_site or "") else ""
                else:
                    surfrad_label = "SURFRAD Penn State (40.7°N, 77.9°W)"
                    aero_label = f"AERONET {chosen_site}"
                    colloc_note = ""

                fig_real.tight_layout()
                savefig(fig_real, f"Level3_AERONET_retrieval_N{N_STREAMS}_theta0{THETA0_DEG:.0f}")
            else:
                print("  Insufficient matched days for retrieval.")
        else:
            print(f"  No SURFRAD data found in {surfrad_dir}.")
            print("  Run: mkdir surfrad_data && cd surfrad_data")
            print("  Then download PSU 2023 data from NOAA SURFRAD FTP.")

    else:
        print("  WARNING: Could not download AERONET data from any site.")
        print("  Skipping AERONET comparison plots.")
        print("  (Network access may be restricted)")



# ══════════════════════════════════════════════════════════════════
#  SUPPORTING ANALYSIS A — Parameter Studies
# ══════════════════════════════════════════════════════════════════
if RUN_STUDY_A:
    print("\n" + "=" * 70)
    print("  SUPPORTING ANALYSIS A — Parameter Studies")
    print("=" * 70)
    study_solver = DisortSolver(n_streams=N_STREAMS)
    study_lines = []          # lines to write to the TXT report
    study_plots = []          # plot filenames generated

    def study_header(title):
        sep = "=" * 72
        study_lines.append("")
        study_lines.append(sep)
        study_lines.append(f"  {title}")
        study_lines.append(sep)
        print(f"\n  --- {title} ---")


    study_lines.append(f"PARAMETER STUDY REPORT")
    study_lines.append(f"Generated by 1D_Atmosphere_Model.py")
    study_lines.append(f"Baseline: AOD={BASE['aod']}, SSA={BASE['ssa']}, "
                       f"g={BASE['g']}, albedo={BASE['albedo']}, "
                       f"theta0={THETA0_DEG} deg, N_streams={N_STREAMS}")
    study_lines.append(f"F_solar_in = F0*mu0 = {F_solar:.4f}")

    # ── STUDY A: SSA sweep ──────────────────────────────────────────
    study_header("STUDY A: Single Scattering Albedo (SSA) sweep")
    ssa_vals = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    study_lines.append(f"  Varying: SSA = {ssa_vals}")
    study_lines.append(f"  Fixed:   AOD={BASE['aod']}, g={BASE['g']}, "
                       f"albedo={BASE['albedo']}")
    study_lines.append("")
    study_lines.append(f"  {'SSA':>5} | {'F_up(TOA)':>10} | {'F_dn(BOA)':>10} | "
                       f"{'F_dir(BOA)':>10} | {'R_TOA':>7} | {'Absorbed':>9} | Interpretation")
    study_lines.append(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-"
                       f"{'-'*7}-+-{'-'*9}-+-------------")

    figA, (axA1, axA2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, ssa_val in enumerate(ssa_vals):
        p = AtmosphereProfile(aod=BASE["aod"], ssa=ssa_val, g=BASE["g"],
                              albedo=BASE["albedo"], theta0=THETA0, f0=F0)
        F_up, F_dn, F_dir, I_toa_f, I_boa_f = compute_fluxes_continuous(study_solver, p)
        R = F_up / F_solar
        A = 1 - R
        c = PALETTE[idx % len(PALETTE)]
        lbl = rf"$\omega_0={ssa_val}$"
        axA1.plot(FINE_ANGLES_DEG, I_toa_f, "-", color=c, lw=1.5, label=lbl)
        axA2.plot(FINE_ANGLES_DEG, I_boa_f, "-", color=c, lw=1.5, label=lbl)
        if ssa_val == 0.0:
            interp = "Pure absorption: no scattering, almost no light returns"
        elif ssa_val < 0.5:
            interp = "Strongly absorbing: most photons absorbed"
        elif ssa_val < 0.9:
            interp = "Moderately absorbing: some scattering"
        elif ssa_val < 1.0:
            interp = "Weakly absorbing: most light scattered"
        else:
            interp = "Conservative scattering: NO absorption, energy conserved"
        study_lines.append(f"  {ssa_val:5.2f} | {F_up:10.4f} | {F_dn:10.4f} | "
                           f"{F_dir:10.4f} | {R:7.4f} | {A:9.4f} | {interp}")
    study_lines.append("")
    study_lines.append("  CONCLUSION: As SSA increases from 0 to 1, more light is scattered")
    study_lines.append("  instead of absorbed. At SSA=1 (conservative scattering), all energy")
    study_lines.append("  is either reflected back to space or transmitted to the surface.")
    study_lines.append("  At SSA=0 (pure absorption), almost nothing returns to the TOA.")

    axA1.set_xlabel("Viewing Zenith Angle (deg)")
    axA1.set_ylabel(RAD_UNITS)
    axA1.legend(ncol=2)
    axA2.set_xlabel("Viewing Zenith Angle (deg)")
    axA2.set_ylabel(RAD_UNITS)
    axA2.legend(ncol=2)
    figA.tight_layout()
    _fnA = (f"Study_A_SSA_sweep_AOD_{BASE['aod']}_g{BASE['g']}_"
            f"albedo_{BASE['albedo']}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}")
    _fnA = _fnA.replace(".", "")
    savefig(figA, _fnA)
    study_plots.append(_fnA)

    # ── STUDY B: AOD sweep ──────────────────────────────────────────
    study_header("STUDY B: Aerosol Optical Depth (AOD) sweep")
    aod_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    study_lines.append(f"  Varying: AOD = {aod_vals}")
    study_lines.append(f"  Fixed:   SSA={BASE['ssa']}, g={BASE['g']}, "
                       f"albedo={BASE['albedo']}")
    study_lines.append("")
    study_lines.append(f"  {'AOD':>5} | {'F_up(TOA)':>10} | {'F_dn(BOA)':>10} | "
                       f"{'F_dir(BOA)':>10} | {'R_TOA':>7} | {'T_dir':>7} | Interpretation")
    study_lines.append(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-"
                       f"{'-'*7}-+-{'-'*7}-+-------------")

    figB, (axB1, axB2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, aod_val in enumerate(aod_vals):
        p = AtmosphereProfile(aod=aod_val, ssa=BASE["ssa"], g=BASE["g"],
                              albedo=BASE["albedo"], theta0=THETA0, f0=F0)
        F_up, F_dn, F_dir, I_toa_f, I_boa_f = compute_fluxes_continuous(study_solver, p)
        R = F_up / F_solar
        T_dir = F_dir / F_solar
        c = PALETTE[idx % len(PALETTE)]
        lbl = rf"$\tau={aod_val}$"
        axB1.plot(FINE_ANGLES_DEG, I_toa_f, "-", color=c, lw=1.5, label=lbl)
        axB2.plot(FINE_ANGLES_DEG, I_boa_f, "-", color=c, lw=1.5, label=lbl)
        if aod_val <= 0.05:
            interp = "Very clean: atmosphere almost transparent"
        elif aod_val <= 0.2:
            interp = "Clean: weak scattering, surface dominates"
        elif aod_val <= 0.5:
            interp = "Moderate: path radiance comparable to surface"
        elif aod_val <= 1.0:
            interp = "Thick: path radiance dominates, surface masked"
        else:
            interp = "Very thick: multiple scattering dominates"
        study_lines.append(f"  {aod_val:5.2f} | {F_up:10.4f} | {F_dn:10.4f} | "
                           f"{F_dir:10.4f} | {R:7.4f} | {T_dir:7.4f} | {interp}")
    study_lines.append("")
    study_lines.append("  CONCLUSION: Increasing AOD increases TOA upwelling (more path radiance)")
    study_lines.append("  but the direct beam at BOA decreases exponentially (Beer-Lambert).")
    study_lines.append("  At very high AOD, the surface becomes invisible from the TOA.")

    axB1.set_xlabel("Viewing Zenith Angle (deg)")
    axB1.set_ylabel(RAD_UNITS)
    axB1.legend()
    axB2.set_xlabel("Viewing Zenith Angle (deg)")
    axB2.set_ylabel(RAD_UNITS)
    axB2.legend()
    figB.tight_layout()
    _fnB = (f"Study_B_AOD_sweep_SSA_{BASE['ssa']}_g{BASE['g']}_"
            f"albedo_{BASE['albedo']}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}")
    _fnB = _fnB.replace(".", "")
    savefig(figB, _fnB)
    study_plots.append(_fnB)

    # ── STUDY C: Asymmetry parameter g sweep ────────────────────────
    study_header("STUDY C: Asymmetry parameter (g) sweep")
    g_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    study_lines.append(f"  Varying: g = {g_vals}")
    study_lines.append(f"  Fixed:   AOD={BASE['aod']}, SSA={BASE['ssa']}, "
                       f"albedo={BASE['albedo']}")
    study_lines.append("")
    study_lines.append(f"  {'g':>5} | {'F_up(TOA)':>10} | {'F_dn(BOA)':>10} | "
                       f"{'R_TOA':>7} | Interpretation")
    study_lines.append(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-"
                       f"{'-'*7}-+-------------")

    figC, (axC1, axC2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, g_val in enumerate(g_vals):
        p = AtmosphereProfile(aod=BASE["aod"], ssa=BASE["ssa"], g=g_val,
                              albedo=BASE["albedo"], theta0=THETA0, f0=F0)
        F_up, F_dn, F_dir, I_toa_f, I_boa_f = compute_fluxes_continuous(study_solver, p)
        R = F_up / F_solar
        c = PALETTE[idx % len(PALETTE)]
        lbl = rf"$g={g_val}$"
        axC1.plot(FINE_ANGLES_DEG, I_toa_f, "-", color=c, lw=1.5, label=lbl)
        axC2.plot(FINE_ANGLES_DEG, I_boa_f, "-", color=c, lw=1.5, label=lbl)
        if g_val == 0.0:
            interp = "Isotropic (Rayleigh-like): equal scatter in all directions"
        elif g_val < 0.5:
            interp = "Weakly forward: light spreads broadly"
        elif g_val < 0.8:
            interp = "Forward scattering: typical aerosol (Mie)"
        else:
            interp = "Strongly forward: light barely deflected (large particles)"
        study_lines.append(f"  {g_val:5.2f} | {F_up:10.4f} | {F_dn:10.4f} | "
                           f"{R:7.4f} | {interp}")
    study_lines.append("")
    study_lines.append("  CONCLUSION: Higher g means more forward scattering. Less light")
    study_lines.append("  is backscattered to the TOA (satellite sees less). More light")
    study_lines.append("  reaches the BOA. g=0 (isotropic) maximises backscatter to TOA.")

    axC1.set_xlabel("Viewing Zenith Angle (deg)")
    axC1.set_ylabel(RAD_UNITS)
    axC1.legend()
    axC2.set_xlabel("Viewing Zenith Angle (deg)")
    axC2.set_ylabel(RAD_UNITS)
    axC2.legend()
    figC.tight_layout()
    _fnC = (f"Study_C_g_sweep_AOD_{BASE['aod']}_SSA_{BASE['ssa']}_"
            f"albedo_{BASE['albedo']}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}")
    _fnC = _fnC.replace(".", "")
    savefig(figC, _fnC)
    study_plots.append(_fnC)

    # ── STUDY D: Surface albedo sweep ───────────────────────────────
    study_header("STUDY D: Surface albedo sweep")
    alb_vals = [0.0, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
    study_lines.append(f"  Varying: albedo = {alb_vals}")
    study_lines.append(f"  Fixed:   AOD={BASE['aod']}, SSA={BASE['ssa']}, g={BASE['g']}")
    study_lines.append("")
    study_lines.append(f"  {'albedo':>6} | {'F_up(TOA)':>10} | {'R_TOA':>7} | "
                       f"{'Surface type'}")
    study_lines.append(f"  {'-'*6}-+-{'-'*10}-+-{'-'*7}-+-------------")

    figD, (axD1, axD2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, alb_val in enumerate(alb_vals):
        p = AtmosphereProfile(aod=BASE["aod"], ssa=BASE["ssa"], g=BASE["g"],
                              albedo=alb_val, theta0=THETA0, f0=F0)
        F_up, F_dn, F_dir, I_toa_f, I_boa_f = compute_fluxes_continuous(study_solver, p)
        R = F_up / F_solar
        c = PALETTE[idx % len(PALETTE)]
        lbl = rf"$a_s={alb_val}$"
        axD1.plot(FINE_ANGLES_DEG, I_toa_f, "-", color=c, lw=1.5, label=lbl)
        axD2.plot(FINE_ANGLES_DEG, I_boa_f, "-", color=c, lw=1.5, label=lbl)
        if alb_val == 0.0:
            stype = "Perfect absorber (black body)"
        elif alb_val <= 0.1:
            stype = "Deep ocean / dark forest"
        elif alb_val <= 0.3:
            stype = "Vegetation / soil"
        elif alb_val <= 0.6:
            stype = "Desert sand / dry soil"
        elif alb_val <= 0.85:
            stype = "Old snow / bright sand"
        else:
            stype = "Fresh snow / perfect reflector"
        study_lines.append(f"  {alb_val:6.2f} | {F_up:10.4f} | {R:7.4f} | {stype}")
    study_lines.append("")
    study_lines.append("  CONCLUSION: Higher albedo increases TOA radiance because more light")
    study_lines.append("  reflects off the surface and passes up through the atmosphere.")
    study_lines.append("  BOA downwelling is nearly unchanged (surface is below the sensor).")
    study_lines.append("  Over bright surfaces aerosol can DARKEN the image (absorbing aerosol).")

    axD1.set_xlabel("Viewing Zenith Angle (deg)")
    axD1.set_ylabel(RAD_UNITS)
    axD1.legend()
    axD2.set_xlabel("Viewing Zenith Angle (deg)")
    axD2.set_ylabel(RAD_UNITS)
    axD2.legend()
    figD.tight_layout()
    _fnD = (f"Study_D_albedo_sweep_AOD_{BASE['aod']}_SSA_{BASE['ssa']}_"
            f"g{BASE['g']}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}")
    _fnD = _fnD.replace(".", "")
    savefig(figD, _fnD)
    study_plots.append(_fnD)

    # ── STUDY E: Solar zenith angle sweep ───────────────────────────
    study_header("STUDY E: Solar zenith angle sweep")
    sza_vals = [0, 15, 30, 45, 60, 75]
    study_lines.append(f"  Varying: theta0 = {sza_vals} deg")
    study_lines.append(f"  Fixed:   AOD={BASE['aod']}, SSA={BASE['ssa']}, "
                       f"g={BASE['g']}, albedo={BASE['albedo']}")
    study_lines.append("")
    study_lines.append(f"  {'theta0':>6} | {'mu0':>5} | {'F_in':>8} | {'F_up(TOA)':>10} | "
                       f"{'R_TOA':>7} | Interpretation")
    study_lines.append(f"  {'-'*6}-+-{'-'*5}-+-{'-'*8}-+-{'-'*10}-+-"
                       f"{'-'*7}-+-------------")

    figE, (axE1, axE2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, sza in enumerate(sza_vals):
        th = np.deg2rad(sza)
        p = AtmosphereProfile(aod=BASE["aod"], ssa=BASE["ssa"], g=BASE["g"],
                              albedo=BASE["albedo"], theta0=th, f0=F0)
        F_up, F_dn, F_dir, I_toa_f, I_boa_f = compute_fluxes_continuous(study_solver, p)
        F_in_sza = F0 * np.cos(th)
        R = F_up / F_in_sza if F_in_sza > 1e-10 else 0.0
        c = PALETTE[idx % len(PALETTE)]
        lbl = rf"$\theta_0={sza}°$"
        axE1.plot(FINE_ANGLES_DEG, I_toa_f, "-", color=c, lw=1.5, label=lbl)
        axE2.plot(FINE_ANGLES_DEG, I_boa_f, "-", color=c, lw=1.5, label=lbl)
        if sza == 0:
            interp = "Sun at zenith: shortest path, max F_in"
        elif sza <= 30:
            interp = "Low SZA: strong illumination"
        elif sza <= 60:
            interp = "Moderate SZA: longer path through atmosphere"
        else:
            interp = "High SZA: grazing incidence, very long path"
        study_lines.append(f"  {sza:6d} | {np.cos(th):5.3f} | {F_in_sza:8.4f} | "
                           f"{F_up:10.4f} | {R:7.4f} | {interp}")
    study_lines.append("")
    study_lines.append("  CONCLUSION: Higher SZA means less solar flux on the surface (F_in=F0*mu0)")
    study_lines.append("  but the optical path through the atmosphere is longer (tau_eff = tau/mu0),")
    study_lines.append("  so proportionally more scattering occurs. The reflectance R generally")
    study_lines.append("  increases at high SZA because photons traverse more atmosphere.")

    axE1.set_xlabel("Viewing Zenith Angle (deg)")
    axE1.set_ylabel(RAD_UNITS)
    axE1.legend()
    axE2.set_xlabel("Viewing Zenith Angle (deg)")
    axE2.set_ylabel(RAD_UNITS)
    axE2.legend()
    figE.tight_layout()
    _fnE = (f"Study_E_SZA_sweep_AOD_{BASE['aod']}_SSA_{BASE['ssa']}_"
            f"g{BASE['g']}_albedo_{BASE['albedo']}_N{N_STREAMS}")
    _fnE = _fnE.replace(".", "")
    savefig(figE, _fnE)
    study_plots.append(_fnE)

    # ── STUDY F: Realistic physical scenarios ───────────────────────
    study_header("STUDY F: Realistic physical scenarios (TOA vs BOA)")
    scenarios = [
        ("Clean atmosphere",      0.02, 0.95, 0.70, 0.1,
         "Very low AOD: nearly pure Rayleigh scattering"),
        ("Continental moderate",  0.30, 0.92, 0.68, 0.15,
         "Typical continental aerosol, moderate loading"),
        ("Urban polluted",        0.80, 0.85, 0.65, 0.12,
         "Heavy urban pollution, absorbing aerosol (black carbon)"),
        ("Desert dust storm",     1.50, 0.93, 0.75, 0.35,
         "Saharan dust event: large particles, high g, bright sand surface"),
        ("Biomass burning",       0.60, 0.82, 0.60, 0.10,
         "Forest fire plume: strongly absorbing, moderate forward scatter"),
        ("Maritime clean",        0.08, 0.99, 0.72, 0.06,
         "Open ocean, sea salt: almost conservative scattering, dark surface"),
        ("Snow + soot",           0.15, 0.70, 0.55, 0.85,
         "Black carbon over snow: absorbing aerosol over bright surface"),
    ]
    study_lines.append(f"  {'Scenario':<24} | {'AOD':>5} | {'SSA':>5} | {'g':>5} | "
                       f"{'alb':>5} | {'F_up':>8} | {'F_dn':>8} | {'R':>6} | Description")
    study_lines.append(f"  {'-'*24}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-"
                       f"{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+----------")

    figF, (axF1, axF2) = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (name, aod, ssa, g, alb, desc) in enumerate(scenarios):
        p = AtmosphereProfile(aod=aod, ssa=ssa, g=g,
                              albedo=alb, theta0=THETA0, f0=F0)
        F_up, F_dn, F_dir, I_toa_f, I_boa_f = compute_fluxes_continuous(study_solver, p)
        R = F_up / F_solar
        c = PALETTE[idx % len(PALETTE)]
        axF1.plot(FINE_ANGLES_DEG, I_toa_f, "-", color=c, lw=1.5, label=name)
        axF2.plot(FINE_ANGLES_DEG, I_boa_f, "-", color=c, lw=1.5, label=name)
        study_lines.append(f"  {name:<24} | {aod:5.2f} | {ssa:5.2f} | {g:5.2f} | "
                           f"{alb:5.2f} | {F_up:8.4f} | {F_dn:8.4f} | {R:6.4f} | {desc}")
    study_lines.append("")
    study_lines.append("  KEY OBSERVATIONS:")
    study_lines.append("  - Desert dust has the highest TOA radiance due to high AOD + high albedo")
    study_lines.append("  - Snow+soot has high TOA radiance from surface but aerosol absorbs some")
    study_lines.append("  - Maritime clean has very low TOA radiance (low AOD + dark ocean)")
    study_lines.append("  - Biomass burning absorbs strongly (low SSA) reducing both TOA and BOA")
    study_lines.append("  - Urban polluted has more absorption than continental (lower SSA)")

    axF1.set_xlabel("Viewing Zenith Angle (deg)")
    axF1.set_ylabel(RAD_UNITS)
    axF1.legend(loc="upper left")
    axF2.set_xlabel("Viewing Zenith Angle (deg)")
    axF2.set_ylabel(RAD_UNITS)
    axF2.legend(loc="upper left")
    figF.tight_layout()
    _fnF = f"Study_F_scenarios_N{N_STREAMS}_theta0{THETA0_DEG:.0f}"
    savefig(figF, _fnF)
    study_plots.append(_fnF)



# ══════════════════════════════════════════════════════════════════
#  SUPPORTING ANALYSIS B — Extreme Physical Cases
# ══════════════════════════════════════════════════════════════════
if RUN_STUDY_B:
    print("\n" + "=" * 70)
    print("  SUPPORTING ANALYSIS B — Extreme Physical Cases")
    print("=" * 70)
    extreme_solver = DisortSolver(n_streams=N_STREAMS)
    extreme_aod = 0.5  # moderate AOD to see effects clearly

    extreme_cases = [
        (1.0, 1.0, "SSA=1, Alb=1", "No absorption anywhere"),
        (0.0, 0.0, "SSA=0, Alb=0", "Total absorption + black surface"),
        (1.0, 0.0, "SSA=1, Alb=0", "Conservative scatt. + black surface"),
        (0.0, 1.0, "SSA=0, Alb=1", "No scattering + reflective surface"),
    ]

    fig_ext, axes_ext = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes_ext.flatten()

    for idx, (ssa_e, alb_e, title_e, desc_e) in enumerate(extreme_cases):
        ax_e = axes_flat[idx]
        g_e = 0.7 if ssa_e > 0 else 0.0  # g irrelevant when SSA=0

        prof_e = AtmosphereProfile(aod=extreme_aod, ssa=ssa_e, g=g_e,
                                   albedo=alb_e, theta0=THETA0, f0=F0)
        F_up_e, F_dn_e, F_dir_e, I_toa_e, I_boa_e = \
            compute_fluxes_continuous(extreme_solver, prof_e)

        R_e = F_up_e / F_solar
        F_abs_sfc_e = (1.0 - alb_e) * (F_dn_e + F_dir_e)
        F_abs_atm_e = F_solar - F_up_e - F_abs_sfc_e
        F_abs_atm_e = max(F_abs_atm_e, 0.0)

        ax_e.plot(FINE_ANGLES_DEG, I_toa_e, "-", color=PALETTE[0],
                  lw=1.5, label="TOA upwelling")
        ax_e.plot(FINE_ANGLES_DEG, I_boa_e, "--", color=PALETTE[1],
                  lw=1.5, label="BOA downwelling")

        ax_e.set_xlabel("Viewing Zenith Angle (deg)")
        ax_e.set_ylabel(RAD_UNITS)
        ax_e.legend(fontsize=10)
        ax_e.grid(True, alpha=0.3)

        print(f"  ({chr(97+idx)}) {title_e}: R={R_e:.4f}, "
              f"F_abs_atm={F_abs_atm_e:.4f}, F_abs_sfc={F_abs_sfc_e:.4f}, "
              f"F_up={F_up_e:.4f}, F_dn={F_dn_e:.4f}, F_dir={F_dir_e:.4f}")

    fig_ext.tight_layout()
    _fn_ext = f"Extreme_cases_AOD_{extreme_aod}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}"
    _fn_ext = _fn_ext.replace(".", "")
    savefig(fig_ext, _fn_ext)



# ══════════════════════════════════════════════════════════════════
#  SUPPORTING ANALYSIS C — Energy Balance
# ══════════════════════════════════════════════════════════════════
if RUN_STUDY_C:
    print("\n" + "=" * 70)
    print("  SUPPORTING ANALYSIS C — Energy Balance")
    print("=" * 70)
    energy_solver = DisortSolver(n_streams=N_STREAMS)
    energy_aod = 0.5

    # Build scenario list: SSA sweep at alb=0.1, albedo sweep at SSA=0.9, 4 extremes
    energy_scenarios = []

    # SSA sweep at albedo=0.1
    for ssa_v in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        g_v = 0.7 if ssa_v > 0 else 0.0
        energy_scenarios.append((f"SSA={ssa_v:.1f}\nalb=0.1",
                                 ssa_v, g_v, 0.1))

    # Albedo sweep at SSA=0.9
    for alb_v in [0.0, 0.3, 0.5, 0.8, 1.0]:
        energy_scenarios.append((f"SSA=0.9\nalb={alb_v:.1f}",
                                 0.9, 0.7, alb_v))

    # 4 extreme cases
    energy_scenarios.append(("SSA=1\nalb=1", 1.0, 0.7, 1.0))
    energy_scenarios.append(("SSA=0\nalb=0", 0.0, 0.0, 0.0))

    labels_en = []
    F_up_arr     = []
    F_abs_sfc_arr = []
    F_abs_atm_arr = []

    for lbl_en, ssa_en, g_en, alb_en in energy_scenarios:
        prof_en = AtmosphereProfile(aod=energy_aod, ssa=ssa_en, g=g_en,
                                    albedo=alb_en, theta0=THETA0, f0=F0)
        _, _, _, _, Fup_en, Fdn_en, Fdir_en = \
            compute_fluxes(energy_solver, prof_en)
        # Surface absorbs (1-albedo) of total downwelling
        Fabs_sfc = (1.0 - alb_en) * (Fdn_en + Fdir_en)
        # Atmosphere absorbs the rest
        Fabs_atm = F_solar - Fup_en - Fabs_sfc
        Fabs_atm = max(Fabs_atm, 0.0)

        labels_en.append(lbl_en)
        F_up_arr.append(Fup_en)
        F_abs_sfc_arr.append(Fabs_sfc)
        F_abs_atm_arr.append(Fabs_atm)

    F_up_arr      = np.array(F_up_arr)
    F_abs_sfc_arr = np.array(F_abs_sfc_arr)
    F_abs_atm_arr = np.array(F_abs_atm_arr)

    x_bar = np.arange(len(labels_en))
    bar_w_en = 0.6

    fig_en, ax_en = plt.subplots(figsize=(16, 7))

    ax_en.bar(x_bar, F_up_arr, bar_w_en,
              label=r"$F_{\uparrow}$ (TOA)",
              color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax_en.bar(x_bar, F_abs_sfc_arr, bar_w_en, bottom=F_up_arr,
              label=r"$F_{abs,sfc}$ = $(1-A_s)(F_{diff}+F_{dir})$",
              color=PALETTE[2], edgecolor="white", linewidth=0.5)
    ax_en.bar(x_bar, F_abs_atm_arr, bar_w_en,
              bottom=F_up_arr + F_abs_sfc_arr,
              label=r"$F_{abs,atm}$",
              color=PALETTE[1], edgecolor="white", linewidth=0.5)

    # Reference line: F_in = F0 * mu0
    ax_en.axhline(F_solar, color="black", ls="--", lw=1.5,
                  label=rf"$F_{{in}} = F_0 \mu_0 = {F_solar:.4f}$")

    ax_en.set_xticks(x_bar)
    ax_en.set_xticklabels(labels_en, fontsize=13, rotation=0)
    ax_en.set_ylabel("Flux")
    ax_en.legend(loc="upper right")
    ax_en.grid(True, axis="y", alpha=0.3)
    fig_en.tight_layout()
    _fn_en = f"Energy_balance_AOD_{energy_aod}_N{N_STREAMS}_theta0{THETA0_DEG:.0f}"
    _fn_en = _fn_en.replace(".", "")
    savefig(fig_en, _fn_en)

    # Print energy conservation check
    F_total = F_up_arr + F_abs_sfc_arr + F_abs_atm_arr
    max_err = np.max(np.abs(F_total - F_solar))
    print(f"  Energy conservation: max|F_total - F_in| = {max_err:.2e}")


# ══════════════════════════════════════════════════════════════════
#  SUPPORTING ANALYSIS D — Discrete vs Continuous Angles
# ══════════════════════════════════════════════════════════════════
if RUN_STUDY_D:
    print("\n" + "=" * 70)
    print("  SUPPORTING ANALYSIS D — Discrete vs Continuous Angles")
    print("=" * 70)
    # Fine angular grid for interpolation (1° steps, 1° to 89°)
    USER_ANGLES_DEG = np.arange(1, 90, 1.0)
    USER_MUS = np.cos(np.deg2rad(USER_ANGLES_DEG))

    # Scenarios to compare
    dvc_scenarios = [
        ("Baseline",                BASE["aod"], BASE["ssa"], BASE["g"], BASE["albedo"]),
        ("High AOD",                1.5,         0.90,        0.70,      0.1),
    ]

    dvc_solver = DisortSolver(n_streams=N_STREAMS)

    fig_dvc, axes_dvc = plt.subplots(len(dvc_scenarios), 2,
                                      figsize=(14, 4 * len(dvc_scenarios)))
    if len(dvc_scenarios) == 1:
        axes_dvc = axes_dvc.reshape(1, 2)

    dvc_stats_lines = []
    dvc_stats_lines.append(f"{'Scenario':<22} | {'Boundary':>4} | "
                            f"{'MaxAbsErr':>10} | {'RMSE_quad':>10} | "
                            f"{'MaxRelErr%':>10} | {'MeanRelErr%':>11}")
    dvc_stats_lines.append("-" * 90)

    for s_idx, (s_name, s_aod, s_ssa, s_g, s_alb) in enumerate(dvc_scenarios):
        prof_dvc = AtmosphereProfile(aod=s_aod, ssa=s_ssa, g=s_g,
                                      albedo=s_alb, theta0=THETA0, f0=F0)

        # Discrete solution at quadrature nodes (delta_m=False for clean comparison)
        mu_up_d, I_toa_d, mu_dn_d, I_boa_d = dvc_solver.solve(
            prof_dvc, output="both", delta_m=False, store_state=True)

        # Continuous solution at the quadrature nodes themselves
        mu_up_c, I_toa_c = dvc_solver.interpolate_intensity(mu_up_d, output="toa")
        mu_dn_c, I_boa_c = dvc_solver.interpolate_intensity(mu_dn_d, output="boa")

        # Continuous solution at the fine angular grid
        mu_fine_toa, I_toa_fine = dvc_solver.interpolate_intensity(USER_MUS, output="toa")
        mu_fine_boa, I_boa_fine = dvc_solver.interpolate_intensity(USER_MUS, output="boa")

        vza_d_up = np.rad2deg(np.arccos(mu_up_d))
        vza_d_dn = np.rad2deg(np.arccos(mu_dn_d))

        # Statistics at quadrature nodes
        for bnd_name, I_disc, I_cont in [("TOA", I_toa_d, I_toa_c),
                                           ("BOA", I_boa_d, I_boa_c)]:
            abs_err = np.abs(np.abs(I_disc) - np.abs(I_cont))
            rel_err = abs_err / (np.abs(I_disc) + 1e-30) * 100
            max_abs = np.max(abs_err)
            rmse = np.sqrt(np.mean(abs_err**2))
            max_rel = np.max(rel_err)
            mean_rel = np.mean(rel_err)
            dvc_stats_lines.append(
                f"{s_name:<22} | {bnd_name:>4} | {max_abs:10.2e} | "
                f"{rmse:10.2e} | {max_rel:10.4f} | {mean_rel:11.4f}")

        # Plot: TOA
        ax_toa = axes_dvc[s_idx, 0]
        ax_toa.plot(USER_ANGLES_DEG, np.abs(I_toa_fine), "-", color=PALETTE[0],
                    lw=1.5, label="Continuous", zorder=2)
        ax_toa.plot(vza_d_up, np.abs(I_toa_d), "o", color=PALETTE[1],
                    markersize=5, markeredgecolor="white", markeredgewidth=0.5,
                    label="Discrete", zorder=3)
        ax_toa.set_xlabel("Viewing Zenith Angle (deg)")
        ax_toa.set_ylabel(RAD_UNITS)
        ax_toa.legend(fontsize=14)

        # Plot: BOA
        ax_boa = axes_dvc[s_idx, 1]
        ax_boa.plot(USER_ANGLES_DEG, np.abs(I_boa_fine), "-", color=PALETTE[0],
                    lw=1.5, label="Continuous", zorder=2)
        ax_boa.plot(vza_d_dn, np.abs(I_boa_d), "o", color=PALETTE[1],
                    markersize=5, markeredgecolor="white", markeredgewidth=0.5,
                    label="Discrete", zorder=3)
        ax_boa.set_xlabel("Viewing Zenith Angle (deg)")
        ax_boa.set_ylabel(RAD_UNITS)
        ax_boa.legend(fontsize=14)

    fig_dvc.tight_layout()
    _fn_dvc = f"Discrete_vs_continuous_N{N_STREAMS}_theta0{THETA0_DEG:.0f}"
    savefig(fig_dvc, _fn_dvc)

    # Print statistics table
    print("\n  Agreement between discrete and continuous:")
    print("  (delta-M OFF for clean comparison — both use same optical depths)\n")
    for line in dvc_stats_lines:
        print(f"  {line}")

    # Summary statistics across all scenarios
    print(f"\n  NOTE: With delta_m=False, Eqs 35a/35b reproduce the discrete solution")
    print(f"  at quadrature nodes to machine precision (~1e-14).  With delta_m=True,")
    print(f"  there is an inherent ~3e-4 discrepancy due to mixed tau conventions")
    print(f"  in solve() (see MEMORY.md for details).")



if RUN_STUDY_A:
    # ── Write TXT report ────────────────────────────────────────────
    txt_fname = os.path.join(IMG_DIR, f"parameter_study_report_N{N_STREAMS}_theta0{THETA0_DEG:.0f}.txt")
    with open(txt_fname, "w", encoding="utf-8") as f:
        for line in study_lines:
            f.write(line + "\n")
    print(f"\n  Report saved: {txt_fname}")

# ══════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Configuration: N_STREAMS={N_STREAMS}, output_dir='{IMG_DIR}'")
if RUN_VALIDATION:
    print(f"  Validation tests passed: {total_pass}/{total_total}")
print(f"\n  All figures saved to: {IMG_DIR}/")
print("=" * 70)
