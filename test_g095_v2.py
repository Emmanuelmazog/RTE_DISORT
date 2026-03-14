"""Diagnostic v2: understand oscillations at g=0.95 in more detail."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atrt.profile import AtmosphereProfile
from atrt.solver import DisortSolver

FINE_DEG = np.arange(1, 90, 1.0)
FINE_MUS = np.cos(np.deg2rad(FINE_DEG))

AOD, SSA, ALBEDO, F0 = 0.5, 0.90, 0.1, np.pi
THETA0 = np.deg2rad(30)

# ---- Test 1: delta-M effect on g=0.95 ----
print("=" * 60)
print("TEST 1: What delta-M does to g=0.95")
print("=" * 60)
sv = DisortSolver(n_streams=36)
p = AtmosphereProfile(aod=AOD, ssa=SSA, g=0.95,
                      albedo=ALBEDO, theta0=THETA0, f0=F0)
# Check what delta-M produces
g_orig = 0.95
N = 36
f_trunc = g_orig ** N  # HG: chi_M = (2M+1)*g^M, f = g^M
g_prime = (g_orig - f_trunc) / (1 - f_trunc)
tau_prime = (1 - SSA * f_trunc) * AOD
ssa_prime = SSA * (1 - f_trunc) / (1 - SSA * f_trunc)
print(f"  Original:  tau={AOD}, ssa={SSA}, g={g_orig}")
print(f"  f_trunc = g^N = {g_orig}^{N} = {f_trunc:.6e}")
print(f"  delta-M:   tau'={tau_prime:.6f}, ssa'={ssa_prime:.6f}, g'={g_prime:.6f}")
print(f"  -> delta-M barely changes anything because g^36 ~ 0")
print()

# ---- Test 2: compare g values, quadrature only ----
print("=" * 60)
print("TEST 2: Quadrature-node oscillations for different g values")
print("=" * 60)
for g_val in [0.6, 0.7, 0.8, 0.9, 0.95]:
    p = AtmosphereProfile(aod=AOD, ssa=SSA, g=g_val,
                          albedo=ALBEDO, theta0=THETA0, f0=F0)
    sv = DisortSolver(n_streams=36)
    mu_up, I_up, _, _ = sv.solve(p, output="both", delta_m=True, store_state=True)
    # measure oscillation: std of differences between adjacent points
    diffs = np.diff(I_up)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    print(f"  g={g_val:.2f}: I_range=[{I_up.min():.5f}, {I_up.max():.5f}], "
          f"sign_changes_in_diff={sign_changes}/{len(diffs)-1}, "
          f"std(I)={I_up.std():.5f}")

print()

# ---- Test 3: N_streams convergence for g=0.95 (quadrature only, no interp) ----
print("=" * 60)
print("TEST 3: Convergence of quadrature TOA flux with N_streams, g=0.95")
print("=" * 60)
for n_str in [8, 16, 24, 32, 36, 48, 64, 80]:
    sv = DisortSolver(n_streams=n_str)
    p = AtmosphereProfile(aod=AOD, ssa=SSA, g=0.95,
                          albedo=ALBEDO, theta0=THETA0, f0=F0)
    mu_up, I_up, mu_dn, I_dn = sv.solve(p, output="both", delta_m=True)
    w_up = sv.weights[sv.up_idx]
    F_up = 2 * np.pi * np.sum(w_up * mu_up * np.abs(I_up))
    print(f"  N={n_str:3d}: F_up={F_up:.6f}, max(I_toa)={np.abs(I_up).max():.6f}, "
          f"min(I_toa)={np.abs(I_up).min():.6f}")

print()

# ---- Test 4: Interpolation stability ----
print("=" * 60)
print("TEST 4: Source-function interpolation at different N, g=0.95")
print("=" * 60)
for n_str in [16, 24, 32, 36, 48, 64, 80]:
    sv = DisortSolver(n_streams=n_str)
    p = AtmosphereProfile(aod=AOD, ssa=SSA, g=0.95,
                          albedo=ALBEDO, theta0=THETA0, f0=F0)
    try:
        mu_up, I_up, _, _ = sv.solve(p, output="both", delta_m=True, store_state=True)
        _, I_toa = sv.interpolate_intensity(FINE_MUS, output="toa")
        I_toa = np.abs(I_toa)
        has_nan = np.any(np.isnan(I_toa))
        diffs = np.diff(I_toa)
        sign_ch = np.sum(np.diff(np.sign(diffs)) != 0) if not has_nan else -1
        print(f"  N={n_str:3d}: range=[{I_toa.min():.6f}, {I_toa.max():.6f}], "
              f"NaN={has_nan}, sign_changes={sign_ch}")
    except Exception as e:
        print(f"  N={n_str:3d}: ERROR: {e}")

print()

# ---- Test 5: Plot comparison ----
print("=" * 60)
print("TEST 5: Generating comparison plot")
print("=" * 60)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

g_test_vals = [0.7, 0.8, 0.95]
for col, g_val in enumerate(g_test_vals):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    for n_str, ls, c in [(16, "--", "C0"), (36, "-", "C1"), (48, "-", "C2"), (64, "-", "C3")]:
        sv = DisortSolver(n_streams=n_str)
        p = AtmosphereProfile(aod=AOD, ssa=SSA, g=g_val,
                              albedo=ALBEDO, theta0=THETA0, f0=F0)
        mu_up, I_up, _, _ = sv.solve(p, output="both", delta_m=True, store_state=True)
        vza_quad = np.rad2deg(np.arccos(mu_up))

        ax_top.plot(vza_quad, np.abs(I_up), "o", color=c, ms=4,
                    label=f"N={n_str} (quad)")

        try:
            _, I_toa = sv.interpolate_intensity(FINE_MUS, output="toa")
            I_toa = np.abs(I_toa)
            if not np.any(np.isnan(I_toa)):
                ax_bot.plot(FINE_DEG, I_toa, ls, color=c, lw=1.2,
                           label=f"N={n_str} (interp)")
        except:
            pass

    ax_top.set_title(f"Quadrature nodes, g={g_val}")
    ax_top.set_xlabel("VZA (deg)")
    ax_top.set_ylabel("I(TOA)")
    ax_top.legend(fontsize=7)
    ax_top.grid(True, alpha=0.3)

    ax_bot.set_title(f"Interpolated (Eqs 35a/b), g={g_val}")
    ax_bot.set_xlabel("VZA (deg)")
    ax_bot.set_ylabel("I(TOA)")
    ax_bot.legend(fontsize=7)
    ax_bot.grid(True, alpha=0.3)

fig.suptitle(f"Effect of g and N_streams on TOA radiance\n"
             f"(AOD={AOD}, SSA={SSA}, delta-M ON)", fontsize=13)
fig.tight_layout()
fig.savefig("diag_g095_detail.png", dpi=150, bbox_inches="tight")
print("Saved: diag_g095_detail.png")
