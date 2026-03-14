"""Diagnostic: oscillations at g=0.95 — test different N_streams and delta-M."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atrt.profile import AtmosphereProfile
from atrt.solver import DisortSolver

# -- Match main.py plot style (serif / Computer Modern, larger fonts) --
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
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
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

IMG_DIR = "Images_final_report"
os.makedirs(IMG_DIR, exist_ok=True)

FINE_DEG = np.arange(1, 90, 1.0)
FINE_MUS = np.cos(np.deg2rad(FINE_DEG))

AOD, SSA, G_TEST = 0.5, 0.90, 0.95
ALBEDO, THETA0, F0 = 0.1, np.deg2rad(30), np.pi

prof = AtmosphereProfile(aod=AOD, ssa=SSA, g=G_TEST,
                         albedo=ALBEDO, theta0=THETA0, f0=F0)

configs = [
    # (N_streams, delta_m, label)
    (16, True,  "N=16, delta-M ON"),
    (36, True,  "N=36, delta-M ON"),
    (64, True,  "N=64, delta-M ON"),
    (96, True,  "N=96, delta-M ON"),
    (128, True, "N=128, delta-M ON"),
    (36, False, "N=36, delta-M OFF"),
    (64, False, "N=64, delta-M OFF"),
    (96, False, "N=96, delta-M OFF"),
    (128, False,"N=128, delta-M OFF"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ax_toa_on, ax_toa_off = axes[0]
ax_boa_on, ax_boa_off = axes[1]

colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

for n_str, dm, label in configs:
    print(f"Running: {label} ...", end=" ", flush=True)
    sv = DisortSolver(n_streams=n_str)
    p = prof.copy()
    mu_up, I_up, mu_dn, I_dn = sv.solve(p, output="both",
                                          delta_m=dm, store_state=True)
    _, I_toa = sv.interpolate_intensity(FINE_MUS, output="toa")
    _, I_boa = sv.interpolate_intensity(FINE_MUS, output="boa")
    I_toa = np.abs(I_toa)
    I_boa = np.abs(I_boa)
    print(f"TOA range [{I_toa.min():.6f}, {I_toa.max():.6f}]")

    # pick color by N
    n_idx = {16:0, 36:1, 64:2, 96:3, 128:4}[n_str]
    c = colors[n_idx]
    lw = 1.5

    if dm:
        ax_toa_on.plot(FINE_DEG, I_toa, "-", color=c, lw=lw, label=label)
        ax_boa_on.plot(FINE_DEG, I_boa, "-", color=c, lw=lw, label=label)
    else:
        ax_toa_off.plot(FINE_DEG, I_toa, "-", color=c, lw=lw, label=label)
        ax_boa_off.plot(FINE_DEG, I_boa, "-", color=c, lw=lw, label=label)

for ax in [ax_toa_on, ax_toa_off, ax_boa_on, ax_boa_off]:
    ax.set_xlabel("Viewing Zenith Angle (deg)")
    ax.set_ylabel("Radiance")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fname = os.path.join(IMG_DIR, f"diag_g095_streams_AOD_{AOD}_SSA_{SSA}_g{G_TEST}_theta030".replace(".", "") + ".png")
fig.savefig(fname, dpi=200, bbox_inches="tight")
print(f"\nSaved: {fname}")

# Also test: what do quadrature-node intensities look like?
print("\n--- Quadrature node intensities at g=0.95 ---")
for n_str in [36, 64, 96]:
    sv = DisortSolver(n_streams=n_str)
    p = prof.copy()
    mu_up, I_up, _, _ = sv.solve(p, output="both", delta_m=True, store_state=True)
    vza = np.rad2deg(np.arccos(mu_up))
    print(f"  N={n_str}: VZA = {vza.round(1)}")
    print(f"           I   = {I_up.round(6)}")
