"""Diagnostic v3: Legendre expansion truncation analysis for g=0.95."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atrt.phase import legendre_expansion_hg

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

# How many Legendre terms are "significant" for different g?
print("=" * 60)
print("Legendre coefficient magnitude at l=N for HG phase function")
print("=" * 60)
print(f"{'g':>6} | {'chi_36':>12} | {'chi_64':>12} | {'chi_80':>12} | "
      f"{'l where chi<0.1':>17}")
for g_val in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    chi_36 = (2*36+1) * g_val**36
    chi_64 = (2*64+1) * g_val**64
    chi_80 = (2*80+1) * g_val**80
    # Find l where (2l+1)*g^l < 0.1
    l_conv = 0
    for l in range(1, 500):
        if (2*l+1) * g_val**l < 0.1:
            l_conv = l
            break
    print(f"  {g_val:.2f} | {chi_36:12.4f} | {chi_64:12.4f} | {chi_80:12.4f} | "
          f"{l_conv:>17d}")

print()
print("=" * 60)
print("delta-M scaling effect")
print("=" * 60)
for g_val in [0.7, 0.8, 0.9, 0.95]:
    for N in [36, 64, 80]:
        f = g_val**N
        g_prime = (g_val - f) / (1 - f) if abs(1-f)>1e-15 else 0
        last_coeff = (2*(N-1)+1) * g_prime**(N-1)
        print(f"  g={g_val:.2f}, N={N:3d}: f={f:.4e}, g'={g_prime:.4f}, "
              f"chi'_{N-1} = {last_coeff:.4f}")

# Plot: truncated vs full HG phase function
print()
print("=" * 60)
print("Generating Legendre truncation plot")
print("=" * 60)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cos_theta = np.linspace(-1, 1, 1000)

for ax, g_val in zip(axes, [0.7, 0.9, 0.95]):
    # Exact HG
    hg_exact = (1 - g_val**2) / (1 + g_val**2 - 2*g_val*cos_theta)**1.5
    ax.plot(cos_theta, hg_exact, "k-", lw=2, label="Exact HG", zorder=5)

    for N, c in [(16, "C0"), (36, "C1"), (64, "C2"), (80, "C3")]:
        chi = legendre_expansion_hg(g_val, N)
        # Build P_l(cos_theta)
        Pl = np.zeros((N, len(cos_theta)))
        Pl[0] = 1.0
        if N > 1:
            Pl[1] = cos_theta
        for l in range(1, N-1):
            Pl[l+1] = ((2*l+1)*cos_theta*Pl[l] - l*Pl[l-1]) / (l+1)
        p_trunc = chi @ Pl
        ax.plot(cos_theta, p_trunc, "-", color=c, lw=1, alpha=0.8,
                label=f"N={N} terms")

    ax.set_xlabel(r"$\cos\Theta$")
    ax.set_ylabel(r"$P(\cos\Theta)$")
    ax.legend()
    ax.set_ylim(-2, min(30, hg_exact.max()*1.2))
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fname = os.path.join(IMG_DIR, "diag_legendre_truncation_g07_g09_g095.png")
fig.savefig(fname, dpi=200, bbox_inches="tight")
print(f"Saved: {fname}")
