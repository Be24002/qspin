"""Scan the temperature of the 2D Ising model across its phase transition.

Produces:
    figures/classical_transition.png   |M| and C vs T, with Onsager's exact Tc
    figures/classical_snapshots.png    spin configurations below / at / above Tc
    results/classical_ising.csv        raw scan data
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qspin.classical_ising import exact_tc_2d, run_ising

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

L = 48
temperatures = np.linspace(1.6, 3.4, 15)

print(f"Running {len(temperatures)} temperatures, L = {L} (checkerboard Metropolis)...")
runs = [
    run_ising(L=L, T=float(T), n_eq=400, n_prod=800, sample_every=4, seed=int(T * 10))
    for T in temperatures
]
M = np.array([r["M_abs"] for r in runs])
C = np.array([r["C_per_site"] for r in runs])
E = np.array([r["E_per_site"] for r in runs])

tc_exact = exact_tc_2d(1.0)
t_of_max_c = temperatures[int(np.argmax(C))]

# ---- Figure 1: order parameter and specific heat ------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
axes[0].plot(temperatures, M, "o-", color="C0")
axes[0].axvline(tc_exact, color="k", ls="--", lw=1, label=r"Onsager $T_c = 2.269$")
axes[0].set_xlabel("temperature  $T$  ($J/k_B$)")
axes[0].set_ylabel(r"magnetization $\langle |M| \rangle$")
axes[0].set_title("Order parameter melts at $T_c$")
axes[0].legend()

axes[1].plot(temperatures, C, "s-", color="C3")
axes[1].axvline(tc_exact, color="k", ls="--", lw=1, label=r"exact $T_c$")
axes[1].set_xlabel("temperature  $T$  ($J/k_B$)")
axes[1].set_ylabel(r"specific heat  $C$  per site")
axes[1].set_title(f"Heat capacity peaks at $T \\approx {t_of_max_c:.2f}$")
axes[1].legend()

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "classical_transition.png", dpi=150)
plt.close(fig)

# ---- Figure 2: snapshots below / at / above the transition --------------
snapshot_temps = [1.8, tc_exact, 3.0]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, T in zip(axes, snapshot_temps):
    run = run_ising(L=L, T=float(T), n_eq=400, n_prod=50, seed=42)
    ax.imshow(run["spins_final"], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"$T = {T:.2f}$" + ("  (critical)" if T == tc_exact else ""))
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("2D Ising snapshots: ordered, critical, disordered")
fig.tight_layout()
fig.savefig(FIGURES / "classical_snapshots.png", dpi=150)
plt.close(fig)

# ---- Raw data ------------------------------------------------------------
np.savetxt(
    RESULTS / "classical_ising.csv",
    np.column_stack([temperatures, M, C, E]),
    delimiter=",",
    header="T,M_abs,C_per_site,E_per_site",
    comments="",
)

print(f"Exact Tc (Onsager)      : {tc_exact:.4f}")
print(f"Peak of C found at T    : {t_of_max_c:.4f}")
print(f"<|M|> at T=1.6 / T=3.4 : {M[0]:.3f} / {M[-1]:.3f}")
print(f"Wrote figures/classical_transition.png, classical_snapshots.png, results/classical_ising.csv")
