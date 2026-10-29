"""Quench dynamics: prepare the TFIM ground state deep in the ordered phase,
suddenly crank the field into the paramagnet, and watch what happens.

Produces:
    figures/tfim_quench.png    order-parameter melting + entanglement growth
    results/tfim_quench.csv    raw time series
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qspin.dynamics import quench

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

L = 12
out = quench(L=L, h_initial=0.3, h_final=3.0, t_max=8.0, n_steps=80)

print(f"Quench L = {L}: h = 0.3 -> 3.0  (across the critical point h = 1)")
print(f"m_z2 : {out['m_z2'][0]:.3f} -> {out['m_z2'][-1]:.3f}   (order melts)")
print(f"S    : {out['S'][0]:.3f} -> {out['S'][-1]:.3f}   (entanglement grows)")
print(f"norm drift (unitarity check): max |1 - ||psi||| = "
      f"{np.max(np.abs(out['norm'] - 1.0)):.2e}")

# ---- Figure ---------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(out["t"], out["m_z2"], "o-", ms=3, color="C0", label=r"$m_{z2}(t)$ — order parameter")
ax1.set_xlabel("time  $t$  ($\\hbar/J$)")
ax1.set_ylabel(r"$m_{z2}(t)$", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")

ax2 = ax1.twinx()
ax2.plot(out["t"], out["S"], "s--", ms=3, color="C2", label=r"$S(t)$ — entanglement")
ax2.set_ylabel("half-chain entropy  $S(t)$", color="C2")
ax2.tick_params(axis="y", labelcolor="C2")

lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labels, loc="center right")
ax1.set_title(f"Quench across the transition:  $h = 0.3 \\to 3.0$,  L = {L}")
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "tfim_quench.png", dpi=150)
plt.close(fig)

np.savetxt(
    RESULTS / "tfim_quench.csv",
    np.column_stack([out["t"], out["m_z2"], out["m_x"], out["S"]]),
    delimiter=",",
    header="t,m_z2,m_x,S",
    comments="",
)
print("Wrote figures/tfim_quench.png, results/tfim_quench.csv")
