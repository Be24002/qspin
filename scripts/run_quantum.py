"""Scan the transverse field h of the TFIM across its quantum phase transition.

Produces:
    figures/tfim_phase_diagram.png  order parameter, gap and entanglement vs h
    results/tfim_scan.csv           raw scan data
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qspin.tfim import scan_tfim

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

L = 12
h_values = np.linspace(0.2, 2.0, 19)

print(f"Exact diagonalization, L = {L} (dim = {2**L}), {len(h_values)} field values...")
rows = scan_tfim(L=L, h_values=h_values)

h = np.array([r["h"] for r in rows])
m_z2 = np.array([r["m_z2"] for r in rows])
m_x = np.array([r["m_x"] for r in rows])
gap = np.array([r["gap"] for r in rows])
S = np.array([r["S"] for r in rows])

h_of_max_S = h[int(np.argmax(S))]
gap_drop = gap[int(np.argmax(h >= 2.0))] / gap[int(np.argmin(np.abs(h - 1.0)))]

# ---- Figure: two panels, order parameter + excitations ------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].plot(h, m_z2, "o-", label=r"$m_{z2} = \sqrt{\langle M_z^2\rangle}/L$ (order)")
axes[0].plot(h, m_x, "s-", label=r"$m_x = \langle M_x \rangle/L$ (transverse)")
axes[0].axvline(1.0, color="k", ls="--", lw=1, label=r"$h_c = J$ (exact)")
axes[0].set_xlabel("transverse field  $h$  ($J$)")
axes[0].set_ylabel("magnetization")
axes[0].set_title(f"Quantum phase transition, L = {L}")
axes[0].legend()

axes[1].plot(h, gap, "s-", color="C3", label=r"gap $\Delta = E_1 - E_0$")
axes[1].axvline(1.0, color="k", ls="--", lw=1)
axes[1].set_xlabel("transverse field  $h$  ($J$)")
axes[1].set_ylabel(r"energy gap $\Delta$")
axes[1].set_title("The gap closes at criticality")
axes[1].legend(loc="upper left")

ax2 = axes[1].twinx()
ax2.plot(h, S, "^--", color="C2", label=r"entanglement $S$ (half chain)")
ax2.set_ylabel("entanglement entropy $S$")
ax2.legend(loc="upper right")

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "tfim_phase_diagram.png", dpi=150)
plt.close(fig)

# ---- Raw data ------------------------------------------------------------
np.savetxt(
    RESULTS / "tfim_scan.csv",
    np.column_stack([h, m_z2, m_x, gap, S]),
    delimiter=",",
    header="h,m_z2,m_x,gap,S",
    comments="",
)

print(f"Order parameter m_z2 at h=0.2 / h=2.0 : {m_z2[0]:.3f} / {m_z2[-1]:.3f}")
print(f"Entanglement S peaks at h             : {h_of_max_S:.3f}   (exact: 1.000)")
print(f"Gap at h=1 vs h=2                     : {gap[int(np.argmin(np.abs(h - 1.0)))]:.3f} vs "
      f"{gap[-1]:.3f}  ({gap_drop:.1f}x smaller at criticality)")
print(f"Cat-state plateau S(h<1) ~ ln 2 = {np.log(2):.3f}; measured S(h=0.2) = {S[0]:.3f}")
print("Wrote figures/tfim_phase_diagram.png, results/tfim_scan.csv")
