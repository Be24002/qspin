# qspin — from classical magnetism to a quantum phase transition

A small physics engine for spin systems, written from scratch in Python + NumPy/SciPy:

1. **Classical Ising model** on a square lattice, simulated with a vectorized
   checkerboard **Metropolis Monte Carlo** — it reproduces Onsager's exact
   critical temperature `T_c = 2J/ln(1+√2) ≈ 2.269` with no fitting parameters.
2. **Transverse-field Ising chain (TFIM)** solved by **exact diagonalization** —
   it locates the zero-temperature quantum phase transition at `h = J`, where the
   excitation gap softens and the half-chain entanglement entropy peaks.
3. **Quench dynamics** — prepare the ordered ground state, suddenly drive the
   system across the critical point, and watch the order parameter melt while
   entanglement grows in real time.

No physics was hard-coded: the only inputs are the Hamiltonians. Everything else
(order parameters, heat capacity, spectra, entanglement) is *measured* from the
simulation, then checked against exact results by the test suite.

---

## The physics in one minute

**Classical.** Spins `s_i = ±1` on a lattice with energy `E = -J Σ s_i s_j`
spontaneously magnetize below `T_c`. Monte Carlo + Metropolis sampling visits
configurations with probability `∝ exp(-E/k_B T)`; measuring `⟨|M|⟩` and the
heat capacity `C` (from energy fluctuations) across temperatures reveals the
transition — and for the 2D square lattice, Onsager solved it exactly in 1944,
so we can check the code against pen and paper.

**Quantum.** Promote the spins to qubits and add a transverse field:

```
H = -J Σ σ^z_i σ^z_{i+1} - h Σ σ^x_i
```

Now quantum fluctuations (the field `h`) compete with the ferromagnetic
coupling `J`, even at zero temperature. At `h = J` the ground state changes
character — a *quantum phase transition*. Exact diagonalization builds the
`2^L × 2^L` Hamiltonian explicitly; from its ground state we measure the order
parameter `m_z2 = √⟨M_z²⟩/L` (finite chains keep the Z2 symmetry, so ⟨M_z⟩ = 0
identically and one resolves the square instead), the gap `E1 - E0`, and the
entanglement entropy `S = -Tr ρ_A ln ρ_A` of half the chain, which is maximal
at criticality.

**Dynamics.** Sudden quench `h: 0.3 → 3.0`: evolve the initial ground state with
`exp(-iHt)` (exact repeated application of one short time step — no Trotter
error) and watch `m_z2(t)` melt while `S(t)` grows: the quench creates
quasiparticle pairs that entangle the two halves of the chain.

## Results

### Classical transition (Metropolis, L = 48)

| | |
|---|---|
| ![transition](figures/classical_transition.png) | Magnetization melts and the heat capacity peaks right where Onsager said, 2.269 — found by simulation, not inserted by hand. |
| ![snapshots](figures/classical_snapshots.png) | Below `T_c`: ordered domains. At `T_c`: fractal-looking critical fluctuations. Above: noise. |

### Quantum transition (exact diagonalization, L = 12)

| | |
|---|---|
| ![phase diagram](figures/tfim_phase_diagram.png) | `m_z2` collapses past `h = 1` while the transverse magnetization rises; the gap is softest and the entanglement entropy peaks exactly at the critical point. The `ln 2` plateau on the ordered side is the cat-like ground state — a foretaste of GHZ physics. |

### Quench across criticality

| | |
|---|---|
| ![quench](figures/tfim_quench.png) | The order parameter decays and oscillates while the entanglement entropy climbs — the simulation shows a many-body system far from equilibrium. |

Raw data for every figure lives in [`results/`](results/).

## Repository layout

```
qspin/
├── src/qspin/
│   ├── classical_ising.py   # checkerboard Metropolis, observables, Onsager Tc
│   ├── tfim.py              # sparse Hamiltonian builder, ED, entanglement
│   └── dynamics.py          # exact real-time evolution (quench)
├── scripts/                 # one runnable experiment per figure/CSV
├── tests/                   # physics regression tests (pytest)
├── figures/                 # generated plots (committed)
└── results/                 # generated CSV data (committed)
```

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest                        # 14 physics tests, ~2 s
python scripts/run_classical.py
python scripts/run_quantum.py
python scripts/run_quench.py
```

Each script prints a summary comparing simulated numbers with the exact
analytic ones and regenerates `figures/` + `results/`.

## Tests as physics checks

The test suite is written so that a claim in this README fails loudly if the
physics drifts:

- Onsager `T_c` reproduced to 4 decimals (analytic formula).
- Metropolis orders at low T and disorders at high T (fixed seeds).
- `E0(h=0) = -J·L` and `E0(h≫J) ≈ -h·L` limits of the TFIM.
- GHZ cat entanglement is exactly `ln 2`; product states carry none.
- Entanglement entropy peaks near `h = 1`; the gap at criticality is far
  smaller than deep in the paramagnet.
- Time evolution preserves the norm to machine precision.

## Roadmap

- [ ] Free-fermion (Jordan–Wigner) solution as an independent benchmark for the full spectrum
- [ ] 2D TFIM via QuTiP / matrix-product states (DMRG-style truncation)
- [ ] Adiabatic annealing along `h(t)` and Kibble–Zurek scaling of defects
- [ ] Open quantum systems: coupling the chain to a Lindblad bath

## References

- L. Onsager, *Crystal statistics. I. A two-dimensional model...*, Phys. Rev. **65**, 117 (1944)
- S. Sachdev, *Quantum Phase Transitions*, 2nd ed., Cambridge University Press
- W. Krauth, *Statistical Mechanics: Algorithms and Computations*, Oxford University Press
- A. Langari et al., quench-dynamics reviews; Calabrese & Cardy, *Entanglement entropy and quantum field theory*, J. Stat. Mech. (2004)

## License

MIT — see [LICENSE](LICENSE).
