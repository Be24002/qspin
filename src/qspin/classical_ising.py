"""Classical Ising model on a periodic square lattice.

Hamiltonian (units: k_B = 1, J > 0 ferromagnetic):

    E(s) = -J * sum_<ij> s_i s_j,        s_i = +-1

The system is simulated with a *checkerboard* Metropolis algorithm: spins are
updated in place with NumPy operations, two colors at a time. Spins of the
same color never touch each other, so updating a whole color at once is
statistically identical to updating its spins one by one -- but orders of
magnitude faster.

Key physical target: the 2D Ising model has an exact critical temperature
(Onsager, 1944),

    T_c = 2J / ln(1 + sqrt(2)) ~ 2.269 J,

below which the system spontaneously magnetizes and above which it is a
paramagnet. The Monte Carlo scan should reproduce both facts.
"""

from __future__ import annotations

import numpy as np


def exact_tc_2d(J: float = 1.0) -> float:
    """Onsager's exact critical temperature of the 2D Ising model."""
    return 2.0 * J / np.log(1.0 + np.sqrt(2.0))


def total_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """Total energy, counting each nearest-neighbour bond once (PBC)."""
    bonds_x = spins * np.roll(spins, -1, axis=0)
    bonds_y = spins * np.roll(spins, -1, axis=1)
    return float(-J * (bonds_x.sum() + bonds_y.sum()))


def metropolis_sweep(
    spins: np.ndarray, T: float, J: float, rng: np.random.Generator
) -> np.ndarray:
    """One full Metropolis sweep (N flip attempts on average, N = #spins).

    Parameters
    ----------
    spins : (L, L) array of +-1
    T : temperature in units of J/k_B
    J : coupling constant
    rng : NumPy random generator (pass the same one to keep a chain)
    """
    ii, jj = np.indices(spins.shape)
    parity = (ii + jj) % 2  # checkerboard colors

    for color in (0, 1):
        # Sum of the 4 nearest neighbours, recomputed after each color flip.
        neighbors = (
            np.roll(spins, 1, axis=0)
            + np.roll(spins, -1, axis=0)
            + np.roll(spins, 1, axis=1)
            + np.roll(spins, -1, axis=1)
        )
        # Energy change if this spin were flipped: dE = 2 J s_i * sum_neighbors
        dE = 2.0 * J * spins * neighbors
        # Metropolis rule: accept if dE <= 0, else with probability exp(-dE/T).
        accept = (dE <= 0) | (rng.random(spins.shape) < np.exp(-dE / T))
        flips = (parity == color) & accept
        spins[flips] *= -1

    return spins


def run_ising(
    L: int = 48,
    T: float = 2.3,
    J: float = 1.0,
    n_eq: int = 400,
    n_prod: int = 800,
    sample_every: int = 4,
    seed: int = 0,
) -> dict:
    """Run a full Metropolis simulation at temperature ``T`` and measure.

    Returns a dictionary with the mean magnetization |M|, the specific heat
    per site C (from energy fluctuations), the energy per site, and the final
    spin configuration (useful for snapshots).
    """
    rng = np.random.default_rng(seed)
    spins = rng.choice(np.array([-1, 1]), size=(L, L))

    # Equilibration (measurements discarded).
    for _ in range(n_eq):
        metropolis_sweep(spins, T, J, rng)

    # Production: sample every ``sample_every`` sweeps.
    mags, energies = [], []
    for sweep in range(n_prod):
        metropolis_sweep(spins, T, J, rng)
        if sweep % sample_every == 0:
            mags.append(spins.mean())
            energies.append(total_energy(spins, J))

    mags = np.asarray(mags)
    energies = np.asarray(energies)
    n = L * L

    return {
        "L": L,
        "T": T,
        "M_abs": float(np.abs(mags).mean()),          # order parameter
        "M2": float((mags**2).mean()),
        "E_per_site": float(energies.mean() / n),
        # Fluctuation-dissipation: C = (<E^2> - <E>^2) / (N T^2)
        "C_per_site": float(((energies**2).mean() - energies.mean() ** 2) / (n * T**2)),
        "spins_final": spins.copy(),
    }
