"""Real-time quench dynamics of the TFIM.

Prepare the ground state of H(h_i), then suddenly change the field to h_f
and evolve the *same* state with the new Hamiltonian:

    |psi(t)> = exp(-i H(h_f) t) |psi(0)>.

Because we apply the same small time step repeatedly,
U(dt)^n = U(n dt) is exact (no Trotter error); the only error is floating
point, which we monitor through the norm of the state.

Physically this is a model experiment repeated in labs every day: prepare,
quench, watch. The order parameter melts while the entanglement entropy
grows -- the quench creates particle pairs out of the vacuum.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from .tfim import (
    build_tfim_hamiltonian,
    entanglement_entropy,
    ground_state,
    magnetization_x,
    magnetization_z_squared,
)


def quench(
    L: int = 12,
    h_initial: float = 0.3,
    h_final: float = 3.0,
    J: float = 1.0,
    t_max: float = 8.0,
    n_steps: int = 80,
) -> dict:
    """Evolve the ground state of H(h_initial) under H(h_final).

    Returns time series of the order parameter m_z2, transverse
    magnetization m_x, half-chain entanglement entropy S, and the norm
    (a unitarity check).
    """
    _, vecs = ground_state(L, J=J, h=h_initial, k=2)
    psi = vecs[:, 0].astype(np.complex128)

    H_final = build_tfim_hamiltonian(L, J, h_final)
    dt = t_max / n_steps
    times = np.linspace(0.0, t_max, n_steps + 1)

    def measure(state):
        return (
            float(np.sqrt(magnetization_z_squared(state, L)) / L),
            float(magnetization_x(state, L) / L),
            entanglement_entropy(state, L),
        )

    m_z2, m_x, S = [measure(psi)[0]], [measure(psi)[1]], [measure(psi)[2]]
    norms = [float(np.linalg.norm(psi))]

    for _ in range(n_steps):
        # One exact short-time step; iterating gives U(dt)^n = U(n*dt).
        psi = expm_multiply(-1j * dt * H_final, psi)
        a, b, c = measure(psi)
        m_z2.append(a)
        m_x.append(b)
        S.append(c)
        norms.append(float(np.linalg.norm(psi)))

    return {
        "t": times,
        "m_z2": np.array(m_z2),
        "m_x": np.array(m_x),
        "S": np.array(S),
        "norm": np.array(norms),
    }
