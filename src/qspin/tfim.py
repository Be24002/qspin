"""Transverse-field Ising chain (TFIM), solved by exact diagonalization.

Hamiltonian on a periodic chain of L spin-1/2 sites (hbar = 1):

    H = -J * sum_i sigma^z_i sigma^z_{i+1}  -  h * sum_i sigma^x_i

The first term wants all spins to align along z (ferromagnet); the second,
the transverse field h, wants each spin along x (paramagnet). Their
competition produces a *quantum* phase transition at zero temperature at
h_c = J (in the thermodynamic limit):

    h < J : ordered phase, twofold-degenerate 'cat' ground states
    h > J : quantum paramagnet, unique product-like ground state

Finite chains (exact diagonalization) keep the Z2 symmetry, so the ground
state is the symmetric cat combination for any h < J. The plain
magnetization <sigma^z> is then identically zero; the standard workaround is
to use the symmetry-resolved order parameter

    m_z2 = sqrt(< M_z^2 >) / L,   M_z = sum_i sigma^z_i,

which equals 1 in the ordered phase and ~1/sqrt(L) in the paramagnet.

The entanglement entropy of half the chain, S = -Tr(rho_A ln rho_A), is
maximal near h_c -- for the infinite chain it diverges logarithmically with
the conformal charge c = 1/2 of the TFIM universality class.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# Single-qubit Pauli matrices (sparse).
SZ = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
SX = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))


def _operator_at(op: sp.csr_matrix, L: int, site: int) -> sp.csr_matrix:
    """Embed a single-site operator into the 2^L Hilbert space via tensor product."""
    left = sp.identity(2**site, format="csr")
    right = sp.identity(2 ** (L - site - 1), format="csr")
    return sp.kron(left, sp.kron(op, right, format="csr"), format="csr")


def sigma_z_sum(L: int) -> sp.csr_matrix:
    """Total magnetization operator M_z = sum_i sigma^z_i."""
    dim = 2**L
    total = sp.csr_matrix((dim, dim))
    for i in range(L):
        total = total + _operator_at(SZ, L, i)
    return total.tocsr()


def sigma_x_sum(L: int) -> sp.csr_matrix:
    """Total transverse magnetization operator M_x = sum_i sigma^x_i."""
    dim = 2**L
    total = sp.csr_matrix((dim, dim))
    for i in range(L):
        total = total + _operator_at(SX, L, i)
    return total.tocsr()


def build_tfim_hamiltonian(L: int, J: float = 1.0, h: float = 1.0) -> sp.csr_matrix:
    """Sparse TFIM Hamiltonian on a periodic chain of ``L`` sites."""
    dim = 2**L
    H = sp.csr_matrix((dim, dim))
    # ZZ coupling along the ring (site L-1 couples back to site 0).
    for i in range(L):
        j = (i + 1) % L
        H = H - J * (_operator_at(SZ, L, i) @ _operator_at(SZ, L, j))
    # Transverse field.
    for i in range(L):
        H = H - h * _operator_at(SX, L, i)
    return H.tocsr()


def ground_state(L: int, J: float = 1.0, h: float = 1.0, k: int = 2):
    """Lowest ``k`` eigenpairs of the TFIM Hamiltonian (sorted ascending)."""
    H = build_tfim_hamiltonian(L, J, h)
    vals, vecs = eigsh(H, k=k, which="SA")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def magnetization_z_squared(psi: np.ndarray, L: int) -> float:
    """<M_z^2> = |M_z |psi>|^2 (M_z is Hermitian)."""
    Mpsi = sigma_z_sum(L) @ psi
    return float(np.real(np.vdot(Mpsi, Mpsi)))


def magnetization_x(psi: np.ndarray, L: int) -> float:
    """<M_x> for state ``psi``."""
    return float(np.real(np.vdot(psi, sigma_x_sum(L) @ psi)))


def entanglement_entropy(psi: np.ndarray, L: int, cut: int | None = None) -> float:
    """Von Neumann entanglement entropy of the left ``cut`` sites (default L/2).

    S = -Tr(rho_A ln rho_A), computed from the Schmidt values of the state
    reshaped into a (2^cut, 2^(L-cut)) matrix.
    """
    if cut is None:
        cut = L // 2
    matrix = psi.reshape(2**cut, 2 ** (L - cut))
    schmidt = np.linalg.svd(matrix, compute_uv=False)
    probs = schmidt**2
    probs = probs[probs > 1e-14]
    return float(-np.sum(probs * np.log(probs)))


def scan_tfim(
    L: int = 12,
    h_values: np.ndarray | None = None,
    J: float = 1.0,
) -> list[dict]:
    """Ground-state observables across a scan of the transverse field ``h``.

    For each h: ground-state energy, excitation gap E1 - E0, order parameter
    m_z2, transverse magnetization m_x, and half-chain entanglement entropy.
    """
    if h_values is None:
        h_values = np.linspace(0.2, 2.0, 19)

    rows = []
    for h in h_values:
        vals, vecs = ground_state(L, J=J, h=h, k=2)
        psi0 = vecs[:, 0]
        rows.append(
            {
                "h": float(h),
                "E0": float(vals[0]),
                "E0_per_site": float(vals[0] / L),
                "gap": float(vals[1] - vals[0]),
                "m_z2": float(np.sqrt(magnetization_z_squared(psi0, L)) / L),
                "m_x": float(magnetization_x(psi0, L) / L),
                "S": entanglement_entropy(psi0, L),
            }
        )
    return rows
