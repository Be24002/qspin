"""Physics regression tests: every claim in the README is checked here."""

import numpy as np
import pytest

from qspin.classical_ising import exact_tc_2d, run_ising
from qspin.dynamics import quench
from qspin.tfim import entanglement_entropy, ground_state, scan_tfim

# ---------------------------------------------------------------- classical


def test_exact_tc_matches_onsager():
    assert exact_tc_2d(1.0) == pytest.approx(2.269185, abs=1e-4)


def test_metropolis_orders_at_low_temperature():
    result = run_ising(L=16, T=1.5, n_eq=200, n_prod=200, sample_every=4, seed=3)
    assert result["M_abs"] > 0.7  # well inside the ferromagnetic phase


def test_metropolis_disorders_at_high_temperature():
    result = run_ising(L=16, T=5.0, n_eq=200, n_prod=200, sample_every=4, seed=4)
    assert result["M_abs"] < 0.2  # paramagnet


def test_specific_heat_is_positive():
    result = run_ising(L=16, T=2.3, n_eq=200, n_prod=200, sample_every=4, seed=5)
    assert result["C_per_site"] > 0


# ------------------------------------------------------------------ quantum


def test_tfim_h0_energy_is_classical():
    # At h = 0 the ground states are the two all-aligned cats: E0 = -J * (#bonds).
    vals, _ = ground_state(L=8, J=1.0, h=0.0, k=2)
    assert vals[0] == pytest.approx(-8.0, abs=1e-8)


def test_tfim_large_h_energy_is_field_dominated():
    # Deep paramagnet: E0 ~ -h L (ZZ coupling is a tiny perturbation).
    vals, _ = ground_state(L=6, J=1.0, h=100.0, k=2)
    assert vals[0] == pytest.approx(-600.0, rel=1e-3)


def test_cat_state_entanglement_is_ln2():
    # Build the GHZ/cat state explicitly: at h = 0 the ground eigenspace is
    # exactly degenerate and eigsh may return any combination of |all up> and
    # |all down>, so we test the observable on a hand-made cat instead.
    cat = np.zeros(2**6)
    cat[0] = 1.0 / np.sqrt(2.0)  # |000000> (all spins up)
    cat[-1] = 1.0 / np.sqrt(2.0)  # |111111> (all spins down)
    assert entanglement_entropy(cat, 6) == pytest.approx(np.log(2), abs=1e-12)


def test_product_state_entanglement_is_zero():
    up_up_up = np.zeros(2**6)
    up_up_up[0] = 1.0
    assert entanglement_entropy(up_up_up, 6) == pytest.approx(0.0, abs=1e-12)


def test_polarized_state_entanglement_is_zero():
    _, vecs = ground_state(L=6, J=1.0, h=60.0, k=2)
    assert entanglement_entropy(vecs[:, 0], 6) < 0.01


def test_tfim_order_parameter_switches_off():
    rows = scan_tfim(L=10, h_values=np.array([0.3, 3.0]))
    ordered, disordered = rows[0], rows[1]
    assert ordered["m_z2"] > 0.9
    assert disordered["m_z2"] < 0.5


def test_entanglement_peaks_at_critical_point():
    # In a finite chain the ferromagnetic splitting ~ h^L is *smaller* than
    # the critical gap, so the gap minimum sits at the edge of the ordered
    # phase and cannot locate h_c. The entanglement entropy peak does:
    # S ~ (c/3) ln L at criticality (c = 1/2) rises above the ln 2 cat plateau.
    rows = scan_tfim(L=10, h_values=np.linspace(0.5, 1.6, 8))
    h_peak = max(rows, key=lambda r: r["S"])["h"]
    assert 0.7 < h_peak < 1.3


def test_gap_is_small_at_criticality():
    # The excitation gap at h ~ 1 is at least twice smaller than deep in the
    # paramagnet, where Delta -> 2|J - h| (free-fermion result).
    rows = scan_tfim(L=10, h_values=np.array([1.0, 2.0]))
    assert rows[0]["gap"] < 0.5 * rows[1]["gap"]


# ----------------------------------------------------------------- dynamics


def test_quench_preserves_norm():
    out = quench(L=10, t_max=2.0, n_steps=20)
    assert np.allclose(out["norm"], 1.0, atol=1e-8)


def test_quench_melts_the_order():
    out = quench(L=10, h_initial=0.2, h_final=5.0, t_max=4.0, n_steps=40)
    assert out["m_z2"][-1] < 0.7 * out["m_z2"][0]
