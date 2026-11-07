"""The threshold, checked three ways against each other.

The closed form (A-1)*beta_H*V_H > mu is the whole argument of the paper. It is
worth more than an assertion, so it is derived numerically here as well and the
derivations are required to agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tip import BASELINE, PUBLISHED, hiv_only_equilibrium  # noqa: E402
from tip.boundary import (  # noqa: E402
    A_critical,
    R0_closed_form,
    R0_next_gen,
    V_floor,
    budget,
    budget_surface,
    eps_max,
    feasible,
    invasion_rate,
    phi_max,
)

# Parameter sets spanning feasible and infeasible, single and dual route.
CASES = [
    BASELINE,
    BASELINE.with_(eps=0.05),
    BASELINE.with_(eps=0.13),
    BASELINE.with_(phi=1e-3),
    BASELINE.with_(phi=5e-3, eps=0.02),
    BASELINE.with_(A=4.0, Pi=4.0),
    BASELINE.with_(A=50.0, Pi=50.0),
    PUBLISHED,
]


@pytest.mark.parametrize("p", CASES)
def test_next_gen_matches_closed_form(p):
    """With the single route, the NGM radius IS A*p. No wiggle room."""
    q = p.with_(superinfect_IH=False)
    assert R0_next_gen(q) == pytest.approx(R0_closed_form(q), rel=1e-10)


@pytest.mark.parametrize("p", CASES)
def test_invasion_sign_matches_R0(p):
    """R0 > 1 iff the dominant eigenvalue has positive real part."""
    q = p.with_(superinfect_IH=False)
    r0 = R0_closed_form(q)
    lam = invasion_rate(q)
    if abs(r0 - 1.0) < 1e-6:
        pytest.skip("too close to the threshold to compare signs")
    assert (r0 > 1.0) == (lam > 0.0)


@pytest.mark.parametrize("p", CASES)
def test_cross_route_only_helps(p):
    """Dropping V_T -> I_H -> I_D lowers R0, so the published bound is conservative."""
    on = R0_next_gen(p.with_(superinfect_IH=True))
    off = R0_next_gen(p.with_(superinfect_IH=False))
    assert on >= off - 1e-12


def test_closed_form_threshold_equals_inequality():
    """R0 = 1 must coincide with (A-1)*beta_H*V_H = mu."""
    p = BASELINE.with_(superinfect_IH=False)
    V_H = hiv_only_equilibrium(p)[4]
    A = p.Pi * p.N * p.beta_T * p.T_star / p.c
    mu_star = (A - 1.0) * p.beta_H * V_H           # mu that puts R0 exactly at 1
    eps_star = mu_star / p.rho                      # since phi = 0, mu = rho*eps
    q = p.with_(eps=eps_star)
    assert R0_closed_form(q) == pytest.approx(1.0, rel=1e-9)


def test_A_equals_Pi():
    """A = Pi*N*beta_T*T*/c reduces to Pi, which is what makes the file readable."""
    p = BASELINE
    A = p.Pi * p.N * p.beta_T * p.T_star / p.c
    assert A == pytest.approx(p.Pi, rel=1e-12)


# --- the feasible region -----------------------------------------------------


def test_budget_matches_eps_crit():
    """budget() and Params.eps_crit() are the same quantity at phi = 0."""
    assert budget(BASELINE) == pytest.approx(BASELINE.eps_crit(), rel=1e-12)
    assert budget(BASELINE) == pytest.approx(0.1375, rel=1e-9)


def test_phi_and_eps_trade_one_for_one():
    """The wedge is linear: giving up phi/rho buys exactly that much eps."""
    p = BASELINE
    b = budget(p)
    for frac in (0.0, 0.2, 0.5, 0.9):
        phi = frac * b * p.rho
        assert eps_max(p, phi=phi) == pytest.approx(b - frac * b, rel=1e-12)
        assert phi_max(p, eps=b - frac * b) == pytest.approx(phi, abs=1e-15)


def test_feasible_boundary_is_consistent():
    """feasible() must agree with eps_max() on both sides of the edge."""
    p = BASELINE
    for phi in (0.0, 5e-4, 1.5e-3, 2.1e-3, 1e-2):
        e = eps_max(p, phi=phi)
        if e > 1e-9:
            assert feasible(p, phi=phi, eps=e * 0.99)
            assert not feasible(p, phi=phi, eps=e * 1.01 + 1e-9)
        else:
            assert not feasible(p, phi=phi, eps=0.0 + 1e-9)


def test_V_floor_inverts_A_critical():
    """A_critical(mu) is the A at which V_floor(mu) lands on V_target."""
    p = BASELINE
    for mu in (1e-4, 5.2e-4, 2.0625e-3):
        A = A_critical(p, mu=mu)
        assert V_floor(p, mu=mu, A=A) == pytest.approx(p.V_target, rel=1e-9)


def test_reservoir_breakeven_A():
    """The latent reservoir clears the bar only above A = 3.77."""
    assert A_critical(BASELINE, mu=BASELINE.mu_ref) == pytest.approx(3.77, rel=0.01)


def test_budget_surface_shape_and_values():
    """Surface must match the scalar function pointwise."""
    p = BASELINE
    A_grid = np.array([2.0, 12.0, 50.0])
    phi_grid = np.array([0.0, 1e-3, 1e-2])
    AA, PP, EE = budget_surface(p, A_grid, phi_grid)
    assert EE.shape == (len(phi_grid), len(A_grid))
    for i, phi in enumerate(phi_grid):
        for j, A in enumerate(A_grid):
            assert EE[i, j] == pytest.approx(eps_max(p, phi=phi, A=A), rel=1e-12)


def test_mouse_phi_is_infeasible_at_A12():
    """Gossel's mouse inflow kills the design at A = 12, as the gate reports."""
    from tip.params import PHI_MOUSE

    assert eps_max(BASELINE, phi=PHI_MOUSE) < 0.0
