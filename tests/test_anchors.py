"""Regression anchors: numbers that were checked by hand and must not drift.

The five values below were independently verified in the 22 Oct 2026 review.
If a refactor breaks one of these, the refactor is wrong, not the number.

Run:  python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tip import BASELINE, PUBLISHED, hiv_only_equilibrium, rhs  # noqa: E402


def test_target_equilibrium():
    """T* = c / (N beta_H) = 8e5 cells/mL, matching Weinberger Table 1."""
    assert BASELINE.T_star == pytest.approx(8e5, rel=1e-12)


def test_A_reduces_to_packaging_advantage():
    """A = Pi N beta_T T*/c reduces to Pi beta_T/beta_H = Pi, since beta_T = beta_H."""
    assert BASELINE.A_derived == pytest.approx(BASELINE.Pi, rel=1e-12)


def test_mu_bound():
    """mu <= (A-1) beta_H V_target = 2.1e-3 /d. VERIFIED 22 Oct 2026."""
    assert BASELINE.mu_max() == pytest.approx(2.0625e-3, rel=1e-9)
    assert BASELINE.mu_max() == pytest.approx(2.1e-3, rel=0.02)


def test_eps_bound():
    """eps <= 13.7% under d_M = rho. VERIFIED 22 Oct 2026."""
    assert BASELINE.eps_crit() == pytest.approx(0.1375, rel=1e-9)


def test_threshold_superinfection_probability():
    """p* = 1/A = 8.3%, independent of the loss rate."""
    assert 1.0 / BASELINE.A == pytest.approx(0.08333, rel=1e-3)


def test_net_half_life_at_bound():
    """At mu_max the compartment half-life is 336 d, about 11 months."""
    assert np.log(2) / BASELINE.mu_max() == pytest.approx(336.0, rel=0.01)


# --- structural checks on the model itself -----------------------------------


def test_published_case_recovers_short_lived_loss_rate():
    """rho = 0 and d_M = d must reproduce the published TIP-only equation."""
    assert PUBLISHED.rho == 0.0
    assert PUBLISHED.d_M == pytest.approx(PUBLISHED.d, rel=1e-12)
    assert PUBLISHED.mu == pytest.approx(PUBLISHED.d, rel=1e-12)


def test_delta_D_cancels_value():
    """delta_D = (1-D) delta = 0.14 /d."""
    assert BASELINE.delta_D == pytest.approx(0.14, rel=1e-12)


def test_hiv_only_equilibrium_is_a_fixed_point():
    """rhs must vanish at the HIV-only equilibrium."""
    y = hiv_only_equilibrium(BASELINE)
    dy = rhs(0.0, y, BASELINE)
    scale = np.maximum(np.abs(y), 1.0)
    assert np.all(np.abs(dy) / scale < 1e-9)


def test_mu_decomposition():
    """mu = phi + rho*eps, the identity the gate script rests on."""
    p = BASELINE.with_(phi=0.003, eps=0.04)
    assert p.mu == pytest.approx(p.phi + p.rho * p.eps, rel=1e-12)


def test_inflow_alone_can_break_the_bound():
    """Large enough phi makes eps_crit negative: infeasible at any eps."""
    assert BASELINE.with_(phi=0.5 * BASELINE.rho).eps_crit() < 0.0
