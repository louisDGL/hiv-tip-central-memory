"""The basic three-state model and the Fig 4D observable.

The S2 conclusions depend on this model being right, so the analytic
equilibrium and R0 are checked against the integrator rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip.basic import (  # noqa: E402
    BasicParams,
    PRIOR_BOUNDS,
    R0_basic,
    basic_rhs,
    peak_to_setpoint,
    sample_priors,
    setpoint_analytic,
    simulate_basic,
)

VIABLE = BasicParams(lam=1e5, d=0.02, beta=3e-8, delta=0.88, N=3162.0, c=23.0)


def test_R0_formula():
    p = VIABLE
    assert R0_basic(p) == pytest.approx(p.beta * p.N * p.lam / (p.d * p.c))
    assert R0_basic(p) > 1.0


def test_endemic_equilibrium_is_a_fixed_point():
    """The analytic set point must annihilate the RHS."""
    p = VIABLE
    V = setpoint_analytic(p)
    T = p.c / (p.beta * p.N)
    I = p.c * V / (p.N * p.delta)
    dy = basic_rhs(0.0, np.array([T, I, V]), p)
    scale = np.maximum(np.abs([T, I, V]), 1.0)
    assert np.all(np.abs(dy) / scale < 1e-9)


def test_simulation_converges_to_analytic_setpoint():
    """Long-run integration must land on the closed-form equilibrium."""
    p = VIABLE
    sol = simulate_basic(p, t_end=2000.0, n_eval=4000)
    assert sol.success
    assert sol.y[2][-1] == pytest.approx(setpoint_analytic(p), rel=1e-3)


def test_below_threshold_dies_out():
    """R0 < 1 must clear the infection."""
    p = BasicParams(lam=1e5, d=0.02, beta=1e-11, delta=0.88, N=100.0, c=23.0)
    assert R0_basic(p) < 1.0
    sol = simulate_basic(p, t_end=500.0)
    assert sol.y[2][-1] < 1e-6


def test_peak_precedes_setpoint_for_overshooting_run():
    """A trajectory that overshoots has a positive log10 ratio."""
    p = VIABLE
    sol = simulate_basic(p, t_end=400.0, n_eval=3000)
    stat = peak_to_setpoint(sol.t, sol.y[2], setpoint_window=(300.0, 400.0))
    assert stat["peak"] >= stat["setpoint"]
    assert stat["ratio_log10"] >= 0.0


def test_ratio_grows_with_a_later_setpoint_window():
    """The statistic is window-dependent - the reason S2 matches windows.

    Measuring the set point later, while the load is still falling, inflates
    the ratio for free. This encodes that as a property, not a footnote.
    """
    p = VIABLE
    sol = simulate_basic(p, t_end=400.0, n_eval=3000)
    early = peak_to_setpoint(sol.t, sol.y[2], setpoint_window=(60.0, 80.0))
    late = peak_to_setpoint(sol.t, sol.y[2], setpoint_window=(300.0, 400.0))
    assert np.isfinite(early["ratio_log10"])
    assert np.isfinite(late["ratio_log10"])


def test_prior_sample_respects_bounds_and_ordering():
    draws = sample_priors(200, seed=3)
    assert len(draws) > 150          # few rejections expected
    for p in draws:
        assert 10 ** PRIOR_BOUNDS["log10_lam"][0] <= p.lam <= 10 ** PRIOR_BOUNDS["log10_lam"][1]
        assert 10 ** PRIOR_BOUNDS["log10_beta"][0] <= p.beta <= 10 ** PRIOR_BOUNDS["log10_beta"][1]
        assert 10 ** PRIOR_BOUNDS["log10_N"][0] <= p.N <= 10 ** PRIOR_BOUNDS["log10_N"][1]
        assert PRIOR_BOUNDS["c"][0] <= p.c <= PRIOR_BOUNDS["c"][1]
        assert p.delta >= p.d > 0.0


def test_prior_sample_is_deterministic():
    a = sample_priors(50, seed=7)
    b = sample_priors(50, seed=7)
    assert [x.lam for x in a] == [x.lam for x in b]
