"""Invasion threshold, feasible region, and the numerical check on both.

The design question is whether the TIP invades and persists when it is rare,
at the HIV-only equilibrium. M2 section 3 gives the answer in closed form:

    (A - 1) * beta_H * V_H  >  mu,        mu = d_M - rho*(1 - eps)

This module derives the same thing three ways, so that the closed form is a
result rather than an assertion:

  1. `R0_closed_form`   -- A * p, the route I_T -> I_D -> V_T -> I_T
  2. `R0_next_gen`      -- spectral radius of the next-generation matrix
  3. `invasion_rate`    -- spectral abscissa of the linearised TIP subsystem

(1) and (2) must agree exactly when the cross route is off. (3) must agree in
SIGN with both, always: R0 > 1 iff the dominant eigenvalue has positive real
part. `tests/test_boundary.py` enforces all of it.

The feasible region follows from the same inequality. Writing d_M = rho + phi
and substituting mu = phi + rho*eps:

    phi/rho + eps  <=  (A - 1) * beta_H * V_target / rho  =:  budget(A)

so the admissible set in (A, phi, eps) is a simple wedge, and `budget` is the
only function needed to draw it.
"""

from __future__ import annotations

import numpy as np

from .model import hiv_only_equilibrium
from .params import Params

__all__ = [
    "budget",
    "eps_max",
    "phi_max",
    "feasible",
    "V_floor",
    "A_critical",
    "tip_jacobian",
    "invasion_rate",
    "R0_closed_form",
    "R0_next_gen",
    "budget_surface",
]


# ---------------------------------------------------------------------------
# The feasible region. Everything here is closed form.
# ---------------------------------------------------------------------------


def budget(p: Params, A=None, V_target=None):
    """Shared budget B(A) with  phi/rho + eps <= B.  Dimensionless.

    Accepts array-like A and broadcasts.
    """
    A = p.A if A is None else np.asarray(A, dtype=float)
    V = p.V_target if V_target is None else V_target
    return (A - 1.0) * p.beta_H * V / p.rho


def eps_max(p: Params, phi=None, A=None, V_target=None):
    """Largest admissible fitness cost at a given inflow rate.

    Negative means infeasible at any eps: the inflow alone breaks the bound.
    """
    phi = p.phi if phi is None else np.asarray(phi, dtype=float)
    return budget(p, A, V_target) - phi / p.rho


def phi_max(p: Params, eps=None, A=None, V_target=None):
    """Largest admissible inflow rate at a given fitness cost, in d^-1."""
    eps = p.eps if eps is None else np.asarray(eps, dtype=float)
    return p.rho * (budget(p, A, V_target) - eps)


def feasible(p: Params, phi=None, eps=None, A=None, V_target=None):
    """Does (A, phi, eps) sit inside the admissible wedge?"""
    phi = p.phi if phi is None else np.asarray(phi, dtype=float)
    eps = p.eps if eps is None else np.asarray(eps, dtype=float)
    return (phi / p.rho + eps) <= budget(p, A, V_target)


def V_floor(p: Params, mu=None, A=None):
    """Viral load below which the TIP goes extinct, copies/mL."""
    mu = p.mu if mu is None else np.asarray(mu, dtype=float)
    A = p.A if A is None else np.asarray(A, dtype=float)
    return mu / (p.beta_H * (A - 1.0))


def A_critical(p: Params, mu=None, V_target=None):
    """Smallest A that holds the load at V_target for a given mu."""
    mu = p.mu if mu is None else np.asarray(mu, dtype=float)
    V = p.V_target if V_target is None else V_target
    return 1.0 + mu / (p.beta_H * V)


def budget_surface(p: Params, A_grid, phi_grid, V_target=None):
    """eps_max on a grid. Returns (AA, PP, EE) with EE[i, j] for phi_grid[i], A_grid[j].

    Values below zero mark the infeasible region; callers usually mask them.
    """
    AA, PP = np.meshgrid(np.asarray(A_grid, float), np.asarray(phi_grid, float))
    EE = budget(p, AA, V_target) - PP / p.rho
    return AA, PP, EE


# ---------------------------------------------------------------------------
# The same threshold, derived numerically. This is the check, not the claim.
# ---------------------------------------------------------------------------


def tip_jacobian(p: Params, eq=None) -> np.ndarray:
    """Jacobian of the TIP subsystem (I_T, I_D, V_T) at the HIV-only equilibrium.

    The TIP states are zero there, so the subsystem decouples and its Jacobian
    is exactly the linearisation that decides invasion.
    """
    eq = hiv_only_equilibrium(p) if eq is None else eq
    T_s, I_H_s = eq[0], eq[1]
    V_H_s = eq[4]
    force_H = p.beta_H * V_H_s

    # d(I_T)/dt = beta_T*T*V_T - (mu + beta_H*V_H)*I_T
    # d(I_D)/dt = beta_H*V_H*I_T - delta_D*I_D  [+ beta_T*I_H*V_T if cross]
    # d(V_T)/dt = Pi*N*delta_D*I_D - c*V_T
    cross = p.beta_T * I_H_s if p.superinfect_IH else 0.0

    return np.array([
        [-(p.mu + force_H), 0.0,          p.beta_T * T_s],
        [force_H,           -p.delta_D,   cross],
        [0.0,               p.Pi * p.N * p.delta_D, -p.c],
    ])


def invasion_rate(p: Params, eq=None) -> float:
    """Spectral abscissa of the TIP subsystem. Positive means the TIP invades."""
    return float(np.max(np.real(np.linalg.eigvals(tip_jacobian(p, eq)))))


def R0_closed_form(p: Params, eq=None) -> float:
    """A * p, with p the probability of superinfection before loss.

    A      = Pi * N * beta_T * T* / c       TIP output per superinfection
    p      = beta_H*V_H / (beta_H*V_H + mu) superinfection before loss

    Uses the single route I_T -> I_D -> V_T -> I_T, which is what M2 section 3
    derives. delta_D cancels: a dual cell emits Pi*N*delta_D TIP per day and
    lives 1/delta_D days.
    """
    eq = hiv_only_equilibrium(p) if eq is None else eq
    T_s, V_H_s = eq[0], eq[4]
    A = p.Pi * p.N * p.beta_T * T_s / p.c
    force_H = p.beta_H * V_H_s
    return A * force_H / (force_H + p.mu)


def R0_next_gen(p: Params, eq=None) -> float:
    """Spectral radius of the next-generation matrix K = F @ inv(V).

    F holds ONLY the event that starts a new infected cell: a free TIP entering
    a target (and, if the cross route is on, entering an HIV-infected cell).
    Everything else is a transition and belongs in V, INCLUDING the production
    of free TIP by a dual cell.

    That placement matters. Putting virion production in F instead makes a
    "generation" half a cycle, and the spectral radius comes back as the square
    root of the per-cell R0. Both have their threshold at 1, so the biology is
    unchanged, but only this placement reproduces A*p.
    """
    eq = hiv_only_equilibrium(p) if eq is None else eq
    T_s, I_H_s, V_H_s = eq[0], eq[1], eq[4]
    cross = p.beta_T * I_H_s if p.superinfect_IH else 0.0
    force_H = p.beta_H * V_H_s

    F = np.array([
        [0.0, 0.0, p.beta_T * T_s],
        [0.0, 0.0, cross],
        [0.0, 0.0, 0.0],
    ])
    V = np.array([
        [p.mu + force_H, 0.0,                     0.0],
        [-force_H,       p.delta_D,               0.0],
        [0.0,            -p.Pi * p.N * p.delta_D, p.c],
    ])
    K = F @ np.linalg.inv(V)
    return float(np.max(np.abs(np.linalg.eigvals(K))))
