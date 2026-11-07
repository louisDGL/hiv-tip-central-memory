"""The six-state TIP model and its central-memory extension.

States, per mL of blood:
    T     uninfected target CD4+
    I_H   HIV-infected
    I_T   TIP-only
    I_D   dually infected
    V_H   free HIV
    V_T   free TIP

Equations are Weinberger, Schaffer & Arkin (2003) as used by Pitchai 2024,
with two changes in the I_T equation only:
    - an added self-renewal term  rho*(1-eps)*I_T
    - the loss rate d_M of the central-memory pool in place of the loss rate d
      of a short-lived target cell

Conditional replication is the absent production term in I_T: a TIP-only cell
makes nothing. Setting rho = 0 and d_M = d recovers the published model exactly.
"""

from __future__ import annotations

import numpy as np

from .params import Params

STATES = ("T", "I_H", "I_T", "I_D", "V_H", "V_T")


def rhs(t: float, y: np.ndarray, p: Params) -> np.ndarray:
    """Right-hand side of the six-state system. Signature matches solve_ivp."""
    T, I_H, I_T, I_D, V_H, V_T = y

    inf_H = p.beta_H * V_H          # force of HIV infection
    inf_T = p.beta_T * V_T          # force of TIP infection

    # Second route into I_D: an HIV-infected cell picking up a TIP. Written in
    # M1/M2 section 2, but ABSENT from the published code (expandedHiv.m has no
    # -k*Vt*I term). Switch it off to verify against the paper's own figures.
    cross = inf_T * I_H if p.superinfect_IH else 0.0

    dT = p.lam - p.d * T - inf_H * T - inf_T * T
    dI_H = inf_H * T - p.delta * I_H - cross
    dI_T = inf_T * T + p.rho * (1.0 - p.eps) * I_T - p.d_M * I_T - inf_H * I_T
    dI_D = cross + inf_H * I_T - p.delta_D * I_D
    dV_H = p.N * p.delta * I_H + (1.0 - p.D) * p.N * p.delta_D * I_D - p.c * V_H
    dV_T = p.Pi * p.N * p.delta_D * I_D - p.c * V_T

    return np.array([dT, dI_H, dI_T, dI_D, dV_H, dV_T])


def hiv_only_equilibrium(p: Params) -> np.ndarray:
    """The HIV-only (TIP-free) steady state, used as the linearisation point.

    T* = c / (N * beta_H) exactly. The remaining components follow from
    setting the TIP states to zero.
    """
    T = p.T_star
    # I_H from dT = 0 with V_T = 0:  lam - d*T - beta_H*V_H*T = 0
    # and from dV_H = 0 with I_D = 0: V_H = N*delta*I_H / c
    # substitute: lam - d*T - beta_H*T*(N*delta*I_H/c) = 0
    I_H = (p.lam - p.d * T) * p.c / (p.beta_H * T * p.N * p.delta)
    V_H = p.N * p.delta * I_H / p.c
    return np.array([T, I_H, 0.0, 0.0, V_H, 0.0])


def persists(p: Params, V_H: float | None = None, A: float | None = None) -> bool:
    """Does the TIP invade and persist?

    Linearising the three TIP equations about the HIV-only equilibrium and
    keeping only the route I_T -> I_D -> V_T -> I_T gives

        (A - 1) * beta_H * V_H  >  mu

    Dropping the second route V_T -> I_H -> I_D only raises R0^TIP, so this
    bound is conservative.
    """
    A = p.A if A is None else A
    V_H = hiv_only_equilibrium(p)[4] if V_H is None else V_H
    return (A - 1.0) * p.beta_H * V_H > p.mu


def simulate(p: Params, y0: np.ndarray, t_end: float = 3650.0, n: int = 2000):
    """Integrate the system. Returns the solve_ivp result object.

    Stiff system (c = 30/d against rho = 0.015/d, three orders apart), so LSODA.
    """
    from scipy.integrate import solve_ivp

    return solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        args=(p,),
        method="LSODA",
        dense_output=True,
        t_eval=np.linspace(0.0, t_end, n),
        rtol=1e-8,
        atol=1e-10,
    )
