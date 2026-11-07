"""The basic three-state viral dynamics model, and the acute-phase observable
that Fig 4D of the base paper rests on.

    T' = lam - d*T - beta*V*T
    I' = beta*V*T - delta*I
    V' = N*delta*I - c*V

Perelson 1996 / Nowak & May 2000, which is what Weinberger 2003 extends. The
base paper's claim is that this model fits the control animals but cannot
reproduce the TIP-treated animals under ANY parameter combination, the
discriminating statistic being the peak-to-set-point ratio in log10.

That claim is testable rather than quotable, and this module is what tests it.
The parameter ranges swept are the base paper's OWN MCMC priors, read out of
SIV-TIP-MCMC/tipmodel1_InfantPVL_A1M1.py (lnprior) and GlobalparamInfantPVL.py,
so the sweep asks the question on the paper's terms, not on ours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "BasicParams",
    "PRIOR_BOUNDS",
    "basic_rhs",
    "R0_basic",
    "setpoint_analytic",
    "simulate_basic",
    "peak_to_setpoint",
    "sample_priors",
]


@dataclass(frozen=True)
class BasicParams:
    """Units: cells/mL, virions/mL, days."""
    lam: float = 1e5       # cells mL^-1 d^-1
    d: float = 0.02        # d^-1
    beta: float = 3e-8     # mL virion^-1 d^-1
    delta: float = 0.88    # d^-1   infant SHIV, GlobalparamInfantPVL.delta
    N: float = 3162.0      # virions cell^-1  (10^3.5, PARAM.Burst)
    c: float = 23.0        # d^-1   PARAM.c


#: Support of the base paper's own priors. lnprior() in tipmodel1 bounds:
#:   4 < log10(lam) < 6     -9 < log10(k) < -6     1 < log10(n) < 6
#:   36 >= c >= 9           delta >= d > 0
#: d and delta carry Gaussian priors (0.02+-0.005, 0.88+-0.4) truncated by
#: delta >= d > 0; we sweep +-3 sd of those, clipped positive.
PRIOR_BOUNDS = {
    "log10_lam": (4.0, 6.0),
    "log10_beta": (-9.0, -6.0),
    "d": (0.005, 0.035),
    "log10_N": (1.0, 6.0),
    "delta": (0.05, 2.08),
    "c": (9.0, 36.0),
}

#: Observed viral load is RNA copies, two per virion (PARAM.v2RNA = 2).
V_TO_RNA = 2.0


def basic_rhs(t: float, y: np.ndarray, p: BasicParams) -> np.ndarray:
    T, I, V = y
    inf = p.beta * V * T
    return np.array([
        p.lam - p.d * T - inf,
        inf - p.delta * I,
        p.N * p.delta * I - p.c * V,
    ])


def R0_basic(p: BasicParams) -> float:
    """R0 = beta*N*lam/(d*c). Above 1 the infection establishes."""
    return p.beta * p.N * p.lam / (p.d * p.c)


def setpoint_analytic(p: BasicParams) -> float:
    """Endemic equilibrium viral load, virions/mL: V* = (d/beta)*(R0 - 1).

    Exact for this model, so it is used to cross-check the numerics rather
    than to replace them.
    """
    return (p.d / p.beta) * (R0_basic(p) - 1.0)


def simulate_basic(p: BasicParams, V0: float = 0.1, t_end: float = 240.0,
                   n_eval: int = 1200):
    """Integrate from a small inoculum into an uninfected host.

    T(0) = lam/d, the disease-free steady state, which is how the base paper's
    own driver initialises (ivE = [lam/d, 0, 1e-6]).
    """
    y0 = np.array([p.lam / p.d, 0.0, V0])
    sol = solve_ivp(
        basic_rhs, (0.0, t_end), y0, args=(p,), method="LSODA",
        t_eval=np.linspace(0.0, t_end, n_eval), rtol=1e-9, atol=1e-12,
    )
    return sol


def peak_to_setpoint(t: np.ndarray, V: np.ndarray,
                     peak_window: tuple[float, float] = (0.0, 70.0),
                     setpoint_window: tuple[float, float] = (140.0, 240.0),
                     floor: float = 1e-6) -> dict:
    """The Fig 4D statistic: log10(peak) - log10(set point).

    peak      : max viral load inside the acute window
    set point : median viral load inside the late window

    Windows are in DAYS and are reported back so the caller can show the
    sensitivity rather than hide the choice.
    """
    V = np.maximum(np.asarray(V, dtype=float), floor)
    t = np.asarray(t, dtype=float)

    acute = (t >= peak_window[0]) & (t <= peak_window[1])
    late = (t >= setpoint_window[0]) & (t <= setpoint_window[1])
    if not acute.any() or not late.any():
        return {"peak": np.nan, "setpoint": np.nan, "ratio_log10": np.nan}

    peak = float(np.max(V[acute]))
    sp = float(np.median(V[late]))
    return {
        "peak": peak,
        "setpoint": sp,
        "ratio_log10": float(np.log10(peak) - np.log10(sp)),
        "peak_window": peak_window,
        "setpoint_window": setpoint_window,
    }


def sample_priors(n: int, seed: int = 0) -> list[BasicParams]:
    """Latin-hypercube sample over the base paper's own prior support.

    Rejects draws with delta < d, which the paper's lnprior also rejects.
    """
    from scipy.stats import qmc

    keys = list(PRIOR_BOUNDS)
    lo = np.array([PRIOR_BOUNDS[k][0] for k in keys])
    hi = np.array([PRIOR_BOUNDS[k][1] for k in keys])
    raw = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    scaled = qmc.scale(raw, lo, hi)

    out = []
    for row in scaled:
        v = dict(zip(keys, row))
        if v["delta"] < v["d"]:
            continue
        out.append(BasicParams(
            lam=10 ** v["log10_lam"],
            d=v["d"],
            beta=10 ** v["log10_beta"],
            delta=v["delta"],
            N=10 ** v["log10_N"],
            c=v["c"],
        ))
    return out
