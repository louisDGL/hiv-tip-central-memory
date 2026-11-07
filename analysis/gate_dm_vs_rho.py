"""GATE G0 — does d_M = rho survive contact with the literature?

This runs BEFORE anything in Aim 3. Reviewer's point, 22 Oct 2026:

    "If the central-memory pool is maintained partly by differentiation from
     stem-memory rather than purely by self-renewal, d_M exceeds rho and mu
     has a floor no eps can get under."

The algebra, written once so the paper can quote it.

Let phi be the per-cell differentiation inflow into T_CM from stem-memory,
in d^-1. At steady state the UNTRANSDUCED compartment satisfies

    rho * M + phi * M = d_M * M        =>     d_M = rho + phi

so phi = 0 is exactly the labelled assumption d_M = rho. The TIP-carrying
compartment does not receive that inflow — a differentiating T_SCM cell is
untransduced — so its net loss rate is

    mu = d_M - rho*(1 - eps) = phi + rho*eps.

Two things follow, and the second is the one worth putting in the paper.

1. mu >= phi at any eps. If phi alone exceeds the bound, the design is
   infeasible and no measurement of eps can rescue it. That is the hard
   infeasibility the reviewer is pointing at.

2. More usefully, the bound mu <= (A-1)*beta_H*V_target reads

       phi/rho + eps  <=  (A-1)*beta_H*V_target / rho

   so the inflow fraction and the fitness cost draw on ONE budget, one for
   one. Every percent of the compartment maintained from outside is a percent
   of eps given up. This is a sharper statement than a yes/no gate, and it
   says what the literature number has to be good enough to decide.

Run:  python analysis/gate_dm_vs_rho.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tip import BASELINE  # noqa: E402
from tip.params import PHI_MOUSE  # noqa: E402


def budget(p=BASELINE, A=None, V_target=None) -> float:
    """The total budget shared by phi/rho and eps, dimensionless."""
    return p.mu_max(A, V_target) / p.rho


def eps_max(phi: float, p=BASELINE, A=None, V_target=None) -> float:
    """Largest admissible fitness cost given an inflow rate phi."""
    return budget(p, A, V_target) - phi / p.rho


def main() -> int:
    p = BASELINE
    B = budget(p)

    print("GATE G0 - d_M = rho")
    print("=" * 62)
    print(f"  rho                      {p.rho:.4g} /d   (Macallan 2004)")
    print(f"  A                        {p.A:.4g}")
    print(f"  V_target                 {p.V_target:.4g} copies/mL")
    print(f"  mu_max = (A-1)bV         {p.mu_max():.4g} /d")
    print(f"  net half-life at mu_max  {np.log(2)/p.mu_max():.0f} d")
    print()
    print(f"  shared budget  phi/rho + eps  <=  {B:.4f}   ({100*B:.2f}%)")
    print()

    print("  phi (/d)    phi/rho     eps_max      verdict")
    print("  " + "-" * 56)
    for frac in (0.0, 0.01, 0.05, 0.10, 0.1375, 0.20, 0.50):
        phi = frac * p.rho
        e = eps_max(phi, p)
        verdict = "feasible" if e > 0 else "INFEASIBLE at any eps"
        print(f"  {phi:9.3e}  {frac:8.4f}  {e:10.4f}   {verdict}")
    print()

    # The reference points that fix the scale, from M2 section 3.
    print("  reference points")
    print("  " + "-" * 56)
    print(f"  published TIP-only loss rate d = {p.d:.4g} /d"
          f"  -> {p.d/p.mu_max():.0f}x outside the bar")
    print(f"  latent reservoir mu_ref = {p.mu_ref:.4g} /d"
          f"  -> {p.mu_max()/p.mu_ref:.1f}x inside it")
    print()

    # A is a ceiling, not a measurement: sweep it rather than trusting 12.
    print("  sensitivity to A (A = 12 is a CEILING, Pitchai Fig 2F)")
    print("  " + "-" * 56)
    for A in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
        print(f"  A = {A:5.1f}   mu_max = {p.mu_max(A=A):.4g} /d"
              f"   budget = {100*budget(p, A=A):6.2f}%")
    print()

    print("  VERDICT - gate is OPEN, not PASSED")
    print("  " + "-" * 56)
    print("  The human literature does NOT force d_M > rho, so the assumption")
    print("  survives. The apparent evidence against it is a misreading:")
    print("  Macallan 2004 reports d* = 4.1%/day for CD4 T_CM against")
    print("  p = 1.5%/day, but d* is the disappearance rate of LABELLED cells")
    print("  only. Macallan says so in the same paper - at steady state the")
    print("  population disappearance rate d must equal p. d* > p is the")
    print("  signature of kinetic heterogeneity inside T_CM, not of inflow.")
    print()
    print("  But phi is not thereby zero. It is UNMEASURED:")
    print("    - no human study estimates the T_SCM -> T_CM flux at all")
    print("    - Costa Del Amo 2018, the only human T_SCM kinetic model,")
    print("      does not model T_CM")
    print("    - Gattinoni 2011 shows the pathway qualitatively only")
    print("    - deuterium cannot separate in-situ division from arrival of")
    print("      an already-labelled upstream cell, even in principle")
    print()
    print(f"  The one quantitative analogue is mouse: phi ~ {PHI_MOUSE:.3g} /d")
    print("  (Gossel 2017, naive -> T_CM, busulfan chimeras). At that value:")
    e_mouse = eps_max(PHI_MOUSE, p)
    print(f"    phi/rho = {PHI_MOUSE/p.rho:.2f}   eps_max = {e_mouse:.3f}"
          f"   -> INFEASIBLE by {PHI_MOUSE/p.mu_max():.0f}x at A = 12")
    print()
    print("  So phi = 0 is admissible on human data and phi ~ rho would kill")
    print("  the design outright, and nothing published distinguishes them.")
    print("  That span IS the Aim 3 result. It belongs in the paper as a")
    print("  stated limit, not absorbed into the line 'assume d_M = rho'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
