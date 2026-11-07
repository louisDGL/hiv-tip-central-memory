"""AIM 1 - the probability p that a TIP-only cell is superinfected before it
is lost, and the implied R0^TIP. No simulation; this is all closed form.

The base paper's Discussion cites 5-10% of infected cells carrying multiple
proviruses at chronic set point. M1 converts that straight into p and gets
R0^TIP = p*A = 0.6 to 1.2, i.e. the published TIP sitting at about 1 in humans.
That conversion is wrong, and M2 section 1 already says why:

    "That is a prevalence conditioned on infection at one instant, not the
     cumulative probability that a single TIP-only cell is superinfected over
     its life; the two differ by the ratio of compartment lifetimes."

This script does the conversion properly and reports all three readings so the
gap between them is visible.

  reading N (naive)      p = q                         -- what M1 did
  reading M (model)      p = bV/(bV + loss) at the model's own equilibrium
  reading I (inverted)   recover the force of superinfection from q, then
                         apply it to a cell with the right lifetime

Reading I is the defensible one. A newly HIV-infected cell lives Exp(delta) and
is superinfected at rate s, so the prevalence of multiply-infected cells among
infected cells is q = s/(s + delta), hence

    s = q*delta/(1 - q)

and that same force applied to a TIP-only cell with net loss mu gives

    p = s/(s + mu).

The lifetime ratio is the whole story: an HIV-infected cell lives 1/delta = 1.4
days, a TIP-carrying central-memory cell lives 1/mu = up to 336 days. Two
hundred times longer exposure to the same hazard.

Run:  python analysis/a1_superinfection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip import BASELINE, hiv_only_equilibrium  # noqa: E402
from tip.boundary import R0_closed_form  # noqa: E402

#: Base paper Discussion: fraction of infected cells carrying multiple proviruses.
Q_RANGE = (0.05, 0.10)


def force_from_prevalence(q: float, delta: float) -> float:
    """Invert q = s/(s + delta) for the force of superinfection s, in d^-1."""
    return q * delta / (1.0 - q)


def p_from_force(s: float, loss: float) -> float:
    """P(superinfected before loss) for a cell with net loss rate `loss`."""
    return s / (s + loss)


def main() -> int:
    p = BASELINE
    eq = hiv_only_equilibrium(p)
    V_H = eq[4]
    force_model = p.beta_H * V_H

    # The two compartments being compared.
    mu_bar = p.mu_max()          # net loss at the design bound, 2.06e-3 /d
    loss_published = p.d         # short-lived TIP-only cell, 0.02 /d

    print("AIM 1 - SUPERINFECTION PROBABILITY AND R0^TIP")
    print("=" * 70)
    print(f"  model HIV-only equilibrium:  V_H = {V_H:.4g} copies/mL"
          f"   (set point used in Table 1: {p.V0:.4g})")
    print(f"  force of superinfection, model-internal: beta_H*V_H ="
          f" {force_model:.4g} /d")
    print()
    print("  compartment lifetimes")
    print("  " + "-" * 64)
    print(f"    HIV-infected cell     1/delta = {1/p.delta:8.2f} d")
    print(f"    published TIP-only    1/d     = {1/p.d:8.2f} d")
    print(f"    TIP in central memory 1/mu    = {1/mu_bar:8.2f} d"
          f"   at the design bound")
    print(f"    ratio memory : HIV-infected   = {(1/mu_bar)/(1/p.delta):8.1f} x")
    print()

    # ------------------------------------------------------------------
    print("  p UNDER EACH READING")
    print("  " + "-" * 64)
    print("  q      reading          p          R0 = p*A   A_crit = 1/p")
    print("  " + "-" * 64)

    rows = []
    for q in Q_RANGE:
        s = force_from_prevalence(q, p.delta)

        # N: prevalence read directly as the probability
        pN = q
        # M: model-internal force, memory lifetime
        pM = p_from_force(force_model, mu_bar)
        # I: force inverted from q, applied to each compartment
        pI_mem = p_from_force(s, mu_bar)
        pI_pub = p_from_force(s, loss_published)

        for label, pv in (("N naive", pN),
                          ("I published", pI_pub),
                          ("I memory", pI_mem),
                          ("M model", pM)):
            print(f"  {q:.2f}   {label:<15} {pv:7.4f}    {pv*p.A:8.2f}"
                  f"   {1/pv:9.2f}")
            rows.append((q, label, pv))
        print(f"         implied force s = {s:.4g} /d"
              f"   ({s/force_model:.1f}x the model's own)")
        print()

    # ------------------------------------------------------------------
    print("  VERDICT")
    print("  " + "-" * 64)
    pI_lo = p_from_force(force_from_prevalence(Q_RANGE[0], p.delta), mu_bar)
    pI_hi = p_from_force(force_from_prevalence(Q_RANGE[1], p.delta), mu_bar)
    print(f"  Central memory, prevalence-inverted:  p = {pI_lo:.3f} to {pI_hi:.3f}")
    print(f"  R0^TIP at A = {p.A:.0f}:                     "
          f"{pI_lo*p.A:.2f} to {pI_hi*p.A:.2f}")
    print()
    pub_lo = p_from_force(force_from_prevalence(Q_RANGE[0], p.delta), loss_published)
    pub_hi = p_from_force(force_from_prevalence(Q_RANGE[1], p.delta), loss_published)
    print(f"  Published TIP-only cell:              p = {pub_lo:.3f} to {pub_hi:.3f}")
    print(f"  R0^TIP at A = {p.A:.0f}:                     "
          f"{pub_lo*p.A:.2f} to {pub_hi*p.A:.2f}")
    print()
    print(f"  M1's naive reading gave R0 = {Q_RANGE[0]*p.A:.1f} to"
          f" {Q_RANGE[1]*p.A:.1f}, i.e. 'about 1'.")
    print("  Done properly the published construct is NOT marginal in humans.")
    print("  The threshold p* = 1/A = "
          f"{1/p.A:.4f} is cleared by every reading except the naive one.")
    print()

    # ------------------------------------------------------------------
    print("  R0^TIP OVER A   (A is a ceiling - sweep it, never fix it)")
    print("  " + "-" * 64)
    print("      A     R0 (published)   R0 (memory)   above 1?")
    for A in (2.0, 4.0, 8.0, 12.0, 20.0, 50.0):
        r_pub = pub_lo * A
        r_mem = pI_lo * A
        flag = "yes" if min(r_pub, r_mem) > 1.0 else ("memory only"
                                                      if r_mem > 1.0 else "no")
        print(f"  {A:7.1f}   {r_pub:13.2f}   {r_mem:11.2f}   {flag}")
    print()
    print(f"  A_crit (published) = {1/pub_lo:.2f}    A_crit (memory) = {1/pI_lo:.2f}")
    print()

    # Cross-check against the independent implementation in tip.boundary.
    r0_code = R0_closed_form(BASELINE.with_(superinfect_IH=False))
    p_code = force_model / (force_model + BASELINE.mu)
    print("  CROSS-CHECK vs tip.boundary (model-internal force, mu as in Table 1)")
    print(f"    p  = {p_code:.6f}      R0 = {r0_code:.4f}"
          f"   (= p*A = {p_code*p.A:.4f})")
    assert abs(r0_code - p_code * p.A) < 1e-9
    print()

    print("  R0 AT SET POINT IS NOT THE MARGIN THE DESIGN BUYS")
    print("  " + "-" * 64)
    print("  M1 motivates the design by the published TIP sitting at")
    print("  'R0^TIP ~ 1 in humans rather than safely above it'. On the")
    print("  correct conversion it sits at 7.8-9.6, so that motivation does")
    print("  not hold and should be dropped rather than defended.")
    print()
    print("  The design still works, for the OTHER reason - the floor. R0 is")
    print("  evaluated at set point; the floor is where the TIP dies as the")
    print("  load falls. Those are different quantities and only the second")
    print("  one responds to mu:")
    print()
    print("    compartment              mu (/d)     V_floor (copies/mL)")
    print("    " + "-" * 56)
    for label, mu in (("published, loss = d", p.d),
                      ("memory at the bound", mu_bar),
                      ("memory at mu_ref", p.mu_ref)):
        vf = mu / (p.beta_H * (p.A - 1.0))
        print(f"    {label:<24} {mu:.4g}      {vf:10.4g}")
    print()
    print(f"    improvement, published -> mu_ref:"
          f" {p.d/p.mu_ref:.0f}x lower floor")
    print()
    print("  So Aim 1's answer is favourable but it reframes the paper: the")
    print("  argument is about persistence at low load, not about invasion at")
    print("  set point. Say that explicitly in M3.")
    print()

    print("  CAVEATS TO CARRY INTO THE WRITE-UP")
    print("  " + "-" * 64)
    print("  - q is itself a prevalence among SAMPLED infected cells. If the")
    print("    assay enriches for productively infected cells it is biased up.")
    print("  - s inverted from q assumes superinfection is Poisson and")
    print("    independent of cell state. Clustering in lymphoid tissue breaks")
    print("    that and would lower the effective s.")
    print("  - A = 12 is a CEILING from a coculture that arranges")
    print("    superinfection. Every R0 above scales linearly in A.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
