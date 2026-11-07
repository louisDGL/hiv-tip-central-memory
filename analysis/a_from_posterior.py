"""A from the base paper's posterior, and the tension it exposes.

M2 section 2 derives   A = Pi*N*beta_T*T*/c = Pi*beta_T/beta_H = Pi = P^2,
so a posterior over P is a posterior over A. The cloned repo has one:
external/HIV-SIV-TIP-Modeling/HIV-TIP/AllPsAndDs.txt, 4660 rows of "col1,col2".

THE COLUMN CONVENTION IS NOT OBVIOUS AND THE REPO CONTRADICTS ITSELF.

  predictHIV_M1_V6.m:19   pdvals = [ round(10.^vv(:,1),2) , round(vv(:,2),2) ]
                          -> treats column 1 as log10(P)
  predictHIV_M1_V6.m:79   pvaal.P = rho_scaling * pClusterVals(tip)
                          with pClusterVals = vv(:,1), rho_scaling = 3
                          -> uses the RAW column as P, times 3
  predictHIV_M1_V6.m:21   commented-out literal  tipvals = [37.73, 0.037; ...]
                          -> 37.73 = 10^1.577, consistent with log10(P)

  SIV-TIP-MCMC: the sampled parameter is `rho` = log10(P) (tipmodel1 line 86,
  P = 10**rho1) with prior bound 0 < rho <= 2, i.e. P in (1, 100].

Column 1 has median 1.581, which sits inside (0, 2] as a log10 and inside
(1, 100] as a raw P. Both readings are numerically admissible, so this script
reports BOTH and does not pick one. Resolving it is a one-email question to
the authors and it is worth asking before M3.

Run:  python analysis/a_from_posterior.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip import BASELINE  # noqa: E402

POSTERIOR = ROOT / "external" / "HIV-SIV-TIP-Modeling" / "HIV-TIP" / "AllPsAndDs.txt"


def load_posterior(path: Path = POSTERIOR) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",")
    return raw[:, 0], raw[:, 1]


def summarise(x: np.ndarray) -> tuple[float, float, float]:
    return (float(np.percentile(x, 2.5)), float(np.median(x)),
            float(np.percentile(x, 97.5)))


def budget(A: np.ndarray | float, p=BASELINE) -> np.ndarray | float:
    """Shared budget for phi/rho + eps, dimensionless."""
    return (np.asarray(A) - 1.0) * p.beta_H * p.V_target / p.rho


def report(name: str, A: np.ndarray, p=BASELINE) -> None:
    lo, med, hi = summarise(A)
    b = budget(A, p)
    blo, bmed, bhi = summarise(b)
    mu_max = (A - 1.0) * p.beta_H * p.V_target
    frac_fail = float(np.mean(p.mu_ref > mu_max))

    print(f"  {name}")
    print(f"    A            median {med:10.4g}   95% [{lo:.4g}, {hi:.4g}]")
    print(f"    eps budget   median {100*bmed:9.3g}%  95% [{100*blo:.3g}%, {100*bhi:.3g}%]")
    print(f"    vs M2 budget {100*p.eps_crit():.2f}%  ->  factor {bmed/p.eps_crit():.3g}")
    print(f"    mu_ref above the bar in {100*frac_fail:.1f}% of draws")
    print()


def main() -> int:
    p = BASELINE
    col1, col2 = load_posterior()

    print("POSTERIOR FILE")
    print("=" * 70)
    print(f"  rows {len(col1)}, unique {len(np.unique(np.c_[col1, col2], axis=0))}")
    print(f"  col1  median {np.median(col1):.4f}  range [{col1.min():.3f}, {col1.max():.3f}]")
    print(f"  col2  median {np.median(col2):.4f}  range [{col2.min():.3f}, {col2.max():.3f}]")
    print()
    print("  col2 is D in the repo convention (HIV output from a dual cell")
    print(f"  scales as D). M2's D is the inhibition, so D_M2 = 1 - D_repo")
    print(f"  = {1-np.median(col2):.4f} against Table 1's {p.D}.")
    print()

    print("A UNDER EACH READING OF COLUMN 1")
    print("=" * 70)
    report("reading 1 -- col1 = log10(P), so A = P^2 = 10^(2*col1)",
           10.0 ** (2.0 * col1))
    report("reading 2 -- col1 = P directly, so A = P^2", col1**2)

    print("WHAT M2 ASSUMES")
    print("=" * 70)
    print(f"  Table 1 sets Pi = A = {p.A:.0f}, taken from Pitchai Fig 2F, and")
    print(f"  notes Pi = P^2 gives P = {np.sqrt(p.A):.2f}, inside Weinberger's 1-100.")
    print()
    print("  Under reading 1 the fit puts P near"
          f" {10**np.median(col1):.1f}, also inside 1-100,")
    print(f"  but Pi = {10**(2*np.median(col1)):.0f} -- about"
          f" {10**(2*np.median(col1))/p.A:.0f}x above Table 1.")
    print()

    print("THE POINT, EITHER WAY")
    print("=" * 70)
    print("  M2's identity A = Pi forces one number to serve two roles:")
    print("    - A, TIP output per superinfection, read off Fig 2F  -> 12")
    print("    - Pi, the packaging advantage the MCMC fits          -> see above")
    print("  Under reading 1 these differ by orders of magnitude. That gap is")
    print("  a real result about the base paper, not about your extension, and")
    print("  it cuts in your favour: a larger A makes the bound LOOSER, so")
    print("  Table 1's eps <= 13.7% would be conservative rather than generous.")
    print()
    print("  The reviewer's point stands regardless and is the safe move:")
    print("  A = 12 is a ceiling from an assay that ARRANGES superinfection.")
    print("  Report the budget as a curve over A. Do not stake the paper on")
    print("  any single value of A until the column convention is confirmed.")
    print()

    print("SENSITIVITY -- the figure to actually publish")
    print("=" * 70)
    print("     A        mu_max (/d)   half-life (d)   eps budget")
    print("  " + "-" * 60)
    for A in (2.0, 4.0, 8.0, 12.0, 50.0, 200.0, 1000.0):
        m = (A - 1.0) * p.beta_H * p.V_target
        print(f"  {A:8.1f}   {m:11.4g}   {np.log(2)/m:11.0f}   {100*budget(A):10.4g}%")
    print()
    A_break = 1.0 + p.mu_ref / (p.beta_H * p.V_target)
    print(f"  A at which the latent reservoir itself clears the bar: {A_break:.2f}")
    print(f"  Below that, mu_ref alone breaks the bound at eps = 0, phi = 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
