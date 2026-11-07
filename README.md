# TIP maintenance in the central-memory compartment

M2 project, Berkeley. A therapeutic interfering particle (TIP) mobilises only
from dually infected cells, so its reproduction scales with wild-type load. In
suppressing HIV it starves its own source, and there is a floor viral load
below which it goes extinct:

```
V_H_floor = µ / (β_H · (A − 1))
```

The published floor is set by `d`, the loss rate of a TIP-only cell. This
project partitions transduced cells into the **central-memory** pool, where an
integrated provirus is copied by cell division rather than packaged. The
numerator becomes the net loss rate of that compartment,

```
µ = d_M − ρ(1 − ε)
```

which can be driven small without touching conditional replication, the
packaging advantage, or the safety argument. The design enters the published
model through `µ` and nothing else.

**Base paper.** Nagoor Pitchai, Tanner, Khetan et al., *Engineered deletions of
HIV replicate conditionally to reduce disease in nonhuman primates*,
Science **385**, eadn5866 (2024). Model inherited from Weinberger, Schaffer &
Arkin, *J. Virol.* **77**, 10028 (2003).

---

## Where things stand

Two inherited claims have been re-derived from the data rather than quoted.
Both results, with their sensitivity analyses and caveats, are in
**[CONCLUSIONS.md](CONCLUSIONS.md)**.

- **`R0^TIP` is 4–11, not "about 1".** Reading the published 5–10% multiply-
  infected prevalence directly as a per-cell probability throws away a 340×
  lifetime ratio. Corrected, the published construct is not marginal in
  humans — so the design's motivation is the **floor** (38× lower), not the
  invasion threshold.
- **Fig 4D's separation is real but does not exclude the basic model.** It
  survives window matching, yet 4 000 draws from the paper's own priors land
  in the treated band 8.9% of the time. Prior-predictive odds 8.3 : 1 —
  substantial evidence, not an exclusion under all parameters.

Two feasibility axes remain unmeasured and are reported as a region rather
than a number: `A`, which is a ceiling from an assay that *arranges*
superinfection, and `φ`, the T_SCM → T_CM inflow, which has never been
quantified in humans.

## Layout

| Path | |
|---|---|
| [CONCLUSIONS.md](CONCLUSIONS.md) | S1 and S2 results, sensitivity, caveats |
| [src/tip/params.py](src/tip/params.py) | Table 1 — single source of truth |
| [src/tip/model.py](src/tip/model.py) | six-state TIP model |
| [src/tip/boundary.py](src/tip/boundary.py) | threshold + feasible region, derived three ways |
| [src/tip/basic.py](src/tip/basic.py) | three-state basic model, Fig 4D observable |
| [src/tip/data.py](src/tip/data.py) | macaque viral-load loader |
| [analysis/](analysis/) | the runnable studies |
| [tests/](tests/) | 58 tests; hand-verified numbers frozen as regressions |

## Running

```bash
pip install -r requirements.txt

python analysis/a1_superinfection.py   # S1: p, R0^TIP, the prevalence conversion
python analysis/gate_dm_vs_rho.py      # the φ/ε budget
python analysis/s3_budget_surface.py   # the (A, φ) region -> figures/
python -m pytest tests/ -q             # 58 tests
```

`s2_verification.py` and `a_from_posterior.py` need the upstream repository
for its data and its posterior:

```bash
git clone https://github.com/khetanneha/HIV-SIV-TIP-Modeling.git \
          external/HIV-SIV-TIP-Modeling

python analysis/s2_verification.py     # S2: does Fig 4D's statistic discriminate
python analysis/a_from_posterior.py    # A read off the paper's own MCMC
```

It is LGPL-3.0 and is read for reference only — the model here is reimplemented
from the published equations, not vendored.

## Conventions

1. No number appears twice. If it is in `params.py` it is imported.
2. Every parameter carries a source or the word `ASSUMPTION`.
3. `A` is swept, never fixed. Any plot with `A = 12` hard-coded is wrong by
   construction.
