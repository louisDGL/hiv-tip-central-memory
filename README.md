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

Two gates have been run. Both came back against the proposal as written — see
**[FINDINGS.md](FINDINGS.md)**.

- `d_M = ρ` survives the human literature, but the inflow term `φ` has **never
  been measured in humans**. The one quantitative analogue is mouse, where
  `φ ≈ ρ` and the design fails by 7×.
- `A = 12` is not supported by the base paper's own MCMC posterior, which puts
  `A = P² ≈ 2.50` (95% 2.07–3.43). The `ε` budget falls from 13.75% to 1.87%,
  and the latent reservoir — the anchor for what is achievable — sits outside
  the bound in 100% of posterior draws.

The project is not sunk by this; it is relocated. The result becomes the
admissible region over `(A, φ)` rather than a bound on `ε`.

## Layout

| Path | |
|---|---|
| [PIPELINE.md](PIPELINE.md) | execution plan, stage by stage |
| [FINDINGS.md](FINDINGS.md) | gate results with the numbers |
| [m2-proposal-feedback.md](m2-proposal-feedback.md) | the review this responds to |
| [src/tip/params.py](src/tip/params.py) | Table 1 — single source of truth |
| [src/tip/model.py](src/tip/model.py) | six-state RHS, equilibrium, threshold |
| [analysis/](analysis/) | the gate scripts |
| [tests/](tests/) | hand-verified numbers, frozen as regressions |

## Running

```bash
pip install -r requirements.txt

python analysis/gate_dm_vs_rho.py     # the φ/ε budget
python analysis/a_from_posterior.py   # A from the paper's posterior
python -m pytest tests/ -q            # 11 anchors
```

`a_from_posterior.py` needs the upstream repository:

```bash
git clone https://github.com/khetanneha/HIV-SIV-TIP-Modeling.git \
          external/HIV-SIV-TIP-Modeling
```

It is LGPL-3.0 and is read for reference only — the model here is reimplemented
from the published equations, not vendored.

## Conventions

1. No number appears twice. If it is in `params.py` it is imported.
2. Every parameter carries a source or the word `ASSUMPTION`.
3. `A` is swept, never fixed. Any plot with `A = 12` hard-coded is wrong by
   construction.
