# Conclusions — S1 and S2

Two questions inherited from the base paper, re-derived here rather than
quoted. Both results are reproducible from this repository; the commands are
at the bottom.

Each conclusion is stated at the confidence the evidence supports. Where a
check was not run, that is said rather than glossed.

---

## S1 — The superinfection probability, and what `R0^TIP` is worth

### The conclusion

**The published TIP is not marginal in humans.** `R0^TIP` sits between **4 and
11** across every plausible input, against a threshold of 1. The figure of
"about 1" that motivated the central-memory design comes from a single
arithmetic step that does not survive inspection.

### Where "about 1" came from

The base paper's Discussion cites **5–10% of infected cells carrying multiple
proviruses** at chronic set point. Reading that prevalence `q` directly as the
per-cell probability `p` gives `R0^TIP = q·A = 0.6` to `1.2`.

But `q` is a **prevalence conditioned on infection at one instant**, and the
threshold needs the **cumulative probability that one TIP-only cell is
superinfected over its whole life**. Those differ by the ratio of compartment
lifetimes, and that ratio is large.

At steady state an infected cell's age is `Exp(δ)`, so

```
q = s/(s + δ)        ⟹        s = q·δ/(1 − q)
```

recovers the force of superinfection `s`. Applying that same force to a cell
with net loss rate `µ`:

```
p = s/(s + µ)
```

An HIV-infected cell lives `1/δ = 1.4` days. A TIP-carrying central-memory
cell lives up to `1/µ = 485` days — **340× longer exposure to the same
hazard**. Reading `p = q` throws that factor away.

### The numbers

| reading | `p` | `R0^TIP` at `A = 12` |
|---|---|---|
| naive, `p = q` | 0.05 – 0.10 | **0.6 – 1.2** |
| published TIP-only cell (`µ = d`) | 0.65 – 0.80 | **7.8 – 9.6** |
| central memory at the bound | 0.95 – 0.97 | **11.4 – 11.7** |

### Sensitivity — why this is solid

Swept well past the cited ranges, `q ∈ [0.02, 0.20]` and `δ ∈ [0.5, 1.0]`,
for the **published** construct at `A = 12`:

| `δ` \ `q` | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|
| 0.50 | 4.05 | 6.82 | 8.82 | 10.34 |
| 0.70 | 5.00 | 7.78 | 9.55 | 10.77 |
| 0.88 | 5.68 | 8.38 | 9.96 | 11.00 |
| 1.00 | 6.06 | 8.70 | 10.17 | 11.11 |

Worst corner `R0 = 4.05`, and even there `R0 > 1` for any `A > 2.96`. There is
no corner of the input space where the published construct is marginal.

### What this costs, and what it does not

It removes one motivation for the central-memory design. The design should not
be sold as *"the published TIP sits at `R0 ≈ 1`, we push it above"* — it does
not sit there.

**What survives is stronger: the floor.** `R0` is evaluated at set point; the
floor is where the TIP goes extinct *as the load falls*. Only the second
responds to `µ`:

| compartment | `µ` (/d) | `V_floor` (copies/mL) |
|---|---|---|
| published, loss = `d` | 2.0×10⁻² | 9 697 |
| memory at the bound | 2.1×10⁻³ | 1 000 |
| memory at `µ_ref` (latent reservoir) | 5.2×10⁻⁴ | 252 |

**38× lower floor**, which is the whole point of moving the provirus into a
self-renewing compartment. The argument is about persistence at low load, not
invasion at set point.

### Caveats carried forward

- `q` is a prevalence among **sampled** infected cells. An assay enriching for
  productively infected cells biases it upward.
- The inversion assumes superinfection is Poisson and independent of cell
  state. Clustering in lymphoid tissue would lower the effective `s`.
- Every `R0` above scales linearly in `A`, which is a ceiling, not a
  measurement.

---

## S2 — Does the Fig 4D statistic do the work asked of it?

The base paper leans hardest on Fig 4D: the basic three-state model fits the
control animals but *"cannot reproduce the treated animal under any parameter
combination"*, the discriminating statistic being the **peak-to-set-point
ratio in log₁₀**. That claim licenses the human extrapolation of Fig 5G, so
anything built on top of it inherits the claim.

Three checks, using the paper's own data (3 controls, 4 treated infant
macaques) and its own MCMC priors.

### Check 1 — the separation is real

Controls were followed to week 17, treated animals to week 30. A ratio
measured later is larger for free, because the load is still falling, so the
arms must be compared on a matched window.

| window | controls | treated | gap |
|---|---|---|---|
| full follow-up | −0.18 to 0.61 | 1.27 to 3.37 | **+0.67** |
| both truncated to week 17 | −0.18 to 0.61 | 1.02 to 2.82 | **+0.41** |
| both truncated to week 13 | −0.37 to 0.57 | 0.86 to 2.68 | **+0.29** |

**The separation survives window matching.** It narrows, but every treated
animal stays above every control. This part of Fig 4D holds.

Worth noting what the controls actually do: their ratios sit at or **below
zero** — the viral load was still rising at the end of follow-up. In these
infant macaques no set point was established within 17 weeks.

### Check 2 — but the statistic does not exclude the basic model

4 000 Latin-hypercube draws over the base paper's own prior support
(`lnprior` in `tipmodel1_InfantPVL_A1M1.py`), each scored **on the animals'
own weekly grid** with the identical peak window and last-4 set point:

| band | fraction of prior draws |
|---|---|
| control band `[−0.18, +0.61]` | **74.2%** |
| treated band `[+1.02, +2.82]` | **8.9%** |

Prior-predictive odds of **8.3 : 1** in favour of control-like.

That is substantial evidence, and it points the paper's way. It is **not**
"no parameter combination" — roughly **one prior draw in eleven** reaches the
treated band with no TIP at all.

### Check 3 — and it is not an R₀ ceiling

`corr(log₁₀ R₀, ratio) = −0.37`. The ratio runs *against* `R₀`: a high-`R₀`
basic model produces a **flatter** trajectory, not a steeper one. The large
ratios come from near-threshold draws. So the exclusion cannot be restated as
a bound on `R₀`.

### The conclusion

**The separation in Fig 4D is real and favours the TIP explanation by about an
order of magnitude in prior-predictive odds. That is defensible to cite.**

**What is not defensible is the phrase "under any parameter combination."** On
the summary statistic the basic model reaches the treated band often enough
that the exclusion must rest on the **full trajectory fit** — peak timing,
decay shape and set point jointly — which is a stronger and different claim
than the one usually quoted.

### What was not tested

The sweep asks whether *some* parameter set reaches the observed ratio. It does
**not** ask whether that same parameter set simultaneously fits the control
animals, which is the constraint the paper's MCMC actually imposes. Adding the
joint constraint is the obvious next step and **could restore the exclusion**.

Until that is run, the honest statement is **"not established by the summary
statistic"** — not "refuted".

---

## What follows for the design

Neither result damages the central-memory extension, because the extension
never depended on either. It enters the published model through `µ` alone and
acts on the floor:

```
V_floor = µ / (β_H·(A − 1)),      µ = d_M − ρ(1 − ε) = φ + ρ·ε
```

What changes is the framing of the write-up:

1. Drop *"the published TIP sits at `R0 ≈ 1`"*. It does not. Motivate the
   design by the **floor**, which improves 38-fold, not by the invasion
   threshold, which was never the problem.
2. Cite Fig 4D as the authors' full-trajectory claim, not as a verified
   summary-statistic exclusion — or reproduce the joint fit and cite that.

---

## Reproducing

```bash
pip install -r requirements.txt

# S1 - needs nothing beyond this repository
python analysis/a1_superinfection.py

# S2 - needs the upstream data
git clone https://github.com/khetanneha/HIV-SIV-TIP-Modeling.git \
          external/HIV-SIV-TIP-Modeling
python analysis/s2_verification.py

python -m pytest tests/ -q      # 58 tests
```

The upstream repository is read for data and reference only. The models here
are reimplemented from the published equations.
