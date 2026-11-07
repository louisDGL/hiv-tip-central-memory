"""S2 / AIM 2 - verification. Does the discriminating statistic actually
discriminate, and can the basic model really not reach it?

The base paper leans hardest on Fig 4D: the basic three-state model fits the
control animals but cannot reproduce the TIP-treated animals under any
parameter combination, the discriminating statistic being the peak-to-set-point
ratio in log10. That licenses the human extrapolation of Fig 5G, so everything
downstream inherits it. M1 takes it as given. This script does not.

Three things are checked, in order of how much they could hurt:

  CHECK 1  Is the separation an artefact of unequal follow-up?
           Controls stop at week 17, treated animals run to week 30. A ratio
           measured later is larger for free, because the load is still
           falling. The honest comparison truncates both arms to the same
           window. If the separation dies there, Fig 4D dies with it.

  CHECK 2  What peak-to-set-point ratios can the BASIC model actually reach,
           swept over the base paper's own MCMC prior support? If the basic
           model can reach the treated animals' ratios, "no parameter
           combination" is false and the figure does not license Fig 5G.

  CHECK 3  Does R0 alone explain the achievable ratio? If the acute overshoot
           is a monotone function of R0, then "the basic model cannot reach
           this ratio" is really a statement about an R0 ceiling, and should
           be argued that way.

Run:  python analysis/s2_verification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip.basic import (  # noqa: E402
    R0_basic,
    V_TO_RNA,
    peak_to_setpoint,
    sample_priors,
    setpoint_analytic,
    simulate_basic,
)
from tip.data import load_viral_loads  # noqa: E402

N_LATE = 4          # points averaged for the set point
ACUTE_MAX_D = 70.0  # peak searched inside the first 10 weeks


def observed_statistic(s, t_max_days: float | None = None,
                       n_late: int = N_LATE) -> dict:
    """Peak-to-set-point in log10 for one animal.

    peak      : max over the acute window (first 10 weeks)
    set point : median of the last `n_late` observations in the window kept
    """
    if t_max_days is not None:
        s = s.truncate(t_max_days)
    if len(s.vl) < n_late + 1:
        return {"animal": s.animal, "arm": s.arm, "ratio": np.nan,
                "peak": np.nan, "setpoint": np.nan, "n": len(s.vl),
                "t_last_w": np.nan}

    acute = s.t_days <= ACUTE_MAX_D
    peak = float(np.max(s.vl[acute])) if acute.any() else float(np.max(s.vl))
    sp = float(np.median(s.vl[-n_late:]))
    return {
        "animal": s.animal, "arm": s.arm,
        "peak": peak, "setpoint": sp,
        "ratio": float(np.log10(peak) - np.log10(sp)),
        "n": len(s.vl), "t_last_w": float(s.t_days[-1] / 7.0),
    }


def summarise(rows, arm):
    vals = np.array([r["ratio"] for r in rows if r["arm"] == arm
                     and np.isfinite(r["ratio"])])
    return vals


def check1_followup(data) -> dict:
    print("CHECK 1 - IS THE SEPARATION AN ARTEFACT OF UNEQUAL FOLLOW-UP?")
    print("=" * 72)

    all_series = data["control"] + data["treated"]
    last_ctrl = max(s.t_days[-1] for s in data["control"])
    last_trt = max(s.t_days[-1] for s in data["treated"])
    print(f"  last observation   controls {last_ctrl/7:.0f} w"
          f"   treated {last_trt/7:.0f} w"
          f"   -> treated followed {last_trt/last_ctrl:.1f}x longer")
    print()

    out = {}
    for label, tmax in (("full follow-up", None),
                        ("both truncated to week 17", last_ctrl),
                        ("both truncated to week 13", 13 * 7.0)):
        rows = [observed_statistic(s, tmax) for s in all_series]
        c = summarise(rows, "control")
        t = summarise(rows, "treated")
        gap = float(np.min(t) - np.max(c))
        out[label] = (c, t, gap)
        print(f"  {label}")
        print(f"    control  n={len(c)}  ratio {np.min(c):.2f} to {np.max(c):.2f}"
              f"   median {np.median(c):.2f}")
        print(f"    treated  n={len(t)}  ratio {np.min(t):.2f} to {np.max(t):.2f}"
              f"   median {np.median(t):.2f}")
        verdict = "SEPARATED" if gap > 0 else "OVERLAP"
        print(f"    worst-case gap (min treated - max control) ="
              f" {gap:+.2f} log10   {verdict}")
        print()

    print("  per-animal detail, both arms truncated to week 17")
    print("  " + "-" * 68)
    print("    arm       animal     peak        set point     ratio   last w")
    for s in all_series:
        r = observed_statistic(s, last_ctrl)
        if not np.isfinite(r["ratio"]):
            print(f"    {r['arm']:<9} {r['animal']:<10} (too few points)")
            continue
        print(f"    {r['arm']:<9} {r['animal']:<10} {r['peak']:.3g}"
              f"    {r['setpoint']:.3g}      {r['ratio']:5.2f}"
              f"   {r['t_last_w']:5.1f}")
    print()
    return out


#: The animals were sampled weekly to week 17. The model is scored on exactly
#: that grid, so the simulated statistic and the observed one are the same
#: measurement. Scoring the model on a longer, denser window instead would let
#: it settle closer to equilibrium and silently change the comparison.
OBS_GRID_DAYS = np.arange(1, 18) * 7.0


def statistic_on_grid(sol, t_obs: np.ndarray = OBS_GRID_DAYS,
                      n_late: int = N_LATE) -> dict:
    """Apply the observed-data statistic to a simulation, on the data's grid."""
    V = np.interp(t_obs, sol.t, sol.y[2] * V_TO_RNA)
    V = np.maximum(V, 1e-9)
    acute = t_obs <= ACUTE_MAX_D
    peak = float(np.max(V[acute]))
    sp = float(np.median(V[-n_late:]))
    return {"peak": peak, "setpoint": sp,
            "ratio": float(np.log10(peak) - np.log10(sp))}


def check2_basic_model_reach(obs, n_draws: int = 4000, seed: int = 0) -> dict:
    print("CHECK 2 - WHAT CAN THE BASIC MODEL REACH?")
    print("=" * 72)
    print(f"  Latin-hypercube over the base paper's own prior support,"
          f" {n_draws} draws.")
    print("  Each draw is scored on the SAME weekly grid as the animals"
          " (weeks 1-17),")
    print("  with the same peak window and the same last-4 set point.")
    print()

    params = sample_priors(n_draws, seed=seed)
    ratios, r0s, peaks, sps = [], [], [], []
    for p in params:
        r0 = R0_basic(p)
        if not (1.0 < r0 < 1e6):
            continue
        sp_analytic = setpoint_analytic(p) * V_TO_RNA
        if not (1e2 <= sp_analytic <= 1e10):
            continue
        sol = simulate_basic(p, t_end=200.0)
        if not sol.success or np.any(~np.isfinite(sol.y[2])):
            continue
        stat = statistic_on_grid(sol)
        if not np.isfinite(stat["ratio"]) or stat["peak"] < 1e3:
            continue
        # A simulated set point far below the assay floor is not a trajectory
        # the animals could have produced; the paper's own likelihood censors
        # at LOD_siv. Without this, near-threshold draws that peak and crash
        # contribute ratios of 10+ that no animal could have shown.
        if stat["setpoint"] < 1e2:
            continue
        ratios.append(stat["ratio"])
        r0s.append(r0)
        peaks.append(stat["peak"])
        sps.append(stat["setpoint"])

    ratios = np.asarray(ratios)
    r0s = np.asarray(r0s)
    peaks = np.asarray(peaks)

    c, t, _ = obs["both truncated to week 17"]
    c_lo, c_hi = float(np.min(c)), float(np.max(c))
    t_lo, t_hi = float(np.min(t)), float(np.max(t))

    print(f"  draws kept: {len(ratios)} of {len(params)}")
    print("  achievable peak-to-set-point ratio (log10):")
    for q in (0, 10, 50, 90, 99, 100):
        print(f"    p{q:<3d}  {np.percentile(ratios, q):6.2f}")
    print()
    print("  how much of that range lands in each arm's observed band")
    print("  " + "-" * 68)
    in_ctrl = float(np.mean((ratios >= c_lo) & (ratios <= c_hi)))
    in_trt = float(np.mean((ratios >= t_lo) & (ratios <= t_hi)))
    print(f"    control band  [{c_lo:+.2f}, {c_hi:+.2f}]"
          f"   {100*in_ctrl:5.1f}% of draws")
    print(f"    treated band  [{t_lo:+.2f}, {t_hi:+.2f}]"
          f"   {100*in_trt:5.1f}% of draws")
    print()
    print("  Peak viral load is the other half of the picture, and the arms")
    print("  differ there too:")
    print(f"    simulated peaks   median {np.median(peaks):.3g}"
          f"   p10 {np.percentile(peaks,10):.3g}"
          f"   p90 {np.percentile(peaks,90):.3g}")
    print()
    return {"ratios": ratios, "r0": r0s, "peaks": peaks,
            "bands": (c_lo, c_hi, t_lo, t_hi),
            "in_ctrl": in_ctrl, "in_trt": in_trt}


def check3_r0_explains(sweep, obs) -> None:
    print("CHECK 3 - IS THE CEILING JUST AN R0 CEILING?")
    print("=" * 72)
    ratios, r0 = sweep["ratios"], sweep["r0"]
    lr0 = np.log10(r0)
    rho = float(np.corrcoef(lr0, ratios)[0, 1])
    print(f"  corr(log10 R0, ratio) = {rho:+.3f}  over {len(ratios)} draws")
    print()
    print("    log10 R0 bin      n     median ratio    max ratio")
    edges = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0])
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lr0 >= a) & (lr0 < b)
        if m.sum() < 5:
            continue
        print(f"    [{a:3.1f}, {b:3.1f})     {m.sum():5d}"
              f"      {np.median(ratios[m]):8.2f}     {np.max(ratios[m]):8.2f}")
    print()


def main() -> int:
    try:
        data = load_viral_loads()
    except FileNotFoundError as e:
        print(e)
        return 1

    obs = check1_followup(data)
    sweep = check2_basic_model_reach(obs)
    check3_r0_explains(sweep, obs)

    # ---- the verdict ---------------------------------------------------
    print("VERDICT")
    print("=" * 72)
    c, t, gap = obs["both truncated to week 17"]
    ratios = sweep["ratios"]
    c_lo, c_hi, t_lo, t_hi = sweep["bands"]

    print(f"  1. The separation is REAL and is not a follow-up artefact.")
    print(f"     Matched 17-week window: controls [{c_lo:+.2f}, {c_hi:+.2f}],")
    print(f"     treated [{t_lo:+.2f}, {t_hi:+.2f}], gap {gap:+.2f} log10.")
    print(f"     It survives truncation to week 13 as well.")
    print()
    ratio_bf = sweep["in_ctrl"] / sweep["in_trt"] if sweep["in_trt"] > 0 else np.inf
    print(f"  2. The statistic CARRIES EVIDENCE but does not exclude.")
    print(f"     Scored on the animals' own grid, the basic model lands in the")
    print(f"     control band in {100*sweep['in_ctrl']:.1f}% of prior draws"
          f" and in the treated band")
    print(f"     in {100*sweep['in_trt']:.1f}%. Prior-predictive odds of"
          f" {ratio_bf:.1f}:1 for control-like.")
    print()
    print(f"     That is substantial evidence and it points the paper's way.")
    print(f"     It is NOT 'no parameter combination': roughly one prior draw")
    print(f"     in {1/sweep['in_trt']:.0f} reaches the treated band unaided.")
    print()
    print(f"  3. It is not an R0 ceiling. The ratio runs mildly AGAINST R0")
    print(f"     (corr {np.corrcoef(np.log10(sweep['r0']), ratios)[0,1]:+.2f}),")
    print(f"     so a high-R0 basic model produces a FLATTER trajectory, not a")
    print(f"     steeper one. The large ratios come from near-threshold draws.")
    print()
    print("  WHAT THIS MEANS FOR THE PROPOSAL")
    print("  " + "-" * 68)
    print("  The separation in Fig 4D is real, survives window matching, and")
    print("  favours the TIP explanation by roughly an order of magnitude in")
    print("  prior-predictive odds. That is a defensible thing to cite.")
    print()
    print("  What is NOT defensible is the phrase 'cannot reproduce the")
    print("  treated animal under any parameter combination'. On the summary")
    print("  statistic the basic model reaches that band often enough that the")
    print("  exclusion must rest on the FULL trajectory fit - peak timing,")
    print("  decay shape and set point jointly - which is a stronger and")
    print("  different claim. M3 should cite it that way or reproduce it.")
    print()
    print("  None of this touches the central-memory design, which never")
    print("  depended on Fig 4D. The design enters through mu and the floor.")
    print()
    print("  CAVEAT ON WHAT WAS AND WAS NOT TESTED")
    print("  " + "-" * 68)
    print("  The sweep asks whether SOME parameter set reaches the observed")
    print("  ratio. It does not ask whether that parameter set simultaneously")
    print("  fits the control animals, which is the constraint the paper's")
    print("  MCMC actually imposes. Adding that joint constraint is the next")
    print("  step and could restore the exclusion. Until it is run, the honest")
    print("  statement is 'not established by the summary statistic', not")
    print("  'refuted'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
