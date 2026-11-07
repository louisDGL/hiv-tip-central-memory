"""AIM 3 (reframed) - the admissible region over (A, phi), with eps as the value.

The deliverable is no longer a number for eps. Both inputs to that number turn
out to be unmeasured, so the honest object is the region itself:

    phi/rho + eps  <=  (A - 1) * beta_H * V_target / rho

Left panel  : eps_max over (A, phi/rho), with the infeasible wedge hatched.
Right panel : the phi = 0 slice over the full range of A, with the two readings
              of the base paper's posterior marked.

Run:  python analysis/s3_budget_surface.py
Writes: figures/budget_surface.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from tip import BASELINE  # noqa: E402
from tip.boundary import budget, budget_surface  # noqa: E402
from tip.params import PHI_MOUSE  # noqa: E402

# --- palette: single-hue sequential blue, light -> dark --------------------
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INFEASIBLE = "#f0efec"
ACCENT = "#eb6834"       # categorical slot 2, used only for annotation marks

EPS_CLIP = 0.25          # above this the budget stops constraining anything


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.8, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def main() -> int:
    p = BASELINE
    outdir = ROOT / "figures"
    outdir.mkdir(exist_ok=True)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.42},
    )

    # ------------------------------------------------------------------
    # LEFT: the surface
    # ------------------------------------------------------------------
    A_grid = np.logspace(np.log10(1.5), np.log10(60.0), 400)
    phi_frac = np.linspace(0.0, 1.0, 400)          # phi / rho, dimensionless
    AA, FF, EE = budget_surface(p, A_grid, phi_frac * p.rho)

    EEc = np.clip(EE, 0.0, EPS_CLIP)
    EEc = np.ma.masked_where(EE < 0.0, EEc)

    axL.set_facecolor(INFEASIBLE)
    im = axL.contourf(AA, FF / p.rho, EEc, levels=np.linspace(0, EPS_CLIP, 11),
                      cmap=CMAP, extend="max")
    # the boundary itself
    axL.contour(AA, FF / p.rho, EE, levels=[0.0], colors=[INK],
                linewidths=1.6)
    # a few interior contours, labelled
    cs = axL.contour(AA, FF / p.rho, EE, levels=[0.05, 0.1375],
                     colors=[INK_2], linewidths=0.8, linestyles="--")
    # label only the 5% line; the 13.75% line runs through the Table 1 callout
    axL.clabel(cs, levels=[0.05], fmt=lambda v: f"ε={v:.0%}", fontsize=8,
               colors=INK_2)

    # hatch the infeasible wedge, lightly - the region is background, not data
    matplotlib.rcParams["hatch.linewidth"] = 0.5
    matplotlib.rcParams["hatch.color"] = "#cfcec6"
    axL.contourf(AA, FF / p.rho, np.ma.masked_where(EE >= 0.0, EE),
                 levels=[-1e9, 0.0], colors="none", hatches=["///"])
    axL.annotate("infeasible at any ε", xy=(0.30, 0.55), xycoords="axes fraction",
                 fontsize=9.5, color=MUTED, style="italic", ha="center")

    # annotations
    axL.plot([p.A], [0.0], marker="o", ms=7, color=ACCENT, zorder=6,
             markeredgecolor=SURFACE, markeredgewidth=1.4)
    axL.annotate("Table 1\nA=12, φ=0", xy=(p.A, 0.0), xytext=(14, 26),
                 textcoords="offset points", fontsize=8.5, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    axL.axhline(PHI_MOUSE / p.rho, color=ACCENT, lw=1.2, ls=":", zorder=4)
    axL.annotate(f"mouse inflow  φ/ρ = {PHI_MOUSE/p.rho:.2f}  (Gossel 2017)",
                 xy=(0.97, PHI_MOUSE / p.rho), xycoords=("axes fraction", "data"),
                 xytext=(0, -7), textcoords="offset points",
                 fontsize=8.5, color=INK_2, ha="right", va="top")

    A_break = 1.0 + p.mu_ref / (p.beta_H * p.V_target)
    axL.axvline(A_break, color=MUTED, lw=0.9, ls="-.", zorder=4)
    axL.annotate("latent reservoir\nbreakeven  A = 3.77",
                 xy=(A_break, 0.0), xycoords=("data", "axes fraction"),
                 xytext=(5, 40), textcoords="offset points",
                 fontsize=8.5, color=INK_2, ha="left", va="bottom")

    axL.set_xscale("log")
    axL.set_xlim(1.5, 60)
    axL.set_ylim(0, 1)
    axL.set_xticks([2, 3, 5, 10, 20, 40, 60])
    axL.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axL.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axL.set_xlabel("A — TIP output per superinfection", color=INK_2, fontsize=10)
    axL.set_ylabel("φ/ρ — fraction of T$_{CM}$ maintained from outside",
                   color=INK_2, fontsize=10)
    axL.set_title("Admissible fitness cost over the two unmeasured axes",
                  color=INK, fontsize=11, pad=10, loc="left")
    axL.grid(True, color=GRID, lw=0.6, alpha=0.7)
    axL.set_axisbelow(True)
    style_axes(axL)

    cb = fig.colorbar(im, ax=axL, pad=0.025, fraction=0.046)
    cb.set_label("ε$_{max}$ — largest admissible fitness cost", color=INK_2,
                 fontsize=9, labelpad=12)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))

    # ------------------------------------------------------------------
    # RIGHT: the phi = 0 slice, full range of A
    # ------------------------------------------------------------------
    A_full = np.logspace(np.log10(1.5), np.log10(3000.0), 600)
    b = budget(p, A_full)

    axR.plot(A_full, b, color="#2a78d6", lw=2.0, solid_capstyle="round")
    axR.axhline(0.1375, color=MUTED, lw=0.8, ls="--")
    axR.annotate("Table 1: ε ≤ 13.75%", xy=(1.7, 0.1375 * 1.35), fontsize=8.5,
                 color=INK_2)

    for A_mark, label, off, ha in (
        (2.50, "posterior\nread as P\nA ≈ 2.5", (10, 4), "left"),
        (p.A, "Fig 2F\nA = 12", (10, -4), "left"),
        (1452.0, "posterior\nread as log₁₀P\nA ≈ 1452", (-10, 4), "right"),
    ):
        yv = float(budget(p, A_mark))
        axR.plot([A_mark], [yv], marker="o", ms=6.5, color=ACCENT, zorder=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.4)
        axR.annotate(label, xy=(A_mark, yv), xytext=off,
                     textcoords="offset points", fontsize=8.5, color=INK,
                     ha=ha, va="bottom" if off[1] >= 0 else "top")

    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlim(1.5, 3000)
    axR.set_ylim(1e-2, 5e1)
    axR.set_xlabel("A — TIP output per superinfection", color=INK_2, fontsize=10)
    axR.set_ylabel("ε budget at φ = 0", color=INK_2, fontsize=10)
    axR.set_title("The budget is linear in (A−1); the posterior reading\n"
                  "moves it by two orders of magnitude",
                  color=INK, fontsize=11, pad=10, loc="left")
    axR.grid(True, color=GRID, lw=0.6, alpha=0.7)
    axR.set_axisbelow(True)
    axR.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axR.set_yticks([1e-2, 1e-1, 1e0, 1e1])
    axR.set_yticklabels(["1%", "10%", "100%", "1000%"])
    style_axes(axR)

    out = outdir / "budget_surface.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # ---- the same content as numbers, for the table in the paper --------
    print("BUDGET SURFACE")
    print("=" * 62)
    print(f"  written to {out.relative_to(ROOT)}")
    print()
    print("  phi=0 slice")
    print("     A        eps budget")
    for A in (2.0, 2.5, 3.77, 4.0, 8.0, 12.0, 50.0, 1452.0):
        print(f"  {A:8.2f}     {100*float(budget(p, A)):10.4g}%")
    print()
    print(f"  mouse inflow phi/rho = {PHI_MOUSE/p.rho:.3f} exceeds the whole")
    print(f"  budget for any A below {1.0 + PHI_MOUSE/(p.beta_H*p.V_target):.0f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
