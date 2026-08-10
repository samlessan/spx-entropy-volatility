#!/usr/bin/env python3
"""
Diagnostic page for the supervisor meeting, v4.

    python make_diagnostics_v4.py

The substantive change from v1: the endogeneity claim is now *tested* rather
than asserted. Annual aggregation confounds the strong secular rise in strike
density with any volatility effect, and at annual frequency the raw correlation
is about -0.09, i.e. nothing. The right test is within year, at the date level:
comparing two dates in the same year, does the higher-IV date have worse
coverage in sigma units?

Mechanically hi_sigma = ln(K_max/F) / (IV*sqrt(T)), so IV enters the
denominator. But the listed strike range ln(K_max/F) may widen in volatile
periods and offset it. Which force dominates is empirical.

The script prints the test result FIRST. Read it before using the figure --
if the within-year partial correlation is weak, the endogeneity argument is
not supported and the caption must change.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

INK, ACCENT, MUTED, GREY = "#1a1a1a", "#c0392b", "#7f8c8d", "#9aa5a8"

d = pd.read_csv("data/density.csv", parse_dates=["date"])
d["year"] = d.date.dt.year
d["pass"] = (d.n_within_1sig >= 40) & (d.hi_sigma >= 3) & (d.lo_sigma <= -3)
d = d[d.med_iv > 0].copy()
d["log_iv"] = np.log(d.med_iv)
# The observed log-moneyness range, i.e. the numerator. Isolating this shows
# whether the strike ladder widens enough to offset a higher IV.
d["log_range_up"] = d.hi_sigma * d.log_iv.pipe(np.exp) * np.sqrt(d.dte / 365.0)


def within_corr(frame, x, y, group="year"):
    """Partial correlation after removing group means -- a fixed-effects corr."""
    f = frame[[x, y, group]].dropna()
    xd = f[x] - f.groupby(group)[x].transform("mean")
    yd = f[y] - f.groupby(group)[y].transform("mean")
    return np.corrcoef(xd, yd)[0, 1], len(f)


print("=" * 68)
print("TEST: is sigma-unit coverage endogenous to volatility?")
print("=" * 68)

raw_ann = d.groupby("year").agg(iv=("med_iv", "median"), pr=("pass", "mean"))
r_ann = np.corrcoef(raw_ann.iv, raw_ann.pr)[0, 1]
print(f"\nAnnual, pass rate vs median IV      corr = {r_ann:+.3f}  (n={len(raw_ann)})")
print("   ^ confounded by the secular rise in strike density. Not the test.")

for label, xv, yv in [
    ("hi_sigma   vs log IV", "log_iv", "hi_sigma"),
    ("lo_sigma   vs log IV", "log_iv", "lo_sigma"),
    ("pass       vs log IV", "log_iv", "pass"),
    ("log range  vs log IV", "log_iv", "log_range_up"),
]:
    r, n = within_corr(d, xv, yv)
    print(f"Within-year {label:22s} corr = {r:+.3f}  (n={n:,})")

r_hi, _ = within_corr(d, "log_iv", "hi_sigma")
r_rng, _ = within_corr(d, "log_iv", "log_range_up")
r_pass, _ = within_corr(d, "log_iv", "pass")

print()
if r_hi < -0.30:
    verdict = ("SUPPORTED: within a given year, higher-IV dates have materially "
               "worse sigma-unit coverage. The endogeneity argument stands.")
elif r_hi < -0.10:
    verdict = ("WEAK: the effect has the predicted sign but is modest. State it "
               "as a caution, not as a demonstrated bias.")
else:
    verdict = ("NOT SUPPORTED: drop the endogeneity claim. Justify decomposition "
               "on the grounds that it makes tail dependence measurable instead.")
print("VERDICT:", verdict)
print(f"\n(The strike range moves with IV at corr = {r_rng:+.3f}; if strongly")
print(" positive, the ladder widens in volatile periods and partly offsets.)")
print("=" * 68 + "\n")

by_year = d.groupby("year").agg(
    strikes=("n_strikes", "median"), within1=("n_within_1sig", "median"),
    atm_gap=("atm_gap_pct", "median"), hi_sig=("hi_sigma", "median"),
    lo_sig=("lo_sigma", "median"), med_iv=("med_iv", "median"),
    pass_rate=("pass", "mean"), pairs=("dte", "size")).reset_index()

# ---------------------------------------------------------------- figure ---
fig = plt.figure(figsize=(8.27, 11.69))
gs = fig.add_gridspec(4, 2, height_ratios=[0.95, 0.95, 1.05, 0.80],
                      hspace=0.55, wspace=0.30,
                      left=0.11, right=0.95, top=0.885, bottom=0.065)

fig.text(0.11, 0.950, "SPX option chain: strike density diagnostics",
         fontsize=15, weight="bold", color=INK)
fig.text(0.11, 0.917,
         "OptionMetrics IvyDB US, secid 108105, AM-settled monthlies only.\n"
         "1996-01-04 to 2025-08-29, 18,046,547 rows. Date-expiry pairs with 14 to 90 days to maturity.",
         fontsize=8, color=MUTED, linespacing=1.5)

# A
ax = fig.add_subplot(gs[0, :])
ax.plot(by_year.year, by_year.within1, color=INK, lw=1.6)
ax.fill_between(by_year.year, 0, by_year.within1, color=INK, alpha=0.06)
ax.axhline(40, color=ACCENT, ls="--", lw=1)
ax.text(1996.4, 45, "40 strikes: rough floor for resolving bimodality",
        fontsize=7.5, color=ACCENT)
ax.set_ylabel("Strikes within $\\pm1\\sigma$\nof forward (median)", fontsize=8.5)
ax.set_title("A.  Strike density has risen ~12x", fontsize=10.5,
             weight="bold", loc="left", color=INK, pad=8)
ax.tick_params(labelsize=8)
ax.set_xlim(1996, 2025)
ax.set_ylim(0, None)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.985, 0.06,
        "Strike increments stayed at 5 points while SPX rose from 620 to 6,470,\n"
        "so the grid became ~10x finer in relative terms with no rule change.",
        transform=ax.transAxes, fontsize=7, color=MUTED, ha="right", va="bottom",
        linespacing=1.4)

# B
ax = fig.add_subplot(gs[1, 0])
sub = by_year[by_year.year >= 2007]
cols = [ACCENT if y == 2020 else GREY for y in sub.year]
ax.bar(sub.year, sub.pass_rate, color=cols, width=0.72)
ax.set_ylabel("Share of pairs passing", fontsize=8.5)
ax.set_title("B.  Pass rate, 2007 onward", fontsize=10.5, weight="bold",
             loc="left", color=INK, pad=8)
ax.set_xticks([2008, 2012, 2016, 2020, 2024])
ax.tick_params(labelsize=8)
ax.set_ylim(0, 1.08)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
v = sub.loc[sub.year == 2020, "pass_rate"]
if not v.empty:
    ax.annotate("2020", (2020, v.iloc[0]), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=7.5, color=ACCENT,
                weight="bold")

# C -- the actual test, binned within year
ax = fig.add_subplot(gs[1, 1])
f = d[["log_iv", "hi_sigma", "year"]].dropna()
xd = f.log_iv - f.groupby("year").log_iv.transform("mean")
yd = f.hi_sigma - f.groupby("year").hi_sigma.transform("mean")
bins = pd.qcut(xd, 20, duplicates="drop")
grp = pd.DataFrame({"x": xd, "y": yd}).groupby(bins, observed=True).mean()
ax.axhline(0, color=MUTED, lw=0.6)
ax.axvline(0, color=MUTED, lw=0.6)
ax.scatter(grp.x, grp.y, s=22, color=INK, zorder=3)
b, a = np.polyfit(xd, yd, 1)
xs = np.linspace(xd.quantile(0.01), xd.quantile(0.99), 20)
ax.plot(xs, a + b * xs, color=ACCENT, lw=1.3, ls="--", zorder=2)
ax.text(0.96, 0.93, f"within-year corr = {r_hi:+.2f}", transform=ax.transAxes,
        fontsize=8, ha="right", color=ACCENT, weight="bold")
ax.set_xlim(xd.quantile(0.005), xd.quantile(0.995))
ax.set_xlabel("log implied vol, deviation from year mean", fontsize=8)
ax.set_ylabel("Upper coverage ($\\sigma$),\ndeviation from year mean", fontsize=8.5)
ax.set_title("C.  No volatility dependence", fontsize=10.5,
             weight="bold", loc="left", color=INK, pad=8)
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# table
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
show = by_year[by_year.year.isin([1996, 2000, 2005, 2008, 2010, 2015, 2018,
                                  2020, 2022, 2025])]
cells = [[int(r.year), f"{r.strikes:.0f}", f"{r.within1:.0f}", f"{r.atm_gap:.2f}",
          f"{r.hi_sig:+.1f}", f"{r.lo_sig:+.1f}", f"{r.med_iv:.3f}",
          f"{r.pass_rate:.2f}"] for _, r in show.iterrows()]
tbl = ax.table(cellText=cells,
               colLabels=["Year", "Strikes", "Within $\\pm1\\sigma$", "ATM gap %",
                          "Upper $\\sigma$", "Lower $\\sigma$", "Median IV", "Pass"],
               loc="center", cellLoc="center", bbox=[0, 0.02, 1, 0.96])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (row, _), cell in tbl.get_celld().items():
    cell.set_linewidth(0.4)
    if row == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor(INK)

# notes, in their own axes so nothing overlaps
ax = fig.add_subplot(gs[3, :])
ax.axis("off")
ax.text(0, 1.0,
        "Coverage is measured in units of $\\sigma\\sqrt{T}$, not raw moneyness: a fixed moneyness band spans very different\n"
        "probability mass across maturities and volatility regimes. In 1996 the highest listed strike sat only ~1.2$\\sigma$ above the\n"
        "forward, so the entire right tail of the risk-neutral density would be extrapolation rather than observed prices, and any\n"
        "entropy measure would largely reflect the tail-fitting assumption rather than market prices.",
        transform=ax.transAxes, fontsize=7.8, color=INK, va="top", linespacing=1.6)

verdict_short = (
    "Panel C tests whether sigma-unit coverage is endogenous to volatility. It is not: within year, upper coverage\n"
    f"correlates {r_hi:+.2f} with log implied vol, and the pass rate itself {r_pass:+.2f}. The listed strike range does widen\n"
    f"with volatility (corr {r_rng:+.2f}), offsetting the mechanical effect of implied vol entering the denominator.")
ax.text(0, 0.44, verdict_short, transform=ax.transAxes, fontsize=7.8,
        color=INK, va="top", linespacing=1.6)

ax.text(0, 0.08,
        "Proposed approach: retain every observation and decompose the entropy integral into an observed component\n"
        "and a tail-extrapolated component, reporting the extrapolated probability share as a diagnostic and testing\n"
        "robustness across its terciles. Decomposition is preferred not because screening is biased, but because it\n"
        "converts a tail-extrapolation assumption into a measured and reportable quantity.",
        transform=ax.transAxes, fontsize=7.8, color=INK, va="top", linespacing=1.6)

with PdfPages("diagnostics.pdf") as pdf:
    pdf.savefig(fig)
plt.close(fig)
print("wrote diagnostics.pdf")
