#!/usr/bin/env python3
"""Settle the RV question in one run.

    cd ~/urss && source venv/bin/activate && python rv_verdict.py

Three blocks:
  1. GROUND TRUTH  -- print H/L/O/C on sessions whose true values are known.
  2. CONSISTENCY   -- how far H/L exceed the O-C envelope, from the same row.
  3. SCORE         -- every candidate estimator vs VIX, MEAN and MEDIAN agg.

Verdict rules are printed at the end.
"""
import numpy as np
import pandas as pd

d = pd.read_csv("data/spy_iid_ms.csv", low_memory=False)
dc = next(c for c in d.columns if c.upper() == "DATE")

w = d[[dc, "price_high_m", "price_low_m", "O_official", "C_official"]].copy()
w.columns = ["date", "H", "L", "O", "C"]
w["date"] = pd.to_datetime(w.date, errors="coerce")
for c in ["H", "L", "O", "C"]:
    w[c] = pd.to_numeric(w[c], errors="coerce")
w = w.dropna().drop_duplicates("date").set_index("date").sort_index()
w = w[(w.H > 0) & (w.L > 0) & (w.O > 0) & (w.C > 0) & (w.H >= w.L)]
print("n rows:", len(w), "|", w.index.min().date(), "to", w.index.max().date())

# ---------------------------------------------------------------- 1. truth --
KNOWN = {
    "2015-08-24": "flash-crash open, true range ~7-8%",
    "2018-02-05": "Volmageddon, true range ~4-5%",
    "2020-02-19": "pre-covid high, quiet, ~0.5%",
    "2020-03-16": "worst covid day, true range ~7-8%",
    "2020-04-07": "recovery rally, true range ~3-4%",
    "2022-01-25": "volatile, true range ~3%",
    "2025-01-03": "ordinary quiet session, ~1%",
}
print("\n=== GROUND TRUTH ===")
print(f"{'date':12s} {'H':>8s} {'L':>8s} {'O':>8s} {'C':>8s} {'rng':>7s}   expected")
for dt, note in KNOWN.items():
    t = pd.Timestamp(dt)
    if t in w.index:
        r = w.loc[t]
        print(f"{dt:12s} {r.H:8.2f} {r.L:8.2f} {r.O:8.2f} {r.C:8.2f} "
              f"{np.log(r.H / r.L):7.4f}   {note}")
    else:
        print(f"{dt:12s} {'ABSENT':>8s}")

# --------------------------------------------------------- 2. consistency --
hi_oc = w[["O", "C"]].max(axis=1)
lo_oc = w[["O", "C"]].min(axis=1)
w["exc_hi"] = np.log(w.H / hi_oc)
w["exc_lo"] = np.log(lo_oc / w.L)
print("\n=== H/L vs the O-C envelope (same row) ===")
print("exc_hi q:", w.exc_hi.quantile([.5, .9, .99, 1]).round(4).to_dict())
print("exc_lo q:", w.exc_lo.quantile([.5, .9, .99, 1]).round(4).to_dict())
print("violations (H<max(O,C) or L>min(O,C)):",
      int(((w.H < hi_oc - 1e-9) | (w.L > lo_oc + 1e-9)).sum()))

w["rng"] = np.log(w.H / w.L)
w["r"] = np.log(w.C).diff()
w["cc"] = w.r ** 2
w["park"] = 0.360674 * w.rng ** 2
w["gk"] = (0.5 * w.rng ** 2
           - (2 * np.log(2) - 1) * np.log(w.C / w.O) ** 2).clip(lower=0)
w["oc"] = 0.360674 * np.log(hi_oc / lo_oc) ** 2

ratio = (w.rng / w.r.abs().clip(1e-6)).median()
print(f"\nrng / |r| median = {ratio:.2f}   (GBM expects 1.6-1.8)")


def ann(s):
    return np.sqrt(s.median() * 252)


print(f"median annualised vol   cc={ann(w.cc.dropna()):.3f}  "
      f"park={ann(w.park):.3f}  gk={ann(w.gk):.3f}  oc={ann(w.oc):.3f}")
print("   (SPY 2015-2025 truth: roughly 0.11-0.13)")

# ---------------------------------------------------------------- 3. score --
v = pd.read_csv("data/vix_cboe.csv")
v["date"] = pd.to_datetime(v.date, errors="coerce")
v = (v.dropna(subset=["date", "vix"]).drop_duplicates("date")
      .set_index("date").sort_index()[["vix"]])

print("\n=== corr(log VIX, log fwd-21d) ===")
for col, lab in [("cc", "close-to-close"), ("park", "parkinson"),
                 ("gk", "garman-klass"), ("oc", "O-C range")]:
    s = w[col].replace(0, np.nan).dropna()
    m = pd.concat([s.rename("x"), v], axis=1, join="inner").dropna()
    lv = np.log(m.vix / 100)
    f = np.log(m.x.rolling(21).mean().shift(-21))
    g = np.log(m.x.rolling(21).median().shift(-21))
    ok, okm = f.notna(), g.notna()
    print(f"  {lab:16s} n={int(ok.sum()):5d}  "
          f"MEAN={np.corrcoef(lv[ok], f[ok])[0, 1]:.3f}   "
          f"MEDIAN={np.corrcoef(lv[okm], g[okm])[0, 1]:.3f}")

print("\n=== VIX sanity (true: 82.69 / 46.70 / ~16.5) ===")
for dt in ["2020-03-16", "2020-04-07", "2025-01-03"]:
    t = pd.Timestamp(dt)
    print(f"  {dt}: {v.vix.get(t, 'ABSENT')}")

print("""
------------------------------------------------------------------
VERDICT RULES
  A. Known-date rng >> expected  ->  price_high_m/price_low_m corrupt.
     Dependent variable becomes close-to-close from C_official.
  B. close-to-close MEAN >= 0.65 while park/gk ~0.35  ->  same, confirmed.
  C. All four ~0.35 AND known dates look correct  ->  problem is VIX
     or the date index, not the OHLC.
  D. MEDIAN >> MEAN everywhere  ->  fat-tailed daily contamination;
     aggregate in logs.
------------------------------------------------------------------""")
