#!/usr/bin/env python3
"""Rebuild data/spy_rv.csv with a correct dependent variable.

    cd ~/urss && source venv/bin/activate && python rebuild_rv.py

DECISION (pre-registered, 2026-08-05)
  Dependent variable: close-to-close realised variance, r_t^2, where
  r_t = ln(CPrc_t / CPrc_{t-1}) from data/spy_iid_ms.csv.

  Rejected alternatives and why:
    ivol_t          WRDS per-unit-time microstructure variance; units
                    undocumented, required a 1.5e4 rescale, corr with
                    forward VIX 0.432.
    Parkinson / GK  built on price_high_m / price_low_m, which carry
                    erroneous prints. Range inflated ~60% vs returns from
                    the same file (rng/|r| = 2.77 vs GBM 1.6-1.8);
                    2020-03-16 reports a 16.1% range against a true 8.7%.
                    Best achievable after screening: 0.629, at a 20% data cut.
    O_official /    constant 1.00 flags, not prices. Their use silently
    C_official      degenerated Garman-Klass to 0.5*rng^2.

  Validation: corr(log VIX, log forward-21d mean) = 0.671, n = 2671.
  Benchmark for FORWARD realised vol is 0.65-0.72 (not the 0.75-0.85 that
  applies to contemporaneous), so this passes.
  Median annualised vol implied = 0.079 / 0.6745 = 0.117, against a true
  SPY 2015-2025 figure of 0.11-0.13.

  Known limitation to state in the write-up: r^2 is a 1-observation-per-day
  variance estimator, so the daily HAR component is noisy. This attenuates
  HAR-d and biases toward finding nothing, which is the conservative
  direction for a null-friendly design.
"""
import numpy as np
import pandas as pd

SRC = "data/spy_iid_ms.csv"
OUT = "data/spy_rv.csv"

d = pd.read_csv(SRC, low_memory=False)
dc = next(c for c in d.columns if c.upper() == "DATE")

w = d[[dc, "CPrc", "OPrc", "price_high_m", "price_low_m"]].copy()
w.columns = ["date", "C", "O", "H", "L"]
w["date"] = pd.to_datetime(w.date, errors="coerce")
for c in ["C", "O", "H", "L"]:
    w[c] = pd.to_numeric(w[c], errors="coerce")
w = w.dropna(subset=["date", "C"]).drop_duplicates("date")
w = w[w.C > 0].set_index("date").sort_index()

w["r"] = np.log(w.C).diff()
w["rv"] = w.r ** 2                      # <- the dependent variable
w["park"] = 0.360674 * np.log(w.H / w.L) ** 2   # kept for robustness only
w = w.dropna(subset=["rv"])

# guard: a single-day move beyond 20% is a data error, not a session
n0 = len(w)
w = w[w.r.abs() < 0.20]
print(f"rows: {n0} -> {len(w)} ({n0 - len(w)} dropped for |r| >= 0.20)")

med_ann = np.sqrt(w.rv.median() * 252) / 0.6745
print(f"implied median annualised vol: {med_ann:.4f}   (truth 0.11-0.13)")
print(f"date range: {w.index.min().date()} to {w.index.max().date()}")

# ------------------------------------------------------------ validate --
v = pd.read_csv("data/vix_cboe.csv")
v["date"] = pd.to_datetime(v.date, errors="coerce")
v = (v.dropna(subset=["date", "vix"]).drop_duplicates("date")
      .set_index("date").sort_index()[["vix"]])
m = pd.concat([w.rv.rename("x"), v], axis=1, join="inner").dropna()
f = np.log(m.x.rolling(21).mean().shift(-21).clip(1e-12))
k = f.notna()
corr = np.corrcoef(np.log(m.vix[k] / 100), f[k])[0, 1]
print(f"corr(log VIX, log fwd-21d rv) = {corr:.3f}   (gate: >= 0.60)")
assert corr >= 0.60, "VALIDATION FAILED -- do not use this file"

out = w.reset_index()[["date", "rv", "park", "r"]]
out.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  ({len(out):,} rows)")
print("\nNow edit horse_race.py:")
print("  DELETE:  IVOL_SCALE = rv.park.median() / rv.ivol_t.median()")
print("           rv['rv']  = rv.ivol_t * IVOL_SCALE")
print("  The rv column already exists in the new file. No rescale, which")
print("  also removes the full-sample leakage in the old median-matching.")
