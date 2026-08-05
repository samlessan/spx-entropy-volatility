#!/usr/bin/env python3
"""
Resolve the units of WRDS IID `ivol_t` / `ivol_q`.

    cd ~/urss && source venv/bin/activate && python calibrate_ivol.py

The raw magnitudes (~1e-8) are four orders of magnitude below plausible daily
variance, so the variable is on some undocumented scale. Rather than guess,
regress it against a volatility estimator built from the file's own OHLC
columns, whose units are unambiguous.

    ln(RV_range) = a + b * ln(ivol_t)

  b ~ 1.0  -> ivol_t is a VARIANCE; recover by multiplying by exp(a)
  b ~ 0.5  -> ivol_t is a variance and RV_range is a std dev, or vice versa
  b ~ 2.0  -> ivol_t is a STANDARD DEVIATION
  low R^2  -> not a volatility measure at all; abandon and use the range
              estimator directly as the dependent variable

Range estimators used (daily variance of log returns):
  Parkinson  (1980)     0.3607 * [ln(H/L)]^2                needs H, L
  Garman-Klass (1980)   0.5*[ln(H/L)]^2 - 0.3863*[ln(C/O)]^2   needs O,H,L,C

Both are unbiased for diffusive variance and far more efficient than
close-to-close. Parkinson is the fallback when open/close are absent.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

F = "data/spy_iid_ms.csv"
d = pd.read_csv(F)
print(f"{F}: {d.shape[0]:,} rows x {d.shape[1]} cols\n")

# ---------------------------------------------------------------- columns --
date_col = next((c for c in d.columns if c.upper() == "DATE"), d.columns[0])
d[date_col] = pd.to_datetime(d[date_col], format="%Y%m%d", errors="coerce")
if d[date_col].isna().all():
    d[date_col] = pd.to_datetime(pd.read_csv(F, usecols=[date_col])[date_col],
                                 errors="coerce")

ivcols = [c for c in d.columns if c.lower().startswith("ivol")]
print("ivol columns:", ivcols)
print(d[ivcols].describe().T.to_string(float_format=lambda x: f"{x:.4e}"))

pcols = [c for c in d.columns if "price" in c.lower()]
print(f"\nall price columns ({len(pcols)}):")
for c in pcols:
    print("   ", c)


def pick(*subs):
    """First price column containing all substrings."""
    for c in pcols:
        lc = c.lower()
        if all(s in lc for s in subs):
            return c
    return None


hi = pick("high", "_m") or pick("high")
lo = pick("low", "_m") or pick("low")
op = pick("open", "_m") or pick("open")
cl = pick("close", "_m") or pick("close") or pick("last")

print(f"\nusing  high={hi}  low={lo}  open={op}  close={cl}")
if hi is None or lo is None:
    raise SystemExit("no high/low columns -- cannot build a range estimator")

w = d[[date_col, hi, lo] + [c for c in (op, cl) if c] + ivcols].copy()
w = w.rename(columns={date_col: "date", hi: "H", lo: "L"})
if op:
    w = w.rename(columns={op: "O"})
if cl:
    w = w.rename(columns={cl: "C"})

# --------------------------------------------------------------- cleaning --
n0 = len(w)
w = w[(w.H > 0) & (w.L > 0) & (w.H >= w.L)]
w["rng"] = np.log(w.H / w.L)
# 2015-01-07 has price_high_m = 220.25, a bad print on a ~205 stock.
# Kill any day whose log range exceeds 25%, which no SPY session has produced.
w = w[w.rng < 0.25]
print(f"\ncleaned {n0:,} -> {len(w):,} rows "
      f"({n0 - len(w)} dropped for bad/absent prints)")

# ------------------------------------------------------------- estimators --
w["park"] = 0.360674 * w.rng ** 2
if "O" in w and "C" in w:
    w = w[(w.O > 0) & (w.C > 0)]
    w["gk"] = 0.5 * w.rng ** 2 - (2 * np.log(2) - 1) * np.log(w.C / w.O) ** 2
    w["gk"] = w.gk.where(w.gk > 0)
    ref, refname = w.gk, "Garman-Klass"
else:
    ref, refname = w.park, "Parkinson"
w["ref"] = ref

print(f"\nreference estimator: {refname}")
print(f"   daily variance  median {w.ref.median():.3e}   "
      f"implied annual vol {np.sqrt(w.ref.median()*252):.3f}")
print(f"   (SPY annualised vol should land ~0.12-0.25 over 2015-2025)")

# ----------------------------------------------------------- calibration --
print("\n" + "=" * 72)
for iv in ivcols:
    g = w[[iv, "ref"]].dropna()
    g = g[(g[iv] > 0) & (g.ref > 0)]
    if len(g) < 100:
        print(f"{iv}: only {len(g)} usable rows, skipping")
        continue
    X = sm.add_constant(np.log(g[iv]))
    m = sm.OLS(np.log(g.ref), X).fit()
    b = m.params.iloc[1]
    a = m.params.iloc[0]
    print(f"\n{iv}:  n={len(g):,}")
    print(f"   ln(ref) = {a:+.3f} + {b:.3f} * ln({iv})     R2={m.rsquared:.3f}"
          f"   corr={np.corrcoef(np.log(g[iv]), np.log(g.ref))[0,1]:+.3f}")
    # Monthly aggregation: GK/Parkinson are unbiased but noisy day by day, so
    # a low daily R2 is expected even when the relationship is exact.
    # Averaging both sides over a month strips the estimator noise and makes
    # the slope and fit interpretable.
    gm = g.copy()
    gm["ym"] = w.loc[g.index, "date"].dt.to_period("M")
    gm = gm.groupby("ym")[[iv, "ref"]].mean()
    gm = gm[(gm[iv] > 0) & (gm.ref > 0)]
    mm = sm.OLS(np.log(gm.ref), sm.add_constant(np.log(gm[iv]))).fit()
    bm, am = mm.params.iloc[1], mm.params.iloc[0]
    print(f"   monthly: ln(ref) = {am:+.3f} + {bm:.3f} * ln({iv})"
          f"     R2={mm.rsquared:.3f}   n={len(gm)}")

    b, a = bm, am          # judge on the monthly fit
    if mm.rsquared < 0.15:
        verdict = "NOT a volatility measure -- do not use"
    elif abs(b - 1) < 0.15:
        verdict = (f"VARIANCE on a scaled grid. Recover with "
                   f"var_daily = {iv} * {np.exp(a):.4g}")
    elif abs(b - 0.5) < 0.15:
        verdict = f"ivol is a VARIANCE, ref is std-dev-like; b~0.5"
    elif abs(b - 2) < 0.3:
        verdict = f"STANDARD DEVIATION. var_daily = ({iv})^2 * {np.exp(a):.4g}"
    else:
        verdict = f"non-standard exponent b={b:.2f}; investigate before use"
    print(f"   -> {verdict}")

    if abs(b - 1) < 0.15:
        implied = g[iv] * np.exp(a)
        print(f"   sanity: implied annual vol median "
              f"{np.sqrt(implied.median()*252):.3f}")

print("\n" + "=" * 72)
out = w[["date", "ref", "park"] + ivcols].rename(columns={"ref": "rv_daily"})
out.to_csv("data/spy_rv.csv", index=False)
print(f"wrote data/spy_rv.csv  ({len(out):,} rows, {out.date.min().date()} "
      f"to {out.date.max().date()})")
print("\nIf no ivol column calibrates cleanly, use rv_daily "
      f"({refname}) as the dependent variable. It is standard, unbiased,\n"
      "and its units are known -- which is more than can be said for ivol_t.")
