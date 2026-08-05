#!/usr/bin/env python3
"""Build the dependent variable properly.

    cd ~/urss && source venv/bin/activate && python rv_build.py

OPrc / CPrc are the genuine session open and close (O_official / C_official
are flags, constant 1.00). Uses them to detect stub prints in
price_high_m / price_low_m on a per-row basis, then scores every estimator.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

d = pd.read_csv("data/spy_iid_ms.csv", low_memory=False)
dc = next(c for c in d.columns if c.upper() == "DATE")

w = d[[dc, "price_high_m", "price_low_m", "OPrc", "CPrc"]].copy()
w.columns = ["date", "H", "L", "O", "C"]
w["date"] = pd.to_datetime(w.date, errors="coerce")
for c in ["H", "L", "O", "C"]:
    w[c] = pd.to_numeric(w[c], errors="coerce")
w = w.dropna().drop_duplicates("date").set_index("date").sort_index()
w = w[(w.H > 0) & (w.L > 0) & (w.O > 0) & (w.C > 0)]
print("n rows:", len(w), "|", w.index.min().date(), "to", w.index.max().date())

# ------------------------------------------------ stub-print detection --
# A true session high >= max(O,C); a true low <= min(O,C). Measure how far
# each breaches that bound. Real breaches are small; stubs are enormous.
hi_oc = w[["O", "C"]].max(axis=1)
lo_oc = w[["O", "C"]].min(axis=1)
w["gap_hi"] = np.log(w.H / hi_oc)     # >= 0 normally, large => bad high
w["gap_lo"] = np.log(lo_oc / w.L)     # >= 0 normally, large => bad low

print("\n=== gap of H above max(O,C), and of min(O,C) above L ===")
print("gap_hi q:", w.gap_hi.quantile([.5, .9, .99, .999, 1]).round(4).to_dict())
print("gap_lo q:", w.gap_lo.quantile([.5, .9, .99, .999, 1]).round(4).to_dict())
print("H < max(O,C):", int((w.gap_hi < -1e-9).sum()),
      " L > min(O,C):", int((w.gap_lo < -1e-9).sum()))

print("\n=== known dates with real O and C ===")
for dt, note in [("2020-02-19", "quiet ~0.5%"), ("2020-03-16", "covid ~7-8%"),
                 ("2025-01-03", "quiet ~1%"), ("2017-10-04", "quiet ~0.3%")]:
    t = pd.Timestamp(dt)
    if t in w.index:
        r = w.loc[t]
        print(f"  {dt}  H={r.H:7.2f} L={r.L:7.2f} O={r.O:7.2f} C={r.C:7.2f}"
              f"  rng={np.log(r.H / r.L):.4f}  gap_lo={r.gap_lo:.4f}  ({note})")

# --------------------------------------------------------- estimators --
w["rng"] = np.log(w.H / w.L)
w["r"] = np.log(w.C).diff()
w["cc"] = w.r ** 2
w["park"] = 0.360674 * w.rng ** 2
w["gk"] = (0.5 * w.rng ** 2
           - (2 * np.log(2) - 1) * np.log(w.C / w.O) ** 2).clip(lower=0)

print("\nrng / |r| median =", round((w.rng / w.r.abs().clip(1e-6)).median(), 2),
      " (GBM expects 1.6-1.8)")


def ann(s):
    return np.sqrt(s.median() * 252)


print(f"median annualised vol  cc={ann(w.cc.dropna()):.3f}  "
      f"park={ann(w.park):.3f}  gk={ann(w.gk):.3f}   (truth ~0.11-0.13)")

# -------------------------------------------------------------- score --
v = pd.read_csv("data/vix_cboe.csv")
v["date"] = pd.to_datetime(v.date, errors="coerce")
v = (v.dropna(subset=["date", "vix"]).drop_duplicates("date")
      .set_index("date").sort_index()[["vix"]])


def score(s, lab):
    s = s.replace(0, np.nan).dropna()
    if len(s) < 200:
        print(f"  {lab:34s} SKIP n={len(s)}")
        return
    m = pd.concat([s.rename("x"), v], axis=1, join="inner").dropna()
    lv = np.log(m.vix / 100)
    out = []
    for agg in ("mean", "median"):
        f = np.log(getattr(m.x.rolling(21), agg)().shift(-21))
        k = f.notna()
        out.append(np.corrcoef(lv[k], f[k])[0, 1])
    print(f"  {lab:34s} n={len(m):5d}  MEAN={out[0]:.3f}  MEDIAN={out[1]:.3f}")


print("\n=== corr(log VIX, log fwd-21d) -- close-to-close needs no H/L ===")
score(w.cc, "close-to-close (no screen)")

print("\n=== range estimators, stub screen on gap ===")
for thr in [None, 0.05, 0.03, 0.02, 0.01]:
    if thr is None:
        m = pd.Series(True, index=w.index)
        lab = "no screen"
    else:
        m = (w.gap_hi <= thr) & (w.gap_lo <= thr)
        lab = f"gap <= {thr:.2f}  ({100 * (~m).mean():.1f}% cut)"
    score(w.park.where(m), "park  " + lab)

print()
for thr in [0.05, 0.03, 0.02]:
    m = (w.gap_hi <= thr) & (w.gap_lo <= thr)
    score(w.gk.where(m), f"gk    gap <= {thr:.2f}")

print("\n=== worst gap_lo offenders ===")
print(w.nlargest(8, "gap_lo")[["H", "L", "O", "C", "rng", "gap_lo"]]
      .round(3).to_string())
