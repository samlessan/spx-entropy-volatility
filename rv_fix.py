#!/usr/bin/env python3
"""Find the real O/C price columns, then screen bad prints properly.

    cd ~/urss && source venv/bin/activate && python rv_fix.py

O_official / C_official are FLAGS (constant 1.00), not prices. This locates
genuine open/close columns by testing every numeric column for price-like
behaviour against price_high_m / price_low_m, then applies a neighbour-based
bad-print screen and rescores against VIX.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

d = pd.read_csv("data/spy_iid_ms.csv", low_memory=False)
dc = next(c for c in d.columns if c.upper() == "DATE")
dates = pd.to_datetime(d[dc], errors="coerce")

H = pd.to_numeric(d["price_high_m"], errors="coerce")
L = pd.to_numeric(d["price_low_m"], errors="coerce")
mid = (H + L) / 2
ok = H.notna() & L.notna() & (H > 0) & (L > 0)

# ------------------------------------------------- 1. find price columns --
print("=== columns that behave like a daily price ===")
cands = []
for c in d.columns:
    if c == dc:
        continue
    s = pd.to_numeric(d[c], errors="coerce")
    if s.notna().sum() < len(d) * 0.9:
        continue
    m = ok & s.notna() & (s > 0)
    if m.sum() < len(d) * 0.9:
        continue
    inside = ((s >= L * 0.995) & (s <= H * 1.005))[m].mean()
    rel = (s[m] / mid[m])
    if inside > 0.90 and 0.9 < rel.median() < 1.1:
        cands.append((c, inside, rel.median(), s[m].std()))

for c, ins, rel, sd in sorted(cands, key=lambda x: -x[1])[:25]:
    print(f"  {c:28s} inside_HL={ins:.3f}  med(x/mid)={rel:.4f}  sd={sd:.2f}")
if not cands:
    print("  none found -- open/close are not in this file")

# ------------------------------------------------------ 2. screen prints --
w = pd.DataFrame({"date": dates, "H": H, "L": L}).dropna()
w = w.drop_duplicates("date").set_index("date").sort_index()
w = w[(w.H > 0) & (w.L > 0) & (w.H >= w.L)]
w["rng"] = np.log(w.H / w.L)

# a stub print makes today's range huge relative to its own neighbourhood
nb = w.rng.rolling(21, center=True, min_periods=10).median()
w["excess"] = w.rng / nb
print("\n=== rng / local-median-rng ===")
print(w.excess.quantile([.5, .9, .99, .999, 1]).round(2).to_dict())

for thr in [3, 4, 5, 6, 8]:
    print(f"  excess > {thr}: {int((w.excess > thr).sum()):4d} days "
          f"({100 * (w.excess > thr).mean():.2f}%)")

# ------------------------------------------------------------- 3. score --
v = pd.read_csv("data/vix_cboe.csv")
v["date"] = pd.to_datetime(v.date, errors="coerce")
v = (v.dropna(subset=["date", "vix"]).drop_duplicates("date")
      .set_index("date").sort_index()[["vix"]])


def score(mask, lab):
    s = (0.360674 * w.rng[mask] ** 2).replace(0, np.nan).dropna()
    m = pd.concat([s.rename("x"), v], axis=1, join="inner").dropna()
    lv = np.log(m.vix / 100)
    out = []
    for agg in ("mean", "median"):
        r = getattr(m.x.rolling(21), agg)()
        f = np.log(r.shift(-21))
        k = f.notna()
        out.append(np.corrcoef(lv[k], f[k])[0, 1])
    print(f"  {lab:26s} n={len(m):5d}  MEAN={out[0]:.3f}  MEDIAN={out[1]:.3f}")


print("\n=== corr(log VIX, log fwd-21d parkinson) ===")
score(w.index == w.index, "no screen")
for thr in [8, 6, 5, 4, 3]:
    score(w.excess <= thr, f"excess <= {thr}")

print("\n=== worst offenders (excess > 5) ===")
bad = w[w.excess > 5].nlargest(10, "excess")
print(bad[["H", "L", "rng", "excess"]].round(3).to_string())
