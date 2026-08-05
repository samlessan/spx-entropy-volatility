#!/usr/bin/env python3
"""
Find what is breaking the orthogonalisation.

    cd ~/urss && source venv/bin/activate && python diagnose_panel.py
"""
import glob
import numpy as np
import pandas as pd

ent = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/entropy_*.csv"))])
ent["date"] = pd.to_datetime(ent.date)
ent["gap"] = (ent.dte - 30).abs()
ent = ent.sort_values(["date", "gap"]).drop_duplicates("date", keep="first")

ent["liv"] = np.log(ent.atm_iv.where(ent.atm_iv > 0))
ent["lsp"] = np.log(ent.half_spread.clip(lower=1e-8))

cols = ["xh", "liv", "lsp", "n_used", "dte", "bkm_skew", "bkm_kurt",
        "atm_iv", "half_spread", "tail_share", "h", "h_ln"]

print(f"rows: {len(ent):,}\n")
print(f"{'column':>12}{'n_nan':>8}{'n_inf':>8}{'min':>14}{'p1':>12}"
      f"{'p50':>12}{'p99':>12}{'max':>14}")
for c in cols:
    v = ent[c].astype(float)
    n_nan = int(v.isna().sum())
    n_inf = int(np.isinf(v).sum())
    f = v[np.isfinite(v)]
    print(f"{c:>12}{n_nan:8d}{n_inf:8d}{f.min():14.4g}{f.quantile(.01):12.4g}"
          f"{f.median():12.4g}{f.quantile(.99):12.4g}{f.max():14.4g}")

print("\n--- rows with any non-finite control ---")
oc = ["liv", "lsp", "n_used", "dte", "bkm_skew", "bkm_kurt"]
bad = ent[~np.isfinite(ent[oc].astype(float)).all(axis=1)]
print(f"{len(bad)} rows")
if len(bad):
    print(bad[["date"] + oc + ["atm_iv", "half_spread"]].head(10).to_string(index=False))

print("\n--- extreme BKM values (|skew|>20 or kurt>500 or kurt<0) ---")
ext = ent[(ent.bkm_skew.abs() > 20) | (ent.bkm_kurt > 500) | (ent.bkm_kurt < 0)]
print(f"{len(ext)} rows ({100*len(ext)/len(ent):.1f}%)")
if len(ext):
    print(ext[["date", "bkm_skew", "bkm_kurt", "atm_iv", "n_used",
               "tail_share"]].head(10).to_string(index=False))

print("\n--- condition number of the control matrix ---")
g = ent[np.isfinite(ent[oc].astype(float)).all(axis=1)]
A = np.column_stack([np.ones(len(g)), g[oc].values.astype(float)])
print(f"full sample:  cond = {np.linalg.cond(A):.3e}")
for n in (200, 400, 800):
    An = A[:n]
    print(f"first {n:4d}:   cond = {np.linalg.cond(An):.3e}"
          f"   (constant cols: {[oc[i] for i in range(len(oc)) if np.std(An[:, i+1]) == 0]})")
print("\ncond > 1e10 means the least-squares solve is numerically unstable.")
