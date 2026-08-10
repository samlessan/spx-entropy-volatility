#!/usr/bin/env python3
"""Condition of the out-of-sample orthogonalisation design matrix.

    cd ~/urss && source venv/bin/activate && python check_orth.py

horse_race.py emits a RuntimeWarning from numpy.matmul at the orthogonalisation
step. matmul performs no division, so this checks whether the least-squares
solve behind it is degenerate in the early expanding windows.
"""
import numpy as np
import pandas as pd

OCOLS = ["liv", "lsp", "n_used", "dte", "bkm_skew", "bkm_kurt"]

p = pd.read_csv("data/panel.csv", parse_dates=["date"]).set_index("date")
missing = [c for c in OCOLS if c not in p.columns]
if missing:
    raise SystemExit(f"panel.csv missing {missing}; has {list(p.columns)}")

for end in pd.date_range("2018-01-31", "2019-12-31", freq="ME"):
    tr = p.loc[:end].dropna(subset=OCOLS + ["xh"])
    if len(tr) < 10:
        continue
    A = np.column_stack([np.ones(len(tr)), tr[OCOLS].values.astype(float)])
    orth = np.linalg.lstsq(A, tr.xh.values, rcond=None)[0]
    print(f"{end.date()} n={len(tr):5d} "
          f"rank={np.linalg.matrix_rank(A)}/{A.shape[1]} "
          f"cond={np.linalg.cond(A):.2e} "
          f"max|orth|={np.abs(orth).max():.3e} "
          f"finite={np.isfinite(orth).all()}")
