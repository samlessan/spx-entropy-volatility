import numpy as np, pandas as pd
p = pd.read_csv("data/panel.csv", parse_dates=["date"]).set_index("date")
ocols = [c for c in ["log_iv","log_hs","n_used","dte","bkm_skew","bkm_kurt"] if c in p.columns]
print("ocols:", ocols)
for end in pd.date_range("2018-01-31", "2019-12-31", freq="ME"):
    tr = p.loc[:end].dropna(subset=ocols+["xh"])
    if len(tr) < 10: continue
    A = np.column_stack([np.ones(len(tr)), tr[ocols].values.astype(float)])
    orth = np.linalg.lstsq(A, tr.xh.values, rcond=None)[0]
    print(f"{end.date()} n={len(tr):5d} rank={np.linalg.matrix_rank(A)}/{A.shape[1]} "
          f"cond={np.linalg.cond(A):.2e} max|orth|={np.abs(orth).max():.3e} "
          f"finite={np.isfinite(orth).all()}")
