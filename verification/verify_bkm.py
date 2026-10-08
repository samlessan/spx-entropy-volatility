#!/usr/bin/env python3
"""Check of the BKM moments and the entropy extraction's strike selection.

    cd ~/urss && source venv/bin/activate
    curl -sL -o data/SKEW_History.csv https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv
    python verification/verify_bkm.py

1. External benchmark: pipeline BKM skewness vs the Cboe SKEW index
   (SKEW = 100 - 10 * S, S = 30-day risk-neutral skewness of SPX log returns).
2. Mechanism: re-run the pipeline's own process_chain() on sampled dates
   (A) exactly as the pipeline does, (B) without zero-bid quotes,
   (C) Cboe/VIX-style: zero bids dropped and each wing cut after two
   consecutive zero bids. (A) must reproduce panel.csv, or nothing below counts.
   Since October 2026 the pipeline itself applies (C), so A and C coincide;
   before the fix they correlated 0.68 for XH and BKM kurtosis exceeded 500.
3. The most extreme XH and BKM-kurtosis dates, under A, B and C.
4. Does the headline entropy result survive winsorising XH and BKM, or
   dropping BKM from the model?
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.path.insert(0, os.getcwd())
import extract_entropy as ee            # the pipeline's own functions

H, OOS, N_PER_YEAR = 21, "2018-01-01", 20
rng = np.random.default_rng(0)


def line(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


pan = pd.read_csv("data/panel.csv", index_col=0, parse_dates=True).sort_index()
pan["exdate"] = pd.to_datetime(pan.exdate)

# ============================================== 1. Cboe SKEW benchmark --
line("1. PIPELINE BKM SKEWNESS vs CBOE SKEW INDEX (S = (100 - SKEW) / 10)")
fn = "data/SKEW_History.csv"
try:
    with open(fn, "rb") as fh:
        head = fh.read(300)
    print("first bytes of the Cboe file:", head[:120])
    sk = pd.read_csv(fn)
    sk.columns = [str(c).strip().upper() for c in sk.columns]
    dcol = next(c for c in sk.columns if "DATE" in c)
    vcol = next(c for c in sk.columns if "SKEW" in c or "CLOSE" in c)
    sk[dcol] = pd.to_datetime(sk[dcol], errors="coerce")
    sk[vcol] = pd.to_numeric(sk[vcol], errors="coerce")
    sk = sk.dropna(subset=[dcol, vcol]).drop_duplicates(dcol).set_index(dcol)[vcol]
    print(f"Cboe SKEW rows: {len(sk):,}  ({sk.index.min().date()} to {sk.index.max().date()})")
    s_cboe = ((100 - sk) / 10).rename("cboe")
    j = pan[["bkm_skew"]].join(s_cboe, how="inner").dropna()
    print(f"matched dates: {len(j):,}")
    print(f"  level corr  = {j.bkm_skew.corr(j.cboe):+.2f}   "
          f"daily-change corr = {j.bkm_skew.diff().corr(j.cboe.diff()):+.2f}")
    print(f"  {'year':>6} {'pipeline median':>16} {'Cboe median':>12}")
    for y, g in j.groupby(j.index.year):
        print(f"  {y:>6} {g.bkm_skew.median():16.2f} {g.cboe.median():12.2f}")
    print(f"  {'all':>6} {j.bkm_skew.median():16.2f} {j.cboe.median():12.2f}")
except Exception as e:
    print(f"  SECTION 1 FAILED: {type(e).__name__}: {e}  (continuing; is data/SKEW_History.csv the CSV?)")


# ========================================== 2. zero-bid mechanism test --
def variant(g, mode):
    g = g[g.best_offer > g.best_bid].copy()
    g["mid"] = 0.5 * (g.best_bid + g.best_offer)
    if mode == "B":
        g = g[g.best_bid > 0]
    elif mode == "C":
        keep = []
        F = g.attrs["F"]
        for side, asc in [("P", False), ("C", True)]:
            s = g[(g.cp_flag == side) & ((g.strike < F) if side == "P" else (g.strike >= F))]
            s = s.sort_values("strike", ascending=asc)
            zeros = 0
            for idx, b in zip(s.index, s.best_bid.values):
                if b <= 0:
                    zeros += 1
                    if zeros >= 2:
                        break
                    continue
                zeros = 0
                keep.append(idx)
        g = g.loc[keep]
    return g


def run(g, F, T, mode):
    g = g.copy()
    g.attrs["F"] = F
    try:
        return ee.process_chain(variant(g, mode), F, T)
    except Exception:
        return None


# sample dates: N_PER_YEAR per year + the extreme dates
ext = list(pan.xh.nsmallest(6).index) + list(pan.bkm_kurt.nlargest(6).index)
samp = []
for y, g in pan.groupby(pan.index.year):
    samp += list(g.sample(min(N_PER_YEAR, len(g)), random_state=1).index)
dates = sorted(set(samp) | set(ext))

rows = []
for yr in sorted({d.year for d in dates}):
    f = f"data/raw/spx_{yr}.parquet"
    if not os.path.exists(f):
        print(f"  {f} missing, skipping {yr}")
        continue
    raw = pd.read_parquet(f)
    raw["date"] = pd.to_datetime(raw.date)
    raw["exdate"] = pd.to_datetime(raw.exdate)
    raw["strike"] = raw.strike_price / 1000.0
    for d in [d for d in dates if d.year == yr]:
        r0 = pan.loc[d]
        g = raw[(raw.date == d) & (raw.exdate == r0.exdate)]
        if g.empty:
            continue
        F, T = float(r0.fwd), float(r0.dte_yr)
        otm = g[((g.cp_flag == "C") & (g.strike >= F)) | ((g.cp_flag == "P") & (g.strike < F))]
        otm = otm[(otm.best_offer > otm.best_bid) & (0.5 * (otm.best_bid + otm.best_offer) > 0.05)
                  & (otm.strike / F > 0.4) & (otm.strike / F < 1.6)]
        row = {"date": d, "extreme": d in ext, "zero_bid_share": (otm.best_bid <= 0).mean(),
               "xh_panel": r0.xh, "kurt_panel": r0.bkm_kurt}
        for m in "ABC":
            res = run(g, F, T, m)
            for k in ["xh", "bkm_skew", "bkm_kurt", "n_used"]:
                row[f"{k}_{m}"] = res[k] if res else np.nan
        rows.append(row)
    del raw
R = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

line("2. DO ZERO-BID QUOTES INFLATE BKM OR XH?  (sampled dates)")
S = R[~R.extreme]
okA = (np.abs(S.xh_A - S.xh_panel) < 1e-6).mean()
print(f"sampled dates: {len(S)}   variant A reproduces panel.csv xh on {okA:.0%} "
      f"(must be ~100%)")
print(f"median share of OTM quotes with zero bid: {S.zero_bid_share.median():.1%} "
      f"(90th pct {S.zero_bid_share.quantile(.9):.1%})")
print(f"\n  {'':28s} {'A pipeline':>11} {'B no 0-bid':>11} {'C Cboe-style':>13}")
for k, lab in [("bkm_skew", "BKM skew, median"), ("bkm_kurt", "BKM kurt, median"),
               ("xh", "XH, median"), ("n_used", "strikes used, median")]:
    print(f"  {lab:28s} {S[k+'_A'].median():11.2f} {S[k+'_B'].median():11.2f} "
          f"{S[k+'_C'].median():13.2f}")
print(f"  {'BKM kurt, 99th pct':28s} {S.bkm_kurt_A.quantile(.99):11.1f} "
      f"{S.bkm_kurt_B.quantile(.99):11.1f} {S.bkm_kurt_C.quantile(.99):13.1f}")
print(f"\n  corr(XH under A, XH under C)       = {S.xh_A.corr(S.xh_C):+.3f}")
print(f"  corr(skew under A, skew under C)   = {S.bkm_skew_A.corr(S.bkm_skew_C):+.3f}")

line("3. EXTREME DATES (6 most negative XH, 6 largest BKM kurtosis)")
E = R[R.extreme].copy()
E["ivA"] = pan.loc[E.index, "atm_iv"] * 100
print(E[["zero_bid_share", "xh_A", "xh_B", "xh_C", "bkm_kurt_A", "bkm_kurt_C",
         "n_used_A", "n_used_C", "ivA"]].round(3).to_string())

# ============================================== 4. headline robustness --
line("4. DOES THE HEADLINE SURVIVE?  (corrected spec, 21-row embargo)")
HAR = ["har_d", "har_w", "har_m"]


def oos_t(d, small, big, ocols):
    idx = d.index
    oidx = idx[idx >= pd.Timestamp(OOS)]
    mon = oidx.to_period("M").asi8
    ys, p1, p2 = [], [], []
    for mo in np.unique(mon):
        te = oidx[mon == mo]
        tr = d[idx < te[0]].iloc[:-H]
        if len(tr) <= 200:
            continue
        A = np.column_stack([np.ones(len(tr)), tr[ocols].values.astype(float)])
        o = np.linalg.lstsq(A, tr.xh.values, rcond=None)[0]
        tr = tr.assign(xh_o=tr.xh.values - A @ o)
        dt = d.loc[te]
        At = np.column_stack([np.ones(len(dt)), dt[ocols].values.astype(float)])
        dt = dt.assign(xh_o=dt.xh.values - At @ o)
        b1 = sm.OLS(tr.y, sm.add_constant(tr[small])).fit().params.values
        b2 = sm.OLS(tr.y, sm.add_constant(tr[big])).fit().params.values
        ys.append(dt.y.values)
        p1.append(np.column_stack([np.ones(len(dt)), dt[small].values]) @ b1)
        p2.append(np.column_stack([np.ones(len(dt)), dt[big].values]) @ b2)
    y, a, b = map(np.concatenate, (ys, p1, p2))
    f = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    t = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": H}).tvalues[0]
    inc = (np.mean((y - a) ** 2) - np.mean((y - b) ** 2)) / np.var(y)
    return t, inc


def wins(s):
    return s.clip(s.quantile(.01), s.quantile(.99))


OC = ["liv", "lsp", "n_used", "dte", "bkm_skew", "bkm_kurt"]
OC_NOBKM = ["liv", "lsp", "n_used", "dte"]
base = HAR + ["liv", "bkm_skew", "bkm_kurt"]
cases = [
    ("headline (as horse_race.py)", pan, base, OC),
    ("XH winsorised 1/99 pct", pan.assign(xh=wins(pan.xh)), base, OC),
    ("XH and BKM winsorised 1/99", pan.assign(xh=wins(pan.xh), bkm_skew=wins(pan.bkm_skew),
                                              bkm_kurt=wins(pan.bkm_kurt)), base, OC),
    ("BKM dropped entirely", pan, HAR + ["liv"], OC_NOBKM),
]
print(f"  {'case':32s} {'entropy t':>10} {'OOS incR2':>10}")
for lab, d, b, oc in cases:
    t, inc = oos_t(d, b, b + ["xh_o"], oc)
    print(f"  {lab:32s} {t:+10.2f} {inc:+10.4f}")
t, inc = oos_t(pan.assign(bkm_skew=wins(pan.bkm_skew), bkm_kurt=wins(pan.bkm_kurt)),
               HAR + ["liv"], HAR + ["liv", "bkm_skew", "bkm_kurt"], OC)
print(f"  {'BKM over IV, BKM winsorised':32s} {t:+10.2f} {inc:+10.4f}")
print("\n" + "=" * 72 + "\nDONE\n" + "=" * 72)
