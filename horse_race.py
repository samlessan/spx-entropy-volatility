#!/usr/bin/env python3
"""
The horse race. Does orthogonalised excess entropy forecast realised
volatility beyond HAR-RV, implied vol and BKM moments?

    cd ~/urss && source venv/bin/activate && python horse_race.py
    python horse_race.py published     # original August 2026 specification

Inputs   data/entropy_YYYY.csv   (from extract_entropy.py)
         data/spy_rv.csv          (from rebuild_rv.py)
Output   data/panel.csv           merged daily panel
         results.json             headline numbers, keyed (results_published.json
                                  for the original specification)

DESIGN
  Dependent variable: log of mean daily close-to-close realised variance of
    SPY (official close) over trading days t+1 .. t+21. 21 days matches the
    ~30 calendar-day option horizon.
  Zero-return days: rv = r^2 is exactly 0 when SPY closes unchanged (8 days in
    2015-2025). rv is floored at the 0.5th percentile of its positive values
    before logs are taken, so those days do not enter HAR at log(1e-12) = -27.6.
  One observation per date: the date-expiry pair with dte closest to 30.
  HAR-RV (Corsi 2009): lagged log RV at daily, weekly (5d) and monthly (22d)
    horizons.
  Excess entropy enters ORTHOGONALISED: the residual from regressing xh on
    log IV, log half-spread, n_used, dte, BKM skew and BKM kurtosis,
    re-estimated inside each training window.
  Evaluation: expanding window, out-of-sample from 2018-01-01, refit monthly.
  Embargo: a training row dated s has a target spanning s+1 .. s+21. At a
    refit on date t the last H training rows are dropped, since their targets
    overlap the test period.
  Metrics: MSE and R2 on log variance. QLIKE (Patton 2011) on the variance
    scale, reported at exp(forecast) (median forecast, as originally published)
    and at exp(forecast + s2/2) (mean forecast, lognormal correction, s2 the
    training residual variance). Clark-West (2007) for nested models, HAC with
    21 lags.

The 'published' option reproduces the August 2026 table: floor 1e-12 and no
embargo. It is kept so the effect of both corrections can be reported.
"""

import glob
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

H = 21                       # forecast horizon, trading days
TARGET_DTE = 30
OOS_START = "2018-01-01"
PUBLISHED = len(sys.argv) > 1 and sys.argv[1] == "published"
EMBARGO = 0 if PUBLISHED else H

# ------------------------------------------------------------------ load --
ent = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/entropy_*.csv"))])
ent["date"] = pd.to_datetime(ent.date)
ent = ent.dropna(subset=["xh", "atm_iv", "half_spread", "bkm_skew", "bkm_kurt"])

# one row per date: the expiry nearest 30 days
ent["gap"] = (ent.dte - TARGET_DTE).abs()
ent = ent.sort_values(["date", "gap"]).drop_duplicates("date", keep="first")
ent = ent.drop(columns="gap").set_index("date").sort_index()
print(f"entropy panel: {len(ent):,} dates, "
      f"{ent.index.min().date()} to {ent.index.max().date()}")
print(f"  share of date-expiry rows with xh > 0: {(ent.xh > 0).mean():.1%} "
      f"(max xh {ent.xh.max():+.4f})")

rv = pd.read_csv("data/spy_rv.csv")
rv["date"] = pd.to_datetime(rv.date)
rv = rv.set_index("date").sort_index()
assert "rv" in rv.columns, "spy_rv.csv is missing the rv column"
print(f"dependent variable: close-to-close RV from CPrc, "
      f"median annualised {np.sqrt(rv.rv.median()*252)/0.6745:.4f}")

# ------------------------------------------------------- build the target --
n_zero = int((rv.rv == 0).sum())
FLOOR = 1e-12 if PUBLISHED else float(rv.rv[rv.rv > 0].quantile(0.005))
print(f"specification: {'PUBLISHED (Aug 2026)' if PUBLISHED else 'CORRECTED'}"
      f"   floor {FLOOR:.3e}   embargo {EMBARGO} rows   zero-return days {n_zero}")
rv["rvc"] = rv.rv.clip(lower=FLOOR)
rv["lrv"] = np.log(rv.rvc)
# forward average variance over the next H days (excludes today)
rv["rv_fwd"] = rv.rvc.shift(-1).rolling(H).mean().shift(-(H - 1))
rv["y"] = np.log(rv.rv_fwd.clip(lower=1e-12))

# HAR components, all backward-looking as of the close of t
rv["har_d"] = rv.lrv
rv["har_w"] = rv.lrv.rolling(5).mean()
rv["har_m"] = rv.lrv.rolling(22).mean()

d = ent.join(rv[["y", "rv", "har_d", "har_w", "har_m"]], how="inner")
d["liv"] = np.log(d.atm_iv)
d["lsp"] = np.log(d.half_spread.clip(lower=1e-8))
d = d.dropna(subset=["y", "har_d", "har_w", "har_m", "liv", "lsp",
                     "xh", "bkm_skew", "bkm_kurt", "n_used", "dte"])
print(f"merged panel:  {len(d):,} observations "
      f"({d.index.min().date()} to {d.index.max().date()})")

# ------------------------------------------- orthogonalise excess entropy --
ocols = ["liv", "lsp", "n_used", "dte", "bkm_skew", "bkm_kurt"]
om = sm.OLS(d.xh, sm.add_constant(d[ocols])).fit()
d["xh_o"] = om.resid          # FULL SAMPLE: in-sample table only
print(f"orthogonalisation R2 (full sample) = {om.rsquared:.3f}, "
      f"residual sd = {om.resid.std():.4f}")
d.to_csv("data/panel.csv")

# ------------------------------------------------------------- the models --
MODELS = {
    "HAR":            ["har_d", "har_w", "har_m"],
    "HAR+IV":         ["har_d", "har_w", "har_m", "liv"],
    "HAR+IV+BKM":     ["har_d", "har_w", "har_m", "liv", "bkm_skew", "bkm_kurt"],
    "HAR+IV+BKM+XH":  ["har_d", "har_w", "har_m", "liv", "bkm_skew", "bkm_kurt",
                       "xh_o"],
}


def qlike(actual_var, pred_var):
    """Patton (2011) QLIKE."""
    r = actual_var / pred_var
    return float(np.mean(r - np.log(r) - 1))


# --------------------------------------------------------- in-sample fit --
print("\n" + "=" * 74)
print("IN-SAMPLE, full sample (Newey-West, 21 lags)")
print("=" * 74)
r2_in = {}
for name, cols in MODELS.items():
    m = sm.OLS(d.y, sm.add_constant(d[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": H})
    r2_in[name] = float(m.rsquared)
    extra = ""
    if "xh_o" in cols:
        extra = (f"   xh_o coef={m.params['xh_o']:+.3f} "
                 f"t={m.tvalues['xh_o']:+.2f} p={m.pvalues['xh_o']:.4f}")
    print(f"{name:16s} R2={m.rsquared:.4f}  adjR2={m.rsquared_adj:.4f}{extra}")
print(f"in-sample increment of IV = {r2_in['HAR+IV'] - r2_in['HAR']:+.4f}")

# ----------------------------------------------------- out-of-sample race --
print("\n" + "=" * 74)
print(f"OUT-OF-SAMPLE (expanding window from {OOS_START}, refit monthly, "
      f"embargo {EMBARGO})")
print("=" * 74)

oos_idx = d.index[d.index >= OOS_START]
refit_points = pd.Series(oos_idx).groupby(
    [oos_idx.year, oos_idx.month]).min().values


def run_oos(panel, models):
    """Expanding-window forecasts. Returns (predictions, training s2)."""
    pr = {k: pd.Series(index=oos_idx, dtype=float) for k in models}
    s2 = {k: pd.Series(index=oos_idx, dtype=float) for k in models}
    for name, cols in models.items():
        beta = None
        orth = None                  # orthogonalisation fitted on train only
        sig2 = np.nan
        for t in oos_idx:
            if t in refit_points:
                tr = panel[panel.index < t]
                if EMBARGO:
                    tr = tr.iloc[:-EMBARGO]
                if len(tr) > 200:
                    trX = tr[cols].copy()
                    if "xh_o" in cols:
                        A = np.column_stack([np.ones(len(tr)),
                                             tr[ocols].values.astype(float)])
                        orth = np.linalg.lstsq(A, tr.xh.values, rcond=None)[0]
                        trX["xh_o"] = tr.xh.values - A @ orth
                    fit = sm.OLS(tr.y, sm.add_constant(trX)).fit()
                    beta, sig2 = fit.params, float(fit.mse_resid)
            if beta is None:
                continue
            row = panel.loc[t, cols].copy()
            if "xh_o" in cols:
                a = np.concatenate([[1.0], panel.loc[t, ocols].values.astype(float)])
                row["xh_o"] = float(panel.loc[t, "xh"] - a @ orth)
            x = np.concatenate([[1.0], row.values.astype(float)])
            pr[name][t] = float(x @ beta.values)
            s2[name][t] = sig2
    return pr, s2


preds, s2 = run_oos(d, MODELS)

act = d.loc[oos_idx, "y"]
valid = act.notna()
for k in preds:
    valid &= preds[k].notna()
act = act[valid]
av = np.exp(act)
den = float(np.mean((act - act.mean()) ** 2))
print(f"OOS n = {len(act)}  ({act.index.min().date()} to {act.index.max().date()})")

print(f"\n{'model':16s}{'MSE(log)':>10}{'R2_oos':>9}{'QLIKE':>9}{'QLIKE_bc':>10}")
res = {}
for name in MODELS:
    p = preds[name][valid]
    mse = float(np.mean((act - p) ** 2))
    r2 = 1 - mse / den
    q = qlike(av, np.exp(p))
    qbc = qlike(av, np.exp(p + s2[name][valid] / 2))
    res[name] = dict(pred=p, mse=mse, r2=r2, qlike=q, qlike_bc=qbc)
    print(f"{name:16s}{mse:10.4f}{r2:9.4f}{q:9.3f}{qbc:10.3f}")


# ------------------------------------------------------------ Clark-West --
def clark_west(y, p_small, p_large):
    """Clark & West (2007) MSPE-adjusted statistic for NESTED models."""
    e1 = (y - p_small) ** 2
    e2 = (y - p_large) ** 2
    adj = (p_small - p_large) ** 2
    f = e1 - (e2 - adj)
    m = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC",
                                       cov_kwds={"maxlags": H})
    return float(np.asarray(m.params)[0]), float(np.asarray(m.tvalues)[0])


# ------------------------------------------------------------- placebo --
# one fixed draw of white noise with the sd of xh, through identical machinery
rngp = np.random.default_rng(12345)
dp = d.copy()
dp["xh"] = dp.xh.mean() + rngp.normal(0, d.xh.std(), len(dp))
pp, _ = run_oos(dp, {"P": MODELS["HAR+IV+BKM+XH"]})
pp = pp["P"]

print("\nClark-West (nested, HAC 21 lags)")
pairs = [("HAR", "HAR+IV"), ("HAR+IV", "HAR+IV+BKM"),
         ("HAR+IV+BKM", "HAR+IV+BKM+XH"), ("HAR", "HAR+IV+BKM+XH")]
cwt = {}
for a, b in pairs:
    st, t = clark_west(act.values, res[a]["pred"].values, res[b]["pred"].values)
    cwt[f"{a} -> {b}"] = t
    flag = "**" if t > 1.645 else ""
    print(f"  {a:14s} -> {b:16s} CW={st:+.5f}  t={t:+.2f} {flag}")

stp, tp = clark_west(act.values, res["HAR+IV+BKM"]["pred"].values,
                     pp[valid].values)
print(f"  {'PLACEBO (noise)':14s} -> {'HAR+IV+BKM+noise':16s} "
      f"CW={stp:+.5f}  t={tp:+.2f}")

# --------------------------------------------- robustness: excluding 2020 --
ex = act.index.year != 2020
_, t_iv_ex = clark_west(act[ex].values, res["HAR"]["pred"][ex].values,
                        res["HAR+IV"]["pred"][ex].values)
_, t_xh_ex = clark_west(act[ex].values, res["HAR+IV+BKM"]["pred"][ex].values,
                        res["HAR+IV+BKM+XH"]["pred"][ex].values)
print(f"\nexcluding 2020 (n = {int(ex.sum())}): IV t = {t_iv_ex:+.2f}   "
      f"XH t = {t_xh_ex:+.2f}")

inc = {"IV": res["HAR+IV"]["r2"] - res["HAR"]["r2"],
       "BKM": res["HAR+IV+BKM"]["r2"] - res["HAR+IV"]["r2"],
       "XH": res["HAR+IV+BKM+XH"]["r2"] - res["HAR+IV+BKM"]["r2"]}
print("OOS incremental R2: " + "   ".join(f"{k} {v:+.4f}" for k, v in inc.items()))

out = {
    "spec": "published" if PUBLISHED else "corrected",
    "floor": FLOOR, "embargo": EMBARGO, "zero_return_days": n_zero,
    "panel_n": len(d), "panel_start": str(d.index.min().date()),
    "panel_end": str(d.index.max().date()),
    "oos_n": len(act), "oos_start": str(act.index.min().date()),
    "oos_end": str(act.index.max().date()),
    "orth_r2_full_sample": float(om.rsquared),
    "share_xh_positive": float((ent.xh > 0).mean()),
    "r2_in_sample": r2_in,
    "oos": {k: {kk: v[kk] for kk in ["mse", "r2", "qlike", "qlike_bc"]}
            for k, v in res.items()},
    "oos_increment": inc,
    "cw_t": cwt, "placebo_t": tp,
    "excl_2020": {"n": int(ex.sum()), "iv_t": t_iv_ex, "xh_t": t_xh_ex},
}
fn = "results_published.json" if PUBLISHED else "results.json"
with open(fn, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {fn}")
