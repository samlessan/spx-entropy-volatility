#!/usr/bin/env python3
"""
The horse race. Does orthogonalised excess entropy forecast realised
volatility beyond HAR-RV, implied vol and BKM moments?

    cd ~/urss && source venv/bin/activate && python horse_race.py

Inputs   data/entropy_YYYY.csv   (from extract_entropy.py)
         data/spy_rv.csv          (from calibrate_ivol.py)
Output   data/panel.csv           merged daily panel

DESIGN -- fixed in advance:
  Dependent variable: log realised variance over the next 21 trading days,
    from ivol_t (rescaled by median-matching to Parkinson). 21 days matches
    the ~30 calendar-day option horizon.
  One observation per date: the date-expiry pair with dte closest to 30.
  HAR-RV (Corsi 2009): lagged log RV at daily, weekly (5d) and monthly (22d)
    horizons.
  Excess entropy enters ORTHOGONALISED -- residual from regressing xh on
    log IV, log half-spread, n_used, dte, BKM skew and BKM kurtosis. That
    residual is the only form in which entropy is allowed to compete.
  Evaluation: expanding-window out-of-sample from 2018-01-01, refit monthly.
    MSE and QLIKE (Patton 2011) on the variance scale, plus Clark-West for
    the nested comparison. Overlapping horizons -> Newey-West lags = 21 for
    any in-sample t-stats.

Nested models are compared with Clark-West, not Diebold-Mariano: under the
null the larger model's extra parameters are pure noise, which biases DM
toward the small model.
"""

import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

H = 21                       # forecast horizon, trading days
TARGET_DTE = 30
OOS_START = "2018-01-01"
IVOL_SCALE = None  # unused; retained for provenance            # set by median-matching below

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

rv = pd.read_csv("data/spy_rv.csv")
rv["date"] = pd.to_datetime(rv.date)
rv = rv.set_index("date").sort_index()

# rescale ivol_t to daily variance by matching medians to Parkinson
# rv column comes straight from spy_rv.csv (close-to-close from CPrc)
assert "rv" in rv.columns, "spy_rv.csv is missing the rv column"
print(f"dependent variable: close-to-close RV from CPrc, "
      f"median annualised {np.sqrt(rv.rv.median()*252)/0.6745:.4f}")

# ------------------------------------------------------- build the target --
rv["lrv"] = np.log(rv.rv.clip(lower=1e-12))
# forward average variance over the next H days (excludes today)
rv["rv_fwd"] = rv.rv.shift(-1).rolling(H).mean().shift(-(H - 1))
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
d["xh_o"] = om.resid          # FULL SAMPLE -- in-sample table only
print(f"orthogonalisation R2 = {om.rsquared:.3f}, "
      f"residual sd = {om.resid.std():.4f}")
print("NOTE: the OOS loop below re-estimates this orthogonalisation inside")
print("      the expanding window; the full-sample version would leak.")
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
    """Patton (2011) QLIKE. Robust to noise in the volatility proxy."""
    r = actual_var / pred_var
    return float(np.mean(r - np.log(r) - 1))


# --------------------------------------------------------- in-sample fit --
print("\n" + "=" * 74)
print("IN-SAMPLE (Newey-West, 21 lags for the overlapping horizon)")
print("=" * 74)
for name, cols in MODELS.items():
    m = sm.OLS(d.y, sm.add_constant(d[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": H})
    extra = ""
    if "xh_o" in cols:
        extra = (f"   xh_o coef={m.params['xh_o']:+.3f} "
                 f"t={m.tvalues['xh_o']:+.2f} p={m.pvalues['xh_o']:.4f}")
    print(f"{name:16s} R2={m.rsquared:.4f}  adjR2={m.rsquared_adj:.4f}{extra}")

# ----------------------------------------------------- out-of-sample race --
print("\n" + "=" * 74)
print(f"OUT-OF-SAMPLE (expanding window from {OOS_START}, refit monthly)")
print("=" * 74)

oos_idx = d.index[d.index >= OOS_START]
refit_points = pd.Series(oos_idx).groupby(
    [oos_idx.year, oos_idx.month]).min().values

preds = {k: pd.Series(index=oos_idx, dtype=float) for k in MODELS}

for name, cols in MODELS.items():
    beta = None
    orth = None                      # orthogonalisation fitted on train only
    for t in oos_idx:
        if t in refit_points:
            tr = d[d.index < t]
            if len(tr) > 200:
                if "xh_o" in cols:
                    # Re-fit the projection of xh on the controls using ONLY
                    # data available at t, so the regressor carries no
                    # information from the forecast period.
                    # Explicit design matrix: sm.add_constant SKIPS adding an
                    # intercept when any column is already constant, which
                    # would silently desync fit and predict.
                    A = np.column_stack([np.ones(len(tr)),
                                         tr[ocols].values.astype(float)])
                    orth = np.linalg.lstsq(A, tr.xh.values, rcond=None)[0]
                    trX = tr[cols].copy()
                    trX["xh_o"] = tr.xh.values - A @ orth
                else:
                    trX = tr[cols]
                beta = sm.OLS(tr.y, sm.add_constant(trX)).fit().params
        if beta is None:
            continue
        row = d.loc[t, cols].copy()
        if "xh_o" in cols and orth is not None:
            a = np.concatenate([[1.0], d.loc[t, ocols].values.astype(float)])
            row["xh_o"] = float(d.loc[t, "xh"] - a @ orth)
        x = np.concatenate([[1.0], row.values.astype(float)])
        preds[name][t] = float(x @ beta.values)

act = d.loc[oos_idx, "y"]
valid = act.notna()
for k in preds:
    valid &= preds[k].notna()
act = act[valid]
av = np.exp(act)

print(f"\n{'model':16s}{'MSE(log)':>11}{'R2_oos':>9}{'QLIKE':>11}"
      f"{'dQLIKE vs prev':>16}")
prev_q = None
res = {}
for name in MODELS:
    p = preds[name][valid]
    mse = float(np.mean((act - p) ** 2))
    r2 = 1 - mse / float(np.mean((act - act.mean()) ** 2))
    q = qlike(av, np.exp(p))
    res[name] = dict(pred=p, mse=mse, qlike=q)
    dq = "" if prev_q is None else f"{q - prev_q:+.5f}"
    print(f"{name:16s}{mse:11.4f}{r2:9.4f}{q:11.5f}{dq:>16}")
    prev_q = q

# ------------------------------------------------------------ Clark-West --
def clark_west(y, p_small, p_large):
    """
    Clark & West (2007) MSPE-adjusted statistic for NESTED models.
    Positive, significant -> the larger model genuinely helps.
    """
    e1 = (y - p_small) ** 2
    e2 = (y - p_large) ** 2
    adj = (p_small - p_large) ** 2
    f = e1 - (e2 - adj)
    m = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC",
                                       cov_kwds={"maxlags": H})
    return float(np.asarray(m.params)[0]), float(np.asarray(m.tvalues)[0])


# ------------------------------------------------------------- placebo --
# Sanity: replace xh with white noise of the same sd and re-run the final
# comparison. If the placebo is significant, the test procedure is broken.
rngp = np.random.default_rng(12345)
dp = d.copy()
dp["xh"] = dp.xh.mean() + rngp.normal(0, d.xh.std(), len(dp))
pp = pd.Series(index=oos_idx, dtype=float)
beta = None; orth = None
cols = MODELS["HAR+IV+BKM+XH"]
for t in oos_idx:
    if t in refit_points:
        tr = dp[dp.index < t]
        if len(tr) > 200:
            A = np.column_stack([np.ones(len(tr)), tr[ocols].values.astype(float)])
            orth = np.linalg.lstsq(A, tr.xh.values, rcond=None)[0]
            trX = tr[cols].copy(); trX["xh_o"] = tr.xh.values - A @ orth
            beta = sm.OLS(tr.y, sm.add_constant(trX)).fit().params
    if beta is None: continue
    row = dp.loc[t, cols].copy()
    a = np.concatenate([[1.0], dp.loc[t, ocols].values.astype(float)])
    row["xh_o"] = float(dp.loc[t, "xh"] - a @ orth)
    pp[t] = float(np.concatenate([[1.0], row.values.astype(float)]) @ beta.values)

print("\nClark-West (nested, HAC 21 lags)")
pairs = [("HAR", "HAR+IV"), ("HAR+IV", "HAR+IV+BKM"),
         ("HAR+IV+BKM", "HAR+IV+BKM+XH"), ("HAR", "HAR+IV+BKM+XH")]
for a, b in pairs:
    st, t = clark_west(act.values, res[a]["pred"].values, res[b]["pred"].values)
    flag = "**" if t > 1.645 else ""
    print(f"  {a:14s} -> {b:16s} CW={st:+.5f}  t={t:+.2f} {flag}")

stp, tp = clark_west(act.values, res["HAR+IV+BKM"]["pred"].values,
                     pp[valid].values)
print(f"  {'PLACEBO (noise)':14s} -> {'HAR+IV+BKM+noise':16s} "
      f"CW={stp:+.5f}  t={tp:+.2f}")
print("  ^ placebo must NOT be significant; if it is, the test is broken")

print("\n" + "=" * 74)
print("The line that matters: HAR+IV+BKM -> HAR+IV+BKM+XH.")
print("t > 1.645 means excess entropy adds out-of-sample forecasting power")
print("beyond implied vol and the model-free moments. Anything else is a null,")
print("and a null gets written up as a null.")
