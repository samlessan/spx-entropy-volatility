#!/usr/bin/env python3
"""Persistence-adjusted power curve. The last number.

    cd ~/urss && source venv/bin/activate && python verification/verify_power.py

verify_rv_sensitivity.py planted an iid signal. xh_o is persistent, and with 21-day
overlapping horizons a persistent regressor has fewer effective independent
observations, so true power is lower. This measures xh_o's autocorrelation
and re-runs the power curve with a planted signal matched to it.

Every out-of-sample fit uses a 21-row embargo: training rows whose 21-day
targets overlap the test month are dropped. 200 replications per effect size;
power is reported with its binomial standard error. The planted effect is
reported in two units: its share of the residual variance after HAR+IV+BKM
(the design parameter) and the out-of-sample incremental R2 it actually
achieves (the same units as the headline table).
"""
import glob

import numpy as np
import pandas as pd

H, NREP = 21, 200
rng = np.random.default_rng(7)

rvf = pd.read_csv("data/spy_rv.csv")
rvf["date"] = pd.to_datetime(rvf.date)
rv = rvf.set_index("date").sort_index().rv
FLOOR = rv[rv > 0].quantile(0.005)
r = rv.clip(lower=FLOOR)
lrv = np.log(r)

fwd = pd.Series(np.nan, index=r.index)
v = r.values
for i in range(len(r) - H):
    fwd.iloc[i] = v[i + 1:i + 1 + H].mean()

p = pd.DataFrame({"y": np.log(fwd), "har_d": lrv,
                  "har_w": lrv.rolling(5).mean(),
                  "har_m": lrv.rolling(22).mean()})
ent = pd.concat([pd.read_csv(f) for f in
                 sorted(glob.glob("data/entropy_*.csv"))], ignore_index=True)
ent["date"] = pd.to_datetime(ent.date)
ent = ent.dropna(subset=["xh", "atm_iv", "half_spread", "bkm_skew", "bkm_kurt"])
ent["gap"] = (ent.dte - 30).abs()
ent = (ent.sort_values(["date", "gap"]).drop_duplicates("date")
          .set_index("date").sort_index())
p = p.join(ent[["xh", "atm_iv", "half_spread", "n_used", "dte",
                "bkm_skew", "bkm_kurt"]], how="inner").dropna()
p["liv"] = np.log(p.atm_iv)
p["lsp"] = np.log(p.half_spread.clip(lower=1e-12))

HAR = ["har_d", "har_w", "har_m"]
ORTH = ["liv", "lsp", "n_used", "dte", "bkm_skew", "bkm_kurt"]
BASE = HAR + ["liv", "bkm_skew", "bkm_kurt"]


def ols(X, y):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    b, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return b


def pred(b, X):
    return np.column_stack([np.ones(len(X)), np.asarray(X, float)]) @ b


def nw_t(f, lags=21):
    f = np.asarray(f, float)
    n, m = len(f), f.mean()
    e = f - m
    g = (e @ e) / n
    for L in range(1, lags + 1):
        g += 2 * (1 - L / (lags + 1)) * ((e[L:] @ e[:-L]) / n)
    return m / np.sqrt(max(g, 1e-30) / n)


def oos_t(cs, cb, panel, start="2018-01-01"):
    """Clark-West t and OOS incremental R2 of cb over cs.

    Expanding window, refit monthly, 21-row embargo: at the first date of a
    test month the training set is every earlier row except the last H, whose
    21-day targets overlap the test month. Panel must be sorted by date.
    """
    idx = panel.index
    y = panel.y.values.astype(float)
    Xs = np.column_stack([np.ones(len(panel)), panel[cs].values.astype(float)])
    Xb = np.column_stack([np.ones(len(panel)), panel[cb].values.astype(float)])
    pos = np.flatnonzero(idx >= pd.Timestamp(start))
    per = idx[pos].to_period("M").asi8
    ys, p1, p2 = [], [], []
    for mo in np.unique(per):
        te = pos[per == mo]
        ntr = te[0] - H                 # rows 0 .. te[0]-1, minus the embargo
        if ntr < 250:
            continue
        b1 = np.linalg.lstsq(Xs[:ntr], y[:ntr], rcond=None)[0]
        b2 = np.linalg.lstsq(Xb[:ntr], y[:ntr], rcond=None)[0]
        ys.append(y[te])
        p1.append(Xs[te] @ b1)
        p2.append(Xb[te] @ b2)
    yy, a, b = np.concatenate(ys), np.concatenate(p1), np.concatenate(p2)
    f = (yy - a) ** 2 - ((yy - b) ** 2 - (a - b) ** 2)
    inc = (np.mean((yy - a) ** 2) - np.mean((yy - b) ** 2)) / np.var(yy)
    return nw_t(f), inc, len(yy)


def orth_oos(panel, col="xh"):
    out = pd.Series(np.nan, index=panel.index)
    idx = panel.index
    per = pd.Series(idx).dt.to_period("M")
    for mo in per.unique():
        te = idx[per.values == mo]
        tr = panel.loc[idx < te[0]]
        if len(tr) < 250:
            continue
        b = ols(tr[ORTH], tr[col])
        out.loc[te] = panel.loc[te, col].values - pred(b, panel.loc[te, ORTH])
    return out


p["xh_o"] = orth_oos(p)
q = p.dropna(subset=["xh_o"])
s = q.xh_o
rho1 = s.autocorr(1)
print("=" * 62)
print("PERSISTENCE OF THE ACTUAL REGRESSOR")
print("=" * 62)
print(f"AR(1) of xh_o        : {rho1:.3f}")
for L in [5, 21, 63]:
    print(f"AR({L:2d}) of xh_o       : {s.autocorr(L):.3f}")
print(f"AR(1) of raw xh      : {q.xh.autocorr(1):.3f}   "
      f"(orthogonalisation removes some persistence)")
eff = len(q) * (1 - rho1) / (1 + rho1) if rho1 > 0 else len(q)
print(f"\nn = {len(q)},  effective n at AR(1)={rho1:.2f} is roughly {eff:.0f}")

print("\n" + "=" * 62)
print(f"POWER CURVE WITH PERSISTENCE MATCHED TO xh_o ({NREP} reps)")
print("=" * 62)
b0 = ols(p[BASE], p.y)
resid = p.y.values - pred(b0, p[BASE])
rs = (resid - resid.mean()) / resid.std()
n = len(p)

base_r2 = 1 - resid.var() / p.y.values.var()
print(f"in-sample R2 of HAR+IV+BKM = {base_r2:.4f}; a planted share k of the "
      f"residual variance is ~{1 - base_r2:.2f}k of total variance\n")
print(f"{'planted k':>10s} {'mean t':>7s} {'power':>6s} {'+/- se':>7s} "
      f"{'OOS incR2':>10s} {'IS incR2':>9s}")
GRID = [0.000, 0.005, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040, 0.050]
power, oos_inc = {}, {}
for target in GRID:
    ts, incs, iss = [], [], []
    for _ in range(NREP):
        # AR(1) noise matched to xh_o's persistence
        e = rng.standard_normal(n)
        z = np.empty(n)
        z[0] = e[0]
        for i in range(1, n):
            z[i] = rho1 * z[i - 1] + np.sqrt(max(1 - rho1 ** 2, 1e-9)) * e[i]
        z = (z - z.mean()) / z.std()
        # persistent component of the residual, so the signal is realistic
        rp = pd.Series(rs).rolling(max(int(1 / max(1 - rho1, 0.02)), 1),
                                   min_periods=1).mean().values
        rp = (rp - rp.mean()) / rp.std()
        sig = np.sqrt(target) * rp + np.sqrt(max(1 - target, 0)) * z
        pa = p.assign(plant=sig)
        t, inc, _ = oos_t(BASE, BASE + ["plant"], pa)
        ts.append(t)
        incs.append(inc)
        bb = ols(pa[BASE + ["plant"]], pa.y)
        iss.append((1 - (pa.y.values - pred(bb, pa[BASE + ["plant"]])).var()
                    / pa.y.values.var()) - base_r2)
    ts = np.array(ts)
    pw = (ts > 1.645).mean()
    power[target], oos_inc[target] = pw, float(np.mean(incs))
    print(f"{target:10.4f} {ts.mean():+7.2f} {pw:6.2f} "
          f"{np.sqrt(pw * (1 - pw) / NREP):7.3f} {np.mean(incs):+10.4f} "
          f"{np.mean(iss):+9.4f}")

m80 = next((k for k, val in power.items() if val >= 0.80), None)
m50 = next((k for k, val in power.items() if val >= 0.50), None)

b = ols(q[BASE], q.y)
e0 = q.y.values - pred(b, q[BASE])
b = ols(q[BASE + ["xh_o"]], q.y)
e1 = q.y.values - pred(b, q[BASE + ["xh_o"]])
obs_is = (1 - e1.var() / q.y.values.var()) - (1 - e0.var() / q.y.values.var())
tobs, obs_oos, n_oos = oos_t(BASE, BASE + ["xh_o"], q)
HAR_IV = HAR + ["liv"]
t_iv, iv_oos, _ = oos_t(HAR, HAR_IV, p)
b = ols(p[HAR], p.y)
r2h = 1 - (p.y.values - pred(b, p[HAR])).var() / p.y.values.var()
b = ols(p[HAR_IV], p.y)
r2hi = 1 - (p.y.values - pred(b, p[HAR_IV])).var() / p.y.values.var()

print("\n" + "=" * 62)
print("NUMBERS FOR THE REPORT")
print("=" * 62)
print(f"  rows after 250-row burn-in            : {len(q)}  (from {q.index.min().date()})")
print(f"  out-of-sample n (2018 onwards)        : {n_oos}")
print(f"  OOS incremental R2 of xh_o            : {obs_oos:+.5f}")
print(f"  IN-SAMPLE incremental R2 of xh_o      : {obs_is:+.5f}")
print(f"  observed Clark-West t                 : {tobs:+.2f}")
print(f"  IV: OOS incremental R2 {iv_oos:+.4f} (t {t_iv:+.2f}), "
      f"in-sample {r2hi - r2h:+.4f}")
print(f"  rejection rate at true zero (size)    : {power[0.000]:.3f}")
print(f"  planted k at 50% power                : {m50}")
print(f"  planted k at 80% power                : {m80}"
      + (f"  (= OOS incremental R2 of {oos_inc[m80]:+.4f})" if m80 else ""))
