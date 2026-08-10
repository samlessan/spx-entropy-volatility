#!/usr/bin/env python3
"""Persistence-adjusted power curve. The last number.

    cd ~/urss && source venv/bin/activate && python verification/verify_power.py

verify_rv_sensitivity.py planted an iid signal. xh_o is persistent, and with 21-day
overlapping horizons a persistent regressor has fewer effective independent
observations, so true power is lower. This measures xh_o's autocorrelation
and re-runs the power curve with a planted signal matched to it.
"""
import glob

import numpy as np
import pandas as pd

H, NREP = 21, 40
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
    idx = panel.index
    dates = idx[idx >= pd.Timestamp(start)]
    per = pd.Series(dates).dt.to_period("M")
    out = []
    for mo in per.unique():
        te = dates[per.values == mo]
        tr = panel.loc[idx < te[0]]
        if len(tr) < 250:
            continue
        te = panel.loc[te]
        b1, b2 = ols(tr[cs], tr.y), ols(tr[cb], tr.y)
        y1, y2 = pred(b1, te[cs]), pred(b2, te[cb])
        out.extend((yv - a) ** 2 - ((yv - bb) ** 2 - (a - bb) ** 2)
                   for yv, a, bb in zip(te.y.values, y1, y2))
    return nw_t(out)


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

print(f"\n{'planted R2':>11s} {'mean t':>8s} {'sd t':>7s} {'power':>7s}")
power = {}
for target in [0.000, 0.002, 0.005, 0.010, 0.020, 0.050]:
    ts = []
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
        ts.append(oos_t(BASE, BASE + ["plant"], p.assign(plant=sig)))
    ts = np.array(ts)
    power[target] = (ts > 1.645).mean()
    print(f"{target:11.3f} {ts.mean():+8.2f} {ts.std():7.2f} {power[target]:7.2f}")

m80 = next((k for k, val in power.items() if val >= 0.80), None)
m50 = next((k for k, val in power.items() if val >= 0.50), None)

b = ols(q[BASE], q.y)
e0 = q.y.values - pred(b, q[BASE])
b = ols(q[BASE + ["xh_o"]], q.y)
e1 = q.y.values - pred(b, q[BASE + ["xh_o"]])
obs = (1 - e1.var() / q.y.values.var()) - (1 - e0.var() / q.y.values.var())
tobs = oos_t(BASE, BASE + ["xh_o"], q)

print("\n" + "=" * 62)
print("NUMBERS FOR THE REPORT")
print("=" * 62)
print(f"  n (out-of-sample panel)              : {len(q)}")
print(f"  observed incremental R2 of xh_o      : {obs:+.5f}")
print(f"  observed Clark-West t                : {tobs:+.2f}")
print(f"  size at true zero (should be <= 0.10): {power[0.000]:.2f}")
print(f"  MDE at 50% power                     : {m50}")
print(f"  MDE at 80% power                     : {m80}")
print(f"  IV's own incremental R2              : +0.13060")
if m80:
    print(f"\n  ratio: the 80%-power threshold is {m80 / max(obs, 1e-9):.0f}x "
          f"the observed effect")
