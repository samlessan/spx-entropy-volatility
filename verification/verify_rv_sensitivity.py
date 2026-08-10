#!/usr/bin/env python3
"""Two fixes to verify_panel.py, then the final numbers.

    cd ~/urss && source venv/bin/activate && python verification/verify_rv_sensitivity.py

FIX 1  zero-return days. rv = r^2 is exactly 0 when SPY closes unchanged.
       log(clip(1e-12)) = -27.6 against a typical -9.8, so those days enter
       HAR as extreme outliers. Quantify, then floor sensibly and re-test.

FIX 2  power curve with replications. verify_panel.py used ONE draw per effect
       size, giving non-monotonic t-stats and an unreliable MDE. Use 40
       draws per size and report the detection RATE, which is actual power.
"""
import glob

import numpy as np
import pandas as pd

H = 21
NREP = 40
rng = np.random.default_rng(20260805)

# ------------------------------------------------------------------ data --
rvf = pd.read_csv("data/spy_rv.csv")
rvf["date"] = pd.to_datetime(rvf.date)
rvf = rvf.set_index("date").sort_index()
rv = rvf.rv.copy()

print("=" * 66)
print("FIX 1 -- ZERO / NEAR-ZERO RETURN DAYS")
print("=" * 66)
n0 = int((rv == 0).sum())
print(f"exactly zero rv:      {n0}")
for thr, lab in [(1e-10, "1e-10"), (1e-8, "1e-8"), (1e-7, "1e-7")]:
    print(f"rv < {lab:6s}:        {int((rv < thr).sum())}")
print(f"rv 0.1th pctile:      {rv.quantile(0.001):.3e}")
print(f"rv 1st  pctile:       {rv.quantile(0.01):.3e}")
print(f"rv median:            {rv.median():.3e}")
if n0:
    print("\ndates with exactly zero return:")
    print("  " + ", ".join(str(x.date()) for x in rv[rv == 0].index[:12]))

FLOOR = rv[rv > 0].quantile(0.005)     # a realistic small-move day
print(f"\nusing FLOOR = {FLOOR:.3e}  (0.5th pctile of strictly positive rv)")
print(f"  log(1e-12) = {np.log(1e-12):.2f}   vs   log(FLOOR) = {np.log(FLOOR):.2f}"
      f"   vs   log(median) = {np.log(rv.median()):.2f}")


def build(floor, h=H):
    r = rv.clip(lower=floor)
    lrv = np.log(r)
    fwd = pd.Series(np.nan, index=r.index)
    v = r.values
    for i in range(len(r) - h):
        fwd.iloc[i] = v[i + 1:i + 1 + h].mean()
    p = pd.DataFrame({"y": np.log(fwd), "har_d": lrv,
                      "har_w": lrv.rolling(5).mean(),
                      "har_m": lrv.rolling(22).mean()})
    ent = pd.concat([pd.read_csv(f) for f in
                     sorted(glob.glob("data/entropy_*.csv"))], ignore_index=True)
    ent["date"] = pd.to_datetime(ent.date)
    ent = ent.dropna(subset=["xh", "atm_iv", "half_spread",
                             "bkm_skew", "bkm_kurt"])
    ent["gap"] = (ent.dte - 30).abs()
    ent = (ent.sort_values(["date", "gap"]).drop_duplicates("date")
              .set_index("date").sort_index())
    p = p.join(ent[["xh", "atm_iv", "half_spread", "n_used", "dte",
                    "bkm_skew", "bkm_kurt"]], how="inner").dropna()
    p["liv"] = np.log(p.atm_iv)
    p["lsp"] = np.log(p.half_spread.clip(lower=1e-12))
    return p


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
    return np.mean(out), nw_t(out)


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


print("\n--- sensitivity of the verdict to the floor ---")
print(f"{'floor':>12s} {'n':>6s} {'HAR R2':>8s} {'IV gate t':>10s} {'XH t':>8s}")
panels = {}
for lab, fl in [("1e-12", 1e-12), ("0.5 pctile", FLOOR), ("1st pctile",
                                                          rv[rv > 0].quantile(0.01))]:
    p = build(fl)
    p["xh_o"] = orth_oos(p)
    q = p.dropna(subset=["xh_o"])
    b = ols(p[HAR], p.y)
    r2 = 1 - (p.y.values - pred(b, p[HAR])).var() / p.y.values.var()
    _, tiv = oos_t(HAR, HAR + ["liv"], p)
    _, txh = oos_t(BASE, BASE + ["xh_o"], q)
    print(f"{lab:>12s} {len(q):6d} {r2:8.4f} {tiv:+10.2f} {txh:+8.2f}")
    panels[lab] = (p, q)

p, q = panels["0.5 pctile"]

# --------------------------------------------------------------- power --
print("\n" + "=" * 66)
print(f"FIX 2 -- POWER CURVE, {NREP} replications per effect size")
print("=" * 66)
b0 = ols(p[BASE], p.y)
resid = p.y.values - pred(b0, p[BASE])
rs = (resid - resid.mean()) / resid.std()
print(f"residual sd of y after HAR+IV+BKM = {resid.std():.4f}\n")
print(f"{'planted R2':>11s} {'mean t':>8s} {'sd t':>7s} "
      f"{'power':>7s}   (fraction of draws with t > 1.645)")
power = {}
for target in [0.000, 0.002, 0.005, 0.010, 0.020, 0.050]:
    ts = []
    for _ in range(NREP):
        z = rng.standard_normal(len(p))
        z = (z - z.mean()) / z.std()
        sig = np.sqrt(target) * rs + np.sqrt(max(1 - target, 0)) * z
        _, t = oos_t(BASE, BASE + ["plant"], p.assign(plant=sig))
        ts.append(t)
    ts = np.array(ts)
    pw = (ts > 1.645).mean()
    power[target] = pw
    print(f"{target:11.3f} {ts.mean():+8.2f} {ts.std():7.2f} {pw:7.2f}")

mde80 = next((k for k, v in power.items() if v >= 0.80), None)
mde50 = next((k for k, v in power.items() if v >= 0.50), None)

# ------------------------------------------------------------- observed --
b = ols(q[BASE], q.y)
e0 = q.y.values - pred(b, q[BASE])
b = ols(q[BASE + ["xh_o"]], q.y)
e1 = q.y.values - pred(b, q[BASE + ["xh_o"]])
obs = (1 - e1.var() / q.y.values.var()) - (1 - e0.var() / q.y.values.var())
_, txh = oos_t(BASE, BASE + ["xh_o"], q)

print("\n" + "=" * 66)
print("FINAL")
print("=" * 66)
print(f"observed incremental R2 from orthogonalised entropy : {obs:+.5f}")
print(f"observed Clark-West t                              : {txh:+.2f}")
print(f"false-positive rate at true zero                   : {power[0.000]:.2f}"
      f"   (should be ~0.05-0.10)")
print(f"effect size detectable with 50% power              : "
      f"{mde50 if mde50 else 'none tested'}")
print(f"effect size detectable with 80% power              : "
      f"{mde80 if mde80 else 'none tested'}")
print(f"\nfor scale: IV's own incremental R2 is +0.1306, i.e. "
      f"{0.1306 / max(abs(obs), 1e-9):.0f}x the entropy effect")
print("""
HOW TO STATE THIS IN THE WRITE-UP
  Not "entropy has no predictive power" -- that overclaims.
  Instead: "the orthogonalised excess entropy of the risk-neutral density
  contributes an incremental out-of-sample R2 of X, statistically
  indistinguishable from zero, in a test with 80% power to detect an
  incremental R2 of Y. Effects larger than Y can be ruled out; effects
  smaller than Y cannot be distinguished from zero with this sample."
""")
