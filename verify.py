#!/usr/bin/env python3
"""INDEPENDENT VERIFICATION of the excess-entropy null.

    cd ~/urss && source venv/bin/activate && python verify.py

Reads ONLY data files. Does not import horse_race.py, calibrate_ivol.py, or
any project module. HAR, orthogonalisation and Clark-West are reimplemented
from scratch so that agreement with the original is genuine corroboration
rather than a shared bug.

BLOCK A  external ground truth   -- is the price series really SPY?
BLOCK B  look-ahead audit        -- does y use only future information?
BLOCK C  statistical power       -- could this test have found a signal?
BLOCK D  independent replication -- does the null reproduce?
BLOCK E  horizon robustness      -- is the null specific to H=21?

Block C is the one that matters most. A null with no power is not evidence.
"""
import numpy as np
import pandas as pd
import glob

pd.set_option("display.width", 200)
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


# =====================================================================
# BLOCK A -- external ground truth
# =====================================================================
print("=" * 70)
print("BLOCK A -- EXTERNAL GROUND TRUTH")
print("=" * 70)

d = pd.read_csv("data/spy_iid_ms.csv", low_memory=False)
dc = next(c for c in d.columns if c.upper() == "DATE")
px = pd.DataFrame({"date": pd.to_datetime(d[dc], errors="coerce"),
                   "C": pd.to_numeric(d["CPrc"], errors="coerce")})
px = px.dropna().drop_duplicates("date").set_index("date").sort_index().C

# SPY calendar-year PRICE returns (dividends excluded). These are widely
# known and a corrupted series would not reproduce them.
TRUE_ANNUAL = {2015: -0.007, 2016: 0.095, 2017: 0.194, 2018: -0.062,
               2019: 0.289, 2020: 0.163, 2021: 0.269, 2022: -0.195,
               2023: 0.242, 2024: 0.246}
print("\nSPY calendar-year price return, computed vs known:")
yr_ok = 0
for y, truth in TRUE_ANNUAL.items():
    s = px[px.index.year == y]
    if len(s) < 200:
        print(f"  {y}: only {len(s)} obs, skipped")
        continue
    got = s.iloc[-1] / s.iloc[0] - 1
    hit = abs(got - truth) < 0.035
    yr_ok += hit
    print(f"  {y}:  computed {got:+7.3f}   known {truth:+7.3f}   "
          f"{'ok' if hit else '<-- MISMATCH'}")
check("annual returns match known SPY", yr_ok >= 8, f"{yr_ok}/10 within 3.5pp")

# level sanity at two well-known points
for dt, lo, hi in [("2020-03-23", 215, 235), ("2024-12-31", 575, 600)]:
    t = pd.Timestamp(dt)
    if t in px.index:
        v = px.loc[t]
        check(f"level on {dt}", lo < v < hi, f"CPrc={v:.2f}, expected {lo}-{hi}")

rv_file = pd.read_csv("data/spy_rv.csv")
rv_file["date"] = pd.to_datetime(rv_file.date)
rv_file = rv_file.set_index("date").sort_index()
r_indep = np.log(px).diff()
merged = pd.concat([rv_file.rv.rename("file"),
                    (r_indep ** 2).rename("mine")], axis=1).dropna()
check("spy_rv.csv rv == my recomputation",
      np.allclose(merged.file, merged.mine, rtol=1e-9),
      f"max diff {np.abs(merged.file - merged.mine).max():.2e}")

med_ann = np.sqrt(rv_file.rv.median() * 252) / 0.6745
check("implied median annualised vol in 0.10-0.14",
      0.10 < med_ann < 0.14, f"{med_ann:.4f}")

# =====================================================================
# BLOCK B -- look-ahead audit
# =====================================================================
print("\n" + "=" * 70)
print("BLOCK B -- LOOK-AHEAD AUDIT")
print("=" * 70)

H = 21
rv = rv_file.rv.copy()
# my own construction: mean of rv over t+1 .. t+H
fwd = pd.Series(np.nan, index=rv.index)
vals = rv.values
for i in range(len(rv) - H):
    fwd.iloc[i] = vals[i + 1:i + 1 + H].mean()
# the project's construction
proj = rv.shift(-1).rolling(H).mean().shift(-(H - 1))
both = pd.concat([fwd.rename("mine"), proj.rename("proj")], axis=1).dropna()
check("forward-RV construction has no look-ahead",
      np.allclose(both.mine, both.proj, rtol=1e-9),
      f"n={len(both)}, max diff {np.abs(both.mine - both.proj).max():.2e}")

# explicit: y at date t must be independent of rv at t
lag0 = np.corrcoef(np.log(rv[both.index]), np.log(both.mine))[0, 1]
print(f"  corr(log rv_t, log y_t) = {lag0:.3f}  "
      f"(should be moderate ~0.3-0.5 from persistence, NOT ~1.0)")
check("y is not contemporaneous rv", lag0 < 0.75, f"{lag0:.3f}")

# =====================================================================
# build the panel, independently
# =====================================================================
ent = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/entropy_*.csv"))],
                ignore_index=True)
ent["date"] = pd.to_datetime(ent.date)
ent = ent.dropna(subset=["xh", "atm_iv", "half_spread", "bkm_skew", "bkm_kurt"])
ent["gap"] = (ent.dte - 30).abs()
ent = (ent.sort_values(["date", "gap"]).drop_duplicates("date")
          .set_index("date").sort_index())

lrv = np.log(rv.clip(lower=1e-12))
p = pd.DataFrame({
    "y": np.log(fwd.clip(lower=1e-12)),
    "har_d": lrv,
    "har_w": lrv.rolling(5).mean(),
    "har_m": lrv.rolling(22).mean(),
})
p = p.join(ent[["xh", "atm_iv", "half_spread", "n_used", "dte",
                "bkm_skew", "bkm_kurt"]], how="inner").dropna()
p["liv"] = np.log(p.atm_iv)
p["lsp"] = np.log(p.half_spread.clip(lower=1e-12))
print(f"\npanel: {len(p):,} obs, {p.index.min().date()} to {p.index.max().date()}")

HAR = ["har_d", "har_w", "har_m"]
ORTH = ["liv", "lsp", "n_used", "dte", "bkm_skew", "bkm_kurt"]


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
        c = (e[L:] @ e[:-L]) / n
        g += 2 * (1 - L / (lags + 1)) * c
    return m / np.sqrt(max(g, 1e-30) / n)


def oos(cols_small, cols_big, panel, start="2018-01-01"):
    """Expanding window, refit monthly. Returns (CW mean, CW t)."""
    idx = panel.index
    dates = idx[idx >= pd.Timestamp(start)]
    months = pd.Series(dates).dt.to_period("M").unique()
    rows = []
    for mo in months:
        te = dates[pd.Series(dates).dt.to_period("M").values == mo]
        tr = panel.loc[idx < te[0]]
        if len(tr) < 250:
            continue
        te = panel.loc[te]
        b1 = ols(tr[cols_small], tr.y)
        b2 = ols(tr[cols_big], tr.y)
        y1 = pred(b1, te[cols_small])
        y2 = pred(b2, te[cols_big])
        for yv, a, bb in zip(te.y.values, y1, y2):
            rows.append(((yv - a) ** 2 - ((yv - bb) ** 2 - (a - bb) ** 2),
                         (yv - a) ** 2, (yv - bb) ** 2))
    a = np.array(rows)
    return a[:, 0].mean(), nw_t(a[:, 0]), a[:, 1].mean(), a[:, 2].mean()


def orthogonalise_oos(panel, target="xh", start="2018-01-01"):
    """Residualise `target` on ORTH using only past data, expanding window."""
    out = pd.Series(np.nan, index=panel.index)
    idx = panel.index
    months = pd.Series(idx).dt.to_period("M").unique()
    for mo in months:
        te = idx[pd.Series(idx).dt.to_period("M").values == mo]
        tr = panel.loc[idx < te[0]]
        if len(tr) < 250:
            continue
        b = ols(tr[ORTH], tr[target])
        out.loc[te] = panel.loc[te, target].values - pred(b, panel.loc[te, ORTH])
    return out


# =====================================================================
# BLOCK C -- STATISTICAL POWER  (the critical block)
# =====================================================================
print("\n" + "=" * 70)
print("BLOCK C -- STATISTICAL POWER")
print("=" * 70)
print("Plant a synthetic regressor with a KNOWN incremental R2 and check the")
print("test detects it. If it cannot, the null carries no information.\n")

base = HAR + ["liv", "bkm_skew", "bkm_kurt"]
rng = np.random.default_rng(20260805)

# residual of y after the base model -- the variance a new regressor could explain
b0 = ols(p[base], p.y)
resid = p.y.values - pred(b0, p[base])
print(f"residual sd of y after HAR+IV+BKM = {resid.std():.4f}")

print(f"\n{'planted inc. R2':>16s} {'CW mean':>10s} {'CW t':>8s}   verdict")
detect = {}
for target_r2 in [0.000, 0.002, 0.005, 0.010, 0.020, 0.050]:
    # signal correlated with the residual at the level implying target_r2
    rho = np.sqrt(target_r2)
    z = rng.standard_normal(len(p))
    z = (z - z.mean()) / z.std()
    rs = (resid - resid.mean()) / resid.std()
    sig = rho * rs + np.sqrt(max(1 - target_r2, 0)) * z
    q = p.copy()
    q["plant"] = sig
    cw, t, _, _ = oos(base, base + ["plant"], q)
    detect[target_r2] = t
    v = "DETECTED" if t > 1.645 else ("marginal" if t > 1.0 else "not detected")
    print(f"{target_r2:16.3f} {cw:+10.5f} {t:+8.2f}   {v}")

mde = min([k for k, v in detect.items() if v > 1.645], default=None)
check("test has power (placebo at R2=0 not detected)", detect[0.000] < 1.645,
      f"t={detect[0.000]:+.2f}")
check("minimum detectable effect found", mde is not None,
      f"MDE = incremental R2 of {mde}" if mde else "NO effect size detected -- test is broken")

# =====================================================================
# BLOCK D -- independent replication
# =====================================================================
print("\n" + "=" * 70)
print("BLOCK D -- INDEPENDENT REPLICATION")
print("=" * 70)

print("\nIn-sample R2 (my own OLS):")
prev = None
for lab, cols in [("HAR", HAR), ("HAR+IV", HAR + ["liv"]),
                  ("HAR+IV+BKM", HAR + ["liv", "bkm_skew", "bkm_kurt"])]:
    b = ols(p[cols], p.y)
    e = p.y.values - pred(b, p[cols])
    r2 = 1 - e.var() / p.y.values.var()
    print(f"  {lab:14s} R2={r2:.4f}" + (f"   (+{r2 - prev:.4f})" if prev else ""))
    prev = r2

p["xh_o"] = orthogonalise_oos(p, "xh")
q = p.dropna(subset=["xh_o"])
b = ols(q[HAR + ["liv", "bkm_skew", "bkm_kurt"]], q.y)
e0 = q.y.values - pred(b, q[HAR + ["liv", "bkm_skew", "bkm_kurt"]])
b = ols(q[HAR + ["liv", "bkm_skew", "bkm_kurt", "xh_o"]], q.y)
e1 = q.y.values - pred(b, q[HAR + ["liv", "bkm_skew", "bkm_kurt", "xh_o"]])
r2_inc = (1 - e1.var() / q.y.values.var()) - (1 - e0.var() / q.y.values.var())
print(f"  incremental R2 from xh_o = {r2_inc:+.5f}")

print("\nClark-West, out-of-sample (my own implementation):")
tests = [
    ("HAR -> HAR+IV", HAR, HAR + ["liv"], p),
    ("HAR+IV -> +BKM", HAR + ["liv"], HAR + ["liv", "bkm_skew", "bkm_kurt"], p),
    ("HAR+IV+BKM -> +XH", base, base + ["xh_o"], q),
    ("HAR+IV+BKM -> +noise", base, base + ["noise"],
     q.assign(noise=rng.standard_normal(len(q)))),
]
res = {}
for lab, cs, cb, panel in tests:
    cw, t, m1, m2 = oos(cs, cb, panel)
    res[lab] = t
    print(f"  {lab:22s} CW={cw:+.5f}  t={t:+.2f}  "
          f"MSE {m1:.4f} -> {m2:.4f}")

check("calibration gate: IV beats HAR", res["HAR -> HAR+IV"] > 1.645,
      f"t={res['HAR -> HAR+IV']:+.2f}")
check("placebo not significant", res["HAR+IV+BKM -> +noise"] < 1.645,
      f"t={res['HAR+IV+BKM -> +noise']:+.2f}")
print(f"\n  ENTROPY VERDICT: t={res['HAR+IV+BKM -> +XH']:+.2f}  -> "
      f"{'SIGNAL' if res['HAR+IV+BKM -> +XH'] > 1.645 else 'NULL'}")

# =====================================================================
# BLOCK E -- horizon robustness
# =====================================================================
print("\n" + "=" * 70)
print("BLOCK E -- HORIZON ROBUSTNESS (is the null specific to H=21?)")
print("=" * 70)
print(f"\n{'H':>4s} {'n':>6s} {'IV gate t':>10s} {'XH t':>8s}   verdict")
for h in [5, 10, 21, 42, 63]:
    f2 = pd.Series(np.nan, index=rv.index)
    v2 = rv.values
    for i in range(len(rv) - h):
        f2.iloc[i] = v2[i + 1:i + 1 + h].mean()
    pp = p.drop(columns=["y", "xh_o"]).join(
        np.log(f2.clip(lower=1e-12)).rename("y"), how="inner").dropna()
    pp["xh_o"] = orthogonalise_oos(pp, "xh")
    qq = pp.dropna(subset=["xh_o"])
    _, tiv, _, _ = oos(HAR, HAR + ["liv"], pp)
    _, txh, _, _ = oos(base, base + ["xh_o"], qq)
    print(f"{h:4d} {len(qq):6d} {tiv:+10.2f} {txh:+8.2f}   "
          f"{'SIGNAL' if txh > 1.645 else 'null'}")

# =====================================================================
print("\n" + "=" * 70)
if FAIL:
    print("VERIFICATION FAILED on:", "; ".join(FAIL))
    print("Do NOT write up until these are resolved.")
else:
    print("ALL CHECKS PASSED.")
    print("The null is verified subject to the stated minimum detectable effect.")
print("=" * 70)
print("""
WHAT THIS DOES NOT VERIFY
  1. The Breeden-Litzenberger extraction itself. xh comes from
     data/entropy_*.csv, which this script takes as given. The synthetic
     harness covered BL recovery; an independent re-extraction of a few
     dates with different code is still worth doing.
  2. Whether SPX option data are correct at source (OptionMetrics).
  3. Whether entropy forecasts something OTHER than realised volatility
     (e.g. tail risk, jump risk, option returns). The null is specific to
     the stated dependent variable.
""")
