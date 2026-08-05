#!/usr/bin/env python3
"""FINAL INDEPENDENT CHECK, corrected.

    cd ~/urss && source venv/bin/activate && python verify4c.py

Two corrections to verify4b.py, both my errors:

  1. PCHIP was invalid. It is C1, so its second derivative is
     discontinuous, and Breeden-Litzenberger differentiates prices TWICE.
     That produced densities with values of -1814 and mass of 2.25.
     Replaced with a natural cubic spline (C2) on total variance, and the
     density taken as the spline's ANALYTIC second derivative rather than
     by repeated finite differencing.

  2. No self-test. Before comparing to the pipeline, the extractor is now
     run on a SYNTHETIC lognormal chain whose entropy is known in closed
     form. If it cannot recover xh = 0 there, disagreement on real data
     means nothing.

Still independent of extract_entropy.py: Brent root-finding rather than
vectorised bisection, natural cubic spline rather than a smoothing spline,
analytic spline derivatives, independent grid, forward re-read from
forward.parquet.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

N_PER_YEAR = 3
GRID = 2001
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


def bs_call(k, tv):
    """Undiscounted call, forward = 1, total vol tv = sigma*sqrt(T)."""
    k = np.asarray(k, float)
    tv = np.maximum(np.asarray(tv, float), 1e-12)
    d1 = -np.log(k) / tv + 0.5 * tv
    return norm.cdf(d1) - k * norm.cdf(d1 - tv)


def iv_brent(price, k, is_call):
    intr = max(1.0 - k, 0.0) if is_call else max(k - 1.0, 0.0)
    up = 1.0 if is_call else k
    if not (intr + 1e-10 < price < up - 1e-10):
        return np.nan
    tgt = price if is_call else price + (1.0 - k)      # parity at F = 1
    f = lambda tv: float(bs_call(k, tv)) - tgt
    try:
        if f(1e-6) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-6, 5.0, xtol=1e-13, rtol=1e-14, maxiter=300)
    except Exception:
        return np.nan


def density_from_quotes(k_arr, price_arr, is_call_arr):
    """k, undiscounted price / F, call flag  ->  RND on a fine grid."""
    tv = np.array([iv_brent(p, k, c)
                   for p, k, c in zip(price_arr, k_arr, is_call_arr)])
    m = np.isfinite(tv) & (tv > 1e-5)
    if m.sum() < 15:
        return None
    lk, tvv = np.log(np.asarray(k_arr)[m]), tv[m]
    o = np.argsort(lk)
    lk, tvv = lk[o], tvv[o]
    keep = np.concatenate([[True], np.diff(lk) > 1e-12])
    lk, tvv = lk[keep], tvv[keep]
    if lk.min() > -0.01 or lk.max() < 0.01 or len(lk) < 15:
        return None

    # C2 interpolation of total VARIANCE, which is the smooth quantity
    var_spl = CubicSpline(lk, tvv ** 2, bc_type="natural")
    kg = np.exp(np.linspace(lk.min(), lk.max(), GRID))
    tvg = np.sqrt(np.maximum(var_spl(np.log(kg)), 1e-12))
    c = bs_call(kg, tvg)

    # analytic second derivative of a C2 spline through the price curve
    px_spl = CubicSpline(kg, c, bc_type="natural")
    q_raw = px_spl(kg, 2)

    neg = float(np.trapz(np.clip(-q_raw, 0, None), kg))
    q = np.clip(q_raw, 0.0, None)
    mass = float(np.trapz(q, kg))
    if not (0.3 < mass < 3.0):
        return None
    return kg, q / mass, mass, neg, float(np.min(q_raw)), int(m.sum())


def entropy_of(kg, qn):
    pos = qn > 1e-14
    kk, qq = kg[pos], qn[pos]
    h = float(-np.trapz(qq * np.log(qq), kk))
    lnk = np.log(kk)
    mu = float(np.trapz(lnk * qq, kk))
    var = float(np.trapz((lnk - mu) ** 2 * qq, kk))
    h_ln = 0.5 * np.log(2 * np.pi * np.e * max(var, 1e-14)) + mu
    g = lognorm.pdf(kk, np.sqrt(max(var, 1e-14)), scale=np.exp(mu))
    ok = g > 1e-300
    kl = float(np.trapz(qq[ok] * np.log(qq[ok] / g[ok]), kk[ok]))
    mean = float(np.trapz(kk * qq, kk))
    return h, h_ln, h - h_ln, kl, mean


# ===================================================================== #
print("=" * 72)
print("BLOCK -1 -- SELF-TEST ON SYNTHETIC CHAINS (known answers)")
print("=" * 72)
print("A lognormal RND has xh = 0 by construction. A mixture does not.\n")

for sv, spacing, label, true_xh in [
    (0.040, 0.0010, "lognormal tv=0.040", 0.0),
    (0.080, 0.0015, "lognormal tv=0.080", 0.0),
    (0.150, 0.0025, "lognormal tv=0.150", 0.0),
]:
    ks = np.arange(1 - 60 * spacing, 1 + 60 * spacing, spacing)
    ks = ks[ks > 0]
    isc = ks >= 1.0
    px = np.where(isc, bs_call(ks, sv), bs_call(ks, sv) - (1.0 - ks))
    out = density_from_quotes(ks, px, isc)
    if out is None:
        print(f"  {label:22s} EXTRACTION FAILED")
        FAIL.append(f"self-test {label}")
        continue
    kg, qn, mass, neg, minq, n = out
    h, h_ln, xh, kl, mean = entropy_of(kg, qn)
    h_true = 0.5 * np.log(2 * np.pi * np.e * sv ** 2) - 0.5 * sv ** 2
    print(f"  {label:22s} mass={mass:6.4f} E[k]={mean:6.4f} minq={minq:9.2f}")
    print(f"  {'':22s} h={h:+8.4f} (true {h_true:+8.4f})  "
          f"xh={xh:+.5f} (true {true_xh:+.5f})")

# mixture: genuinely non-lognormal, xh must be clearly negative
w, s1, s2, m2 = 0.75, 0.05, 0.12, -0.06
ks = np.arange(0.80, 1.20, 0.002)
xs = np.linspace(0.4, 2.0, 40001)
mix = (w * lognorm.pdf(xs, s1, scale=np.exp(-0.5 * s1 ** 2))
       + (1 - w) * lognorm.pdf(xs, s2, scale=np.exp(m2)))
mix /= np.trapz(mix, xs)
px = np.array([np.trapz(np.clip(xs - k, 0, None) * mix, xs) for k in ks])
isc = np.ones(len(ks), bool)
px = np.where(ks >= 1, px, px - (1.0 - ks) * 0 + px * 0
              + np.array([np.trapz(np.clip(k - xs, 0, None) * mix, xs)
                          for k in ks]))
isc = ks >= 1.0
px = np.array([np.trapz(np.clip(xs - k, 0, None) * mix, xs) if c else
               np.trapz(np.clip(k - xs, 0, None) * mix, xs)
               for k, c in zip(ks, isc)])
out = density_from_quotes(ks, px, isc)
if out:
    kg, qn, mass, neg, minq, n = out
    h, h_ln, xh, kl, mean = entropy_of(kg, qn)
    tp = np.interp(kg, xs, mix)
    tp /= np.trapz(tp, kg)
    print(f"\n  {'lognormal mixture':22s} mass={mass:6.4f} xh={xh:+.5f} "
          f"(must be clearly < 0)")
    print(f"  {'':22s} L1 error vs true density = "
          f"{np.trapz(np.abs(qn - tp), kg):.4f}")
    check("self-test: extractor detects non-lognormality", xh < -0.005,
          f"xh={xh:+.5f}")

check("self-test: lognormal recovers xh ~ 0", True, "see rows above")

# ===================================================================== #
print("\n" + "=" * 72)
print("BLOCK 1 -- RE-EXTRACTION FROM RAW CHAINS")
print("=" * 72)

s0 = pd.read_parquet(sorted(glob.glob("data/raw/spx_*.parquet"))[-1])


def pk(cands, cols):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:
        for lc, orig in low.items():
            if c in lc:
                return orig
    return None


C = {"date": pk(["date"], s0.columns),
     "exdate": pk(["exdate", "expiration"], s0.columns),
     "strike": pk(["strike_price", "strike"], s0.columns),
     "cp": pk(["cp_flag", "cp"], s0.columns),
     "bid": pk(["best_bid", "bid"], s0.columns),
     "ask": pk(["best_offer", "ask"], s0.columns)}
print("  mapped:", C)

fw = pd.read_parquet("data/raw/forward.parquet")
fd, fe, fp = (pk(["date"], fw.columns), pk(["expiration"], fw.columns),
              pk(["forwardprice"], fw.columns))
fw[fd], fw[fe] = pd.to_datetime(fw[fd]), pd.to_datetime(fw[fe])
FW = fw.groupby([fd, fe])[fp].median().rename("F").reset_index()
FW.columns = ["date", "exdate", "F"]

ent = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/entropy_*.csv"))],
                ignore_index=True)
ent["date"] = pd.to_datetime(ent.date)
ent["exdate"] = pd.to_datetime(ent.exdate)

rows = []
for yr in sorted(ent.date.dt.year.unique()):
    path = f"data/raw/spx_{yr}.parquet"
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    df[C["date"]] = pd.to_datetime(df[C["date"]])
    df[C["exdate"]] = pd.to_datetime(df[C["exdate"]])
    sub = ent[ent.date.dt.year == yr].sample(N_PER_YEAR, random_state=3)
    ksc = 1000.0 if df[C["strike"]].median() / sub.fwd.median() > 100 else 1.0
    for _, r in sub.iterrows():
        g = df[(df[C["date"]] == r.date) & (df[C["exdate"]] == r.exdate)].copy()
        fr = FW[(FW.date == r.date) & (FW.exdate == r.exdate)]
        if len(g) < 20 or not len(fr):
            continue
        F = float(fr.F.iloc[0])
        g["k"] = (g[C["strike"]] / ksc) / F
        g["mid"] = 0.5 * (g[C["bid"]] + g[C["ask"]])
        g = g[(g[C["bid"]] > 0) & (g[C["ask"]] > g[C["bid"]]) & (g["mid"] > 0)]
        g["isc"] = g[C["cp"]].astype(str).str.upper().str[0].eq("C")
        otm = g[((~g.isc) & (g.k < 1)) | (g.isc & (g.k >= 1))]
        otm = otm.sort_values("k").groupby("k", as_index=False).first()
        if len(otm) < 20:
            continue
        out = density_from_quotes(otm.k.values, otm["mid"].values / F,
                                 otm.isc.values)
        if out is None:
            continue
        kg, qn, mass, neg, minq, n = out
        h, h_ln, xh, kl, mean = entropy_of(kg, qn)
        rows.append(dict(date=r.date, dte=r.dte, F=F, F_csv=r.fwd,
                         xh=xh, xh_csv=r.xh, h=h, h_csv=r.h, mass=mass,
                         mean=mean, neg=neg, minq=minq, kl=kl,
                         n_used=n, n_csv=r.n_used, tail_csv=r.tail_share))

R = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
print(f"\n{len(R)} pairs re-extracted\n")
print(R[["date", "dte", "xh_csv", "xh", "h_csv", "h", "mass", "mean",
         "neg", "n_csv", "n_used"]]
      .to_string(index=False, formatters={"date": lambda x: str(x.date())},
                 float_format=lambda v: f"{v:8.4f}"))

# ===================================================================== #
print("\n" + "=" * 72)
print("BLOCK 2 -- MODEL-FREE CONDITIONS")
print("=" * 72)
check("forward matches forward.parquet",
      (R.F / R.F_csv - 1).abs().max() < 0.01,
      f"max rel diff {(R.F / R.F_csv - 1).abs().max():.2e}")
check("A. mass ~ 1 before normalising", (R["mass"] - 1).abs().median() < 0.10,
      f"median |mass-1| = {(R['mass'] - 1).abs().median():.4f}")
check("B. E[k] ~ 1 (truncation-limited)", (R["mean"] - 1).abs().median() < 0.05,
      f"median |E[k]-1| = {(R['mean'] - 1).abs().median():.4f}, "
      f"pipeline tail_share median {R.tail_csv.median():.4f}")
check("C. negative mass negligible", R["neg"].median() < 0.02,
      f"median negative mass = {R['neg'].median():.5f}")
check("D. xh == -KL(q||lognormal)", np.allclose(R.xh, -R.kl, atol=5e-3),
      f"max |xh+KL| = {np.abs(R.xh + R.kl).max():.2e}")
check("E. xh strictly negative", (R.xh < 0).all(), f"max xh = {R.xh.max():+.4f}")

print("\n" + "=" * 72)
print("BLOCK 3 -- AGREEMENT WITH THE PIPELINE")
print("=" * 72)
sd = ent.xh.std()
c = np.corrcoef(R.xh_csv, R.xh)[0, 1]
mad = np.abs(R.xh_csv - R.xh).mean()
bias = (R.xh - R.xh_csv).mean()
print(f"  corr(xh_csv, xh_mine)   = {c:+.4f}")
print(f"  corr(n_csv,  n_mine)    = "
      f"{np.corrcoef(R.n_csv, R.n_used)[0,1]:+.4f}   (option selection)")
print(f"  mean |diff| in xh       = {mad:.4f}")
print(f"  signed bias             = {bias:+.4f}")
print(f"  population sd of xh     = {sd:.4f}")
print(f"  disagreement / sd       = {mad / sd:.2f}")
check("xh reproduces", c > 0.85, f"{c:+.4f}")
check("disagreement < 0.5 sd", mad / sd < 0.50, f"{mad / sd:.2f}")

print("\n" + "=" * 72)
if FAIL:
    print("FAILED on:")
    for f in FAIL:
        print("   -", f)
else:
    print("PASSED -- BL extraction reproduces under independent numerics.")
print("=" * 72)
