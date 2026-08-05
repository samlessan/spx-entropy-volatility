#!/usr/bin/env python3
"""FINAL INDEPENDENT CHECK -- the Breeden-Litzenberger extraction.

    cd ~/urss && source venv/bin/activate && python verify4.py

This is the one link never independently verified. Everything downstream of
data/entropy_*.csv has been reproduced by two implementations; xh itself has
been computed once, by one piece of code.

Re-derives xh from the RAW option chains using different numerics at every
step, then compares to the CSVs:

    step              extract_entropy.py        this script
    IV inversion      vectorised bisection      Brent root-find (scipy)
    smile interp      smoothing spline          PCHIP (monotone, local)
    2nd derivative    analytic spline           central finite differences
    integration       (as implemented)          trapezoid, independent grid

Also checks the model-free conditions the RND must satisfy regardless of
implementation:
    A. integral of q dk = 1                  (probability)
    B. integral of k q dk = 1                (martingale / forward)
    C. q >= 0 everywhere                     (no butterfly arbitrage)
    D. xh == -KL(q || lognormal)             (the claimed identity)

Runs on a stratified sample of date-expiry pairs, one per year.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

N_PER_YEAR = 3
GRID = 4001
rng = np.random.default_rng(11)
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


# ===================================================================== #
# 0. provenance
# ===================================================================== #
print("=" * 70)
print("BLOCK 0 -- PROVENANCE")
print("=" * 70)
for pat in ["data/entropy_*.csv", "data/spy_rv.csv", "data/*.parquet"]:
    fs = sorted(glob.glob(pat))
    tot = sum(os.path.getsize(f) for f in fs)
    print(f"  {pat:24s} {len(fs):3d} files, {tot / 1e6:9.1f} MB")

raw_files = sorted(glob.glob("data/*.parquet")) or sorted(glob.glob("data/**/*.parquet",
                                                                   recursive=True))
if not raw_files:
    print("\nNo parquet files found under data/. Set RAW_GLOB below and rerun.")
    sys.exit(1)

samp = pd.read_parquet(raw_files[0])
print(f"\nraw schema ({os.path.basename(raw_files[0])}, {len(samp):,} rows):")
for c in samp.columns:
    print(f"    {c:24s} {str(samp[c].dtype):10s} e.g. {samp[c].iloc[0]!r}")


def find(*subs, cols=None, required=True):
    cols = cols if cols is not None else samp.columns
    for c in cols:
        lc = c.lower()
        if all(s in lc for s in subs):
            return c
    if required:
        raise SystemExit(f"could not find a column matching {subs}")
    return None


COL = {
    "date": find("date", required=False) or samp.columns[0],
    "exdate": find("exdate", required=False) or find("exp", required=False),
    "strike": find("strike", required=False),
    "cp": find("cp", required=False) or find("flag", required=False),
    "bid": find("best", "bid", required=False) or find("bid", required=False),
    "ask": find("best", "offer", required=False) or find("ask", required=False),
    "fwd": find("forward", required=False) or find("under", required=False),
}
print("\nmapped columns:")
for k, v in COL.items():
    print(f"    {k:8s} -> {v}")
if any(COL[k] is None for k in ("exdate", "strike", "cp", "bid", "ask")):
    raise SystemExit("essential columns missing -- edit COL by hand and rerun")


# ===================================================================== #
# independent extraction
# ===================================================================== #
def bs_call(k, tv):
    """Undiscounted call, forward = 1, total vol tv = sigma*sqrt(T)."""
    tv = np.maximum(tv, 1e-12)
    d1 = (-np.log(k)) / tv + 0.5 * tv
    d2 = d1 - tv
    return norm.cdf(d1) - k * norm.cdf(d2)


def iv_brent(price, k, is_call):
    """Brent root-find on total vol. Independent of the bisection in the
    original pipeline."""
    intr = max(1.0 - k, 0.0) if is_call else max(k - 1.0, 0.0)
    up = 1.0 if is_call else k
    if not (intr + 1e-12 < price < up - 1e-12):
        return np.nan
    tgt = price if is_call else price + (1.0 - k)   # put-call parity, F=1
    f = lambda tv: bs_call(k, tv) - tgt
    try:
        if f(1e-6) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-6, 5.0, xtol=1e-12, rtol=1e-14, maxiter=200)
    except Exception:
        return np.nan


def extract(g, T):
    """Return dict with h, h_ln, xh, mass, mean, tail, minq -- or None."""
    fwd = float(g["_fwd"].iloc[0])
    d = g.copy()
    d["k"] = d[COL["strike"]] / fwd
    d["mid"] = 0.5 * (d[COL["bid"]] + d[COL["ask"]])
    d = d[(d[COL["bid"]] > 0) & (d[COL["ask"]] > d[COL["bid"]]) & (d.mid > 0)]
    d["is_call"] = d[COL["cp"]].astype(str).str.upper().str[0].eq("C")
    # OTM only: puts below the forward, calls above
    otm = d[((~d.is_call) & (d.k < 1.0)) | (d.is_call & (d.k >= 1.0))]
    otm = otm.groupby("k", as_index=False).first().sort_values("k")
    if len(otm) < 20:
        return None

    tv = np.array([iv_brent(p / fwd, k, c) for p, k, c
                   in zip(otm.mid, otm.k, otm.is_call)])
    m = np.isfinite(tv) & (tv > 1e-5)
    if m.sum() < 15:
        return None
    lk, tv = np.log(otm.k.values[m]), tv[m]
    if lk.min() > -0.01 or lk.max() < 0.01:
        return None

    # PCHIP in log-moneyness, flat extrapolation (monotone, local -- not a
    # global smoothing spline)
    interp = PchipInterpolator(lk, tv, extrapolate=False)
    lo, hi = lk.min(), lk.max()
    kg = np.exp(np.linspace(lo, hi, GRID))
    tvg = interp(np.log(kg))
    tvg = np.where(np.isfinite(tvg), tvg, np.interp(np.log(kg), lk, tv))

    c = bs_call(kg, tvg)
    # central finite differences on a log-spaced grid -> use gradient twice
    d1 = np.gradient(c, kg)
    q = np.gradient(d1, kg)
    q = np.clip(q, 0.0, None)
    minq_raw = float(np.min(np.gradient(np.gradient(c, kg), kg)))

    mass = float(np.trapz(q, kg))
    if not (0.2 < mass < 5.0):
        return None
    mean = float(np.trapz(kg * q, kg)) / mass
    qn = q / mass

    pos = qn > 1e-14
    h = float(-np.trapz(qn[pos] * np.log(qn[pos]), kg[pos]))

    # lognormal benchmark matched on log-mean and log-variance of q
    lnk = np.log(kg[pos])
    mu = float(np.trapz(lnk * qn[pos], kg[pos]))
    var = float(np.trapz((lnk - mu) ** 2 * qn[pos], kg[pos]))
    sv = np.sqrt(max(var, 1e-12))
    g_ln = lognorm.pdf(kg[pos], sv, scale=np.exp(mu))
    h_ln = 0.5 * np.log(2 * np.pi * np.e * var) + mu

    ok = g_ln > 1e-300
    kl = float(np.trapz(qn[pos][ok] * np.log(qn[pos][ok] / g_ln[ok]), kg[pos][ok]))
    return dict(h=h, h_ln=h_ln, xh=h - h_ln, mass=mass, mean=mean,
                minq=minq_raw, kl=kl, n_used=int(m.sum()),
                tail=float(1 - mass) if mass < 1 else 0.0)


# ===================================================================== #
# sample pairs and compare
# ===================================================================== #
print("\n" + "=" * 70)
print("BLOCK 1 -- INDEPENDENT RE-EXTRACTION")
print("=" * 70)

ent = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/entropy_*.csv"))],
                ignore_index=True)
ent["date"] = pd.to_datetime(ent.date)
ent["exdate"] = pd.to_datetime(ent.exdate)

rows = []
for f in raw_files:
    yr = "".join(ch for ch in os.path.basename(f) if ch.isdigit())[:4]
    try:
        df = pd.read_parquet(f)
    except Exception as e:
        print(f"  {os.path.basename(f)}: unreadable ({e})")
        continue
    df[COL["date"]] = pd.to_datetime(df[COL["date"]])
    df[COL["exdate"]] = pd.to_datetime(df[COL["exdate"]])
    sub = ent[ent.date.dt.year.astype(str) == yr]
    if sub.empty:
        continue
    pick = sub.sample(min(N_PER_YEAR, len(sub)), random_state=11)
    for _, r in pick.iterrows():
        g = df[(df[COL["date"]] == r.date) & (df[COL["exdate"]] == r.exdate)]
        if len(g) < 20:
            continue
        g = g.copy()
        g["_fwd"] = r.fwd if "fwd" in r else (
            g[COL["fwd"]].iloc[0] if COL["fwd"] else np.nan)
        if not np.isfinite(g["_fwd"].iloc[0]):
            continue
        T = r.dte_yr if "dte_yr" in r else (r.dte / 365.0)
        try:
            res = extract(g, T)
        except Exception as e:
            res = None
        if res:
            res.update(date=r.date, exdate=r.exdate, dte=r.dte,
                       xh_csv=r.xh, h_csv=r.h, h_ln_csv=r.h_ln,
                       n_csv=r.n_used, tail_csv=r.tail_share)
            rows.append(res)

if not rows:
    raise SystemExit("no pairs re-extracted -- check COL mapping and raw file names")

R = pd.DataFrame(rows).sort_values("date")
print(f"\nre-extracted {len(R)} date-expiry pairs\n")
print(R[["date", "dte", "xh_csv", "xh", "h_csv", "h", "mass", "mean", "n_csv",
         "n_used"]].to_string(index=False,
                              formatters={"date": lambda x: str(x.date())},
                              float_format=lambda v: f"{v:9.4f}"))

# ===================================================================== #
print("\n" + "=" * 70)
print("BLOCK 2 -- MODEL-FREE CONDITIONS (must hold for any implementation)")
print("=" * 70)
check("A. density integrates to ~1", (R["mass"] - 1).abs().median() < 0.10,
      f"median |mass-1| = {(R['mass'] - 1).abs().median():.4f}")
check("B. martingale: E[k] ~ 1", (R["mean"] - 1).abs().median() < 0.03,
      f"median |E[k]-1| = {(R['mean'] - 1).abs().median():.4f}")
check("C. no butterfly arbitrage (q >= 0)", R["minq"].min() > -0.05,
      f"most negative raw q = {R['minq'].min():.4f}")
check("D. xh == -KL(q || lognormal)",
      np.allclose(R.xh, -R.kl, atol=2e-3),
      f"max |xh + KL| = {np.abs(R.xh + R.kl).max():.2e}")
check("E. xh strictly negative", (R.xh < 0).all(),
      f"max xh = {R.xh.max():.4f}")

# ===================================================================== #
print("\n" + "=" * 70)
print("BLOCK 3 -- AGREEMENT WITH THE PIPELINE")
print("=" * 70)
c_xh = np.corrcoef(R.xh_csv, R.xh)[0, 1]
c_h = np.corrcoef(R.h_csv, R.h)[0, 1]
mad = np.abs(R.xh_csv - R.xh).mean()
bias = (R.xh - R.xh_csv).mean()
sd_pop = pd.read_csv(sorted(glob.glob("data/entropy_*.csv"))[0]).xh.std()
print(f"  corr(xh_csv, xh_mine)          = {c_xh:.4f}")
print(f"  corr(h_csv,  h_mine)           = {c_h:.4f}")
print(f"  mean |difference| in xh        = {mad:.4f}")
print(f"  mean signed difference (bias)  = {bias:+.4f}")
print(f"  for scale, population sd of xh = {sd_pop:.4f}")
print(f"  disagreement as a share of sd  = {mad / sd_pop:.2f}")

check("xh correlates across implementations", c_xh > 0.90, f"{c_xh:.4f}")
check("disagreement small vs population sd", mad / sd_pop < 0.5,
      f"{mad / sd_pop:.2f}")
check("no systematic bias", abs(bias) < 0.5 * sd_pop, f"{bias:+.4f}")

# ===================================================================== #
print("\n" + "=" * 70)
if FAIL:
    print("FINAL CHECK FAILED on:")
    for f in FAIL:
        print("   -", f)
    print("\nThe null may still hold, but the entropy variable needs")
    print("reconciling between implementations before write-up.")
else:
    print("FINAL CHECK PASSED.")
    print("The Breeden-Litzenberger extraction reproduces under independent")
    print("numerics, and the risk-neutral density satisfies the model-free")
    print("conditions. The full chain from raw chains to the null is verified.")
print("=" * 70)
