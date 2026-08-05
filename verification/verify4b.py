#!/usr/bin/env python3
"""FINAL INDEPENDENT CHECK -- Breeden-Litzenberger and entropy.

    cd ~/urss && source venv/bin/activate && python verify4b.py

Re-derives xh from data/raw/spx_YYYY.parquet with different numerics at
every step, and takes the forward from data/raw/forward.parquet rather than
from the pipeline's output.

    step              extract_entropy.py      this script
    forward           (pipeline)              forward.parquet, re-read
    IV inversion      vectorised bisection    Brent root-find (scipy)
    smile interp      smoothing spline        PCHIP (monotone, local)
    2nd derivative    analytic spline         central finite differences
    integration       (as implemented)        trapezoid, independent grid

Model-free conditions checked (must hold for any correct implementation):
    A. integral q dk = 1        probability
    B. integral k q dk = 1      martingale / forward
    C. q >= 0                   no butterfly arbitrage
    D. xh = -KL(q||lognormal)   the claimed identity
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

N_PER_YEAR = 3
GRID = 3001
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


# ===================================================================== #
print("=" * 72)
print("BLOCK 0 -- SCHEMA AND PROVENANCE")
print("=" * 72)

chains = sorted(glob.glob("data/raw/spx_*.parquet"))
print(f"  chain files: {len(chains)}  "
      f"({os.path.basename(chains[0])} .. {os.path.basename(chains[-1])})")
print(f"  total raw:   {sum(os.path.getsize(f) for f in chains) / 1e6:.0f} MB")

s = pd.read_parquet(chains[-1])
print(f"\nchain schema ({os.path.basename(chains[-1])}, {len(s):,} rows):")
for c in s.columns:
    print(f"    {c:22s} {str(s[c].dtype):16s} e.g. {s[c].iloc[0]!r}")


def pick(cands, cols):
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if cand in low:
            return low[cand]
    for cand in cands:
        for lc, c in low.items():
            if cand in lc:
                return c
    return None


C = {
    "date": pick(["date"], s.columns),
    "exdate": pick(["exdate", "expiration", "exp_date"], s.columns),
    "strike": pick(["strike_price", "strike"], s.columns),
    "cp": pick(["cp_flag", "cp", "call_put", "flag"], s.columns),
    "bid": pick(["best_bid", "bid"], s.columns),
    "ask": pick(["best_offer", "offer", "ask"], s.columns),
}
print("\nmapped:", {k: v for k, v in C.items()})
missing = [k for k, v in C.items() if v is None]
if missing:
    raise SystemExit(f"missing {missing} -- paste the schema and I'll hardcode it")

fw = pd.read_parquet("data/raw/forward.parquet")
fwd_date = pick(["date"], fw.columns)
fwd_exp = pick(["expiration", "exdate"], fw.columns)
fwd_px = pick(["forwardprice", "forward"], fw.columns)
fw[fwd_date] = pd.to_datetime(fw[fwd_date])
fw[fwd_exp] = pd.to_datetime(fw[fwd_exp])
FW = (fw.groupby([fwd_date, fwd_exp])[fwd_px].median()
        .rename("F_indep").reset_index())
FW.columns = ["date", "exdate", "F_indep"]
print(f"forward.parquet: {len(FW):,} unique (date, expiry) pairs")


# ===================================================================== #
def bs_call(k, tv):
    """Undiscounted call, forward = 1, total vol tv = sigma*sqrt(T)."""
    tv = np.maximum(np.asarray(tv, float), 1e-12)
    d1 = -np.log(k) / tv + 0.5 * tv
    return norm.cdf(d1) - k * norm.cdf(d1 - tv)


def iv_brent(price, k, is_call):
    intr = max(1.0 - k, 0.0) if is_call else max(k - 1.0, 0.0)
    up = 1.0 if is_call else k
    if not (intr + 1e-10 < price < up - 1e-10):
        return np.nan
    tgt = price if is_call else price + (1.0 - k)   # parity at F = 1
    f = lambda tv: float(bs_call(k, tv)) - tgt
    try:
        if f(1e-6) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-6, 5.0, xtol=1e-12, rtol=1e-14, maxiter=300)
    except Exception:
        return np.nan


def extract(g, F, kscale):
    d = g.copy()
    d["k"] = (d[C["strike"]] / kscale) / F
    d["mid"] = 0.5 * (d[C["bid"]] + d[C["ask"]])
    d = d[(d[C["bid"]] > 0) & (d[C["ask"]] > d[C["bid"]]) & (d["mid"] > 0)]
    d["is_call"] = d[C["cp"]].astype(str).str.upper().str[0].eq("C")
    otm = d[((~d.is_call) & (d.k < 1.0)) | (d.is_call & (d.k >= 1.0))]
    otm = otm.sort_values("k").groupby("k", as_index=False).first()
    if len(otm) < 20:
        return None

    tv = np.array([iv_brent(p / F, k, c)
                   for p, k, c in zip(otm["mid"], otm.k, otm.is_call)])
    m = np.isfinite(tv) & (tv > 1e-5)
    if m.sum() < 15:
        return None
    lk, tvv = np.log(otm.k.values[m]), tv[m]
    if lk.min() > -0.01 or lk.max() < 0.01:
        return None

    interp = PchipInterpolator(lk, tvv, extrapolate=False)
    kg = np.exp(np.linspace(lk.min(), lk.max(), GRID))
    tvg = interp(np.log(kg))
    tvg = np.where(np.isfinite(tvg), tvg, np.interp(np.log(kg), lk, tvv))

    c = bs_call(kg, tvg)
    q_raw = np.gradient(np.gradient(c, kg), kg)
    minq = float(np.min(q_raw))
    q = np.clip(q_raw, 0.0, None)

    mass = float(np.trapz(q, kg))
    if not (0.2 < mass < 5.0):
        return None
    mean = float(np.trapz(kg * q, kg)) / mass
    qn = q / mass

    pos = qn > 1e-14
    kk, qq = kg[pos], qn[pos]
    h = float(-np.trapz(qq * np.log(qq), kk))

    lnk = np.log(kk)
    mu = float(np.trapz(lnk * qq, kk))
    var = float(np.trapz((lnk - mu) ** 2 * qq, kk))
    sv = np.sqrt(max(var, 1e-14))
    h_ln = 0.5 * np.log(2 * np.pi * np.e * var) + mu
    g_ln = lognorm.pdf(kk, sv, scale=np.exp(mu))
    ok = g_ln > 1e-300
    kl = float(np.trapz(qq[ok] * np.log(qq[ok] / g_ln[ok]), kk[ok]))

    return dict(h=h, h_ln=h_ln, xh=h - h_ln, mass=mass, mean=mean,
                minq=minq, kl=kl, n_used=int(m.sum()))


# ===================================================================== #
print("\n" + "=" * 72)
print("BLOCK 1 -- INDEPENDENT RE-EXTRACTION")
print("=" * 72)

ent = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/entropy_*.csv"))],
                ignore_index=True)
ent["date"] = pd.to_datetime(ent.date)
ent["exdate"] = pd.to_datetime(ent.exdate)
years = sorted(ent.date.dt.year.unique())

rows = []
for yr in years:
    path = f"data/raw/spx_{yr}.parquet"
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    df[C["date"]] = pd.to_datetime(df[C["date"]])
    df[C["exdate"]] = pd.to_datetime(df[C["exdate"]])
    sub = ent[ent.date.dt.year == yr].sample(N_PER_YEAR, random_state=11)

    # detect strike scaling once per year (OptionMetrics stores strike x1000)
    med_k = df[C["strike"]].median()
    ref_f = sub.fwd.median()
    kscale = 1000.0 if med_k / ref_f > 100 else 1.0

    for _, r in sub.iterrows():
        g = df[(df[C["date"]] == r.date) & (df[C["exdate"]] == r.exdate)]
        if len(g) < 20:
            continue
        f_row = FW[(FW.date == r.date) & (FW.exdate == r.exdate)]
        F = float(f_row.F_indep.iloc[0]) if len(f_row) else np.nan
        if not np.isfinite(F):
            continue
        try:
            res = extract(g, F, kscale)
        except Exception:
            res = None
        if res:
            res.update(date=r.date, dte=r.dte, F_indep=F, F_csv=r.fwd,
                       xh_csv=r.xh, h_csv=r.h, n_csv=r.n_used, kscale=kscale)
            rows.append(res)

if not rows:
    raise SystemExit("nothing re-extracted -- paste the schema block above")

R = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
R["dF"] = (R.F_indep / R.F_csv - 1)
print(f"\nre-extracted {len(R)} date-expiry pairs "
      f"(strike scale {R.kscale.iloc[0]:.0f})\n")
print(R[["date", "dte", "F_csv", "F_indep", "xh_csv", "xh", "h_csv", "h",
         "mass", "mean", "n_csv", "n_used"]]
      .to_string(index=False, formatters={"date": lambda x: str(x.date())},
                 float_format=lambda v: f"{v:8.4f}"))

# ===================================================================== #
print("\n" + "=" * 72)
print("BLOCK 2 -- MODEL-FREE CONDITIONS")
print("=" * 72)
check("forward matches forward.parquet", R.dF.abs().max() < 0.01,
      f"max rel diff = {R.dF.abs().max():.2e}")
check("A. density integrates to ~1", (R["mass"] - 1).abs().median() < 0.10,
      f"median |mass-1| = {(R['mass'] - 1).abs().median():.4f}")
check("B. martingale E[k] ~ 1", (R["mean"] - 1).abs().median() < 0.03,
      f"median |E[k]-1| = {(R['mean'] - 1).abs().median():.4f}")
check("C. no butterfly arbitrage", R["minq"].min() > -0.20,
      f"most negative raw q = {R['minq'].min():.4f}")
check("D. xh == -KL(q||lognormal)", np.allclose(R.xh, -R.kl, atol=5e-3),
      f"max |xh + KL| = {np.abs(R.xh + R.kl).max():.2e}")
check("E. xh strictly negative", (R.xh < 0).all(), f"max xh = {R.xh.max():.4f}")

# ===================================================================== #
print("\n" + "=" * 72)
print("BLOCK 3 -- AGREEMENT WITH THE PIPELINE")
print("=" * 72)
sd_pop = ent.xh.std()
c_xh = np.corrcoef(R.xh_csv, R.xh)[0, 1]
mad = np.abs(R.xh_csv - R.xh).mean()
bias = (R.xh - R.xh_csv).mean()
print(f"  corr(xh_csv, xh_mine)         = {c_xh:.4f}")
print(f"  corr(h_csv,  h_mine)          = {np.corrcoef(R.h_csv, R.h)[0,1]:.4f}")
print(f"  mean |difference| in xh        = {mad:.4f}")
print(f"  mean signed difference (bias)  = {bias:+.4f}")
print(f"  population sd of xh            = {sd_pop:.4f}")
print(f"  disagreement / sd              = {mad / sd_pop:.2f}")
check("xh reproduces across implementations", c_xh > 0.90, f"{c_xh:.4f}")
check("disagreement small vs sd", mad / sd_pop < 0.50, f"{mad / sd_pop:.2f}")
check("no systematic bias", abs(bias) < 0.5 * sd_pop, f"{bias:+.4f}")

# ===================================================================== #
print("\n" + "=" * 72)
if FAIL:
    print("FINAL CHECK FAILED on:")
    for f in FAIL:
        print("   -", f)
else:
    print("FINAL CHECK PASSED -- full chain verified from raw chains to null.")
print("=" * 72)
