#!/usr/bin/env python3
"""
Extract excess entropy and BKM moments from real SPX option chains.

    cd ~/urss && source venv/bin/activate
    python extract_entropy.py 2020            # one year
    python extract_entropy.py 2018 2025       # range

Writes data/entropy_YYYY.csv, one row per (date, exdate).

DESIGN DECISIONS -- fixed in advance, do not tune:
  * Estimator: cubic smoothing spline on the OTM implied-vol smile in log
    moneyness, smoothing parameter s = 1e-2, then Breeden-Litzenberger by
    second difference of the re-priced call curve.
  * s = 1e-2 was chosen from the synthetic harness as the smallest value at
    which entropy bias and variance are stable at empirically measured quote
    noise (bias +0.01 to +0.06 nats, sd < 0.03). It is NOT to be re-tuned
    after seeing regression output.
  * Tails: lognormal matched to the boundary implied vol. The extrapolated
    probability share is recorded as `tail_share` for robustness splits.
  * OTM only: puts below the forward, calls above. ITM quotes are near
    intrinsic and their time value is swamped by the spread.

OUTPUTS per date-expiry:
  h            entropy of the fitted RND (nats, over price/forward)
  h_ln         entropy of the variance-matched lognormal
  xh           EXCESS ENTROPY = h - h_ln   (<= 0 by construction)
  bkm_vol/skew/kurt   Bakshi-Kapadia-Madan (2003) model-free moments
  atm_iv       implied vol at the forward
  half_spread  median near-the-money half-spread / forward  (noise proxy)
  n_used       strikes actually used
  tail_share   probability mass coming from the extrapolated tails
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate, stats
from scipy.special import ndtr

RAW = Path("data/raw")
OUT = Path("data")
DTE_LO, DTE_HI = 14, 90
SMOOTH = 1e-2                      # LOCKED
NGRID = 4000
MIN_STRIKES = 12

XLO, XHI = 0.05, 3.0               # moneyness grid
XG = np.linspace(XLO, XHI, NGRID)


# ---------------------------------------------------------------- pricing --
# ndtr is the compiled normal CDF; stats.norm.cdf carries ~100x the overhead
# and this is the innermost loop of the whole pipeline.
def bs_call(F, K, s):
    K = np.asarray(K, float)
    s = np.asarray(s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * s ** 2) / s
        d2 = d1 - s
    return F * ndtr(d1) - K * ndtr(d2)


def bs_put(F, K, s):
    return bs_call(F, K, s) - (F - K)


def implied_tv_vec(price, K, is_call, F=1.0, iters=80):
    """
    Vectorised Black-76 inversion for TOTAL vol (sigma*sqrt(T)).

    Bisection on the whole strike vector simultaneously. Monotonicity of the
    option price in vol guarantees convergence; 80 halvings of [1e-4, 5]
    gives far more precision than the quotes justify.
    Returns nan where the price violates no-arbitrage bounds.
    """
    price = np.asarray(price, float)
    K = np.asarray(K, float)
    is_call = np.asarray(is_call, bool)

    intr = np.where(is_call, np.clip(F - K, 0, None), np.clip(K - F, 0, None))
    upper = np.where(is_call, F, K)
    ok = np.isfinite(price) & (price > intr + 1e-12) & (price < upper - 1e-12)

    lo = np.full(K.shape, 1e-4)
    hi = np.full(K.shape, 5.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = bs_call(F, K, mid)
        val = np.where(is_call, c, c - (F - K))
        too_low = val < price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    out = 0.5 * (lo + hi)
    return np.where(ok, out, np.nan)


# ------------------------------------------------------------- estimation --
def fit_rnd(moneyness, totvol, smooth=SMOOTH):
    """
    Spline the smile, re-price densely, differentiate twice, splice lognormal
    tails. Returns (density on XG, tail_share) or (None, nan).
    """
    if len(moneyness) < MIN_STRIKES:
        return None, np.nan
    o = np.argsort(moneyness)
    m, v = np.asarray(moneyness)[o], np.asarray(totvol)[o]
    keep = np.isfinite(m) & np.isfinite(v) & (v > 0)
    m, v = m[keep], v[keep]
    m, idx = np.unique(m, return_index=True)
    v = v[idx]
    if len(m) < MIN_STRIKES:
        return None, np.nan

    lm = np.log(m)
    try:
        spl = interpolate.UnivariateSpline(lm, v, s=smooth * len(m), k=3)
    except Exception:
        return None, np.nan

    lo, hi = m.min(), m.max()
    inner = (XG > lo) & (XG < hi)
    if inner.sum() < 50:
        return None, np.nan
    xk = XG[inner]
    iv = np.clip(spl(np.log(xk)), 1e-4, 5.0)
    cp = bs_call(1.0, xk, iv)

    dens = np.zeros_like(XG)
    d2 = np.gradient(np.gradient(cp, xk), xk)
    dens[inner] = np.clip(d2, 0.0, None)

    s_lo = float(np.clip(spl(np.log(lo)), 1e-4, 5.0))
    s_hi = float(np.clip(spl(np.log(hi)), 1e-4, 5.0))
    for mask, sv in ((XG <= lo, s_lo), (XG >= hi, s_hi)):
        dens[mask] = stats.lognorm.pdf(XG[mask], sv, scale=np.exp(-0.5 * sv ** 2))

    tot = np.trapezoid(dens, XG)
    if not np.isfinite(tot) or tot <= 0:
        return None, np.nan
    dens = dens / tot
    tail_share = float(np.trapezoid(dens[~inner], XG[~inner]))
    return dens, tail_share


def entropy(p):
    q = np.clip(p, 1e-300, None)
    return float(-np.trapezoid(q * np.log(q), XG))


def excess_entropy(p):
    """h(p) - h(lognormal with the same mean and variance)."""
    mu = np.trapezoid(XG * p, XG)
    var = np.trapezoid((XG - mu) ** 2 * p, XG)
    if mu <= 0 or var <= 0:
        return np.nan, np.nan, np.nan
    s2 = np.log(1.0 + var / mu ** 2)
    m = np.log(mu) - 0.5 * s2
    h_ln = m + 0.5 * np.log(2 * np.pi * np.e * s2)
    return entropy(p), float(h_ln), entropy(p) - float(h_ln)


def bkm_moments(m_otm, price_otm, is_call, T, r=0.0):
    """
    Bakshi, Kapadia & Madan (2003) model-free moments from OTM options.
    All quantities in units of the forward, so F = 1 and the log contract is
    centred at 0.
    """
    o = np.argsort(m_otm)
    K = np.asarray(m_otm, float)[o]
    P = np.asarray(price_otm, float)[o]
    C = np.asarray(is_call, bool)[o]
    if len(K) < MIN_STRIKES:
        return (np.nan,) * 3
    lk = np.log(K)
    wV = np.where(C, 2 * (1 - lk) / K ** 2, 2 * (1 + (-lk)) / K ** 2)
    wW = np.where(C, (6 * lk - 3 * lk ** 2) / K ** 2,
                  (6 * lk - 3 * lk ** 2) / K ** 2)
    wX = np.where(C, (12 * lk ** 2 - 4 * lk ** 3) / K ** 2,
                  (12 * lk ** 2 - 4 * lk ** 3) / K ** 2)
    V = np.trapezoid(wV * P, K)
    W = np.trapezoid(wW * P, K)
    X = np.trapezoid(wX * P, K)
    er = np.exp(r * T)
    mu = er - 1 - er * V / 2 - er * W / 6 - er * X / 24
    var = er * V - mu ** 2
    if var <= 0:
        return (np.nan,) * 3
    skew = (er * W - 3 * mu * er * V + 2 * mu ** 3) / var ** 1.5
    kurt = (er * X - 4 * mu * er * W + 6 * mu ** 2 * er * V - 3 * mu ** 4) / var ** 2
    return float(np.sqrt(var / T)), float(skew), float(kurt)


# ------------------------------------------------------------------ chain --
def process_chain(g, fwd, T):
    """One (date, exdate). Returns a dict of measures or None."""
    g = g[g.best_offer > g.best_bid].copy()
    g["mid"] = 0.5 * (g.best_bid + g.best_offer)
    g["m"] = g.strike / fwd
    g = g[(g.m > 0.4) & (g.m < 1.6) & (g.mid > 0.05)]

    otm = g[((g.cp_flag == "C") & (g.m >= 1.0)) |
            ((g.cp_flag == "P") & (g.m < 1.0))].copy()
    if len(otm) < MIN_STRIKES:
        return None
    otm["px"] = otm.mid / fwd                     # normalise by forward

    otm["tv"] = implied_tv_vec(otm.px.values, otm.m.values,
                               (otm.cp_flag == "C").values)
    otm = otm[np.isfinite(otm.tv)]
    if len(otm) < MIN_STRIKES:
        return None

    dens, tail = fit_rnd(otm.m.values, otm.tv.values)
    if dens is None:
        return None
    h, h_ln, xh = excess_entropy(dens)
    if not np.isfinite(xh):
        return None

    bv, bs_, bk = bkm_moments(otm.m.values, otm.px.values,
                              (otm.cp_flag == "C").values, T)

    ntm = otm[(otm.m > 0.95) & (otm.m < 1.05)]
    half = float(((ntm.best_offer - ntm.best_bid) / 2 / fwd).median()) \
        if len(ntm) else np.nan

    # ATM implied vol.  np.interp REQUIRES xp to be increasing -- with
    # unsorted input its behaviour is undefined, which was producing
    # implausible values (ATM IV above 100% on ~1% of dates).
    so = np.argsort(otm.m.values)
    lm_s = np.log(otm.m.values[so])
    tv_s = otm.tv.values[so]
    if lm_s.min() > 0 or lm_s.max() < 0:
        return None                      # OTM set does not bracket the forward
    atm = float(np.interp(0.0, lm_s, tv_s)) / np.sqrt(T)
    if not (0.02 < atm < 1.5):
        return None                      # implausible for 14-90d SPX

    return dict(fwd=fwd, dte_yr=T, n_used=len(otm), h=h, h_ln=h_ln, xh=xh,
                tail_share=tail, atm_iv=atm, bkm_vol=bv, bkm_skew=bs_,
                bkm_kurt=bk, half_spread=half)


def parity_forward(df):
    c = (df.loc[df.cp_flag == "C", ["date", "exdate", "strike", "mid"]]
           .rename(columns={"mid": "c"}))
    p = (df.loc[df.cp_flag == "P", ["date", "exdate", "strike", "mid"]]
           .rename(columns={"mid": "p"}))
    m = c.merge(p, on=["date", "exdate", "strike"])
    if m.empty:
        return pd.DataFrame(columns=["date", "exdate", "fwd"])
    m["gap"] = (m.c - m.p).abs()
    m = m.sort_values("gap").drop_duplicates(["date", "exdate"], keep="first")
    m["fwd"] = m.strike + (m.c - m.p)
    return m[["date", "exdate", "fwd"]]


def run_year(yr):
    f = RAW / f"spx_{yr}.parquet"
    if not f.exists():
        print(f"  {f} missing")
        return
    df = pd.read_parquet(f)
    df["strike"] = df.strike_price / 1000.0
    df["date"] = pd.to_datetime(df.date)
    df["exdate"] = pd.to_datetime(df.exdate)
    df["dte"] = (df.exdate - df.date).dt.days
    df = df[(df.dte >= DTE_LO) & (df.dte <= DTE_HI)].copy()
    df = df[df.best_offer > df.best_bid].copy()
    df["mid"] = 0.5 * (df.best_bid + df.best_offer)

    fwd = parity_forward(df).set_index(["date", "exdate"]).fwd.to_dict()

    out = []
    groups = list(df.groupby(["date", "exdate"], sort=True))
    for i, ((d, ex), g) in enumerate(groups):
        if i % 100 == 0:
            print(f"    {yr}: {i}/{len(groups)} pairs", flush=True)
        F = fwd.get((d, ex))
        if F is None or not np.isfinite(F) or F <= 0:
            continue
        T = (ex - d).days / 365.0
        try:
            res = process_chain(g, F, T)
        except Exception:
            res = None
        if res:
            res.update(date=d, exdate=ex, dte=(ex - d).days)
            out.append(res)

    if not out:
        print(f"  {yr}: nothing extracted")
        return
    o = pd.DataFrame(out)
    cols = ["date", "exdate", "dte", "dte_yr", "fwd", "n_used", "h", "h_ln",
            "xh", "tail_share", "atm_iv", "bkm_vol", "bkm_skew", "bkm_kurt",
            "half_spread"]
    o = o[cols].sort_values(["date", "exdate"])
    o.to_csv(OUT / f"entropy_{yr}.csv", index=False)
    print(f"  {yr}: {len(o):,} date-expiry pairs -> data/entropy_{yr}.csv")
    print(f"      xh    mean {o.xh.mean():+.4f}  sd {o.xh.std():.4f}  "
          f"[{o.xh.quantile(.01):+.3f}, {o.xh.quantile(.99):+.3f}]")
    print(f"      tail  mean {o.tail_share.mean():.3f}   "
          f"atm_iv mean {o.atm_iv.mean():.3f}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [2020]
    years = range(args[0], args[-1] + 1)
    for y in years:
        run_year(y)
