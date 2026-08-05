#!/usr/bin/env python3
"""
Audit the SPX extract and build the strike-density screen.

    python audit_spx.py

Produces
--------
data/audit_yearly.csv    one row per year: completeness and quality checks
data/density.csv         one row per (date, exdate): the estimability screen

The yearly table is the integrity check -- run it while re-pulling is still
cheap. The density table is what sets your sample, and it is a figure in the
report rather than an arbitrary start date.
"""

import glob
import os

import numpy as np
import pandas as pd

RAW = "data/raw"
MIN_DTE = 14          # skip near-expiry: IV is missing and the smile is unstable
MAX_DTE = 90          # keep the liquid part of the term structure
BAND = 0.10           # +/-10% moneyness window for the density count


def year_summary(df, year):
    """Completeness and sanity checks. Anything anomalous here means re-pull."""
    d = df.copy()
    d["K"] = d.strike_price / 1000.0
    d["dte"] = (d.exdate - d.date).dt.days
    spread = d.best_offer - d.best_bid

    return {
        "year": year,
        "rows": len(d),
        "trading_days": d.date.nunique(),
        "first": d.date.min().date(),
        "last": d.date.max().date(),
        "max_gap_days": int(pd.Series(sorted(d.date.unique())).diff().dt.days.max()),
        "expiries_per_day": round(d.groupby("date").exdate.nunique().mean(), 1),
        "rows_per_day": int(len(d) / d.date.nunique()),
        # quality flags -- all should be 0 or near it
        "dup_optionid": int(d.duplicated(["date", "optionid"]).sum()),
        "crossed": int((spread < 0).sum()),
        "zero_bid_pct": round(100 * (d.best_bid <= 0).mean(), 1),
        "iv_null_pct": round(100 * d.impl_volatility.isna().mean(), 1),
        "neg_dte": int((d.dte < 0).sum()),
        "non_european": int((d.am_settlement != 1).sum()),
    }


def density(df):
    """Per (date, expiry) estimability metrics.

    Coverage is measured in units of sigma*sqrt(T), not raw moneyness, because a
    fixed moneyness band covers wildly different probability mass at different
    maturities and vol levels. That distinction is what separates 1996 (right
    tail truncated at +1.2 sigma, entropy dominated by the tail-fitting
    assumption) from 2025 (+12 sigma, entropy determined by observed prices).
    """
    d = df.copy()
    d["K"] = d.strike_price / 1000.0
    d["dte"] = (d.exdate - d.date).dt.days
    d = d[(d.dte >= MIN_DTE) & (d.dte <= MAX_DTE) & d.impl_volatility.notna()]
    if d.empty:
        return pd.DataFrame()

    d["mid"] = (d.best_bid + d.best_offer) / 2.0
    out = []

    for (dt, ex), g in d.groupby(["date", "exdate"], sort=False):
        calls = g.loc[g.cp_flag == "C"].groupby("K").mid.first()
        puts = g.loc[g.cp_flag == "P"].groupby("K").mid.first()
        joint = pd.concat([calls, puts], axis=1, keys=["c", "p"]).dropna()
        if len(joint) < 5:
            continue

        # ATM proxy: strike where call and put mids are closest. A proper
        # put-call-parity forward comes later from the fwdprd table; this is
        # only for bucketing strikes.
        fwd = (joint.c - joint.p).abs().idxmin()
        ks = np.sort(g.K.unique())
        iv = g.impl_volatility.median()
        dte = int(g.dte.iloc[0])
        sig = iv * np.sqrt(dte / 365.0)
        if sig <= 0 or not np.isfinite(sig):
            continue

        logm = np.log(ks / fwd)
        band = ks[np.abs(ks / fwd - 1.0) < BAND]

        out.append({
            "date": dt,
            "exdate": ex,
            "dte": dte,
            "fwd": fwd,
            "med_iv": round(iv, 4),
            "n_strikes": len(ks),
            "n_within_1sig": int((np.abs(logm) < sig).sum()),
            "n_in_band": len(band),
            "atm_gap_pct": round(100 * np.median(np.diff(band)) / fwd, 4) if len(band) > 1 else np.nan,
            "lo_sigma": round(logm.min() / sig, 2),
            "hi_sigma": round(logm.max() / sig, 2),
            "iv_null_pct": round(100 * g.impl_volatility.isna().mean(), 1),
        })

    return pd.DataFrame(out)


def main():
    files = sorted(glob.glob(f"{RAW}/spx_*.parquet"))
    print(f"{len(files)} files\n")

    summaries, densities = [], []
    for path in files:
        year = int(os.path.basename(path).split("_")[1][:4])
        df = pd.read_parquet(path)
        summaries.append(year_summary(df, year))
        densities.append(density(df))
        print(f"  {year} done", end="\r")

    audit = pd.DataFrame(summaries)
    dens = pd.concat(densities, ignore_index=True)

    audit.to_csv("data/audit_yearly.csv", index=False)
    dens.to_csv("data/density.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== INTEGRITY ===")
    print(audit.to_string(index=False))

    print("\n=== FLAGS (want all zero) ===")
    for col in ["dup_optionid", "crossed", "neg_dte", "non_european"]:
        bad = audit.loc[audit[col] > 0, ["year", col]]
        print(f"{col}: {'clean' if bad.empty else bad.to_dict('records')}")

    # A trading year is ~252 days. Anything materially short means a partial
    # pull, which is a re-pull, not a data feature.
    short = audit[(audit.trading_days < 240) & (audit.year < 2025)]
    print(f"\nshort years: {'none' if short.empty else short[['year','trading_days']].to_dict('records')}")

    print("\n=== DENSITY BY YEAR (median across date-expiry pairs) ===")
    dens["year"] = pd.to_datetime(dens.date).dt.year
    by_year = dens.groupby("year").agg(
        pairs=("dte", "size"),
        strikes=("n_strikes", "median"),
        within_1sig=("n_within_1sig", "median"),
        atm_gap_pct=("atm_gap_pct", "median"),
        hi_sigma=("hi_sigma", "median"),
        lo_sigma=("lo_sigma", "median"),
    ).round(2)
    print(by_year.to_string())

    # Candidate screen. Tune the thresholds, then pre-register them and do not
    # touch them again once you have seen a regression result.
    ok = (dens.n_within_1sig >= 40) & (dens.hi_sigma >= 3) & (dens.lo_sigma <= -3)
    print("\n=== PASS RATE UNDER CANDIDATE SCREEN ===")
    print("(>=40 strikes within +/-1 sigma, coverage beyond +/-3 sigma)")
    print(ok.groupby(dens.year).mean().round(3).to_string())


if __name__ == "__main__":
    main()
