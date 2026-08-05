#!/usr/bin/env python3
"""
Calibrate the synthetic harness against the real SPX chains.

The harness conclusions depend on assumed values for quote noise, strike
spacing and quote usability.  This measures them directly.

    cd ~/urss && source venv/bin/activate && python calibrate_noise.py

Writes data/noise_calibration.csv.

Schema assumed (confirmed from the parquet files):
  secid date symbol exdate cp_flag strike_price best_bid best_offer
  volume open_interest impl_volatility am_settlement optionid
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
YEARS = [2015, 2018, 2020, 2022, 2025]
DTE_LO, DTE_HI = 14, 90


def parity_forward(df):
    """
    Forward per (date, exdate) from put-call parity.

    Merge calls to puts on strike, pick the strike where |C - P| is smallest
    (that is the strike closest to the forward), then F = K + (C - P).
    Fully vectorised -- no groupby.apply.
    """
    c = (df.loc[df.cp_flag == "C", ["date", "exdate", "strike", "mid"]]
           .rename(columns={"mid": "c"}))
    p = (df.loc[df.cp_flag == "P", ["date", "exdate", "strike", "mid"]]
           .rename(columns={"mid": "p"}))
    m = c.merge(p, on=["date", "exdate", "strike"], how="inner")
    if m.empty:
        return pd.DataFrame(columns=["date", "exdate", "fwd"])
    m["gap"] = (m.c - m.p).abs()
    m = m.sort_values("gap").drop_duplicates(["date", "exdate"], keep="first")
    m["fwd"] = m.strike + (m.c - m.p)
    return m[["date", "exdate", "fwd"]]


def median_spacing(df):
    """Median gap between adjacent unique strikes, per (date, exdate)."""
    u = (df[["date", "exdate", "strike"]]
         .drop_duplicates()
         .sort_values(["date", "exdate", "strike"]))
    u["d"] = u.groupby(["date", "exdate"]).strike.diff()
    return u.d.dropna()


rows = []
for yr in YEARS:
    f = RAW / f"spx_{yr}.parquet"
    if not f.exists():
        print(f"  missing {f}, skipping")
        continue

    df = pd.read_parquet(f)
    df["strike"] = df["strike_price"] / 1000.0
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    df["dte"] = (df.exdate - df.date).dt.days
    df = df[(df.dte >= DTE_LO) & (df.dte <= DTE_HI)].copy()
    if df.empty:
        print(f"  {yr}: no rows in dte window")
        continue

    n_all = len(df)
    zero_bid = float((df.best_bid <= 0).mean())
    iv_null = float(df.impl_volatility.isna().mean())

    df = df[df.best_offer > df.best_bid].copy()      # drop crossed/locked
    df["mid"] = 0.5 * (df.best_bid + df.best_offer)
    df["spread"] = df.best_offer - df.best_bid
    df["half"] = 0.5 * df.spread

    fwd = parity_forward(df)
    df = df.merge(fwd, on=["date", "exdate"], how="inner")
    df = df[df.fwd > 0].copy()
    df["moneyness"] = df.strike / df.fwd

    otm = df[((df.cp_flag == "C") & (df.moneyness >= 1.0)) |
             ((df.cp_flag == "P") & (df.moneyness < 1.0))].copy()
    otm["half_fwd"] = otm.half / otm.fwd
    ntm = otm[(otm.moneyness > 0.90) & (otm.moneyness < 1.10)]

    sp = median_spacing(df)

    rows.append(dict(
        year=yr,
        n_rows=n_all,
        fwd_med=float(df.fwd.median()),
        half_abs_med=float(otm.half.median()),
        half_abs_p90=float(otm.half.quantile(0.90)),
        half_fwd_med=float(otm.half_fwd.median()),
        half_fwd_p90=float(otm.half_fwd.quantile(0.90)),
        half_fwd_ntm=float(ntm.half_fwd.median()) if len(ntm) else np.nan,
        spread_over_mid=float((otm.spread / otm.mid.clip(lower=1e-9)).median()),
        spacing_med=float(sp.median()) if len(sp) else np.nan,
        spacing_over_fwd=(float(sp.median()) / float(df.fwd.median())
                          if len(sp) else np.nan),
        zero_bid_share=zero_bid,
        mid_under_5c=float((df.mid < 0.05).mean()),
        iv_null_share=iv_null,
    ))
    print(f"{yr}: {n_all:,} rows -> {len(df):,} usable, "
          f"forward ~{df.fwd.median():.0f}")

if not rows:
    raise SystemExit("no data processed")

out = pd.DataFrame(rows)
Path("data").mkdir(exist_ok=True)
out.to_csv("data/noise_calibration.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
print("\n" + "=" * 84)
print(out[["year", "fwd_med", "half_abs_med", "half_abs_p90", "half_fwd_ntm",
           "spread_over_mid", "spacing_med", "spacing_over_fwd",
           "zero_bid_share"]].to_string(index=False))
print("=" * 84)

hn = out.half_fwd_ntm.median()
print("\nPLUG INTO HARNESS")
print(f"  abs_noise, median NTM half-spread (forward units) = {hn:.3e}")
print(f"  abs_noise, p90 stress case                        = {out.half_fwd_p90.median():.3e}")
print(f"\n  Harness assumed 8.0e-06.  Reality is {hn / 8e-6:.1f}x that.")
print(f"  Strike spacing / forward = {out.spacing_over_fwd.median():.3e}")
print(f"  Zero-bid share = {out.zero_bid_share.median():.1%}  (harness assumed 0%)")
print(f"  Mid < $0.05 share = {out.mid_under_5c.median():.1%}")
