#!/usr/bin/env python3
"""
Pull SPX (secid 108105) AM-settled option chains from OptionMetrics IvyDB US.

Usage:
    python pull_spx.py --list                 # discover schema + table names FIRST
    python pull_spx.py --probe 2000-01-04     # single-day sanity check
    python pull_spx.py                        # full pull, resumable

Notes
-----
* Run --list before anything else. WRDS has migrated OptionMetrics between
  schemas ('optionm' with yearly tables opprcd1996..opprcdYYYY, and 'optionm_all'
  with a single partitioned opprcd). Do not trust hardcoded names, including mine.
* Resumable: each year writes its own parquet file and is skipped if present.
  Kill it and restart at will; the WRDS connection will drop at some point.
* am_settlement = 1 keeps standard SPX monthlies and excludes SPXW weeklies.
  This filter is stable across the whole 1996-2025 history.
"""

import argparse
import os
import sys
import time

import pandas as pd
import wrds

SECID = 108105
START_YEAR = 1996
END_YEAR = 2025
OUTDIR = "data/raw"

# 12 columns, not 38. The full row set is ~9GB of CSV across the history;
# this is roughly a third of that before parquet compression.
COLS = [
    "secid", "date", "symbol", "exdate", "cp_flag", "strike_price",
    "best_bid", "best_offer", "volume", "open_interest",
    "impl_volatility", "am_settlement", "optionid",
]


def connect():
    # First run prompts for username/password and offers to create ~/.pgpass.
    # Say yes. Do this in the UK, not on hotel wifi.
    return wrds.Connection(wrds_username="samlessan")


def list_tables(db):
    for schema in ("optionm", "optionm_all", "optionm_all_eur"):
        try:
            tables = db.list_tables(library=schema)
        except Exception as exc:
            print(f"  {schema}: unavailable ({exc.__class__.__name__})")
            continue
        opt = [t for t in tables if "opprcd" in t]
        fwd = [t for t in tables if "fwdprd" in t]
        sec = [t for t in tables if "secprd" in t]
        zero = [t for t in tables if "zerocd" in t]
        print(f"\n[{schema}] {len(tables)} tables")
        print(f"  option prices : {opt[:3]}{' ...' if len(opt) > 3 else ''}  (n={len(opt)})")
        print(f"  forward price : {fwd[:3]}{' ...' if len(fwd) > 3 else ''}  (n={len(fwd)})")
        print(f"  security price: {sec[:3]}{' ...' if len(sec) > 3 else ''}  (n={len(sec)})")
        print(f"  zero curve    : {zero}")


def opt_table(year):
    """Adjust after running --list."""
    return f"optionm.opprcd{year}"


def fetch_year(db, year):
    sql = f"""
        SELECT {', '.join(COLS)}
        FROM {opt_table(year)}
        WHERE secid = {SECID}
          AND am_settlement = 1
    """
    return db.raw_sql(sql, date_cols=["date", "exdate"])


def fetch_support(db):
    """Forward prices and the zero-coupon curve. Both small, both required.

    forward_price is 100% NULL in the option price table -- it lives here.
    Cross-check these against a put-call-parity forward computed from the
    chain itself; if they disagree materially, trust neither until you know why.
    """
    out = {}
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            frames.append(db.raw_sql(
                f"SELECT * FROM optionm.fwdprd{year} WHERE secid = {SECID}",
                date_cols=["date", "expiration"],
            ))
        except Exception as exc:
            print(f"  fwdprd{year}: {exc.__class__.__name__}")
    if frames:
        out["forward"] = pd.concat(frames, ignore_index=True)
    try:
        out["zerocd"] = db.raw_sql("SELECT * FROM optionm.zerocd", date_cols=["date"])
    except Exception as exc:
        print(f"  zerocd: {exc.__class__.__name__}")
    return out


def probe(db, day):
    year = int(day[:4])
    sql = f"""
        SELECT {', '.join(COLS)}
        FROM {opt_table(year)}
        WHERE secid = {SECID} AND date = '{day}'
    """
    df = db.raw_sql(sql, date_cols=["date", "exdate"])
    if df.empty:
        print(f"{day}: no rows -- check table name and that this is a trading day")
        return
    df["K"] = df.strike_price / 1000.0
    df["dte"] = (df.exdate - df.date).dt.days
    df = df[df.am_settlement == 1]
    near = df[df.dte >= 14].sort_values("dte")
    if near.empty:
        print(f"{day}: no expiry beyond 14 days")
        return
    chain = df[df.exdate == near.exdate.iloc[0]]
    mid = (chain.best_bid + chain.best_offer) / 2
    calls = mid[chain.cp_flag == "C"].groupby(chain.K).first()
    puts = mid[chain.cp_flag == "P"].groupby(chain.K).first()
    joint = pd.concat([calls, puts], axis=1, keys=["c", "p"]).dropna()
    fwd = (joint.c - joint.p).abs().idxmin()          # crude ATM proxy
    ks = sorted(chain.K.unique())
    band = [k for k in ks if 0.9 * fwd < k < 1.1 * fwd]
    gaps = pd.Series(band).diff().dropna()
    print(
        f"{day}: dte={chain.dte.iloc[0]:>4}  spot~{fwd:>7.0f}  "
        f"strikes={len(ks):>4}  in +/-10%={len(band):>4}  "
        f"ATMgap={100 * gaps.median() / fwd:>6.3f}% of spot  "
        f"coverage={min(ks) / fwd:.2f}x-{max(ks) / fwd:.2f}x  "
        f"IVnull={chain.impl_volatility.isna().mean():.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe", metavar="YYYY-MM-DD")
    args = ap.parse_args()

    db = connect()
    try:
        if args.list:
            list_tables(db)
            return
        if args.probe:
            probe(db, args.probe)
            return

        os.makedirs(OUTDIR, exist_ok=True)
        total = 0
        for year in range(START_YEAR, END_YEAR + 1):
            path = f"{OUTDIR}/spx_{year}.parquet"
            if os.path.exists(path):
                print(f"{year}: skip (exists)")
                continue
            t0 = time.time()
            try:
                df = fetch_year(db, year)
            except Exception as exc:
                # Reconnect once, then move on. Rerun the script to fill gaps.
                print(f"{year}: FAILED {exc.__class__.__name__}: {exc}", file=sys.stderr)
                db = connect()
                continue
            df.to_parquet(path, compression="snappy", index=False)
            total += len(df)
            mb = os.path.getsize(path) / 1e6
            print(f"{year}: {len(df):>9,} rows  {mb:>7.1f} MB  {time.time() - t0:>5.1f}s")

        print(f"\ntotal {total:,} rows")
        for name, frame in fetch_support(db).items():
            frame.to_parquet(f"{OUTDIR}/{name}.parquet", index=False)
            print(f"{name}: {len(frame):,} rows")
    finally:
        db.close()


if __name__ == "__main__":
    main()
