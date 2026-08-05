# Does the entropy of the option-implied density forecast volatility?

Research code for a URSS-funded summer project (University of Warwick /
Warwick Business School, supervised by Dr Arie Gozluklu): does the Shannon
entropy of the S&P 500 risk-neutral density — extracted from SPX option
chains via Breeden–Litzenberger (1978) — forecast realised volatility
beyond implied volatility and the model-free implied moments?

**Answer: no, and the null is quantified.** Orthogonalised excess entropy
contributes an incremental out-of-sample R² of **+0.0008** (Clark–West
t = −0.17, n = 2,369, 2018–2025) to 21-day realised-variance forecasts
beyond HAR-RV, ATM implied volatility and the Bakshi–Kapadia–Madan
moments. A Monte Carlo with the planted regressor's persistence matched to
the entropy series (AR(1) = 0.58) shows the test has **80% power against an
incremental R² of 0.02** — so effects larger than ~2% of forecast variance
are ruled out. For scale, implied volatility's own contribution is 0.131.
The null is robust across horizons of 5 to 63 trading days.

A secondary result: once ATM implied volatility is in the model, the BKM
model-free moments *also* add nothing (CW t = −1.45). Everything
forecastable in the shape of the risk-neutral density appears to be
subsumed by the level of implied volatility.

## Why a null result is worth publishing code for

The design was built to make a null informative rather than ambiguous:

- **A pre-committed calibration gate.** Before reading the entropy
  coefficient, the pipeline must reproduce a known result: implied
  volatility beating HAR-RV out of sample (Christensen–Prabhala and a
  large literature since). Any run failing that gate is treated as a
  pipeline defect, not evidence.
- **The gate caught five real defects** before any result was written up,
  including an unsorted-input bug in the IV interpolation, a dependent
  variable built from the wrong WRDS field, and price columns that turned
  out to be flags. Each fix was validated by re-running the gate
  (final: CW t = +4.89).
- **A placebo regressor** (pure noise through the identical machinery)
  must come back insignificant on every run. It does (t = +0.23).
- **A synthetic harness** validates the Breeden–Litzenberger extraction
  against densities with closed-form entropy, and its measurement-error
  model predicted the real-data spread coefficient out of sample to 4%
  (0.0237 predicted vs 0.0227 realised per log unit of half-spread).

## Repository map

| File | Role |
|---|---|
| `pull_spx.py` | Pulls SPX chains, forwards, zero curve from WRDS/OptionMetrics |
| `audit_spx.py` | Data-quality audit of the raw chains |
| `extract_entropy.py` | Breeden–Litzenberger extraction: smile smoothing, RND, Shannon entropy vs lognormal benchmark, BKM moments |
| `calibrate_noise.py`, `calibrate_ivol.py` | Quote-noise measurement-error model; dependent-variable unit resolution |
| `rebuild_rv.py` | Realised-variance construction (close-to-close from the official session close) |
| `rv_diagnostics/` | The diagnostics that selected close-to-close over corrupted range estimators, plus panel checks |
| `horse_race.py` | HAR-RV / +IV / +BKM / +entropy nested comparison: in-sample (Newey–West), out-of-sample expanding window, Clark–West, placebo |
| `verification/verify.py` … `verify3.py` | Independent re-implementation of the panel, look-ahead audit, ground-truth checks against known SPY returns, persistence-matched power analysis |
| `verification/verify4b.py` … `verify8.py` | Independent re-extraction of the entropy measure from raw chains under spline-free numerics (butterfly histograms + arbitrage-repair projection), with gated self-tests |
| `HANDOVER.md` | Running research log: defects, decisions, verification ledger |

The verification suite is deliberately kept, failures included: attempts
4b–6 document quantitatively why unregularised density extraction fails
(the entropy functional is *bias*-fragile under quote noise — E[ln(q+δ)]
≈ ln q − σ²/2q² — and per-bin signal-to-noise scales with the level of
volatility), which is the empirical case for the smoothing step in the
main pipeline. The final independent extractor reproduces the pipeline's
BKM moments to median errors of 8% (skew) and 21% (kurtosis) and the
entropy measure ordinally (Spearman 0.65 post-2018).

## Data

Raw inputs are **OptionMetrics IvyDB and WRDS Intraday Indicators under
institutional licence and are not distributed** — the `data/` tree is
excluded. With WRDS access, `pull_spx.py` rebuilds it; without, the
synthetic harness in the `verify*` scripts exercises the full extraction
and testing machinery on generated chains with known answers.

```
pip install -r requirements.txt
python pull_spx.py 2015 2025      # requires WRDS credentials (~/.pgpass)
python extract_entropy.py 2015 2025
python horse_race.py
```

## Headline numbers

| | MSE(log) | R²_oos | QLIKE |
|---|---|---|---|
| HAR | 0.7401 | 0.1918 | 0.768 |
| HAR + IV | 0.6056 | 0.3387 | 0.653 |
| HAR + IV + BKM | 0.6134 | 0.3302 | 0.670 |
| HAR + IV + BKM + entropy | 0.6143 | 0.3292 | 0.659 |

Clark–West (nested, HAC-21): HAR→+IV **t = +4.89**; +IV→+BKM t = −1.45;
+BKM→+entropy **t = −0.22**; placebo t < 0.5 throughout.

Sam Lessan · BSc Economics, University of Warwick
