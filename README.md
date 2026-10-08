# Does the entropy of the option-implied density forecast volatility?

**Beyond Implied Volatility: Option-Implied Entropy and the Predictability of Risk.**
Research code for a URSS-funded summer project (University of Warwick /
Warwick Business School, supervised by Dr Arie Gozluklu). Does the Shannon
entropy of the S&P 500 risk-neutral density, extracted from SPX option
chains via Breeden–Litzenberger (1978), forecast realised volatility beyond
implied volatility and the model-free implied moments?

**Answer: no, and the null is quantified.** Orthogonalised excess entropy
changes out-of-sample R² by **−0.0033** (Clark–West t = −1.63, n = 1,905
trading days, January 2018 to July 2025) when added to 21-day realised
variance forecasts built from HAR-RV, ATM implied volatility and the
Bakshi–Kapadia–Madan moments. A Monte Carlo with the planted regressor's
persistence matched to the entropy series (AR(1) = 0.58, 200 replications)
detects a signal worth about **0.01 of out-of-sample R²** with 85% probability
(±2.5%), so an effect of that size would very likely have been found. For
scale, implied volatility's own out-of-sample contribution is **0.139**
(t = +4.77). The null holds at horizons of 5 to 63 trading days and with
2020 excluded.

A secondary result: once ATM implied volatility is in the model, the BKM
model-free moments *also* add nothing (CW t = −1.47). Once you know the
width of the distribution, its shape adds nothing.

## Why a null result is worth publishing code for

Three checks were built in before the main test was run:

- **A pre-committed calibration gate.** Before reading the entropy
  coefficient, the pipeline must reproduce a known result: implied
  volatility beating HAR-RV out of sample (Christensen–Prabhala and a
  large literature since). Any run failing that gate is treated as a
  pipeline defect, not evidence.
    - **The gate caught five real defects** before any result was written up,
      including an unsorted-input bug in the IV interpolation, a dependent
      variable built from the wrong WRDS field, and price columns that turned
      out to be flags. Each fix was validated by re-running the gate.
- **A placebo regressor** (noise through the identical machinery). The
  single placebo run gives t = −0.54, and across 200 simulated null
  regressors the test rejects 4.5% of the time at a nominal 5%.
- **A synthetic harness** (in the research log, not this repository)
  validated the Breeden–Litzenberger extraction against densities with
  closed-form entropy; its measurement-error model predicted the real-data
  spread coefficient to 4% (0.0237 predicted vs 0.0227 realised per log unit
  of half-spread).

## Corrections (October 2026)

An independent audit of this repository before write-up found two further
defects and three mislabelled statistics. All are fixed in the current code;
`python horse_race.py published` reproduces the original August 2026 table.

| Issue | Effect | Fix |
|---|---|---|
| **Look-ahead through overlapping targets.** At each monthly refit, the last 20 training rows had 21-day targets that ran into the test period. | Flattered every model, the entropy model most, and concentrated in the March 2020 crash. Entropy t moved from −0.22 to −1.63. | 21-row embargo before each refit. Controls that drop 21 rows from the start or at random leave t near −0.22, so the change is leakage, not data loss. |
| **Zero-return days.** SPY closed unchanged on 8 days; log(1e-12) = −27.6 entered HAR as an outlier. | HAR R² was understated, which overstated IV's increment (0.147 → 0.139). Entropy unaffected. | rv floored at the 0.5th percentile of its positive values. |
| **In-sample figures labelled out-of-sample.** The previous summary quoted +0.0008 (entropy) and 0.131 (IV), which are full-sample in-sample increments, and n = 2,369, which counts rows from March 2016. | Labels only. | Summary now quotes out-of-sample figures from `results.json`. |
| HAC lags fixed at 21 for 42- and 63-day horizons; QLIKE evaluated at the median of a log forecast. | No conclusion changes. | Lags = max(h, 21); bias-corrected QLIKE reported alongside. |

## Repository map

| Path | Role |
|---|---|
| `pull_spx.py` | Pulls SPX chains, forwards, zero curve from WRDS/OptionMetrics |
| `audit_spx.py` | Data-quality audit of the raw chains |
| `extract_entropy.py` | Breeden–Litzenberger extraction: smile smoothing, RND, Shannon entropy vs lognormal benchmark, BKM moments |
| `calibrate_noise.py`, `calibrate_ivol.py` | Quote-noise measurement-error model; dependent-variable unit resolution |
| `rebuild_rv.py` | Realised variance construction (close-to-close from the official session close) |
| `rv_diagnostics/` | The diagnostics that selected close-to-close over corrupted range estimators, plus panel checks |
| `strike_density_diagnostics.pdf` | Strike density and coverage of the SPX chain, 1996–2025; why the primary sample starts in 2015 |
| `check_orth.py` | Conditioning check on the out-of-sample orthogonalisation design matrix |
| `make_diagnostics_v4.py` | Builds the strike-density diagnostic page |
| `horse_race.py` | HAR-RV / +IV / +BKM / +entropy nested comparison: in-sample (Newey–West), out-of-sample expanding window with embargo, Clark–West, placebo. Writes `results.json` |
| `results.json`, `results_published.json` | Every headline number from the current and the original specification |
| `verification/` | Checks written separately from the pipeline: panel replication, look-ahead audit of the target, ground-truth tests against known SPY returns, persistence-matched power analysis, floor sensitivity, and an attempted independent re-extraction of the entropy measure from the raw chains |

## Data

Raw inputs are **OptionMetrics IvyDB and WRDS Intraday Indicators under
institutional licence and are not distributed**; the `data/` tree is
excluded. With WRDS access, `pull_spx.py` rebuilds it.

```
pip install -r requirements.txt
python pull_spx.py 2015 2025      # requires WRDS credentials (~/.pgpass)
python extract_entropy.py 2015 2025
python rebuild_rv.py
python horse_race.py
python verification/verify_power.py
```

On macOS with Apple's Accelerate BLAS, numpy can emit spurious `RuntimeWarning:
... encountered in matmul` messages; results are finite and unaffected
(`check_orth.py`).

## Headline numbers

Out of sample, January 2018 to July 2025, n = 1,905, expanding window refit
monthly with a 21-row embargo. Source: `results.json`.

| | MSE(log) | R²_oos | QLIKE (bias-corrected) |
|---|---|---|---|
| HAR | 0.7403 | 0.1916 | 0.566 |
| HAR + IV | 0.6129 | 0.3307 | 0.526 |
| HAR + IV + BKM | 0.6217 | 0.3211 | 0.540 |
| HAR + IV + BKM + entropy | 0.6247 | 0.3178 | 0.541 |

Clark–West (nested, HAC-21): HAR→+IV **t = +4.77**; +IV→+BKM t = −1.47;
+BKM→+entropy **t = −1.63**; placebo t = −0.54. Excluding 2020 (n = 1,652):
IV t = +5.66, entropy t = −1.58.

**Power** (`verification/verify_power.py`, 200 replications, planted signal
with AR(1) matched to the orthogonalised entropy series):

| Planted share of residual variance | Power (± s.e.) | Out-of-sample incremental R² achieved |
|---|---|---|
| 0.010 | 0.65 ± 0.03 | 0.004 |
| 0.0125 | 0.74 ± 0.03 | 0.007 |
| 0.015 | 0.85 ± 0.03 | 0.010 |
| 0.020 | 0.91 ± 0.02 | 0.012 |

**Independent re-implementation** (`verification/verify_panel.py`, separate
code, orthogonalisation fitted from a 250-row burn-in so training starts in
March 2016): entropy t = −0.93, IV t = +4.77. Entropy t by horizon
(5/10/21/42/63 days): −0.15, −0.00, −0.93, −0.71, −0.75, with HAC lags ≥ h.
The two implementations agree on sign and significance; they differ in
magnitude because the re-implementation's training sample starts a year
later.

Sam Lessan · BSc Economics, University of Warwick
