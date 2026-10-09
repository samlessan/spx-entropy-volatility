# Does the entropy of the option-implied density forecast volatility?

**Beyond Implied Volatility: Option-Implied Entropy and the Predictability of Risk.**
Research code for a URSS-funded summer project (University of Warwick /
Warwick Business School, supervised by Dr Arie Gozluklu). Does the Shannon
entropy of the S&P 500 risk-neutral density, extracted from SPX option
chains via Breeden–Litzenberger (1978), forecast realised volatility beyond
implied volatility and the model-free implied moments?

**Answer: no, and the null is quantified.** Orthogonalised excess entropy
changes out-of-sample R² by **−0.011** (Clark–West t = −1.76, n = 1,905
trading days, January 2018 to July 2025) when added to 21-day realised
variance forecasts built from HAR-RV, ATM implied volatility and the
Bakshi–Kapadia–Madan moments. In a Monte Carlo of the re-implemented test,
with the planted regressor's first-order persistence matched to the entropy
series (AR(1) = 0.82, 200 replications), a signal worth about **0.018 of
out-of-sample R²** is detected with 80% probability.
For scale, implied volatility's own out-of-sample contribution is **0.139**
(t = +4.77). The null holds at horizons of 5 to 63 trading days, with 2020
excluded, with outliers winsorised, and with the BKM moments dropped.

A secondary result: once ATM implied volatility is in the model, the BKM
model-free moments *also* add nothing (CW t = −1.07). Once you know the
width of the distribution, its shape adds nothing.

## Why a null result is worth publishing code for

Three checks guard the result:

- **A calibration gate.** Before any entropy result is accepted, the
  pipeline must reproduce a known result: implied
  volatility beating HAR-RV out of sample (Christensen–Prabhala and a
  large literature since). Any run failing that gate is treated as a
  pipeline defect, not evidence.
    - **The gate caught five real defects** before any result was written up,
      including an unsorted-input bug in the IV interpolation, a dependent
      variable built from the wrong WRDS field, and price columns that turned
      out to be flags. Each fix was validated by re-running the gate.
- **A placebo regressor** (noise through the identical machinery). The
  single placebo run in `horse_race.py` gives t = +0.08; separately, across
  200 simulated persistent null regressors in `verification/verify_power.py`,
  the test rejects 6.0% of the time at a nominal 5%.
- **A synthetic harness** (in the research log, not this repository)
  validated the Breeden–Litzenberger extraction against densities with
  closed-form entropy. Figures for the first five defects likewise come
  from August runs recorded in the research log.

The BKM moments are also checked against an external benchmark: pipeline
BKM skewness correlates 0.85 in levels and 0.62 in daily changes with the
Cboe SKEW index. Annual medians differ by 0.1 or less (one SKEW index point)
in 7 of 11 years; in the calm years 2017–2019 pipeline skewness is 0.6 to 1.6
more negative, possibly because Cboe interpolates to exactly 30 days across
all expiries (`verification/verify_bkm.py`).

## Corrections (October 2026)

A fresh review of this repository before write-up, carried out with an AI
assistant, found the problems below. All are fixed in the current code.

| Issue | Effect | Fix |
|---|---|---|
| **Zero-bid quotes in the deep wings.** A quote of 0 bid / 0.10 ask entered at a mid of 0.05, overpricing the tails. | Spurious XH spikes to −0.95 and BKM kurtosis above 500 on affected dates, concentrated in low-volatility 2015–17. On 217 sampled dates, XH before and after the fix correlated only 0.68. Removing the noise raised XH's persistence (AR(1) 0.58 → 0.82), which widened the detectable effect from about 0.01 to about 0.02 of out-of-sample R². | Cboe VIX/SKEW strike selection in `extract_entropy.py`: zero bids dropped, each wing stops after two consecutive zero bids. |
| **Look-ahead through overlapping targets.** At each monthly refit, the last 20 training rows had 21-day targets that ran into the test period. | Flattered every model, concentrated in the March 2020 crash. Dropping 21 rows from the start or at random leaves the result unchanged, so the effect is leakage, not data loss. | 21-row embargo before each refit. |
| **Zero-return days.** SPY closed unchanged on 8 days; log(1e-12) = −27.6 entered HAR as an outlier. | HAR R² understated, IV's increment overstated (0.147 → 0.139). | rv floored at the 0.5th percentile of its positive values. |
| **In-sample figures labelled out-of-sample.** The previous summary quoted +0.0008 (entropy) and 0.131 (IV), which are full-sample in-sample increments, and n = 2,369, which counts rows from March 2016. | Labels only. | Summary now quotes out-of-sample figures from `results.json`. |
| HAC lags fixed at 21 for 42- and 63-day horizons; QLIKE evaluated at the median of a log forecast. | No conclusion changes. | Lags = max(h, 21); bias-corrected QLIKE reported. |

The August 2026 table is reproduced exactly by
`ENT_DIR=data/v1_entropy python horse_race.py published`, where
`data/v1_entropy/` holds the pre-filter extraction.

| Entropy result under each specification | Clark–West t | OOS ΔR² |
|---|---|---|
| August 2026 (pre-filter, no embargo, floor 10⁻¹²) | −0.22 | −0.001 |
| Pre-filter extraction, embargo and floor | −1.63 | −0.003 |
| **Current (zero-bid filter, embargo, floor)** | **−1.76** | **−0.011** |

## Paper

The full write-up, submitted as the URSS 2026 final output, is
[`paper/Lessan_URSS_2026_Option_Implied_Entropy.pdf`](paper/Lessan_URSS_2026_Option_Implied_Entropy.pdf).

The project used an AI assistant, Claude (Anthropic), for most of the code
and the first draft of the paper. The paper's acknowledgements set out who
did what.

## Repository map

| Path | Role |
|---|---|
| `pull_spx.py` | Pulls SPX chains, forwards, zero curve from WRDS/OptionMetrics |
| `audit_spx.py` | Data-quality audit of the raw chains |
| `extract_entropy.py` | Breeden–Litzenberger extraction with Cboe strike selection: smile smoothing, RND, Shannon entropy vs lognormal benchmark, BKM moments |
| `calibrate_noise.py`, `calibrate_ivol.py` | Quote-noise measurement-error model; dependent-variable unit resolution |
| `rebuild_rv.py` | Realised variance construction (close-to-close from the official session close) |
| `rv_diagnostics/` | The diagnostics that selected close-to-close over corrupted range estimators, plus panel checks |
| `strike_density_diagnostics.pdf` | Strike density and coverage of the SPX chain, 1996–2025; why the primary sample starts in 2015 |
| `check_orth.py` | Conditioning check on the out-of-sample orthogonalisation design matrix |
| `make_diagnostics_v4.py` | Builds the strike-density diagnostic page |
| `horse_race.py` | HAR-RV / +IV / +BKM / +entropy nested comparison: in-sample (Newey–West), out-of-sample expanding window with embargo, Clark–West, placebo. Writes `results.json` |
| `results.json`, `results_published.json` | Every headline number from the current and the August 2026 specification |
| `verification/` | Checks written separately from the pipeline: panel replication, look-ahead audit of the target, ground-truth tests against known SPY returns, persistence-matched power analysis, floor sensitivity, BKM against the Cboe SKEW index and the zero-bid test, and an attempted independent re-extraction of the entropy measure |

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
| HAR + IV + BKM | 0.6265 | 0.3158 | 0.526 |
| HAR + IV + BKM + entropy | 0.6370 | 0.3044 | 0.533 |

Clark–West (nested, HAC-21): HAR→+IV **t = +4.77**; +IV→+BKM t = −1.07;
+BKM→+entropy **t = −1.76**; placebo t = +0.08. Excluding 2020 (n = 1,652):
IV t = +5.66, entropy t = −1.78. In sample, the entropy coefficient has a
Newey–West t of −0.65.

**Power** (`verification/verify_power.py`, 200 replications, planted signal
with AR(1) matched to the orthogonalised entropy series):

| Planted share of residual variance | Power (± s.e.) | Out-of-sample incremental R² achieved |
|---|---|---|
| 0 (null) | 0.06 ± 0.02 | −0.004 |
| 0.015 | 0.55 ± 0.04 | 0.007 |
| 0.025 | 0.65 ± 0.03 | 0.011 |
| 0.030 | 0.77 ± 0.03 | 0.016 |
| 0.040 | 0.88 ± 0.02 | 0.021 |

**Independent re-implementation** (`verification/verify_panel.py`, separate
code, orthogonalisation fitted from a 250-row burn-in so training starts in
March 2016): entropy t = −0.59, IV t = +4.77. Entropy t by horizon
(5/10/21/42/63 days): −0.46, −0.64, −0.59, −0.20, −0.25, with HAC lags ≥ h.
The two implementations agree on sign and insignificance. They differ in
training start and in how training rows are residualised; one candidate
explanation for the gap in magnitude is drift in the entropy series (annual
means across all date-expiry pairs −0.19 in 2015, −0.09 in 2022), which was
not tested directly. The power curve above uses the re-implemented test.

Sam Lessan · BSc Economics, University of Warwick
