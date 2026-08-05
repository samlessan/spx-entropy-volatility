
---

## Audit results (28 July 2026) and a screen correction

**Integrity: clean.** 30 files, 18,046,547 rows. Every year 248-253 trading days;
2025 correctly 165 days to 29 Aug. Zero duplicate optionids, zero negative dte,
zero non-European. Only 29 crossed quotes in 18M rows (0.00016%) - filter
`best_offer > best_bid`. The 7-day gap in 2001 is the 9/11 closure, not a fault.

**IMPORTANT - the coverage screen in item 7 above is WRONG. Do not use it.**

Pass rates: 2019 = 0.791, **2020 = 0.598**, 2021 = 0.880. Same pattern 2008 (0.312)
vs 2009 (0.444). The screen is endogenous to volatility: coverage in sigma units is
ln(K_max/F)/(IV*sqrt(T)), and exchanges list strikes on a roughly fixed *moneyness*
grid, so when IV spikes the denominator grows and sigma-unit coverage mechanically
shrinks. The screen therefore preferentially deletes high-volatility days - exactly
the observations the hypothesis is about, and exactly where the RND is most
non-Gaussian. It would truncate the right tail of the dependent variable and flatter
the forecasting results undetectably.

**Replacement: decompose, don't drop.** Per date-expiry, integrate entropy separately
over (a) the observed strike range and (b) the GEV-extrapolated tails. Report entropy
from observed prices as primary, extrapolated probability-mass share as a diagnostic,
total entropy as secondary. Keeps every observation, makes extrapolation dependence
explicit rather than hidden, allows robustness across extrapolation-share terciles.
Also rescues the early sample: 1996 has a high extrapolated share rather than being
excluded outright.~
cd ~/urss
cat >> HANDOVER.md << 'EOF'

---

## Audit results (28 July 2026) and a screen correction

**Integrity: clean.** 30 files, 18,046,547 rows. Every year 248-253 trading days;
2025 correctly 165 days to 29 Aug. Zero duplicate optionids, zero negative dte,
zero non-European. Only 29 crossed quotes in 18M rows (0.00016%) - filter
`best_offer > best_bid`. The 7-day gap in 2001 is the 9/11 closure, not a fault.

**IMPORTANT - the coverage screen in item 7 above is WRONG. Do not use it.**

Pass rates: 2019 = 0.791, **2020 = 0.598**, 2021 = 0.880. Same pattern 2008 (0.312)
vs 2009 (0.444). The screen is endogenous to volatility: coverage in sigma units is
ln(K_max/F)/(IV*sqrt(T)), and exchanges list strikes on a roughly fixed *moneyness*
grid, so when IV spikes the denominator grows and sigma-unit coverage mechanically
shrinks. The screen therefore preferentially deletes high-volatility days - exactly
the observations the hypothesis is about, and exactly where the RND is most
non-Gaussian. It would truncate the right tail of the dependent variable and flatter
the forecasting results undetectably.

**Replacement: decompose, don't drop.** Per date-expiry, integrate entropy separately
over (a) the observed strike range and (b) the GEV-extrapolated tails. Report entropy
from observed prices as primary, extrapolated probability-mass share as a diagnostic,
total entropy as secondary. Keeps every observation, makes extrapolation dependence
explicit rather than hidden, allows robustness across extrapolation-share terciles.
Also rescues the early sample: 1996 has a high extrapolated share rather than being
excluded outright.

**Revised sample:** primary 2015-2025 (RV entitlement binds, not option density);
extended 2010-2025 if the IID splice validates; robustness 2008-2014 with
extrapolation share as a control.

**Density by year (median, 14-90 dte):**

| Year | strikes | within_1sig | atm_gap% | hi_sigma |
|---|---|---|---|---|
| 1996 | 29 | 14 | 0.76 | +1.21 |
| 2000 | 29 | 12 | 1.64 | +2.28 |
| 2005 | 46 | 18 | 0.42 | +2.92 |
| 2010 | 150 | 45 | 0.45 | +3.38 |
| 2015 | 229 | 70 | 0.24 | +3.31 |
| 2020 | 346 | 130 | 0.16 | +3.20 |
| 2025 | 428 | 169 | 0.08 | +7.82 |

**Structural break:** 2022+ has ~100% pass rate and ~3x the strike density of 2015
(0DTE, vol-selling ETF complex). Pre/post-2022 stability testing is necessary.

**Term-structure constraint:** ~630 date-expiry pairs/year vs ~252 trading days = only
~2.5 expiries per date in the 14-90 day window. That is the cost of `am_settlement=1`.
Fine for the primary test; constrains any term-structure-of-entropy secondary result.

**Supervisor:** substantive update sent 28 July 2026 asking (a) how the literature
handles sparse strike coverage, (b) for 20 minutes to discuss whether entropy is
separable from implied volatility. Deliberately plain-language. If no reply within
a week, short follow-up in the same thread.

Outputs: `data/audit_yearly.csv`, `data/density.csv`.

**Screen endogeneity: tested and REFUTED (30 July).** Within-year correlations, n=18,774:
hi_sigma vs log IV = -0.012, pass rate vs log IV = +0.031, log strike range vs log IV = +0.281.
Sigma-unit coverage is NOT volatility-dependent -- the listed strike ladder widens with IV,
offsetting the mechanical effect of IV entering the coverage denominator. The earlier
endogeneity claim (based on 2019/2020/2021 pass rates) was confounded by the secular rise in
strike density and does not survive a within-year test.

One exception worth keeping: lo_sigma vs log IV = +0.195, so downside coverage in sigma units
DOES contract as vol rises. Economically irrelevant -- median lo_sigma runs -15 to -39, so that
tail is covered many times over. Useful answer if asked whether the mechanism exists at all.

Decomposition still preferred, but for the right reason: it converts a tail-extrapolation
assumption into a measured, reportable quantity. Not because screening is biased.

Sent to supervisor 30 July as `diagnostics.pdf`. Script: make_diagnostics_v4.py

---

## Session log: 4-5 August 2026 — supervisor meeting, locked design, harness, go/no-go

### Supervisor meeting (5 Aug, Teams)

Arie endorsed the excess-entropy framing and approved an economic-significance
section (does any forecasting improvement survive transaction costs). He supplied
four papers: Buchen & Kelly (1996, JFQA) and Rompolis (2010, JEF) — maximum-entropy
density estimation; Rompolis & Tzavalis (2008, JFQA) — Gram-Charlier parametric
recovery; Gao & Martin (2021, JF) — what a finished option-implied indicator looks
like. He also mentioned his own work on skewness preferences, which is the economic
mechanism behind non-zero excess entropy (skew demand makes the RND lopsided).

**The circularity point (novelty claim, and the referee-attack line):** Buchen-Kelly
and Rompolis (2010) use entropy as the *estimation objective* — the fitted density
is the maximum entropy consistent with prices, so its entropy is an upper bound by
construction and biased against structure. This project uses entropy of an
*independently estimated* density as a *state variable*. That distinction must be
stated cleanly in the write-up. Reading order: Gao-Martin first, estimation trio
after.

Follow-up owed: five-bullet summary email of the meeting (written record for
September). Status: check sent-mail; send if not.

### Locked design (fixed in advance; do not re-tune)

| Choice | Decision |
|---|---|
| Estimator | Cubic smoothing spline on OTM IV smile in log-moneyness, **s = 1e-2 fixed**, BL second derivative |
| MXE | Robustness only (degenerates under real noise) |
| Bimodality | **Cut.** Harness: 0% detection at SPX precision at every realistic noise level, with false positives on unimodal densities |
| Variable | Excess entropy xh = h(RND) − h(variance-matched lognormal); empirically xh = −KL(RND ‖ lognormal), always negative |
| Sample | 2018–2025 primary, 2015–2025 extended (2015 spread/mid 44% vs ~3–4% later) |
| Controls | log IV, BKM skew/kurt, log half-spread, n_strikes, dte |
| Benchmark | HAR-RV + IV; nested Clark-West, OOS expanding from 2018-01, refit monthly; QLIKE + MSE; NW-21 in-sample |

### Synthetic harness (bl/harness.py) — results of record

Entropy recoverable at s = 1e-2 with bias +0.015 to +0.055 nats depending on quote
noise; measurement sd ≈ 0.013 at 2020-era spreads. Bias slope **0.0237 nats per log
unit of half-spread**, later validated out of sample on real data (realised lsp
coefficient 0.0227, t = 11.2 — a 4% match). This is the project's measurement-error
model and its strongest methodological asset.

### 2020 single-year validation and go/no-go: PASS

xh mean −0.0983, sd 0.0380, tail_share 3.3%, atm_iv 0.304. Orthogonalisation
ladder (R² on xh): IV only 0.220 → +micro 0.414 → +BKM 0.585 → FULL 0.602;
residual sd 0.0240 vs measurement noise 0.013 (S/N 1.85). corr(xh, bkm_skew)
= 0.742 — high but 45% of variance unique. Verdict: real variable, proceed.
Multi-year extraction 2015–2025 launched (~630 pairs/year; 2025 partial to Aug).

---

## Session log: 5 August 2026 — five defects, RV rebuild, the null, verification

### Calibration gate (standing rule)

Before any entropy coefficient is read, the pipeline must reproduce the known
result that implied volatility beats HAR-RV out of sample. A failing gate is a
pipeline defect, not evidence. A placebo (noise regressor through identical
machinery) must be insignificant on every run. The gate refused the pipeline
five times before passing.

### Defect ledger

| # | Defect | How caught | Fix |
|---|---|---|---|
| 1 | `np.interp` on unsorted moneyness → ATM IV corrupt on ~1% of dates (p99 IV > 1.0); also xh blowups to −14.7 at low IV | atm_iv p99 = 1.02; xh outliers 100× sd | Sort before interpolation; plausibility screens; re-extract |
| 2 | Dependent variable built from `ivol_t` (WRDS IID per-unit-time microstructure variance), median-matched by a 1.525e4 scale | Gate failed: HAR→+IV CW t = −0.49; corr(log VIX, fwd-21d ivol_t) = 0.43 vs literature 0.65–0.72 | Abandon ivol_t |
| 3 | `O_official`/`C_official` are constant-1.0 flags, not prices; GK silently degenerated to 0.5·rng² | GK ≡ Parkinson bitwise; H/L outside O-C envelope on 2,680/2,680 rows | Real open/close are `OPrc`/`CPrc` |
| 4 | `calibrate_ivol.py::pick()` searched only columns containing "price", so never found OPrc/CPrc | Code read | n/a (superseded) |
| 5 | `price_high_m`/`price_low_m` carry erroneous prints (e.g. L = 506 on 2025-01-03 vs true ~586); range inflated ~60% (rng/\|r\| = 2.77 vs GBM 1.6–1.8); a minority of days dominates every 21-day mean | Ground-truth check on known sessions; MEDIAN-vs-MEAN aggregation gap (0.62 vs 0.36) | Drop range estimators entirely |

**Dependent variable of record:** close-to-close realised variance r², r = Δln(CPrc),
from `spy_iid_ms.csv`. Validation: corr(log VIX, log fwd-21d rv) = **0.672** (correct
benchmark for *forward* RV is 0.65–0.72; the 0.75–0.85 figure is contemporaneous);
implied median annualised vol 0.1174 vs true 0.11–0.13; `rebuild_rv.py` asserts the
0.60 gate before writing. Note `rv_daily` ≡ `park` in the old file (one estimator,
two names). Median-matching leakage removed along with the rescale.

### Gate after rebuild: PASS

HAR→HAR+IV: CW = +0.295, **t = +4.89**; IV adds 13.1pp in-sample R², 14.7pp OOS,
QLIKE −0.115. Placebo t = −0.41. Pipeline calibrated for the first time.

### Primary result: NULL (pre-registered as reportable)

| | MSE(log) | R²_oos | QLIKE |
|---|---|---|---|
| HAR | 0.7401 | 0.1918 | 0.768 |
| HAR+IV | 0.6056 | 0.3387 | 0.653 |
| HAR+IV+BKM | 0.6134 | 0.3302 | 0.670 |
| HAR+IV+BKM+XH | 0.6143 | 0.3292 | 0.659 |

xh_o: in-sample coef −0.329, t = −0.37, p = 0.715; OOS CW t = −0.22 (independent
replication: −0.17). Incremental OOS R² **+0.00075**. Secondary null: BKM moments
also add nothing beyond IV (CW t = −1.45; OOS R² falls). Not collinearity:
orthogonalisation R² = 0.440, so 56% of xh variance is unique. Robust across
H = 5, 10, 21, 42, 63 (IV gate t decays 7.05 → 3.40 correctly; XH null at all).

### Power (the sentence that makes the null informative)

Persistence-matched Monte Carlo (planted signal AR(1) = 0.58 to match xh_o;
effective n ≈ 630 of 2,369): size at true zero 0.10; **80% power at incremental
R² = 0.02**; 50% at 0.01. Observed effect is 27× below the 80% threshold and 174×
below IV's contribution. Reliability adjustment: cross-implementation corr ≥ 0.65
implies pipeline xh reliability ≥ ~0.7, so for *true* entropy the 80%-power bound
is ≈ 0.03. iid-planted power (0.005) is an overclaim — do not use.

### Verification ledger (independent code, no imports from the pipeline)

| Check | Result |
|---|---|
| Price series is SPY | 10/10 calendar-year returns within 3.5pp; levels on 2020-03-23 and 2024-12-31 correct |
| No look-ahead in y | explicit-loop vs vectorised, max diff 8.7e-19 |
| Independent HAR/CW replication | R² and CW t identical to 4 dp |
| Test size | 0.10 at true zero (CW slightly undersized — safe direction) |
| Zero-return days | 8 days; verdict invariant across rv floors |
| Forward matches forward.parquet | max rel diff 1.1e-3 |
| BKM by pure integration | skew corr 0.94, kurt 0.88 excluding 8 wing-dominated long-dated pairs (median dte 76); several rows match to 5 s.f. (e.g. 2021-06-28 kurt 97.001 vs 97.011); option selection corr(n_used) = 0.987 |
| xh cross-implementation | Spearman ≈ Pearson ≈ 0.65 (2018+, n=24) vs attenuation ceiling ~0.96; gap −0.15 structured (scales with \|skew\| and dte) → attributed to the independent estimator's lognormal-based calibration, second-order in non-lognormality. Level validation of xh rests on the pipeline's own harness (BL recovery + 4% OOS prediction of the noise slope) |
| Martingale | \|xh(mu-matched) − xh(martingale)\| ≤ 0.011 all pairs |

**Vol-scaling law (methods-section paragraph):** per-bin signal in density
extraction is vol-invariant while per-bin noise scales as 1/tv, so SNR ∝ tv and
the Jensen entropy bias ∝ 1/(2·SNR²). Three independent manifestations: the
2015–17 BKM blowups at 7–13% IV, the pre-patch xh = −12.8 crashes at 8% IV, and
the independent extractor's sole failing cell at tv = 0.034. Low-vol regimes are
the hard ones. Entropy is *bias*-fragile under quote noise (E[ln(q+δ)] ≈ ln q −
σ²/2q²): unregularised extraction fails systematically (three failed independent
attempts, kept in `verification/`), which is the empirical case for the smoothing
spline. Stopping rule honoured: no further extractor attempts.

### Publication

Public repo: **github.com/samlessan/spx-entropy-volatility** (account renamed from
RandomFr0g; commits attributed via noreply email). Code only — `data/`, `papers/`
(journal PDFs + IvyDB manual), and logs excluded per OptionMetrics/WRDS licence.
README carries the null, the power bound, the gate story, and the licence note.
verify4b–6 kept deliberately as the documented failure of unregularised extraction.

### Open items

1. Pre-registration document, written as "design fixed in advance" (Bali work).
2. Write-up. Lead methods with the calibration gate; state the null with the MDE;
   include the reliability-adjusted bound and the vol-scaling law; state the
   circularity distinction vs Buchen-Kelly/Rompolis as the novelty claim.
3. Pre-registered robustness still to run: half-spread quintile buckets (entropy
   coefficient within buckets kills the spread-trend story); |dte−30| ≤ 7 band;
   pre/post-2022 stability (structural break in strike density); BKM extreme-row
   screen (|skew|>20 or kurt>500, 8 rows, 0.3%) with/without.
4. Economic-significance / transaction-cost section (Arie approved) — for the
   null this becomes short: nothing to trade on.
5. Extended sample 2010–2025 conditional on the IID splice validating with a
   CPrc-equivalent close in `spy_iid_std_2010_2014.csv`. Optional.
6. Five-bullet meeting-summary email to Arie if not sent; plus a two-line note
   that the repo is public (data excluded per licence).
7. 2025 entropy files end ~Aug 29; refresh near deadline if entitlement allows.
