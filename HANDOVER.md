
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
