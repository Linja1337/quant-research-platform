# Methodology

How a strategy candidate moves from hypothesis to deployment-eligible. The
short version: any backtest result is treated as overfit until proven
otherwise, and the burden of proof is shouldered by formal statistical tests
rather than by the researcher's intuition.

This document is the long version. Each section opens with a plain-English
summary, then gets technical. If you want to skip to a specific topic, the
Table of Contents links jump straight there.

## Table of Contents

1. [Why this exists](#why-this-exists)
2. [Walk-forward validation](#walk-forward-validation)
3. [Combinatorially Symmetric Cross-Validation (CSCV) and PBO](#cscv-and-pbo)
4. [Deflated Sharpe Ratio (DSR)](#deflated-sharpe-ratio-dsr)
5. [Permutation tests](#permutation-tests)
6. [Parameter stability surfaces](#parameter-stability-surfaces)
7. [Cross-framework parity audits](#cross-framework-parity-audits)
8. [Branch tournament and decision logs](#branch-tournament-and-decision-logs)
9. [How the gates compose](#how-the-gates-compose)
10. [What this method does not do](#what-this-method-does-not-do)
11. [References](#references)

---

## Why this exists

Backtests are the single largest source of self-deception in systematic
trading. The mechanism is simple: if you test enough variations of a strategy,
some of them will look profitable on historical data by chance alone. Standard
hold-out and train-test splits do not catch this, because they only run a
single split, and any single split is one draw from a much larger distribution.

The methodology in this platform exists to attack that problem at every layer.
No single technique is sufficient. A strategy must clear several independent
gates, each designed to fail strategies that the others might miss.

So what? If a candidate clears walk-forward, CSCV, DSR, permutation, parameter
stability, and parity audits, it is no longer plausible that the result is from
selection bias on historical data alone. That is the bar.

---

## Walk-forward validation

**Plain English.** Train the strategy's parameters on a window of past data,
then test it on a window of unseen data that immediately follows. Slide both
windows forward, repeat. The strategy is judged on the unseen windows, not on
the windows it was tuned on.

**Technical.** The platform uses Pardo-style rolling and anchored walk-forward
with calendar-day windows (not bar counts, because holidays and session gaps
matter). Each fold optimizes parameters on the in-sample (IS) window and
records out-of-sample (OOS) metrics on the next test window with no
re-optimization. The Walk-Forward Efficiency (WFE) per fold is

```
WFE = OOS_metric / IS_metric
```

Targets:

- WFE > 0.50 — the OOS edge retains at least half its in-sample magnitude.
- WFE distribution across folds should be tight, not bimodal. Bimodal WFE
  usually means the strategy depends on a regime that appears in some folds
  and not others.

A demo of the technique on synthetic data is in `demos/walk_forward_demo.py`.

---

## CSCV and PBO

**Plain English.** Walk-forward gives you one in-sample / out-of-sample split.
But there are many ways to slice a dataset into IS and OOS, and any single
slice can be lucky. CSCV runs every symmetric slice, picks the in-sample
winner each time, and asks how often that winner ends up below median in the
out-of-sample half. That fraction is the Probability of Backtest Overfitting.

**Technical.** Following Bailey, Borwein, Lopez de Prado, and Zhu (2014),
Algorithm 2.3:

1. Build a performance matrix `M` of shape `(T, N)`, where T is the number of
   time blocks and N is the number of strategy configurations under
   consideration.
2. Enumerate all `C(T, T/2)` symmetric partitions of the T blocks into an
   in-sample half and an out-of-sample half.
3. For each partition:
   - Pick `n*`, the configuration with the largest aggregate performance in
     the in-sample half.
   - Compute the rank of `n*` in the out-of-sample performance distribution
     across all N configurations. Normalize the rank to a quantile
     `omega_bar` in `(0, 1)`.
   - Compute `logit = ln(omega_bar / (1 - omega_bar))`.
4. PBO is the fraction of partitions for which `logit <= 0`. Equivalently, the
   fraction of partitions where the in-sample winner ranked at or below the
   out-of-sample median.

Operational targets:

- PBO < 0.10 — strong evidence the in-sample selection generalizes.
- PBO between 0.10 and 0.25 — moderate concern, look at parameter stability
  and DSR before going further.
- PBO > 0.50 — the in-sample winner is essentially uninformative about OOS
  performance. Reject.

CSCV also produces:

- The logit distribution across partitions. Skew tells you the shape of the
  selection bias.
- A Spearman rank correlation between IS and OOS rank, averaged across
  partitions. A negative correlation is a red flag for regime reversal.
- The `prob_loss` statistic: fraction of partitions where the IS winner
  produced negative OOS performance.

The demo at `demos/cscv_pbo_demo.py` runs CSCV on two strategy populations of
equal size, one disciplined and one composed of pure-random PnL. The pure-
random population reliably produces PBO ≈ 0.5, which is the textbook outcome
under the null and the most useful sanity check on the implementation.

### What PBO is not

It is not the probability that any individual strategy is overfit. It is a
property of the *selection process*: across many possible IS / OOS splits, how
often did the procedure of "pick the IS winner" produce a strategy that
generalized? If you only run one split (which is what most backtests do), PBO
is undefined for you.

---

## Deflated Sharpe Ratio (DSR)

**Plain English.** A Sharpe ratio of 2 looks impressive until you remember the
researcher tested 500 variations to find the one that produced it. The
Deflated Sharpe Ratio adjusts the headline number for that selection bias and
for the non-normality of returns.

**Technical.** Following Bailey and Lopez de Prado (2014), the expected
maximum Sharpe under the null hypothesis (no edge) for `N` independent trials
is

```
SR_0 = sqrt(V[SR_n]) * ( (1 - gamma) * Phi^-1(1 - 1/N)
                       + gamma * Phi^-1(1 - 1/(N*e)) )
```

where:

- `gamma` ≈ 0.5772156649 is the Euler-Mascheroni constant.
- `e` ≈ 2.718 is Euler's number.
- `V[SR_n]` is the cross-sectional variance of Sharpe ratios across the N
  trials.
- `Phi^-1` is the inverse standard normal CDF.

The DSR is then

```
DSR = Phi( (SR_observed - SR_0) * sqrt(T - 1)
           / sqrt(1 - skew * SR_observed + ((kurt - 1) / 4) * SR_observed^2) )
```

where `Phi` is the standard normal CDF, T is the number of OOS observations,
and `skew` and `kurt` are the sample skewness and kurtosis of the returns.

Interpretation:

- DSR > 0.95 — significant at the 5% level after correcting for selection
  bias and non-normality.
- DSR > 0.975 — the operational threshold this platform uses for deployment
  consideration.
- DSR < 0.5 — the observed Sharpe is below the expected maximum under the
  null. Reject.

The headline insight from the formula: the expected maximum Sharpe under the
null grows roughly with `sqrt(2 * ln(N))`. If you tested 1,000 strategies, the
"impressive" Sharpe you would expect from pure luck is around 3.7. Anything
below that bar is statistically indistinguishable from noise.

### Why DSR matters more than Sharpe

A junior portfolio that reports "my strategy has Sharpe 3.0" without naming
`N` and `T` is not reporting a result, it is reporting a draw from an
unspecified distribution. DSR forces both numbers to be on the table.

---

## Permutation tests

**Plain English.** Shuffle the daily returns of the underlying instrument,
keeping the bar order within each day intact. Re-run the strategy on the
shuffled data. Repeat 500 times. Compare the actual strategy's Sharpe to the
distribution of shuffled-data Sharpes. If the actual is in the tail, the
strategy is finding a real signal in the temporal structure of the data.

**Technical.** Day-block permutation preserves intraday autocorrelation while
destroying inter-day temporal structure. The resulting null distribution is
the strategy's Sharpe under the hypothesis "the inter-day price evolution is
permutable."

p-value = fraction of permutations whose Sharpe is greater than or equal to
the observed Sharpe. Targets:

- p < 0.05 — typical academic significance threshold.
- p < 0.01 — preferred for live deployment consideration.

This is a complement to DSR, not a substitute. DSR addresses selection bias
across the strategy space; permutation addresses signal vs noise within a
single strategy on a single asset.

---

## Parameter stability surfaces

**Plain English.** A profitable parameter setting that becomes unprofitable
when you nudge any parameter by one step is a knife-edge fit. A profitable
parameter setting whose neighbors are also profitable is a plateau. Plateaus
generalize; knife-edges do not.

**Technical.** For each parameter, perturb the optimum by ±1 and ±2 grid
steps (one parameter at a time, others fixed). Run a lightweight CSCV on the
perturbed configurations and record the mean OOS PnL.

`stability_score` = fraction of one-step neighbors with positive mean OOS
PnL. Target ≥ 0.80.

Visually, plot OOS performance as a function of each parameter. The eye
distinguishes plateaus from peaks instantly. The
`demos/parameter_stability_demo.py` script generates this plot for two
contrasting strategies.

---

## Cross-framework parity audits

**Plain English.** Implement the same strategy in two different backtesting
frameworks. The trade lists must agree to the trade. If they do not, one of
the implementations has a bug, and you have to find which one before
believing either result.

**Technical.** The platform routinely reconciles strategy execution across:

- The execution platform (a commercial C# / .NET trading system used as the
  authority for live order routing).
- A vectorized Python re-implementation that runs orders of magnitude
  faster than the execution platform's GUI backtester.
- Backtrader, a well-known Python backtesting framework.
- backtesting.py, a different Python backtesting framework.
- vectorbt, an array-oriented backtesting library.

A single strategy is implemented in all five environments with the same
entry, exit, position-sizing, and cost rules. Backtests run over the same
window. Trade lists are compared trade-by-trade. Discrepancies are
investigated until reconciled.

The most-cited example from this work, sanitized for the demo: Backtrader's
default `CommInfoBase` uses a contract multiplier of 1.0. For a futures
contract with a $50 point multiplier, this silently makes a one-point price
move look like $1 of P&L instead of $50. A strategy that should have a $5,000
Sharpe-per-year impact looks like $100. Without a parity audit, you would not
notice.

The demo at `demos/parity_audit_demo.py` reproduces this class of bug on
synthetic data and shows the trade-list reconciliation procedure that
catches it.

---

## Branch tournament and decision logs

**Plain English.** Every strategy hypothesis enters a written tournament with
a locked baseline (data window, cost model, execution semantics). Hypotheses
that lose against the baseline are recorded in a decision log along with the
specific reason they failed. They are not retried. The decision log is the
project's long-term memory.

**Technical.** The tournament tracker is a single living document that
records, for every strategy variant tested:

- The hypothesis statement (what was expected and why).
- The exact change to the code or parameters (so the test is reproducible).
- The result with quantified metric deltas relative to the baseline.
- The verdict: confirmed, falsified, or inconclusive.
- If falsified, the specific parameter region marked as a "forbidden
  region." Future hypotheses that fall inside a forbidden region must
  explicitly state why they expect a different result.

This solves a problem that pure backtest pipelines do not. A backtest
pipeline can rank strategies by metric. It cannot remember that an
indistinguishable variant was already rejected last month for a specific
reason. The decision log is what stops the project from drifting back into
already-falsified territory.

The agentic-research process document goes deeper into how AI tooling is
constrained by these logs to prevent it from re-proposing rejected work.

---

## How the gates compose

A candidate strategy is considered deployment-eligible only if it passes all
of the following:

1. Walk-forward: WFE > 0.50, distribution tight across folds.
2. CSCV: PBO < 0.10.
3. DSR: > 0.95.
4. Permutation: p-value < 0.05.
5. Parameter stability: > 0.80 of one-step neighbors profitable.
6. Parity audit: identical trade lists across at least two independent
   frameworks.
7. Decision log: no overlap with a documented forbidden region.

Failing any one is sufficient grounds for rejection. The gates are
deliberately not weighted. Selectively waiving a gate ("the PBO is high but
the Sharpe is so good") is the path back into selection-bias territory and is
explicitly disallowed.

This is a high bar. Most candidates do not clear it. That is the point. The
ratio of rejections to deployments is itself a quality signal for the
research process.

---

## What this method does not do

To be honest about the limits.

- **It does not estimate live drawdowns.** All of these tests measure
  selection bias and statistical robustness on historical data. Live drawdowns
  depend on regime changes that the historical data may not contain.
- **It does not replace position-sizing or risk-management.** The methodology
  is about whether a signal is real, not about how to size a bet on it.
- **It does not eliminate the need for paper trading.** Even a candidate that
  clears every gate gets paper-traded before live capital is committed.
- **It is not a guarantee.** A strategy that passes can still fail in
  production due to regime change, structural shifts, or microstructure
  effects that historical bars do not capture.

The honest claim is narrower: candidates that pass these gates are far less
likely to be artifacts of selection bias than candidates that have not been
tested this way. That is enough to decide what gets a paper-trade slot. It is
not enough to decide what makes money.

---

## References

- Bailey, D.H., Borwein, J.M., Lopez de Prado, M., Zhu, Q.J. (2014). *The
  Probability of Backtest Overfitting.* SSRN 2326253.
- Bailey, D.H., Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.*
  SSRN 2460551.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*
  (2nd ed.). Wiley.
- Harvey, C.R., Liu, Y., Zhu, H. (2016). *... and the Cross-Section of
  Expected Returns.* Review of Financial Studies.
