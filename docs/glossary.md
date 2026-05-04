# Glossary

Plain-English definitions of every technical term used in this repository.
Sorted alphabetically. If you read a term elsewhere in the docs and it is
not defined inline, look it up here.

---

**ATR (Average True Range).** A volatility indicator. The average size of a
typical bar's range over a recent window. Used as a unit for stop losses
and take profits ("set the stop 2.5 ATRs below the entry") so that they
adapt to current volatility instead of being fixed in price points.

**Backtest.** A simulation of a trading strategy on historical data.
Produces an equity curve, a trade list, and a battery of metrics. Subject
to a long list of biases that the validation methodology in this repo is
designed to expose.

**Bar.** A single time slice of price data, summarized by four numbers:
open, high, low, close. Sometimes augmented with volume. A 5-minute bar
covers all the trading activity in a 5-minute window.

**Bayesian optimization.** A class of search algorithms that builds a
probabilistic model of the objective function as it accumulates trials,
then proposes the next trial point that maximizes expected improvement.
Sample-efficient compared to grid or random search. Optuna's TPE sampler
is the implementation this platform uses.

**CSCV (Combinatorially Symmetric Cross-Validation).** A validation
technique due to Bailey, Borwein, Lopez de Prado, and Zhu. Enumerates all
symmetric in-sample / out-of-sample splits of a performance matrix and
asks how often the in-sample winner ranks below median out-of-sample. The
fraction is the Probability of Backtest Overfitting.

**DEAP.** A Python evolutionary-algorithm library. Used here for the
genetic-algorithm optimizer when the parameter space has discontinuous
valid regions that the Bayesian optimizer struggles with.

**Drawdown.** The peak-to-trough decline in cumulative PnL. The headline
risk metric for any trading strategy. "Maximum drawdown" is the worst
peak-to-trough decline observed in the backtest.

**DSR (Deflated Sharpe Ratio).** A correction to the Sharpe ratio that
adjusts for selection bias under multiple testing and for non-normality
of returns. Bailey and Lopez de Prado, 2014. Operational threshold in
this platform: DSR > 0.95.

**EMA (Exponential Moving Average).** A weighted moving average that gives
more weight to recent observations. Used as a smoother for noisy price
series. The "fast EMA" and "slow EMA" in a crossover strategy refer to
two EMAs of different lengths.

**EOD (End of Day).** Behavior at the boundary of a trading session.
"EOD exit" means closing all open positions before the session ends.

**Forbidden region.** A region of parameter space (or a strategy
architecture, or an execution-model assumption) that has been falsified
in a prior iteration and is documented in the decision log. New
hypotheses that overlap a forbidden region must explicitly justify why
they expect a different result.

**Genetic algorithm.** A search algorithm that maintains a population of
candidate solutions, scores each, and produces the next generation by
selecting the fittest, recombining their parameters, and applying random
mutation. Implemented here via DEAP.

**Heikin-Ashi (HA).** A modified candlestick representation in which each
"bar" is computed from a smoothed combination of the previous bar's open
and close, plus the current bar's OHLC. Reduces noise; makes trends
visually more obvious. Used in the platform as a directional gate, not as
a trade trigger.

**In-sample (IS).** The portion of historical data used to choose strategy
parameters or evaluate hypotheses. Always paired with an out-of-sample
window for validation.

**MACD (Moving Average Convergence Divergence).** A momentum indicator
formed by the difference of two EMAs of the price, with a signal line that
is itself an EMA of the MACD line. Crossovers of the MACD line with the
signal line are common momentum triggers.

**Monte Carlo simulation.** A class of methods that uses repeated random
sampling to estimate the distribution of an outcome. Used here to
construct null distributions for hypothesis tests.

**Numba.** A Python library that compiles Python functions to optimized
machine code at runtime. The platform's backtest engine uses Numba's
`@njit` decorator on the bar-by-bar inner loop, achieving roughly
1,000-fold speedup over pure-Python equivalents.

**OHLC.** Open, high, low, close. The four prices that summarize a single
bar.

**OHLCV.** OHLC plus volume.

**Optuna.** A Python hyperparameter optimization library. The platform
uses its TPE (Tree-structured Parzen Estimator) sampler for Bayesian
search.

**Out-of-sample (OOS).** Data that was not used to choose strategy
parameters. The strategy is evaluated on OOS data; if it generalizes from
IS to OOS, the result is more credible.

**Parameter stability.** A property of a strategy's parameter surface.
Stable parameters sit on a plateau where neighboring parameter values are
also profitable. Unstable parameters sit on a knife-edge where any
neighboring value collapses the result. Plateau parameters generalize;
knife-edge parameters do not.

**Parity audit.** The procedure of running the same strategy in multiple
independent backtesting frameworks and reconciling the trade lists. If
two implementations agree to the trade, they agree. If they disagree, one
of them is wrong, and the audit job is to find which.

**PBO (Probability of Backtest Overfitting).** The output of CSCV. The
fraction of symmetric splits in which the in-sample winner ranks at or
below the OOS median. Operational threshold: PBO < 0.10.

**Permutation test.** A statistical test that constructs a null
distribution by repeatedly shuffling the input data, recomputing the
test statistic, and comparing the observed value to the resulting
distribution. The platform uses day-block permutation that preserves
intraday autocorrelation while destroying inter-day temporal structure.

**PnL (Profit and Loss).** The dollar value of trading gains and losses.
"Per-bar PnL" is the dollar gain or loss attributable to each bar.
"Per-trade PnL" is the dollar gain or loss from each completed trade.

**Profit factor (PF).** Gross winning trades divided by gross losing
trades. PF > 1 means winners exceed losers in dollar terms. Literature
range for sustainable trend-following intraday strategies is 1.2 to 1.8.

**Sharpe ratio.** Mean return divided by standard deviation of returns,
annualized. The single most-cited risk-adjusted return metric. Subject to
all the biases the validation methodology is designed to catch, which is
why the Deflated Sharpe Ratio exists.

**Sortino ratio.** Like the Sharpe ratio but using only downside
deviation in the denominator. More relevant for strategies with
asymmetric return distributions.

**SMA (Simple Moving Average).** The arithmetic mean of the most recent N
bars. Used as a smoother. "SMA crossover" is a strategy that buys when a
fast SMA crosses above a slow SMA and sells on the reverse cross.

**Stop loss (SL).** An order to close a losing position when the price
moves against it by a defined amount. Limits per-trade downside.

**Take profit (TP).** An order to close a winning position when the price
moves in favor of it by a defined amount. Limits per-trade upside in
exchange for higher win rate.

**Trail (trailing stop).** A stop loss that moves in the favorable
direction as the position becomes more profitable, locking in gains
without capping the upside.

**vectorbt.** An array-oriented Python backtesting library. Used here for
parity audits, not as the primary engine.

**Walk-forward analysis (WFA).** A validation technique that trains
parameters on a rolling in-sample window, evaluates on the next
out-of-sample window, then slides both windows forward. Repeats over the
full historical period. Concatenated OOS performance is the walk-forward
equity curve.

**WFE (Walk-Forward Efficiency).** The ratio of out-of-sample metric to
in-sample metric, per fold. Pardo (2008). Operational threshold: WFE > 0.50.
