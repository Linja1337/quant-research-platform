# Parity engineering

A working trading platform compares its own output against itself, on
purpose, until everything agrees. This document is about why that is
different from "writing tests" and how the discipline catches the bugs that
unit tests do not.

## The starting position

Any backtest is the composition of three things:

- A signal layer (when to enter, when to exit)
- An execution layer (how the signal is turned into orders, how orders are
  filled, what costs are deducted)
- A bookkeeping layer (cumulative PnL, position state, drawdown)

A test on the signal layer tells you the signal is correctly computed. A
test on the execution layer tells you orders are correctly modeled. A test
on the bookkeeping layer tells you sums are right.

None of those tests tells you that the *composition* is correct, because
the composition is what every backtesting framework implements differently.
A strategy that compiles in five frameworks can produce five different
trade lists from the same OHLC data, and each individual framework's tests
will all pass.

That is the gap parity audits exist to close.

## The procedure

The same strategy is implemented in several independent backtesting
environments, each with the same:

- OHLC bars
- Parameter set
- Entry and exit rules
- Order type (market on next bar open, in this platform's case)
- Commission and slippage model
- Contract multiplier and tick size

The output trade lists are diffed trade-by-trade. The diff includes:

- Entry bar index, entry price, entry quantity
- Exit bar index, exit price, exit quantity
- Per-trade PnL in dollars
- Cumulative PnL after each trade

If two implementations agree on every cell, they agree. If they disagree
anywhere, one of them is wrong. The job is to find which one.

## The bug class this catches

The most instructive example, sanitized for the demo: implicit unit
mismatches.

Backtrader's `CommInfoBase` defaults to `mult=1`. For a futures contract
with a $50 point multiplier, leaving the default in place silently makes a
1.0-point price move worth $1 of P&L instead of $50. There is no error, no
warning, no obvious symptom. The equity curve looks plausible. It is just
50 times smaller than it should be.

A strategy whose true expected Sharpe is 1.5 ends up with an apparent
Sharpe near zero. A strategy whose true expected drawdown is $5,000 looks
like a $100 drawdown. Decisions get made on those numbers. They are wrong
in different directions for different strategies.

The same bug class shows up across frameworks under different names: tick
sizes vs point values, percent commissions vs absolute commissions, share
quantities vs notional dollar amounts, futures multipliers vs cash equity
multipliers. They all silently scale the answer.

The `demos/parity_audit_demo.py` script reproduces this bug class on
synthetic data. Two implementations of the same SMA crossover strategy run
on the same OHLC, with one of them left at the framework default `mult=1`
and the other set to `mult=50`. The equity curves diverge. The diff is
trivial to read once you are looking at it.

## What the platform compares against what

The private platform reconciles, for every strategy considered for live
deployment:

- The C# / .NET strategy running on the commercial trading platform. This
  is the live-execution authority.
- A pure-Python re-implementation of the strategy in the research stack.
  This implementation also runs the optimizers and the validators, so any
  divergence between Python and C# means either the optimizer was not
  actually optimizing the live strategy or the live strategy is not
  actually trading the optimized parameters.
- Backtrader, an independent third-party Python framework.
- backtesting.py, a different independent third-party Python framework.
- vectorbt, an array-oriented backtesting library with a different
  execution model.

A strategy is not eligible for paper trading until at least two of these
agree on its trade list. A strategy is not eligible for live capital until
at least three do.

## Why frameworks disagree, even when they shouldn't

There is no shortage of categories.

- **Bar-close vs intra-bar evaluation.** A signal that fires when the
  close of bar N crosses a level can either fill at the close of bar N
  (intra-bar evaluation) or at the open of bar N+1 (next-bar fill). Both
  are valid. They produce different trade lists.
- **Stop-loss order types.** A stop loss can be a market-next-bar order
  evaluated at bar close, or a server-side stop that triggers intra-bar.
  These produce different exit prices and sometimes different exit bars.
- **EOD handling.** Some frameworks close all positions at the session
  boundary; others let positions run overnight. Strategies that depend on
  EOD behavior need this nailed down explicitly.
- **Margin and contract specifications.** Futures multipliers, tick
  values, lot sizes, currency conversions. Each framework has different
  defaults.
- **Indicator implementations.** Wilder ATR vs simple ATR. EMA seeded
  with SMA vs first-value seeded. Different choices produce slightly
  different indicator values, which propagate to different signal bars.

A parity audit forces the platform owner to make every one of these choices
explicitly. Once the choices are nailed down, the frameworks agree. Until
then, they cannot.

## The engineering payoff

A platform that has been parity-audited has an asset that most retail
research stacks never accumulate: the certainty that "the strategy works
in Python" and "the strategy works in C#" mean the same thing.

That certainty is what lets you optimize a strategy in Python in the
morning and deploy it in C# in the afternoon without re-tuning. Without it,
optimization in Python and execution in C# are two separate research
projects, and the second one usually loses.

## What this is not

Parity audits do not address whether the strategy makes money. They only
address whether different implementations agree on what happened. A bad
strategy that has been parity-audited still loses money; the parity audit
just guarantees that all five implementations lose the same amount.

That is the right kind of guarantee to have.
