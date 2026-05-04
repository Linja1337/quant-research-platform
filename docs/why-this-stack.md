# Why this stack

The choices behind the platform's architecture, and the alternatives that
were considered and rejected. Read this if you want to know how the
engineering judgment shows up.

## Live execution: C# / .NET

**Constrained, not preferred.** The commercial trading platform that handles
broker connectivity and live order routing exposes a strategy SDK in C# /
.NET. Strategies that run live run in C#. There is no toggle for that.

What this gives:

- A mature execution layer with order book state, position bookkeeping,
  margin handling, and EOD reporting that does not need to be rebuilt.
- Direct broker connectivity through a battle-tested adapter.
- A built-in chart and backtest GUI for ad-hoc human inspection.

What this costs:

- Single-threaded per backtest. Large parameter sweeps in the GUI are
  prohibitively slow.
- C# is a slower iteration language for research than Python. Type
  ceremony, build steps, and the GUI's compile-on-save behavior all add
  friction.
- The strategy SDK is opinionated about how indicators, signals, and
  orders compose. Working against the grain produces fragile code.

The decision: accept C# for live execution, do not pretend it is also a
research environment, build a separate research stack in Python.

## Research and validation: Python

**The default choice in 2026.** NumPy, pandas, scipy, Optuna, DEAP, Numba,
and matplotlib together cover almost every numerical and statistical need
the research stack has, with no commercial license cost. The community is
large, the documentation is good, and most quant-finance papers ship
reference code in Python.

Alternatives considered and not chosen:

- **R.** Stronger statistics ecosystem, weaker general-purpose engineering
  ecosystem. Walking back to R from Python costs more than the marginal
  statistics gain.
- **Julia.** Faster than Python for numerical code without Numba. The
  ecosystem is smaller; the deployment story is weaker; the AI tooling
  works less well on Julia code than on Python.
- **C++.** Fastest, most memory-efficient. The development velocity cost is
  too high for research code that may change weekly. Save C++ for
  specialized hot paths if they show up.

The decision: Python everywhere except the live execution layer.

## Engine: custom Numba JIT, not vectorbt or backtesting.py

**Speed and control.** A custom Numba-compiled engine runs ~1,000 backtests
per second on a laptop and lets the platform model exactly the execution
semantics that the live C# strategy uses (bar-close evaluation, next-bar-
open fill, EOD logic, asymmetric trailing stops). Off-the-shelf engines
either run slower (backtesting.py) or have execution models that diverge
from C# in subtle ways (vectorbt's vectorized fills, Backtrader's broker
state machine).

The vectorbt and Backtrader implementations exist anyway, but they exist
*for parity reconciliation*, not as the primary engine. Three independent
implementations are how the platform catches single-implementation bugs;
choosing one as authoritative would defeat the point.

Alternatives considered and not chosen:

- **vectorbt as primary engine.** Faster than the custom Numba engine for
  pure-array strategies. Slower for path-dependent strategies (stops,
  trails) because the vectorized abstraction does not fit. Modeling
  bar-close evaluation with next-bar fills inside the vectorbt API is
  awkward.
- **backtesting.py as primary engine.** Smaller, simpler, and friendly to
  read. Single-threaded; cannot keep up with 200-trial Optuna sweeps.
- **Zipline.** Heavyweight, daily-bar-oriented, last actively maintained
  in 2020. Wrong fit.
- **QuantConnect Lean.** Excellent platform, but it is a hosted service
  with its own opinions about strategy structure. Couples the research
  stack to a vendor in a way the platform actively avoids.

The decision: write the engine. Pay the maintenance cost. Use the off-the-
shelf frameworks as parity checks against it.

## Search: Optuna TPE primary, DEAP genetic fallback

**Sample efficiency, then expressive power.** Optuna's TPE sampler is the
de-facto modern Bayesian optimizer for hyperparameter search. It converges
faster than random or grid for small-to-medium parameter spaces and handles
mixed continuous-discrete spaces well. DEAP exists for cases where the
parameter space has discontinuous valid regions (e.g., constraints like
fast < slow that prune entire grid sections) that the TPE struggles with.

Alternatives considered and not chosen:

- **Grid search.** Hopeless for spaces with more than three or four
  parameters. The combinatorial blowup is its own data-mining accelerator.
- **Random search.** Better than grid but still requires more evaluations
  to find a peak than TPE.
- **scikit-optimize / hyperopt.** Older, less actively maintained, smaller
  user base. No advantage over Optuna.
- **Reinforcement learning hyperparameter search.** Too complex for the
  problem size; introduces another layer of hyperparameters to choose;
  Bailey-Lopez de Prado-style multiple-testing correction does not
  cleanly extend to RL search.

The decision: Optuna TPE as default, with DEAP available for the awkward
cases.

## Validation: CSCV + DSR + walk-forward + permutation + stability + parity

**No single technique catches everything.** Each gate is designed to fail
strategies that the others miss:

- Walk-forward catches parameter sets that work in-sample but fail OOS.
- CSCV catches selection bias in the *process* of picking IS-best.
- DSR catches the multiple-testing inflation in any single Sharpe number.
- Permutation tests catch strategies that capture market drift rather than
  signal.
- Parameter stability catches knife-edge fits.
- Parity audits catch implementation bugs.

A candidate must clear all of them. The methodology doc has the math; this
section is about why a single test would not be enough.

Alternatives considered and not chosen:

- **Bootstrap-only validation.** Bootstrap resamples are good for
  estimating the variance of a metric, but not for distinguishing
  signal from selection bias. Bootstrap a Sharpe of 5 from a noise
  strategy and you get a tight confidence interval around 5.
- **Train-test split, single fold.** The standard ML default. Wrong for
  time series (lookahead leakage), wrong for selection bias (a single
  test fold is one draw from a much larger distribution).
- **Cross-validation, k-fold.** Better than single-fold but still does
  not catch selection bias from picking the IS-best across many
  candidates. CSCV is the time-series-aware extension that does.

The decision: layered validation. Slow, expensive, non-negotiable.

## Decision memory: Markdown decision log

**Tool-agnostic. Diff-able. Outlives any tooling change.** A Markdown file
under git is the simplest possible artifact that gives the project
persistent memory. The decision-log philosophy doc explains the discipline.

Alternatives considered and not chosen:

- **Notion / Confluence / similar.** Couples the project to a vendor.
  Search is good; diff is bad. Hard to load into an AI assistant's
  context window without an export.
- **A SQLite database of iteration records.** More structured. Harder to
  edit by hand. Less useful as the project's onboarding doc.
- **Per-iteration JSON files in a folder.** More machine-friendly. Less
  human-readable. Forces tooling into the read path.
- **A wiki on the trading platform itself.** Couples the research log to
  the live platform. Wrong direction.

The decision: one Markdown file, append-only, in the repo, in git.

## Demo data: synthetic only

**Removes IP exposure and instrument calibration from public artifacts.**
The methodology this repository demonstrates is general; the strategies
that exercise it are not. Synthetic data is reproducible by anyone, runs
without an API key, and contains no identifying information about the
private platform's parameter choices, instrument coverage, or live
performance.

Alternatives considered and not chosen:

- **Public-domain bar data (Yahoo Finance, Stooq).** Risk: a competitor
  who knows the strategy class can re-run the public demo on the real
  underlying and back into the platform's parameters.
- **Anonymized real bars.** Anonymization of price series is harder than
  it sounds. Volatility, autocorrelation, and regime structure all
  fingerprint the source.
- **One real underlying with sanitized parameters.** Same risk as
  public-domain bars.

The decision: synthetic only. Reproducible, safe, sufficient for the
methodology demonstration.

## What was rejected outright

To round out the picture.

- **Deep RL for strategy generation.** Tested in a parallel project,
  rejected for the platform: too many hyperparameters, no clean way to
  apply Bailey-Lopez de Prado multiple-testing correction, and the
  resulting policies were uninterpretable enough that the decision log
  could not record meaningful "why" entries.
- **LLM-generated strategy ideas.** Used as a brainstorming aid only.
  Hard rule: no LLM-suggested code is committed without going through
  the same validation gates as a human-written strategy. Most LLM-
  generated strategies fail at PBO immediately.
- **Live optimization.** Re-tuning parameters on live performance is the
  fastest known path to overfitting your own track record. The platform
  treats live results as evaluation, not training.
- **Single-asset specialization.** The platform is built to scale across
  assets and timeframes. Hard-coding ES futures or BTC perp into the
  research engine would have made the engine cheaper to build but
  much harder to extend.

## What is still open

Honest about the unfinished bets.

- **Pure-Python execution stack.** A long-term direction. The C# layer
  is a constraint, not a preference; the eventual goal is to bypass it
  for direct broker connectivity from Python. Estimated cost is high;
  estimated benefit is research-execution unification.
- **Hardware acceleration.** The Numba engine saturates a single CPU
  core. Larger sweeps could move to a multi-core process pool (already
  used for DEAP) or to GPU array math. Neither is implemented yet
  because the laptop budget has not run out.
- **Streaming bar ingestion.** The platform consumes bars from files
  for research and from broker callbacks for live. A unified
  streaming layer would let research run on the same code path as live
  ingestion. Not yet built.

The unfinished bets are listed because a portfolio that claims everything
works is less credible than one that names what does not.
