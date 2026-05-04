# Demos

Runnable demonstrations of the validation methodology. All synthetic data, no
real instruments, no real strategy parameters. Each demo is self-contained, runs
in well under a minute on a laptop, and writes a chart to `demos/output/`.

If you do not want to run the code, the committed PNGs in `demos/output/` are
the same charts the scripts produce.

## Setup

```
pip install -r requirements.txt
```

That is it. No data download, no environment variables, no API keys.

## What is here

| Demo | What it shows | Tier |
|------|---------------|------|
| `cscv_pbo_demo.py` | Probability of Backtest Overfitting on disciplined vs data-mined parameter searches | 1 |
| `walk_forward_demo.py` | Anchored walk-forward validation with per-fold OOS metrics | 2 |
| `parity_audit_demo.py` | Cross-framework execution parity catching a one-line bug | 2 |
| `permutation_test_demo.py` | Day-shuffled null distribution vs observed Sharpe | 2 |
| `parameter_stability_demo.py` | Plateau vs knife-edge parameter surfaces | 2 |

## CSCV / PBO demo

The headline. Reproduces Bailey, Borwein, Lopez de Prado, and Zhu's
Combinatorially Symmetric Cross-Validation on two strategy populations of equal
size. The first is a disciplined search (parameters chosen inside a hypothesis-
justified window); the second is data-mined (random per-bar PnL, no edge by
construction). Same OHLC, same engine, same number of bars.

```
python demos/cscv_pbo_demo.py
```

Reading the output:

- The logit histogram is the distribution across all C(8, 4) = 70 symmetric
  in-sample / out-of-sample partitions. PBO is the fraction with a non-positive
  logit, that is, splits where the in-sample winner ranked at or below the OOS
  median.
- The scatter on the right shows where each split's in-sample winner landed in
  the OOS performance distribution. With genuine signal, most dots sit above
  the median line. With pure noise, dots scatter symmetrically around it.

The disciplined population produces PBO around 0.15. The pure-noise population
produces PBO around 0.50, which is what you should see if there is no edge.
That contrast is the entire point of the technique.

## A note on the synthetic data

The OHLC generator is a regime-switching geometric process with deliberately
strong drift inside each regime. That is what lets the disciplined SMA
strategies actually have an edge to detect. Real markets do not look like this.
The point of the demo is the validation technique, not the strategy.
