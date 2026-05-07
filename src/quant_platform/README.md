# quant_platform

The library layer of the quant research platform. It collects the
validation methodology that the demos in `demos/` use, packaged so that
external code can import and run the same primitives.

## What is here

- `validation.cscv` — Combinatorially Symmetric Cross-Validation and the
  Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado,
  Zhu 2014, Algorithm 2.3).
- `validation.dsr` — Deflated Sharpe Ratio and the False Strategy Theorem
  (Bailey and Lopez de Prado 2014).
- `validation.walk_forward` — anchored fold-index generator and the
  walk-forward efficiency definition (Pardo 2008).
- `validation.permutation` — day-block shuffle and a generic permutation
  test driver that accepts any scalar statistic.
- `validation.stability` — parameter-surface stability scoring around
  the optimum cell.
- `parity.reconciler` — trade-list diff for cross-framework parity audits.
- `parity.synthetic` — regime-switching OHLC generator and a synthetic
  strategy population used to drive the demos deterministically.
- `strategies.sma_crossover` — reference SMA crossover signal generator,
  used as the worked example in every demo.

## Quick start

```python
import numpy as np
from quant_platform import (
    compute_pbo,
    regime_switching_ohlc,
    sma_crossover_signals,
)

bars = regime_switching_ohlc(n_bars=6_000, seed=42)
signals = sma_crossover_signals(bars["close"], fast_window=8, slow_window=30)

# build a (T, N) performance matrix from your candidate strategies, then
result = compute_pbo(M)
print(f"PBO = {result.pbo:.3f}")
```

## How to read the code

- Methodology references and operational thresholds: `docs/methodology.md`
  in the repository root.
- Worked examples and chart output: every script in `demos/`.
- Usage examples for each public function: the corresponding file under
  `tests/` covers the documented invariants.

## Install

```
pip install -e .
```
