"""Quant research platform: validation methodology in code form.

Public exports cover the validation primitives (CSCV / PBO, DSR, walk-
forward, day-block permutation tests, parameter stability), the parity
reconciliation primitives, and a reference SMA crossover strategy used
by the demos. Every function has a docstring with the algorithm
reference; every module has a corresponding test file under tests/.
"""
from quant_platform.parity.reconciler import reconcile_trade_lists
from quant_platform.parity.synthetic import regime_switching_ohlc
from quant_platform.strategies.sma_crossover import (
    compute_signals as sma_crossover_signals,
)
from quant_platform.validation.cscv import compute_pbo
from quant_platform.validation.dsr import deflated_sharpe_ratio
from quant_platform.validation.permutation import day_block_permutation_test
from quant_platform.validation.stability import parameter_stability_score
from quant_platform.validation.walk_forward import walk_forward_folds

__version__ = "0.1.0"

__all__ = [
    "compute_pbo",
    "deflated_sharpe_ratio",
    "walk_forward_folds",
    "day_block_permutation_test",
    "parameter_stability_score",
    "reconcile_trade_lists",
    "regime_switching_ohlc",
    "sma_crossover_signals",
]
