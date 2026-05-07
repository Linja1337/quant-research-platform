"""Cross-framework trade-list reconciliation.

When two backtest implementations of the same strategy run on the same
data, they should produce identical trade lists. When they don't, one
of them has a bug. The cleanest way to catch the bug is to compare the
trade lists directly: matching entry / exit indices, then matching
dollar PnL per trade.

This module supplies the reconciliation primitive and a small Trade
dataclass and equity_curve helper. Strategy logic that produces trade
lists lives elsewhere (in caller code or in the strategies subpackage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trade:
    """A single closed trade.

    Attributes:
        entry_idx: bar index of the trade's entry fill.
        exit_idx: bar index of the trade's exit fill.
        entry_price: price paid on entry.
        exit_price: price received on exit.
    """

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float

    def points(self) -> float:
        """Price-points difference exit minus entry."""
        return self.exit_price - self.entry_price


@dataclass(frozen=True)
class ReconciliationResult:
    """Output of comparing two trade lists.

    Attributes:
        n_compared: number of trades compared (the shorter length).
        n_matched: trades whose entry and exit indices match exactly.
        n_mismatched: trades whose entry or exit indices differ.
        sample: list of (trade_idx, dollars_a, dollars_b) for the first
            few trades, useful for printing a side-by-side preview.
    """

    n_compared: int
    n_matched: int
    n_mismatched: int
    sample: list[tuple[int, float, float]]


def reconcile_trade_lists(
    trades_a: list[Trade],
    trades_b: list[Trade],
    multiplier_a: float,
    multiplier_b: float,
    commission_per_side: float = 0.0,
    sample_size: int = 5,
) -> ReconciliationResult:
    """Compare two trade lists from competing backtest implementations.

    Trades are compared positionally. For each pair, entry and exit
    indices are checked for exact match. The dollar PnL of each side is
    captured into the sample list for the first sample_size trades so
    the caller can print a preview.

    Args:
        trades_a: trade list from implementation A.
        trades_b: trade list from implementation B.
        multiplier_a: dollars per price point for implementation A
            (e.g. 50.0 for an ES-style futures contract; 1.0 for a
            framework that left the multiplier at its default).
        multiplier_b: dollars per price point for implementation B.
        commission_per_side: round-trip commission has 2 sides; the
            per-side cost is applied to each leg of every trade.
        sample_size: number of leading trades to include in the sample.

    Returns:
        A ReconciliationResult.
    """
    n = min(len(trades_a), len(trades_b))
    matched = 0
    mismatched = 0
    sample: list[tuple[int, float, float]] = []
    for i in range(n):
        ta = trades_a[i]
        tb = trades_b[i]
        if ta.entry_idx != tb.entry_idx or ta.exit_idx != tb.exit_idx:
            mismatched += 1
        else:
            matched += 1
        if i < sample_size:
            sample.append(
                (
                    i,
                    ta.points() * multiplier_a - 2 * commission_per_side,
                    tb.points() * multiplier_b - 2 * commission_per_side,
                )
            )
    return ReconciliationResult(
        n_compared=n,
        n_matched=matched,
        n_mismatched=mismatched,
        sample=sample,
    )


def equity_curve(
    trades: list[Trade],
    multiplier: float,
    n_bars: int,
    commission_per_side: float = 0.0,
) -> np.ndarray:
    """Cumulative dollar PnL by bar.

    Each trade's PnL is recorded at its exit bar. Commissions are
    applied per side per trade.

    Args:
        trades: list of closed trades.
        multiplier: dollars per price point.
        n_bars: length of the underlying bar series.
        commission_per_side: per-side commission cost.

    Returns:
        Length-n_bars array of cumulative PnL.
    """
    pnl = np.zeros(n_bars)
    for t in trades:
        dollars = t.points() * multiplier - 2 * commission_per_side
        pnl[t.exit_idx] += dollars
    return np.cumsum(pnl)
