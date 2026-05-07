"""Tests for quant_platform.parity."""
from __future__ import annotations

import numpy as np

from quant_platform.parity.reconciler import (
    Trade,
    equity_curve,
    reconcile_trade_lists,
)
from quant_platform.parity.synthetic import (
    StrategyConfig,
    regime_switching_ohlc,
    synthetic_population,
)


def test_reconciler_detects_multiplier_bug():
    """The canonical case: two implementations agree on the trade list,
    but one uses multiplier=1.0 and the other uses multiplier=50.0. The
    sample should show a 1/50 ratio in dollars per trade."""
    trades = [
        Trade(entry_idx=10, exit_idx=20, entry_price=100.0, exit_price=110.0),
        Trade(entry_idx=30, exit_idx=45, entry_price=110.0, exit_price=108.0),
        Trade(entry_idx=60, exit_idx=80, entry_price=108.0, exit_price=115.0),
    ]
    result = reconcile_trade_lists(
        trades_a=trades, trades_b=trades, multiplier_a=1.0, multiplier_b=50.0
    )
    assert result.n_compared == 3
    assert result.n_matched == 3
    assert result.n_mismatched == 0
    for _, dollars_a, dollars_b in result.sample:
        assert dollars_a / dollars_b == 0.02


def test_reconciler_counts_index_mismatches():
    """When one implementation's exit index differs by a bar, the
    reconciler should count it as a mismatch."""
    trades_a = [
        Trade(entry_idx=10, exit_idx=20, entry_price=100.0, exit_price=110.0),
        Trade(entry_idx=30, exit_idx=45, entry_price=110.0, exit_price=108.0),
    ]
    trades_b = [
        Trade(entry_idx=10, exit_idx=20, entry_price=100.0, exit_price=110.0),
        Trade(entry_idx=30, exit_idx=46, entry_price=110.0, exit_price=108.0),
    ]
    result = reconcile_trade_lists(
        trades_a=trades_a, trades_b=trades_b, multiplier_a=1.0, multiplier_b=1.0
    )
    assert result.n_matched == 1
    assert result.n_mismatched == 1


def test_reconciler_handles_unequal_list_lengths():
    """Comparison runs over min(len_a, len_b) trades."""
    trades_a = [Trade(0, 5, 100.0, 105.0), Trade(10, 15, 105.0, 110.0)]
    trades_b = [Trade(0, 5, 100.0, 105.0)]
    result = reconcile_trade_lists(
        trades_a=trades_a, trades_b=trades_b, multiplier_a=1.0, multiplier_b=1.0
    )
    assert result.n_compared == 1
    assert result.n_matched == 1


def test_reconciler_sample_size_is_capped():
    trades = [Trade(i * 10, i * 10 + 5, 100.0, 105.0) for i in range(10)]
    result = reconcile_trade_lists(
        trades_a=trades, trades_b=trades,
        multiplier_a=1.0, multiplier_b=1.0,
        sample_size=3,
    )
    assert len(result.sample) == 3


def test_equity_curve_records_pnl_at_exit_bars():
    trades = [
        Trade(entry_idx=2, exit_idx=5, entry_price=100.0, exit_price=110.0),
        Trade(entry_idx=8, exit_idx=12, entry_price=110.0, exit_price=105.0),
    ]
    eq = equity_curve(trades, multiplier=1.0, n_bars=15)
    assert eq[4] == 0.0
    assert eq[5] == 10.0
    assert eq[11] == 10.0
    assert eq[12] == 5.0
    assert eq[14] == 5.0


def test_equity_curve_applies_commission_per_side():
    trades = [Trade(entry_idx=0, exit_idx=5, entry_price=100.0, exit_price=110.0)]
    eq = equity_curve(
        trades, multiplier=1.0, n_bars=10, commission_per_side=2.0
    )
    assert eq[-1] == 10.0 - 4.0


def test_regime_switching_ohlc_shapes_and_determinism():
    bars_a = regime_switching_ohlc(n_bars=500, seed=42)
    bars_b = regime_switching_ohlc(n_bars=500, seed=42)
    for key in ("open", "high", "low", "close"):
        assert bars_a[key].shape == (500,)
        np.testing.assert_array_equal(bars_a[key], bars_b[key])


def test_regime_switching_ohlc_high_low_invariants():
    bars = regime_switching_ohlc(n_bars=500, seed=42)
    assert np.all(bars["high"] >= bars["close"])
    assert np.all(bars["high"] >= bars["open"])
    assert np.all(bars["low"] <= bars["close"])
    assert np.all(bars["low"] <= bars["open"])


def test_synthetic_population_returns_correct_counts():
    configs, truth = synthetic_population(n_signal=10, n_noise=20, seed=0)
    assert len(configs) == 30
    assert truth.sum() == 10
    assert (~truth).sum() == 20


def test_strategy_config_random_kind_is_truth_blind():
    """A random-kind strategy ignores the close series structure: changing
    the close (with the same length) preserves the per-bar PnL distribution
    parameters but shifts the actual values."""
    cfg = StrategyConfig(fast=5, slow=20, kind="random")
    pnl_a, _ = cfg.trades(np.linspace(100, 110, 1000))
    assert pnl_a.shape == (1000,)
    assert abs(pnl_a.mean()) < 1.0
