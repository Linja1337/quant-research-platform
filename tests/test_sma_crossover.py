"""Tests for quant_platform.strategies.sma_crossover."""
from __future__ import annotations

import numpy as np
import pytest

from quant_platform.strategies.sma_crossover import compute_signals


def test_signal_values_in_minus_one_zero_one():
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(size=500))
    signals = compute_signals(close, fast_window=5, slow_window=20)
    assert set(np.unique(signals).tolist()).issubset({-1, 0, 1})


def test_signal_length_matches_input():
    close = np.linspace(100.0, 110.0, 200)
    signals = compute_signals(close, fast_window=3, slow_window=10)
    assert signals.shape == close.shape


def test_signal_first_bar_is_zero_no_lookahead():
    """The signal at bar 0 must not depend on any future data; the
    convention is that bar 0 is always flat."""
    close = np.linspace(100.0, 110.0, 100)
    signals = compute_signals(close, fast_window=3, slow_window=10)
    assert signals[0] == 0


def test_no_lookahead_signal_at_t_uses_only_data_through_t_minus_one():
    """Changing close[t] should not affect signals[t]; it should only
    affect signals[t+1] onward."""
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(size=200))
    base = compute_signals(close, fast_window=3, slow_window=10)

    perturbed = close.copy()
    perturbed[100] += 1000.0
    perturbed_signals = compute_signals(perturbed, fast_window=3, slow_window=10)

    np.testing.assert_array_equal(base[:101], perturbed_signals[:101])


def test_monotone_uptrend_produces_long_signals_after_warmup():
    """A pure uptrend means fast SMA sits above slow SMA after warmup,
    so signals should be +1 once the slow SMA is fully populated."""
    close = np.arange(1, 201, dtype=float)
    signals = compute_signals(close, fast_window=3, slow_window=10)
    assert np.all(signals[15:] == 1)


def test_monotone_downtrend_produces_short_signals_after_warmup():
    close = np.arange(200, 0, -1, dtype=float)
    signals = compute_signals(close, fast_window=3, slow_window=10)
    assert np.all(signals[15:] == -1)


def test_hand_computed_crossover_example():
    """Build a price series that crosses up at a known bar and verify
    the signal flips at the expected position (one bar later, given the
    no-lookahead shift)."""
    close = np.concatenate(
        [
            np.linspace(100.0, 90.0, 50),
            np.linspace(90.0, 110.0, 50),
        ]
    )
    signals = compute_signals(close, fast_window=3, slow_window=10)
    early = signals[20:40]
    late = signals[70:]
    assert (early <= 0).all()
    assert (late == 1).all()


def test_invalid_windows_raise():
    close = np.linspace(100, 110, 100)
    with pytest.raises(ValueError):
        compute_signals(close, fast_window=10, slow_window=5)
    with pytest.raises(ValueError):
        compute_signals(close, fast_window=5, slow_window=5)
    with pytest.raises(ValueError):
        compute_signals(close, fast_window=0, slow_window=5)
    with pytest.raises(ValueError):
        compute_signals(close, fast_window=-1, slow_window=5)
