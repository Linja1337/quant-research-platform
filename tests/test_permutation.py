"""Tests for quant_platform.validation.permutation."""
from __future__ import annotations

import numpy as np
import pytest

from quant_platform.validation.dsr import sharpe_ratio
from quant_platform.validation.permutation import (
    day_block_permutation_test,
    day_shuffled_close,
)

BARS_PER_DAY = 50


def _trending_close(n_bars: int, drift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=drift, scale=0.01, size=n_bars)
    return 100.0 * np.exp(np.cumsum(log_returns))


def _regime_switching_close(
    n_bars: int,
    regime_bars: int,
    drifts: tuple[float, ...],
    seed: int,
) -> np.ndarray:
    """A close series where the drift switches between values every
    regime_bars bars. Day-block shuffling destroys the regime structure,
    so a strategy that catches regimes loses its edge under the null."""
    rng = np.random.default_rng(seed)
    drift_arr = np.array(drifts)
    n_regimes = n_bars // regime_bars + 1
    regime_idx = rng.integers(0, len(drift_arr), size=n_regimes)
    drift_series = np.repeat(drift_arr[regime_idx], regime_bars)[:n_bars]
    log_returns = rng.normal(loc=drift_series, scale=0.005, size=n_bars)
    return 100.0 * np.exp(np.cumsum(log_returns))


def test_day_shuffle_preserves_length_for_full_days():
    close = _trending_close(n_bars=10 * BARS_PER_DAY, drift=0.0, seed=0)
    rng = np.random.default_rng(0)
    shuffled = day_shuffled_close(close, BARS_PER_DAY, rng)
    assert len(shuffled) == len(close)


def test_day_shuffle_truncates_partial_day():
    close = _trending_close(n_bars=10 * BARS_PER_DAY + 7, drift=0.0, seed=0)
    rng = np.random.default_rng(0)
    shuffled = day_shuffled_close(close, BARS_PER_DAY, rng)
    assert len(shuffled) == 10 * BARS_PER_DAY


def test_day_shuffle_returns_continuous_series():
    """Re-stitching should produce a series with no jump discontinuities at
    day boundaries."""
    close = _trending_close(n_bars=8 * BARS_PER_DAY, drift=0.0005, seed=1)
    rng = np.random.default_rng(0)
    shuffled = day_shuffled_close(close, BARS_PER_DAY, rng)
    boundary_jumps = np.abs(
        shuffled[BARS_PER_DAY::BARS_PER_DAY] - shuffled[BARS_PER_DAY - 1 :: BARS_PER_DAY][:-1]
    )
    assert np.all(boundary_jumps < 1e-9)


def test_day_shuffle_preserves_intraday_returns_set():
    """Each day's intraday returns are preserved as a multiset by the
    shuffle, since within-day order is intact and re-stitching only
    shifts levels."""
    close = _trending_close(n_bars=6 * BARS_PER_DAY, drift=0.0, seed=2)
    rng = np.random.default_rng(0)
    shuffled = day_shuffled_close(close, BARS_PER_DAY, rng)

    def intraday_returns(arr: np.ndarray) -> np.ndarray:
        days = arr.reshape(-1, BARS_PER_DAY)
        return np.diff(days, axis=1).flatten()

    orig = np.sort(intraday_returns(close))
    perm = np.sort(intraday_returns(shuffled))
    np.testing.assert_allclose(orig, perm)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    csum = np.cumsum(np.concatenate([[0.0], arr]))
    out = (csum[window:] - csum[:-window]) / window
    pad = np.full(window - 1, arr[0])
    return np.concatenate([pad, out])


def _sma_crossover_pnl(close: np.ndarray, fast: int = 5, slow: int = 25) -> np.ndarray:
    fast_sma = _rolling_mean(close, fast)
    slow_sma = _rolling_mean(close, slow)
    signal = (fast_sma > slow_sma).astype(int)
    position = np.roll(signal, 1)
    position[0] = 0
    bar_returns = np.zeros(len(close))
    bar_returns[1:] = (close[1:] - close[:-1]) / close[:-1]
    return position * bar_returns


def test_permutation_test_rejects_null_for_real_edge():
    """A regime-switching series gives an SMA crossover a real inter-day
    edge. Day-block shuffling destroys the regimes by scrambling whole
    days across regime boundaries, so the p-value should be small."""
    close = _regime_switching_close(
        n_bars=80 * BARS_PER_DAY,
        regime_bars=4 * BARS_PER_DAY,
        drifts=(0.002, -0.002, 0.0015, -0.0015),
        seed=10,
    )
    observed, p_value, null = day_block_permutation_test(
        pnl_fn=lambda c: _sma_crossover_pnl(c, fast=10, slow=80),
        statistic_fn=sharpe_ratio,
        close=close,
        bars_per_day=BARS_PER_DAY,
        n_permutations=300,
        seed=11,
    )
    assert p_value < 0.10
    assert observed > np.median(null)


def test_permutation_test_fails_to_reject_for_random_pnl():
    """A PnL stream drawn from an external rng has no dependence on the
    inter-day structure of the close series. The p-value should sit near
    0.5 rather than near 0.

    Uses an external rng captured by closure so each call (observed plus
    one per permutation) consumes fresh random numbers, putting all
    statistic draws on the same null distribution.
    """
    close = _trending_close(n_bars=20 * BARS_PER_DAY, drift=0.0, seed=20)
    placebo_rng = np.random.default_rng(99)

    def placebo(c: np.ndarray) -> np.ndarray:
        return placebo_rng.normal(loc=0.0, scale=0.01, size=len(c))

    _, p_value, _ = day_block_permutation_test(
        pnl_fn=placebo,
        statistic_fn=sharpe_ratio,
        close=close,
        bars_per_day=BARS_PER_DAY,
        n_permutations=300,
        seed=11,
    )
    assert p_value > 0.10


def test_permutation_test_rejects_zero_iterations():
    with pytest.raises(ValueError):
        day_block_permutation_test(
            pnl_fn=lambda c: np.zeros_like(c),
            statistic_fn=sharpe_ratio,
            close=np.ones(100),
            bars_per_day=10,
            n_permutations=0,
        )
