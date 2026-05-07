"""Day-block permutation tests for trading strategies.

Permutation tests build a null distribution by repeatedly shuffling the
inputs in a way that destroys the structure under test while preserving
nuisance structure. For intraday strategies, the conventional choice is
to shuffle the order of trading days while keeping each day's intraday
bar order intact. This destroys inter-day temporal structure (the thing
the strategy is meant to exploit) and preserves intraday autocorrelation
(which is nuisance structure the strategy must coexist with).

A small p-value, observed under this null, is direct evidence that the
strategy's returns depend on inter-day structure and are unlikely to
have arisen by chance under shuffled data.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def day_shuffled_close(
    close: np.ndarray,
    bars_per_day: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle the order of trading days while preserving intraday order.

    The close series is reshaped into (n_days, bars_per_day), the day
    rows are permuted, and the result is re-stitched so each new day
    starts at the previous day's close. Bars beyond the last full day
    are dropped.

    Args:
        close: 1-D close-price array.
        bars_per_day: number of bars per trading day.
        rng: a numpy Generator (use np.random.default_rng for seeding).

    Returns:
        A 1-D array of length n_days * bars_per_day with shuffled days
        and continuous price levels.
    """
    n_days = len(close) // bars_per_day
    usable = n_days * bars_per_day
    days = close[:usable].reshape(n_days, bars_per_day)
    indices = np.arange(n_days)
    rng.shuffle(indices)
    shuffled = days[indices].copy()
    for i in range(1, n_days):
        shuffled[i] = shuffled[i] + (shuffled[i - 1, -1] - shuffled[i, 0])
    return shuffled.flatten()


def day_block_permutation_test(
    pnl_fn: Callable[[np.ndarray], np.ndarray],
    statistic_fn: Callable[[np.ndarray], float],
    close: np.ndarray,
    bars_per_day: int,
    n_permutations: int,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Run a day-block permutation test on a strategy.

    For each of n_permutations iterations, shuffles the day order of the
    close series, evaluates pnl_fn on the shuffled series, and applies
    statistic_fn (typically a Sharpe ratio) to the resulting per-bar
    PnL. The p-value is the fraction of permutations whose statistic
    is at or above the observed (un-shuffled) statistic.

    Args:
        pnl_fn: callable mapping a close-price array to per-bar PnL.
        statistic_fn: callable mapping a per-bar PnL array to a scalar
            statistic (e.g. Sharpe).
        close: 1-D close-price array.
        bars_per_day: number of bars per trading day.
        n_permutations: number of shuffles. 500 is a common choice.
        seed: seed for the numpy Generator used for shuffling.

    Returns:
        (observed_statistic, p_value, null_statistics) where
        null_statistics is a length-n_permutations array of statistic
        values under the shuffled null.
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations}")

    observed = float(statistic_fn(pnl_fn(close)))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = day_shuffled_close(close, bars_per_day, rng)
        null[i] = float(statistic_fn(pnl_fn(shuffled)))

    p_value = float(np.mean(null >= observed))
    return observed, p_value, null
