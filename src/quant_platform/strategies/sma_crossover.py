"""Simple moving average (SMA) crossover signal generator.

The SMA crossover is the canonical trend-following primitive: long when
the fast SMA sits above the slow SMA, short when below, flat at exact
equality. The function returns one signal per bar already shifted by
one bar, so a caller can multiply by per-bar returns to obtain a
per-bar PnL without introducing look-ahead.

Used as the reference strategy throughout the validation demos. Real
production strategies are not in this repository.
"""
from __future__ import annotations

import numpy as np


def compute_signals(
    close: np.ndarray,
    fast_window: int,
    slow_window: int,
) -> np.ndarray:
    """SMA crossover signals.

    Returns an integer array of -1, 0, +1 indicating short, flat, long
    position for each bar. The signal at bar t is computed from data up
    to bar t-1, so multiplying the signal series by per-bar returns
    gives a strictly causal PnL stream (no look-ahead).

    Args:
        close: 1-D array of close prices.
        fast_window: lookback for the fast SMA. Must be a positive
            integer strictly less than slow_window.
        slow_window: lookback for the slow SMA. Must be a positive
            integer strictly greater than fast_window.

    Returns:
        Integer array of length len(close) with values in {-1, 0, +1}.

    Raises:
        ValueError: if the windows are non-positive or fast >= slow.
    """
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError(
            f"windows must be positive, got fast={fast_window}, slow={slow_window}"
        )
    if fast_window >= slow_window:
        raise ValueError(
            f"fast_window must be strictly less than slow_window, "
            f"got fast={fast_window}, slow={slow_window}"
        )

    close = np.asarray(close, dtype=float)
    fast_sma = _rolling_mean(close, fast_window)
    slow_sma = _rolling_mean(close, slow_window)

    raw = np.sign(fast_sma - slow_sma)
    shifted = np.roll(raw, 1)
    shifted[0] = 0.0
    return shifted.astype(int)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean with arr[0] padding so the output keeps len(arr).

    During the warmup region (bars 0..window-2) both SMAs are flat at
    arr[0], so the comparison fast_sma > slow_sma yields False there
    and compute_signals returns 0. Callers can rely on the length match
    without skipping bars.
    """
    if window <= 1:
        return arr.copy()
    csum = np.cumsum(np.concatenate([[0.0], arr]))
    out = (csum[window:] - csum[:-window]) / window
    pad = np.full(window - 1, arr[0])
    return np.concatenate([pad, out])
