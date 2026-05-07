"""Synthetic OHLC and strategy populations for the validation demos.

Deterministic given a seed, so a given demo produces the same chart on
every machine. Nothing here touches real market data and nothing is
calibrated to any real instrument; the goal is to exercise the
validation machinery on data with known statistical properties.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def regime_switching_ohlc(
    n_bars: int = 5_000,
    base_vol_per_bar: float = 0.0010,
    regime_switch_every: int = 600,
    regime_drifts: tuple[float, ...] = (0.0008, -0.0006, 0.0004, -0.0003),
    regime_vols: tuple[float, ...] = (0.7, 1.4, 0.9, 1.2),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate OHLC bars from a regime-switching log-return process.

    Each regime has its own drift and volatility multiplier. A trend-
    following strategy with a sensible parameter setting will catch some
    regimes; one with a poorly chosen setting will not. That separation
    is what lets the validation demos distinguish signal from noise.

    Args:
        n_bars: number of bars to generate.
        base_vol_per_bar: base log-return standard deviation per bar.
        regime_switch_every: regime length in bars.
        regime_drifts: per-regime drift values; cycled by random index.
        regime_vols: per-regime volatility multipliers; cycled by random
            index.
        seed: seed for the numpy Generator.

    Returns:
        Dict with keys "open", "high", "low", "close" (each a 1-D array
        of length n_bars).
    """
    rng = np.random.default_rng(seed)

    n_regimes = max(1, n_bars // regime_switch_every) + 1
    drifts = np.array(regime_drifts)
    vols = np.array(regime_vols)
    regime_idx = rng.integers(0, len(drifts), size=n_regimes)

    drift_series = np.repeat(drifts[regime_idx], regime_switch_every)[:n_bars]
    vol_series = np.repeat(vols[regime_idx], regime_switch_every)[:n_bars] * base_vol_per_bar

    log_returns = rng.normal(loc=drift_series, scale=vol_series)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    half_range = np.abs(rng.normal(loc=0.0, scale=vol_series)) * close
    high = np.maximum(open_, close) + half_range
    low = np.minimum(open_, close) - half_range

    return {"open": open_, "high": high, "low": low, "close": close}


@dataclass(frozen=True)
class StrategyConfig:
    """A synthetic strategy configuration.

    kind = "sma" runs an SMA crossover on the close series with the
    given fast and slow windows. kind = "random" ignores the close
    series and generates a deterministic random per-bar PnL whose seed
    is derived from (fast, slow). The "random" kind is the noise
    control: it has no edge by construction.
    """

    fast: int
    slow: int
    kind: str = "sma"

    def trades(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-bar PnL and the bar indices on which positions close."""
        if self.kind == "random":
            return self._random_pnl(close), np.array([], dtype=int)
        return self._sma_pnl(close)

    def _sma_pnl(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Long when fast SMA > slow SMA, flat otherwise. No costs."""
        if self.fast >= self.slow or self.slow >= len(close):
            return np.zeros(len(close)), np.array([], dtype=int)

        fast_sma = _rolling_mean(close, self.fast)
        slow_sma = _rolling_mean(close, self.slow)
        long_signal = (fast_sma > slow_sma).astype(int)
        position = np.roll(long_signal, 1)
        position[0] = 0

        bar_returns = np.zeros(len(close))
        bar_returns[1:] = (close[1:] - close[:-1]) / close[:-1]
        per_bar_pnl = position * bar_returns

        diff = np.diff(np.concatenate([[0], position, [0]]))
        close_bars = np.where(diff == -1)[0] - 1
        return per_bar_pnl, close_bars

    def _random_pnl(self, close: np.ndarray) -> np.ndarray:
        seed = (self.fast * 1_000 + self.slow * 31 + 17) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        bar_returns = np.zeros(len(close))
        bar_returns[1:] = (close[1:] - close[:-1]) / close[:-1]
        return rng.normal(loc=0.0, scale=bar_returns.std() * 0.6, size=len(close))


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean. Output bars 0..window-2 are filled with arr[0] so
    the array stays the same length as the input. Demos only use bars
    where both the fast and slow SMAs are defined, so the warmup region
    is harmless."""
    if window <= 1:
        return arr.copy()
    csum = np.cumsum(np.concatenate([[0.0], arr]))
    out = (csum[window:] - csum[:-window]) / window
    pad = np.full(window - 1, arr[0])
    return np.concatenate([pad, out])


def synthetic_population(
    n_signal: int = 60,
    n_noise: int = 140,
    fast_range: tuple[int, int] = (3, 30),
    slow_range: tuple[int, int] = (10, 120),
    noise_kind: str = "random",
    seed: int = 7,
) -> tuple[list[StrategyConfig], np.ndarray]:
    """Build a population of strategy configurations.

    Signal strategies are SMA crossovers in a parameter region empirically
    observed during demo development to capture some of the regime-
    switching drift in the synthetic OHLC. Noise strategies have no edge
    by construction. The truth labels are for chart annotation only;
    CSCV never sees them.

    Args:
        n_signal: number of signal strategies.
        n_noise: number of noise strategies.
        fast_range: (low, high) for the fast SMA window of noise
            strategies, used when noise_kind = "sma_far".
        slow_range: (low, high) for the slow SMA window of noise
            strategies, used when noise_kind = "sma_far".
        noise_kind: "random" for pure-random noise PnL, or "sma_far" for
            SMA crossovers sampled from outside the signal region.
        seed: seed for the numpy Generator.

    Returns:
        (configs, truth) where configs is the list of StrategyConfig and
        truth is a boolean array (True = signal, False = noise).
    """
    rng = np.random.default_rng(seed)
    configs: list[StrategyConfig] = []
    truth: list[bool] = []

    for _ in range(n_signal):
        fast = int(rng.integers(5, 13))
        slow = int(rng.integers(20, 51))
        if fast < slow:
            configs.append(StrategyConfig(fast=fast, slow=slow, kind="sma"))
            truth.append(True)

    while len(configs) < n_signal + n_noise:
        fast = int(rng.integers(*fast_range))
        slow = int(rng.integers(*slow_range))
        if fast >= slow:
            continue
        if noise_kind == "random":
            configs.append(StrategyConfig(fast=fast, slow=slow, kind="random"))
            truth.append(False)
        elif noise_kind == "sma_far":
            in_signal_region = (5 <= fast <= 12) and (20 <= slow <= 50)
            if in_signal_region:
                continue
            configs.append(StrategyConfig(fast=fast, slow=slow, kind="sma"))
            truth.append(False)
        else:
            raise ValueError(f"unknown noise_kind: {noise_kind}")

    return configs, np.array(truth, dtype=bool)
