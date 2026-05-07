"""Walk-forward validation: anchored IS / OOS folds and walk-forward
efficiency.

Walk-forward is a sliding-window cross-validation procedure for time
series. The bar series is split into fixed-size folds. Each fold is
itself partitioned into an in-sample (IS) region used for parameter
selection and an out-of-sample (OOS) region used only for scoring. The
ratio of OOS Sharpe to IS Sharpe across folds is the walk-forward
efficiency. Pardo (2008), "The Evaluation and Optimization of Trading
Strategies", is the canonical reference.

This module provides the index-generation logic and the walk-forward
efficiency definition. Strategy fitting and PnL calculation are caller
responsibilities, since they vary by strategy family.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardFold:
    """A single anchored walk-forward fold.

    Attributes:
        fold: 1-based fold index.
        is_start: inclusive start bar of the in-sample region.
        is_end: exclusive end bar of the in-sample region.
        oos_start: inclusive start bar of the out-of-sample region.
            Always equals is_end.
        oos_end: exclusive end bar of the out-of-sample region.
    """

    fold: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int

    @property
    def is_slice(self) -> slice:
        return slice(self.is_start, self.is_end)

    @property
    def oos_slice(self) -> slice:
        return slice(self.oos_start, self.oos_end)

    @property
    def is_size(self) -> int:
        return self.is_end - self.is_start

    @property
    def oos_size(self) -> int:
        return self.oos_end - self.oos_start


def walk_forward_folds(
    n_bars: int,
    n_folds: int,
    is_fraction: float,
    min_oos_bars: int = 50,
) -> list[WalkForwardFold]:
    """Generate anchored walk-forward folds covering the bar range.

    The bar range [0, n_bars) is split into n_folds equal-size chunks.
    Within each chunk, the first is_fraction of the bars is the IS
    region and the remainder is the OOS region. Folds whose OOS region
    is shorter than min_oos_bars are dropped.

    Args:
        n_bars: total number of bars.
        n_folds: number of folds. Must be at least 1.
        is_fraction: IS share of each fold's bars, in (0, 1).
        min_oos_bars: minimum OOS length (in bars) for a fold to be kept.

    Returns:
        A list of WalkForwardFold records.

    Raises:
        ValueError: if any input is out of range or no folds survive
            the OOS-length filter.
    """
    if n_bars <= 0:
        raise ValueError(f"n_bars must be positive, got {n_bars}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1, got {n_folds}")
    if not 0.0 < is_fraction < 1.0:
        raise ValueError(f"is_fraction must be in (0, 1), got {is_fraction}")

    fold_size = n_bars // n_folds
    is_size = int(fold_size * is_fraction)

    folds: list[WalkForwardFold] = []
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n_bars
        is_end = start + is_size
        if end - is_end < min_oos_bars:
            continue
        folds.append(
            WalkForwardFold(
                fold=f + 1,
                is_start=start,
                is_end=is_end,
                oos_start=is_end,
                oos_end=end,
            )
        )

    if not folds:
        raise ValueError(
            f"no folds produced an OOS region of at least {min_oos_bars} bars; "
            f"check n_bars, n_folds, is_fraction"
        )
    return folds


def walk_forward_efficiency(
    is_sharpe: float,
    oos_sharpe: float,
    is_floor: float = 0.1,
) -> float:
    """Walk-forward efficiency (WFE) for a single fold.

    WFE is OOS Sharpe divided by IS Sharpe. The ratio is meaningful only
    when the IS Sharpe is materially nonzero; when |IS Sharpe| falls
    below is_floor, the ratio becomes uninformative and the function
    returns NaN. Callers should aggregate using nanmean.

    Args:
        is_sharpe: in-sample Sharpe.
        oos_sharpe: out-of-sample Sharpe.
        is_floor: threshold below which the ratio is treated as
            undefined.

    Returns:
        OOS Sharpe / IS Sharpe, or NaN if |is_sharpe| < is_floor.
    """
    if abs(is_sharpe) < is_floor:
        return math.nan
    return oos_sharpe / is_sharpe
