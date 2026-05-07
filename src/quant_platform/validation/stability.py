"""Parameter-stability scoring on profit surfaces.

A parameter "surface" is a 2-D grid where each cell is the cumulative
PnL (or any scalar performance metric) of one parameter combination.
The optimum is the cell with the highest value. The stability score is
the fraction of one-step neighbors of the optimum that are also
profitable. A high score means the optimum sits on a plateau; a low
score means it sits on a knife-edge.

Knife-edge optima are typical of overfit parameter selection. The cell
that won was the best because of luck specific to that combination,
not because the underlying parameter region is good.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityResult:
    """Result of scoring a parameter surface.

    Attributes:
        score: fraction of one-step neighbors that are profitable, in
            [0, 1]. NaN if the optimum has no defined neighbors.
        optimum_index: (row, col) location of the optimum cell.
        optimum_value: value at the optimum.
        n_neighbors_evaluated: count of one-step neighbors that were
            in-bounds and not NaN.
    """

    score: float
    optimum_index: tuple[int, int]
    optimum_value: float
    n_neighbors_evaluated: int


def parameter_stability_score(
    surface: np.ndarray,
    profitable_threshold: float = 0.0,
) -> StabilityResult:
    """Score the stability of a parameter surface around its maximum.

    NaN cells are treated as out-of-domain and excluded from neighbor
    evaluation (they typically arise from invalid parameter combinations
    such as fast >= slow in an SMA crossover grid).

    Args:
        surface: 2-D array of metric values per parameter cell.
        profitable_threshold: value above which a neighbor is considered
            profitable. Defaults to 0.0.

    Returns:
        A StabilityResult bundling the score, the optimum location, and
        diagnostic counts.

    Raises:
        ValueError: if surface is not 2-D or contains no finite values.
    """
    surface = np.asarray(surface, dtype=float)
    if surface.ndim != 2:
        raise ValueError(f"surface must be 2-D, got shape {surface.shape}")
    if not np.any(np.isfinite(surface)):
        raise ValueError("surface contains no finite values")

    masked = np.where(np.isnan(surface), -np.inf, surface)
    flat = int(masked.argmax())
    i_star, j_star = np.unravel_index(flat, surface.shape)
    optimum_value = float(surface[i_star, j_star])

    neighbors: list[float] = []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        i, j = i_star + di, j_star + dj
        if 0 <= i < surface.shape[0] and 0 <= j < surface.shape[1]:
            v = surface[i, j]
            if not np.isnan(v):
                neighbors.append(float(v))

    if not neighbors:
        return StabilityResult(
            score=float("nan"),
            optimum_index=(int(i_star), int(j_star)),
            optimum_value=optimum_value,
            n_neighbors_evaluated=0,
        )

    profitable = sum(1 for v in neighbors if v > profitable_threshold)
    return StabilityResult(
        score=profitable / len(neighbors),
        optimum_index=(int(i_star), int(j_star)),
        optimum_value=optimum_value,
        n_neighbors_evaluated=len(neighbors),
    )
