"""Combinatorially Symmetric Cross-Validation (CSCV) and Probability of
Backtest Overfitting (PBO).

The procedure follows Bailey, Borwein, Lopez de Prado, and Zhu (2014),
"The Probability of Backtest Overfitting", Algorithm 2.3. The input is a
performance matrix M of shape (T, N) where T is the number of time blocks
and N is the number of strategy configurations. M[t, n] is the aggregate
PnL of configuration n during time block t.

For every symmetric partition of the T blocks into in-sample and out-of-
sample halves, the algorithm picks the in-sample best configuration and
records where it lands in the out-of-sample distribution. PBO is the
fraction of partitions where that in-sample best lands at or below the
out-of-sample median. Operational target: PBO < 0.10.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class CSCVResult:
    """Output of a CSCV run.

    Attributes:
        pbo: Probability of Backtest Overfitting in [0, 1].
        logits: log(omega / (1 - omega)) for each symmetric split.
        winner_oos_quantile: For each split, the OOS quantile (in (0, 1))
            of the strategy that won in-sample.
        full_is_rank: Per-strategy in-sample rank using the full (un-split)
            performance matrix; rank 1 is worst, N is best.
        oos_avg_rank: Per-strategy OOS rank averaged across all splits in
            which the strategy was on the OOS side.
        spearman_rho: Spearman correlation between full_is_rank and
            oos_avg_rank. Low correlation is a separate symptom of
            performance degradation.
        n_splits: Number of symmetric splits enumerated, which equals
            C(T, T/2).
    """

    pbo: float
    logits: np.ndarray
    winner_oos_quantile: np.ndarray
    full_is_rank: np.ndarray
    oos_avg_rank: np.ndarray
    spearman_rho: float
    n_splits: int


def compute_pbo(M: np.ndarray) -> CSCVResult:
    """Run CSCV on a (T, N) performance matrix and return the full result.

    Args:
        M: Performance matrix of shape (T, N). T must be even. M[t, n] is
            the aggregate performance of strategy n during time block t.

    Returns:
        A CSCVResult bundling the PBO and supporting per-split arrays.

    Raises:
        ValueError: if T is odd or N < 2.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D, got shape {M.shape}")
    T, N = M.shape
    if T % 2 != 0:
        raise ValueError(f"T (rows of M) must be even, got T={T}")
    if N < 2:
        raise ValueError(f"need at least 2 strategies, got N={N}")

    half = T // 2
    splits = list(itertools.combinations(range(T), half))

    logits = np.empty(len(splits))
    winner_oos_quantile = np.empty(len(splits))
    full_is_perf = M.sum(axis=0)
    full_is_rank = full_is_perf.argsort().argsort() + 1

    oos_avg_rank = np.zeros(N)
    oos_count = np.zeros(N)

    for k, is_idx in enumerate(splits):
        is_idx_arr = list(is_idx)
        oos_idx_arr = [t for t in range(T) if t not in is_idx]
        is_perf = M[is_idx_arr].sum(axis=0)
        oos_perf = M[oos_idx_arr].sum(axis=0)

        n_star = int(np.argmax(is_perf))

        oos_ranks = oos_perf.argsort().argsort() + 1
        rank_n_star = int(oos_ranks[n_star])
        omega = rank_n_star / (N + 1)
        omega_safe = float(np.clip(omega, 1e-6, 1 - 1e-6))
        logits[k] = float(np.log(omega_safe / (1 - omega_safe)))
        winner_oos_quantile[k] = omega

        oos_avg_rank += oos_ranks
        oos_count += 1

    oos_avg_rank /= oos_count
    pbo = float(np.mean(logits <= 0))
    rho, _ = spearmanr(full_is_rank, oos_avg_rank)

    return CSCVResult(
        pbo=pbo,
        logits=logits,
        winner_oos_quantile=winner_oos_quantile,
        full_is_rank=full_is_rank,
        oos_avg_rank=oos_avg_rank,
        spearman_rho=float(rho),
        n_splits=len(splits),
    )


def aggregate_per_block_pnl(
    per_bar_pnl: np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    """Split a per-bar PnL series into n_blocks equal-length blocks and sum.

    Used to convert raw per-bar strategy PnL into the (T, N) performance
    matrix that compute_pbo consumes.

    Args:
        per_bar_pnl: 1-D array of per-bar PnL.
        n_blocks: number of equal-length time blocks.

    Returns:
        Length-n_blocks array; each entry is the sum of one block.
    """
    n = len(per_bar_pnl)
    block_size = n // n_blocks
    out = np.empty(n_blocks)
    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size if b < n_blocks - 1 else n
        out[b] = per_bar_pnl[start:end].sum()
    return out
