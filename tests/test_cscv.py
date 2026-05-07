"""Tests for quant_platform.validation.cscv."""
from __future__ import annotations

import numpy as np
import pytest

from quant_platform.validation.cscv import (
    aggregate_per_block_pnl,
    compute_pbo,
)


def test_pbo_returns_05_under_null():
    """Pure-noise strategies have no edge. CSCV should report a PBO close
    to 0.5 because the in-sample winner is no more likely than chance to
    rank above the OOS median."""
    rng = np.random.default_rng(0)
    T, N = 12, 40
    M = rng.normal(size=(T, N))
    result = compute_pbo(M)
    assert 0.30 <= result.pbo <= 0.70


def test_pbo_returns_low_under_genuine_edge():
    """When a strategy genuinely dominates every block, CSCV should report
    a PBO well below 0.5 because the IS winner is the same one and lands
    at the top of the OOS distribution every time."""
    rng = np.random.default_rng(1)
    T, N = 10, 20
    M = rng.normal(size=(T, N))
    M[:, 0] += 5.0
    result = compute_pbo(M)
    assert result.pbo < 0.10


def test_n_splits_equals_binomial():
    """C(T, T/2) symmetric partitions for an even T."""
    M = np.zeros((8, 5))
    result = compute_pbo(M)
    from math import comb

    assert result.n_splits == comb(8, 4)


def test_full_is_rank_orders_strategies():
    """full_is_rank should reproduce the rank order of column sums."""
    M = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )
    M = np.vstack([M, M, M, M])
    result = compute_pbo(M)
    assert tuple(result.full_is_rank.tolist()) == (1, 2, 3, 4)


def test_compute_pbo_rejects_odd_T():
    with pytest.raises(ValueError):
        compute_pbo(np.zeros((7, 5)))


def test_compute_pbo_rejects_too_few_strategies():
    with pytest.raises(ValueError):
        compute_pbo(np.zeros((6, 1)))


def test_aggregate_per_block_pnl_partitions_correctly():
    pnl = np.arange(20, dtype=float)
    blocks = aggregate_per_block_pnl(pnl, n_blocks=4)
    assert blocks.shape == (4,)
    assert blocks.sum() == pnl.sum()


def test_aggregate_per_block_pnl_handles_remainder():
    """When len(pnl) is not a multiple of n_blocks, the last block absorbs
    the remainder."""
    pnl = np.ones(11, dtype=float)
    blocks = aggregate_per_block_pnl(pnl, n_blocks=3)
    assert blocks[0] == 3.0
    assert blocks[1] == 3.0
    assert blocks[2] == 5.0
