"""Tests for quant_platform.validation.walk_forward."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quant_platform.validation.walk_forward import (
    walk_forward_efficiency,
    walk_forward_folds,
)


def test_folds_cover_full_bar_range_without_overlap():
    """Anchored folds with is_fraction = 0.5 should cover [0, n_bars)
    contiguously when no folds get dropped."""
    folds = walk_forward_folds(
        n_bars=1000, n_folds=4, is_fraction=0.5, min_oos_bars=10
    )
    assert len(folds) == 4
    assert folds[0].is_start == 0
    assert folds[-1].oos_end == 1000
    for prev, curr in zip(folds, folds[1:]):
        assert prev.oos_end == curr.is_start


def test_oos_starts_immediately_after_is():
    folds = walk_forward_folds(n_bars=500, n_folds=3, is_fraction=0.6)
    for fold in folds:
        assert fold.oos_start == fold.is_end


def test_no_folds_raises():
    with pytest.raises(ValueError):
        walk_forward_folds(
            n_bars=100, n_folds=4, is_fraction=0.99, min_oos_bars=50
        )


def test_invalid_is_fraction_raises():
    with pytest.raises(ValueError):
        walk_forward_folds(n_bars=100, n_folds=2, is_fraction=0.0)
    with pytest.raises(ValueError):
        walk_forward_folds(n_bars=100, n_folds=2, is_fraction=1.0)


def test_walk_forward_efficiency_basic_ratio():
    assert walk_forward_efficiency(is_sharpe=2.0, oos_sharpe=1.0) == 0.5


def test_walk_forward_efficiency_returns_nan_when_is_collapses():
    assert math.isnan(walk_forward_efficiency(is_sharpe=0.05, oos_sharpe=1.0))


def test_known_strategy_produces_expected_wfe_on_synthetic_series():
    """Build a synthetic IS/OOS structure where the OOS mean return is
    roughly half the IS mean. The resulting WFE should sit near 0.5."""
    rng = np.random.default_rng(0)
    is_returns = rng.normal(loc=0.05, scale=0.1, size=2000)
    oos_returns = rng.normal(loc=0.025, scale=0.1, size=2000)
    is_sharpe = is_returns.mean() / is_returns.std()
    oos_sharpe = oos_returns.mean() / oos_returns.std()
    wfe = walk_forward_efficiency(is_sharpe, oos_sharpe)
    assert wfe == pytest.approx(oos_sharpe / is_sharpe, abs=1e-12)
    assert 0.3 < wfe < 0.7
