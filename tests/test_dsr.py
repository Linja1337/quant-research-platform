"""Tests for quant_platform.validation.dsr."""
from __future__ import annotations

import numpy as np
import pytest

from quant_platform.validation.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    sharpe_ratio,
)


def test_dsr_at_sr_zero_is_one_half_under_normal_returns():
    """When the observed Sharpe equals SR_0 and returns are normal, the
    z-statistic in the DSR formula is exactly zero, so DSR = Phi(0) = 0.5."""
    sr_zero = expected_max_sharpe_under_null(
        sharpe_variance_across_trials=0.04,
        n_trials=50,
    )
    dsr = deflated_sharpe_ratio(
        observed_sharpe=sr_zero,
        n_observations=1000,
        n_trials=50,
        sharpe_variance_across_trials=0.04,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert dsr == pytest.approx(0.5, abs=1e-9)


def test_dsr_is_monotonically_increasing_in_observed_sharpe():
    """All else equal, a larger observed Sharpe must produce a larger DSR.

    Uses a small T so the DSR stays in the active part of the Phi range
    rather than saturating to 1.0.
    """
    common = dict(
        n_observations=20,
        n_trials=50,
        sharpe_variance_across_trials=0.04,
        skewness=0.0,
        kurtosis=3.0,
    )
    values = [
        deflated_sharpe_ratio(observed_sharpe=sr, **common)
        for sr in (0.0, 0.2, 0.4, 0.6, 0.8)
    ]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_dsr_falls_below_half_when_observed_below_sr_zero():
    sr_zero = expected_max_sharpe_under_null(
        sharpe_variance_across_trials=0.04,
        n_trials=50,
    )
    dsr = deflated_sharpe_ratio(
        observed_sharpe=sr_zero * 0.5,
        n_observations=1000,
        n_trials=50,
        sharpe_variance_across_trials=0.04,
    )
    assert dsr < 0.5


def test_dsr_approaches_one_for_dominant_strategies():
    sr_zero = expected_max_sharpe_under_null(
        sharpe_variance_across_trials=0.04,
        n_trials=50,
    )
    dsr = deflated_sharpe_ratio(
        observed_sharpe=sr_zero + 5.0,
        n_observations=1000,
        n_trials=50,
        sharpe_variance_across_trials=0.04,
    )
    assert dsr > 0.999


def test_expected_max_sharpe_grows_with_n_trials():
    """The False Strategy Theorem says the expected maximum rises with N."""
    sr_small = expected_max_sharpe_under_null(0.04, n_trials=10)
    sr_large = expected_max_sharpe_under_null(0.04, n_trials=1000)
    assert sr_large > sr_small > 0


def test_expected_max_sharpe_zero_when_variance_zero():
    """Zero cross-trial variance means every trial has the same Sharpe, so
    the expected maximum equals the (zero) baseline."""
    assert expected_max_sharpe_under_null(0.0, n_trials=100) == 0.0


def test_expected_max_sharpe_rejects_negative_variance():
    with pytest.raises(ValueError):
        expected_max_sharpe_under_null(-0.1, n_trials=10)


def test_expected_max_sharpe_rejects_too_few_trials():
    with pytest.raises(ValueError):
        expected_max_sharpe_under_null(0.04, n_trials=1)


def test_negative_skew_and_high_kurtosis_lower_dsr():
    """Negative skew and fat tails inflate the variance of the Sharpe
    estimator, so the DSR for the same observed value is lower."""
    common = dict(
        observed_sharpe=0.6,
        n_observations=20,
        n_trials=50,
        sharpe_variance_across_trials=0.04,
    )
    dsr_normal = deflated_sharpe_ratio(skewness=0.0, kurtosis=3.0, **common)
    dsr_non_normal = deflated_sharpe_ratio(skewness=-1.5, kurtosis=8.0, **common)
    assert dsr_non_normal < dsr_normal


def test_sharpe_ratio_zero_for_zero_returns():
    assert sharpe_ratio(np.zeros(100)) == 0.0


def test_sharpe_ratio_matches_mean_over_std():
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.001, scale=0.01, size=10_000)
    expected = returns.mean() / returns.std()
    assert sharpe_ratio(returns) == pytest.approx(expected, abs=1e-12)
