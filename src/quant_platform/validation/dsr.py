"""Deflated Sharpe Ratio (DSR).

Adjusts an observed Sharpe ratio for two known sources of optimism: the
selection bias from running many trials and choosing the best, and the
non-normality of strategy returns. The procedure follows Bailey and Lopez
de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection
Bias, Backtest Overfitting and Non-Normality", SSRN 2460551.

The expected maximum Sharpe under the null hypothesis of no edge across
N independent trials is given by the False Strategy Theorem:

    SR_0 = sqrt(V[SR_n]) * ((1 - gamma) * Phi^-1(1 - 1/N)
                            + gamma * Phi^-1(1 - 1/(N * e)))

where gamma is the Euler-Mascheroni constant, V[SR_n] is the cross-
sectional variance of Sharpe ratios across the N trials, Phi^-1 is the
inverse standard normal CDF, and e is Euler's number.

The DSR is then

    DSR = Phi((SR_obs - SR_0) * sqrt(T - 1)
              / sqrt(1 - skew * SR_obs + ((kurt - 1) / 4) * SR_obs^2))

where Phi is the standard normal CDF, T is the number of return
observations, skew and kurt are the sample skewness and (non-excess)
kurtosis of the returns. Operational target: DSR > 0.95.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe_under_null(
    sharpe_variance_across_trials: float,
    n_trials: int,
) -> float:
    """Expected maximum Sharpe ratio under the null of no edge.

    Implements the False Strategy Theorem of Bailey and Lopez de Prado
    (2014). The result rises with the number of trials and with the
    cross-sectional variance of Sharpe ratios across those trials.

    Args:
        sharpe_variance_across_trials: V[SR_n], the variance of Sharpe
            ratios across the N trials. Must be non-negative.
        n_trials: N, the number of strategy variations tested. Must be
            at least 2.

    Returns:
        The expected maximum Sharpe under the null, on the same time
        scale (per-bar, per-day, per-year) as the input variance.

    Raises:
        ValueError: if sharpe_variance_across_trials is negative or
            n_trials is less than 2.
    """
    if sharpe_variance_across_trials < 0:
        raise ValueError(
            f"sharpe_variance_across_trials must be non-negative, "
            f"got {sharpe_variance_across_trials}"
        )
    if n_trials < 2:
        raise ValueError(f"n_trials must be at least 2, got {n_trials}")

    sigma = math.sqrt(sharpe_variance_across_trials)
    term_a = (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    term_b = EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * (term_a + term_b))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_trials: int,
    sharpe_variance_across_trials: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio for an observed strategy.

    Returns the probability that the observed Sharpe is greater than the
    expected maximum Sharpe under the null. A DSR above 0.95 supports
    rejecting the hypothesis that the observed Sharpe was achieved by
    selection on noise.

    Args:
        observed_sharpe: SR_obs, the observed Sharpe of the candidate
            strategy. Same time scale as the inputs to
            sharpe_variance_across_trials.
        n_observations: T, the number of return observations used to
            compute observed_sharpe. Must be at least 2.
        n_trials: N, the number of strategy variations evaluated during
            selection. Must be at least 2.
        sharpe_variance_across_trials: V[SR_n], the variance of Sharpe
            ratios across the N trials.
        skewness: sample skewness of the returns (third standardized
            moment). Defaults to 0.0 (normal returns).
        kurtosis: sample kurtosis of the returns (fourth standardized
            moment, NOT excess kurtosis). Defaults to 3.0 (normal
            returns).

    Returns:
        DSR in [0, 1].

    Raises:
        ValueError: on invalid n_observations, or if the non-normality
            denominator collapses to a non-positive value (which would
            make the formula undefined).
    """
    if n_observations < 2:
        raise ValueError(f"n_observations must be at least 2, got {n_observations}")

    sr_zero = expected_max_sharpe_under_null(
        sharpe_variance_across_trials=sharpe_variance_across_trials,
        n_trials=n_trials,
    )

    denom_arg = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * observed_sharpe ** 2
    )
    if denom_arg <= 0:
        raise ValueError(
            f"non-normality denominator is non-positive ({denom_arg:.4f}); "
            f"check skewness and kurtosis inputs"
        )

    z = (observed_sharpe - sr_zero) * math.sqrt(n_observations - 1) / math.sqrt(denom_arg)
    return float(norm.cdf(z))


def sharpe_ratio(returns: np.ndarray) -> float:
    """Sample Sharpe ratio (mean / std) of a return series.

    No annualization is applied; the result is on the same time scale as
    the input returns. Returns 0.0 if the standard deviation is too small
    to compute meaningfully.

    Args:
        returns: 1-D array of returns.

    Returns:
        The Sharpe ratio.
    """
    arr = np.asarray(returns, dtype=float)
    sd = float(arr.std())
    if sd < 1e-12:
        return 0.0
    return float(arr.mean() / sd)
