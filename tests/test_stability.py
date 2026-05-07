"""Tests for quant_platform.validation.stability."""
from __future__ import annotations

import numpy as np
import pytest

from quant_platform.validation.stability import parameter_stability_score


def test_uniform_profitable_surface_scores_one():
    """All neighbors of any cell are profitable, so the score is 1.0."""
    surface = np.full((5, 5), 1.0)
    result = parameter_stability_score(surface)
    assert result.score == 1.0
    assert result.n_neighbors_evaluated > 0


def test_isolated_peak_scores_zero():
    """A single positive cell surrounded by losses produces a score of 0."""
    surface = np.full((5, 5), -1.0)
    surface[2, 2] = 10.0
    result = parameter_stability_score(surface)
    assert result.optimum_index == (2, 2)
    assert result.score == 0.0
    assert result.n_neighbors_evaluated == 4


def test_corner_optimum_has_two_neighbors():
    """An optimum at (0, 0) should evaluate exactly 2 in-bounds neighbors."""
    surface = np.array(
        [
            [10.0, 5.0, -1.0],
            [5.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    result = parameter_stability_score(surface)
    assert result.optimum_index == (0, 0)
    assert result.n_neighbors_evaluated == 2
    assert result.score == 1.0


def test_nan_neighbors_are_excluded():
    """NaN cells (out-of-domain combinations) should not count toward
    the neighbor total."""
    surface = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 5.0, 1.0],
            [np.nan, 1.0, -1.0],
        ]
    )
    result = parameter_stability_score(surface)
    assert result.optimum_index == (1, 1)
    assert result.n_neighbors_evaluated == 2
    assert result.score == 1.0


def test_all_nan_surface_raises():
    surface = np.full((3, 3), np.nan)
    with pytest.raises(ValueError):
        parameter_stability_score(surface)


def test_one_dimensional_surface_raises():
    with pytest.raises(ValueError):
        parameter_stability_score(np.array([1.0, 2.0, 3.0]))


def test_threshold_controls_what_counts_as_profitable():
    """With threshold=0, two of the four neighbors of (1, 1) are positive."""
    surface = np.array(
        [
            [-1.0, 0.5, -1.0],
            [-0.5, 5.0, 0.5],
            [-1.0, -0.5, -1.0],
        ]
    )
    default = parameter_stability_score(surface)
    assert default.score == pytest.approx(2 / 4)

    strict = parameter_stability_score(surface, profitable_threshold=0.4)
    assert strict.score == pytest.approx(2 / 4)

    very_strict = parameter_stability_score(surface, profitable_threshold=1.0)
    assert very_strict.score == 0.0
