from __future__ import annotations

import pytest

from mekubbal.statistics import bootstrap_mean_confidence, paired_permutation_stats


def test_bootstrap_mean_confidence_can_include_sign_probabilities():
    summary = bootstrap_mean_confidence(
        [0.1, 0.2, 0.3],
        confidence_level=0.9,
        n_bootstrap=200,
        seed=7,
        include_sign_probabilities=True,
    )

    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
    assert 0.0 <= summary["p_gt_zero"] <= 1.0
    assert 0.0 <= summary["p_lt_zero"] <= 1.0


def test_paired_permutation_stats_returns_tail_probabilities():
    summary = paired_permutation_stats(
        [0.1, 0.2, -0.05],
        n_permutation=500,
        seed=11,
    )

    assert "mean_diff" in summary
    assert "p_two_sided" in summary
    assert "p_observed_or_higher" in summary
    assert "p_observed_or_lower" in summary
    assert 0.0 <= summary["p_two_sided"] <= 1.0


def test_bootstrap_mean_confidence_rejects_empty_values():
    with pytest.raises(ValueError, match="empty values"):
        bootstrap_mean_confidence([], confidence_level=0.9, n_bootstrap=200, seed=1)
