from __future__ import annotations

import pandas as pd

from mekubbal.profile.stats import _aggregate_profile_rows, _pairwise_profile_rows


def test_pairwise_profile_rows_builds_pairwise_summary():
    frame = _pairwise_profile_rows(
        {
            'base': {'AAPL': 0.01, 'MSFT': 0.00},
            'candidate': {'AAPL': 0.03, 'MSFT': 0.02},
        },
        confidence_level=0.9,
        n_bootstrap=200,
        n_permutation=500,
        seed=7,
    )

    assert len(frame) == 1
    assert set([
        'profile_a',
        'profile_b',
        'paired_symbol_count',
        'mean_gap_diff_a_minus_b',
        'p_value_two_sided',
    ]).issubset(frame.columns)


def test_aggregate_profile_rows_ranks_profiles_and_tracks_significant_wins():
    symbol_rows = pd.DataFrame(
        [
            {'symbol': 'AAPL', 'profile': 'base', 'symbol_rank': 2, 'avg_equity_gap': 0.01},
            {'symbol': 'AAPL', 'profile': 'candidate', 'symbol_rank': 1, 'avg_equity_gap': 0.03},
            {'symbol': 'MSFT', 'profile': 'base', 'symbol_rank': 2, 'avg_equity_gap': 0.00},
            {'symbol': 'MSFT', 'profile': 'candidate', 'symbol_rank': 1, 'avg_equity_gap': 0.02},
        ]
    )
    pairwise_rows = pd.DataFrame(
        [
            {
                'profile_a': 'candidate',
                'profile_b': 'base',
                'profile_a_better_significant': True,
                'profile_b_better_significant': False,
            }
        ]
    )

    ranked = _aggregate_profile_rows(symbol_rows, pairwise_rows)

    assert list(ranked['profile']) == ['candidate', 'base']
    assert list(ranked['rank']) == [1, 2]
    assert list(ranked['net_significant_wins']) == [1, -1]
