"""Unit tests for neurospatial.confidence.compute_low_confidence."""
import numpy as np

from neurospatial.confidence import compute_low_confidence


def test_full_consensus_tiny_margin_not_flagged():
    # tiny margin but all 3 metrics agree (consensus==3) and corr is high -> NOT flagged
    low = compute_low_confidence(
        best_corr=[0.9, 0.9], margin=[0.0001, 0.5], consensus=[3, 3])
    assert low.tolist() == [False, False]


def test_low_margin_and_split_consensus_flagged():
    # smallest margin AND consensus<3 -> flagged; the high-margin sibling is not
    low = compute_low_confidence(
        best_corr=[0.9, 0.9], margin=[0.0001, 0.9], consensus=[1, 1])
    assert low.tolist() == [True, False]


def test_best_corr_below_floor_always_flagged():
    # corr below the absolute floor is flagged regardless of margin/consensus
    low = compute_low_confidence(
        best_corr=[0.1], margin=[0.99], consensus=[3], min_corr=0.3)
    assert bool(low[0]) is True


def test_margin_percentile_bottom_quartile():
    # only the bottom-25% margin (with consensus<3) is flagged; corr all high
    margins = [0.001, 0.5, 0.6, 0.7]  # 25th pctile ~0.375 -> only 0.001 qualifies
    low = compute_low_confidence(
        best_corr=[0.9] * 4, margin=margins, consensus=[1, 1, 1, 1],
        margin_percentile=25.0)
    assert low.tolist() == [True, False, False, False]


def test_min_margin_floor_optional_path():
    # with consensus==3 (percentile branch off) and corr high, only the absolute
    # min_margin_floor catches the sub-floor margin
    low = compute_low_confidence(
        best_corr=[0.9, 0.9, 0.9], margin=[0.02, 0.5, 0.6], consensus=[3, 3, 3],
        min_margin_floor=0.05)
    assert low.tolist() == [True, False, False]

    # floor disabled (default None) -> that cluster is no longer flagged
    low_nofloor = compute_low_confidence(
        best_corr=[0.9, 0.9, 0.9], margin=[0.02, 0.5, 0.6], consensus=[3, 3, 3])
    assert low_nofloor.tolist() == [False, False, False]


def test_returns_numpy_bool_array():
    low = compute_low_confidence(best_corr=[0.9], margin=[0.5], consensus=[3])
    assert isinstance(low, np.ndarray) and low.dtype == bool
