"""Unit tests for neurospatial.signaling.select_signaling_cols and bh_fdr."""
import numpy as np

from neurospatial.signaling import select_signaling_cols, bh_fdr

PATHWAYS = {"WNT", "TGFB"}
PAIRS = {"TGFB1-TGFBR1_TGFBR2"}
COLS = ["s-WNT", "r-WNT", "s-TGFB1-TGFBR1_TGFBR2", "s-total-total", "r-total-total"]


def test_pathway_level_keeps_pathways_drops_pairs_and_totals():
    out = select_signaling_cols(COLS, "pathway", PATHWAYS, PAIRS)
    assert out == ["s-WNT", "r-WNT"]


def test_pair_level_keeps_only_pairs():
    out = select_signaling_cols(COLS, "pair", PATHWAYS, PAIRS)
    assert out == ["s-TGFB1-TGFBR1_TGFBR2"]


def test_all_level_keeps_everything_except_totals():
    out = select_signaling_cols(COLS, "all", PATHWAYS, PAIRS)
    assert out == ["s-WNT", "r-WNT", "s-TGFB1-TGFBR1_TGFBR2"]


def test_total_total_always_dropped():
    assert "s-total-total" not in select_signaling_cols(COLS, "all", PATHWAYS, PAIRS)


def test_unknown_suffix_fallback_by_hyphen():
    # unknown-to-DB suffixes: no hyphen -> treated as pathway; hyphen -> pair
    cols = ["s-FOO", "s-A-B"]
    assert select_signaling_cols(cols, "pathway", PATHWAYS, PAIRS) == ["s-FOO"]
    assert select_signaling_cols(cols, "pair", PATHWAYS, PAIRS) == ["s-A-B"]


def test_bh_fdr_known_uniform_vector():
    # p = [0.01,0.02,0.03,0.04,0.05], n=5 -> every BH q collapses to 0.05
    q = bh_fdr([0.01, 0.02, 0.03, 0.04, 0.05])
    assert np.allclose(q, 0.05)


def test_bh_fdr_preserves_input_order():
    # input order [0.9, 0.01] -> q aligned to input, not to sorted order
    q = bh_fdr([0.9, 0.01])
    assert np.allclose(q, [0.9, 0.02])


def test_bh_fdr_bounded_and_monotone_in_rank():
    q = bh_fdr([0.2, 0.001, 0.7, 0.05, 0.9])
    assert q.min() >= 0.0 and q.max() <= 1.0
    # q sorted by p is non-decreasing (BH step-up property)
    p = np.array([0.2, 0.001, 0.7, 0.05, 0.9])
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_bh_fdr_empty():
    assert bh_fdr([]).size == 0
