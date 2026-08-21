"""Unit tests for neurospatial.metrics: purity, boundary agreement, persistence classify."""
import numpy as np

from neurospatial.metrics import size_weighted_purity, boundary_metrics, classify


def test_size_weighted_purity_hand_computed(toy_labels):
    pred, truth, expected = toy_labels
    assert size_weighted_purity(pred, truth) == expected


def test_size_weighted_purity_perfect():
    pred = np.array([0, 0, 1, 1])
    truth = np.array(["a", "a", "b", "b"])
    assert size_weighted_purity(pred, truth) == 1.0


def test_boundary_metrics_perfect_alignment(line_knn):
    codes, knn_idx = line_knn
    recall, precision, f1 = boundary_metrics(codes, codes, knn_idx, bnd_frac=0.5)
    # domains == layers, every cell equally straddles the block boundary -> all match
    assert recall == 1.0 and precision == 1.0 and f1 == 1.0


def test_boundary_metrics_no_domains_gives_nan_precision(line_knn):
    codes, knn_idx = line_knn
    single_domain = np.zeros_like(codes)  # one domain -> no domain boundaries
    recall, precision, f1 = boundary_metrics(single_domain, codes, knn_idx, bnd_frac=0.5)
    assert recall == 0.0            # real layer boundaries exist, none captured
    assert np.isnan(precision)      # no domain-boundary cells -> precision undefined
    assert np.isnan(f1)


def test_classify_persistent():
    assert classify(0.3, 0.3, 0.001, 0.001, eff_thr=0.05, q_thr=0.05) == "persistent"


def test_classify_reversed():
    assert classify(0.3, -0.3, 0.001, 0.001, eff_thr=0.05, q_thr=0.05) == "reversed"


def test_classify_lost():
    # significant at GW20, not at GW34 (q34 above threshold)
    assert classify(0.3, 0.3, 0.001, 0.5, eff_thr=0.05, q_thr=0.05) == "lost"


def test_classify_gained():
    assert classify(0.3, 0.3, 0.5, 0.001, eff_thr=0.05, q_thr=0.05) == "gained"


def test_classify_ns_both():
    assert classify(0.3, 0.3, 0.5, 0.5, eff_thr=0.05, q_thr=0.05) == "ns_both"


def test_classify_effect_size_gate():
    # q's are significant but the effect is below eff_thr -> not counted as significant
    assert classify(0.01, 0.01, 0.001, 0.001, eff_thr=0.05, q_thr=0.05) == "ns_both"
