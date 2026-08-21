"""Shared evaluation metrics: clustering purity, boundary agreement, and the
GW20-vs-GW34 per-pathway persistence classifier.

Extracted from the layer-recovery and persistence entry scripts so the metric
logic is importable and unit-testable.
"""
import numpy as np


def size_weighted_purity(pred, truth):
    """sum_d (max_l count(d,l)) / N  -- standard clustering purity, size-weighted.
    Matches banksy_domains.py/stagate_domains.py's layer_purity."""
    import pandas as pd
    ct = pd.crosstab(pred, truth)
    return float(ct.max(axis=1).sum() / ct.values.sum())


def boundary_metrics(pred_codes, truth_codes, knn_idx, bnd_frac):
    """Return (layer_boundary_recall, domain_boundary_precision, f1).
    pred_codes/truth_codes: int per-cell labels; knn_idx: (n, k) neighbor indices."""
    dom_cross = (pred_codes[knn_idx] != pred_codes[:, None]).mean(axis=1)
    lay_cross = (truth_codes[knn_idx] != truth_codes[:, None]).mean(axis=1)
    dom_b = dom_cross >= bnd_frac
    lay_b = lay_cross >= bnd_frac
    inter = int((dom_b & lay_b).sum())
    recall = inter / int(lay_b.sum()) if lay_b.sum() else float("nan")       # layer bnds captured by domain edges
    precision = inter / int(dom_b.sum()) if dom_b.sum() else float("nan")     # domain edges at real layer bnds
    if precision and recall and not (precision != precision or recall != recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return recall, precision, f1


def classify(rb20, rb34, q20, q34, eff_thr, q_thr):
    """Classify a pathway/direction's V1-vs-V2 effect as it changes GW20->GW34:
    persistent (sig both, same sign) / reversed (sig both, opposite sign) /
    lost (sig GW20 only) / gained (sig GW34 only) / ns_both."""
    sig20 = (q20 < q_thr) and abs(rb20) >= eff_thr
    sig34 = (q34 < q_thr) and abs(rb34) >= eff_thr
    if sig20 and sig34:
        return "persistent" if np.sign(rb20) == np.sign(rb34) else "reversed"
    if sig20 and not sig34:
        return "lost"
    if sig34 and not sig20:
        return "gained"
    return "ns_both"
