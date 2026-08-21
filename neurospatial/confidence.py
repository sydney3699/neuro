"""Cluster-to-reference-type confidence flagging (shared by the annotation stage).

Extracted from the annotation entry scripts so the flag logic is importable and
unit-testable independent of the embedding/Leiden machinery.
"""


def compute_low_confidence(best_corr, margin, consensus, *, min_corr=0.3,
                           margin_percentile=25.0, min_margin_floor=None):
    """Adaptive low-confidence flag for cluster->reference-type assignments.

    A cluster is flagged when ANY of:
      - best_corr < min_corr : the cluster matches no reference type well
        (absolute floor -- a genuine "nothing fits" catch).
      - margin at/below the within-run `margin_percentile` AND metric_consensus < 3 :
        the top-two reference types are near-tied *for this run* AND the three
        metrics (Pearson/Spearman/cosine) don't all agree on the pick.
      - min_margin_floor is not None and margin < min_margin_floor : optional
        absolute margin floor (disabled by default; the old fixed-0.05 behavior).

    The consensus gate is the key change from the old `margin < 0.05` rule: a small
    best-vs-second margin no longer flags a cluster on its own when Pearson, Spearman
    and cosine still agree on the same label. Percentile makes the margin cut adapt
    to each run's own margin distribution instead of a fixed constant that over-flags
    (the old rule flagged 59-84% of clusters, driven entirely by margin<0.05).
    """
    import numpy as np
    best_corr = np.asarray(best_corr, dtype=float)
    margin = np.asarray(margin, dtype=float)
    consensus = np.asarray(consensus, dtype=int)
    low = best_corr < min_corr
    if margin.size:
        thr = np.percentile(margin, margin_percentile)
        low = low | ((margin <= thr) & (consensus < 3))
    if min_margin_floor is not None:
        low = low | (margin < min_margin_floor)
    return low
