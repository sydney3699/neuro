"""Shared helpers for COMMOT signaling columns + multiple-testing correction.

Extracted from the niche/signaling entry scripts so the column-classification and
BH-FDR logic is importable and unit-testable.
"""


def select_signaling_cols(cols, level, pathway_names, pair_suffixes):
    """Classify merged COMMOT sum columns (e.g. 's-WNT', 's-TGFB1-TGFBR1_TGFBR2',
    's-total-total') by level against the actual CellChatDB pathway/pair names.

    level: 'pathway' | 'pair' | 'all'. Drops the '*-total-total' aggregate columns.
    """
    out = []
    for c in map(str, cols):
        if c.endswith("-total-total"):
            continue
        suffix = c[2:] if (c.startswith("s-") or c.startswith("r-")) else c
        if suffix in pathway_names:
            is_pathway = True
        elif suffix in pair_suffixes:
            is_pathway = False
        else:
            is_pathway = "-" not in suffix
        if level == "all" or (level == "pathway" and is_pathway) or (level == "pair" and not is_pathway):
            out.append(c)
    return out


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR. Returns q-values aligned to input order."""
    import numpy as np
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    return q
