"""Shared expression-normalization helpers for the annotation / label-transfer path.

annotate_cells.py's reference-centroid transfer normalizes both sides with
log1p-CP10k computed over the SHARED gene set only (subset first, then normalize,
then log1p -- the order matters, since the per-cell total is taken over the subset).
embedding_discriminability.py has to apply the exact same transform to audit that
transfer, so the semantics live here, pinned by unit tests, rather than being
reimplemented per script.

Deliberately numpy/pandas-only (no scanpy, no scipy): the test extra installs only
those, and sparse inputs are handled by duck-typing `.toarray()`.
"""
import numpy as np
import pandas as pd


def lognorm_profiles(X, var_names, shared_genes, target_sum=1e4):
    """Per-cell log1p-CP10k profiles over `shared_genes`, as a DataFrame.

    Order of operations (matches annotate_cells.py): subset to shared_genes ->
    scale each cell so its total over THOSE genes is target_sum -> log1p. Cells with
    zero total over the subset are left as all-zero rather than producing NaN/inf.
    """
    pos = {str(v): i for i, v in enumerate(var_names)}
    missing = [g for g in shared_genes if g not in pos]
    if missing:
        raise KeyError(f"{len(missing)} shared genes absent from var_names, e.g. {missing[:5]}")
    idx = [pos[g] for g in shared_genes]

    Xa = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    sub = np.asarray(Xa[:, idx], dtype="float64")

    total = sub.sum(axis=1, keepdims=True)
    scale = np.divide(target_sum, total, out=np.zeros_like(total), where=total > 0)
    return pd.DataFrame(np.log1p(sub * scale), columns=list(shared_genes))


def lognorm_mean_by(X, var_names, groups, shared_genes, target_sum=1e4):
    """Mean log1p-CP10k profile per group: DataFrame [group x shared_genes].

    Equivalent to annotate_cells.py's local `lognorm_mean_by` -- per-cell normalize
    first, then average within group (NOT average-then-normalize, which differs).
    """
    df = lognorm_profiles(X, var_names, shared_genes, target_sum=target_sum)
    df["_g"] = np.asarray(groups)
    return df.groupby("_g").mean()


def corr_to_centroids(profiles, centroids):
    """Pearson correlation of each row of `profiles` against each reference centroid.

    profiles:  (n_cells, n_genes) array-like; centroids: (n_types, n_genes) DataFrame.
    Returns (n_cells, n_types) float array. Rows/centroids with zero variance across
    genes yield NaN (no information to correlate), left for the caller to handle.
    """
    A = np.asarray(profiles, dtype="float64")
    B = np.asarray(centroids, dtype="float64")
    Az = A - A.mean(axis=1, keepdims=True)
    Bz = B - B.mean(axis=1, keepdims=True)
    An = np.sqrt((Az ** 2).sum(axis=1, keepdims=True))
    Bn = np.sqrt((Bz ** 2).sum(axis=1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Az @ Bz.T) / (An * Bn.T)
