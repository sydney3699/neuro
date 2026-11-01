"""Unit tests for neurospatial.expression (log1p-CP10k profiles + centroid correlation)."""
import numpy as np
import pytest

from neurospatial.expression import corr_to_centroids, lognorm_mean_by, lognorm_profiles

VARS = ["GENE_A", "GENE_B", "GENE_C"]


def test_profiles_normalize_over_subset_not_all_genes():
    """Per-cell total is taken over the SHARED subset only -- the gene dropped from
    `shared_genes` must not influence the surviving columns' scaling."""
    X = np.array([[1.0, 1.0, 98.0]])            # GENE_C dominates the full-row total
    out = lognorm_profiles(X, VARS, ["GENE_A", "GENE_B"])
    # subset total = 2 -> each gene scales to 5000, not to 100
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out.values, np.log1p([[5000.0, 5000.0]]))


def test_profiles_column_order_follows_shared_genes():
    X = np.array([[1.0, 3.0, 0.0]])
    out = lognorm_profiles(X, VARS, ["GENE_B", "GENE_A"])
    assert list(out.columns) == ["GENE_B", "GENE_A"]
    # total 4 -> B = 3/4*1e4 = 7500, A = 2500
    np.testing.assert_allclose(out.values, np.log1p([[7500.0, 2500.0]]))


def test_zero_total_cell_stays_zero_not_nan():
    X = np.array([[0.0, 0.0, 5.0], [2.0, 2.0, 0.0]])
    out = lognorm_profiles(X, VARS, ["GENE_A", "GENE_B"])
    assert np.isfinite(out.values).all()
    np.testing.assert_allclose(out.values[0], [0.0, 0.0])


def test_missing_shared_gene_raises():
    with pytest.raises(KeyError):
        lognorm_profiles(np.zeros((1, 3)), VARS, ["GENE_A", "NOT_A_GENE"])


def test_sparse_input_via_duck_typed_toarray():
    class FakeSparse:
        def __init__(self, dense):
            self._d = dense

        def toarray(self):
            return self._d

    dense = np.array([[1.0, 1.0, 0.0]])
    out_sparse = lognorm_profiles(FakeSparse(dense), VARS, ["GENE_A", "GENE_B"])
    out_dense = lognorm_profiles(dense, VARS, ["GENE_A", "GENE_B"])
    np.testing.assert_allclose(out_sparse.values, out_dense.values)


def test_mean_by_normalizes_per_cell_before_averaging():
    """Two cells with identical composition but 10x different depth must average to
    the same profile as either one alone -- i.e. normalize-then-average."""
    X = np.array([[1.0, 1.0, 0.0], [10.0, 10.0, 0.0]])
    out = lognorm_mean_by(X, VARS, ["g", "g"], ["GENE_A", "GENE_B"])
    assert list(out.index) == ["g"]
    np.testing.assert_allclose(out.values, np.log1p([[5000.0, 5000.0]]))


def test_mean_by_groups_independently():
    X = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    out = lognorm_mean_by(X, VARS, ["a", "b"], ["GENE_A", "GENE_B"])
    assert list(out.index) == ["a", "b"]
    np.testing.assert_allclose(out.loc["a"].values, np.log1p([1e4, 0.0]))
    np.testing.assert_allclose(out.loc["b"].values, np.log1p([0.0, 1e4]))


def test_corr_to_centroids_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    prof, cent = rng.normal(size=(5, 12)), rng.normal(size=(3, 12))
    got = corr_to_centroids(prof, cent)
    want = np.corrcoef(np.vstack([prof, cent]))[:5, 5:]
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_corr_identical_profile_is_one():
    prof = np.array([[1.0, 2.0, 3.0, 4.0]])
    got = corr_to_centroids(prof, prof)
    np.testing.assert_allclose(got, [[1.0]])


def test_corr_constant_row_is_nan_not_error():
    got = corr_to_centroids(np.ones((1, 4)), np.array([[1.0, 2.0, 3.0, 4.0]]))
    assert np.isnan(got).all()
