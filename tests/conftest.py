"""Shared fixtures for the neurospatial unit tests.

All helpers under test are pure (numpy/pandas only), so fixtures are tiny
hand-built arrays with known expected outputs.
"""
import numpy as np
import pytest


@pytest.fixture
def toy_labels():
    """A small (pred, truth) label pair with a hand-computed size-weighted purity.

    pred a: {x:2, y:1} -> max 2;  pred b: {y:2} -> max 2;  sum=4, N=5 -> 0.8
    """
    pred = np.array(["a", "a", "a", "b", "b"])
    truth = np.array(["x", "x", "y", "y", "y"])
    return pred, truth, 0.8


@pytest.fixture
def line_knn():
    """4 cells in two blocks {0,1}|{2,3}, each with its 2 nearest neighbours.
    With pred==truth==[0,0,1,1] every cell straddles the block boundary equally."""
    knn_idx = np.array([[1, 2], [0, 2], [1, 3], [2, 1]])
    codes = np.array([0, 0, 1, 1])
    return codes, knn_idx
