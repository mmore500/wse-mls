import numpy as np
import pandas as pd
from phyloframe import legacy as pfl
import pytest

from pylib import screen_mdcs_fitch


def _make_tree(rng, n):
    # random topologically sorted tree with polytomies and multiple roots
    ancestor_ids = np.array(
        [0] + [rng.integers(0, i) for i in range(1, n)], dtype=np.int64
    )
    roots = rng.random(n) < 0.05
    ancestor_ids[roots] = np.flatnonzero(roots)
    return ancestor_ids


def _reference(ancestor_ids, absent, present):
    df = pd.DataFrame(
        {"id": np.arange(len(ancestor_ids)), "ancestor_id": ancestor_ids},
    )
    return pfl.alifestd_screen_trait_defined_clades_fitch_asexual(
        df, mask_trait_absent=absent, mask_trait_present=present
    ).astype(bool)


def test_simple():
    # root 0 with leaves 1 (absent) and node 2, which has present leaves 3, 4
    ancestor_ids = np.array([0, 0, 0, 2, 2])
    absent = np.array([False, True, False, False, False])
    present = np.array([False, False, False, True, True])
    np.testing.assert_array_equal(
        screen_mdcs_fitch(ancestor_ids, absent, present),
        [False, False, True, False, False],
    )


def test_conflicting_masks():
    with pytest.raises(ValueError):
        screen_mdcs_fitch(np.array([0]), np.array([True]), np.array([True]))


def test_unsorted():
    with pytest.raises(ValueError):
        screen_mdcs_fitch(
            np.array([1, 1]), np.array([True, False]), np.array([False, True])
        )


@pytest.mark.parametrize("seed", range(200))
def test_matches_phyloframe_random(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 300))
    ancestor_ids = _make_tree(rng, n)
    is_leaf = np.ones(n, dtype=bool)
    is_leaf[ancestor_ids[ancestor_ids != np.arange(n)]] = False
    # leaves only, or any node, may carry trait information
    has_info = is_leaf if rng.random() < 0.5 else rng.random(n) < 0.7
    present = has_info & (rng.random(n) < rng.random())
    absent = has_info & ~present
    np.testing.assert_array_equal(
        screen_mdcs_fitch(ancestor_ids, absent, present),
        _reference(ancestor_ids, absent, present),
    )
