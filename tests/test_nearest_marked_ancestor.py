import numpy as np
import pytest

from pylib import nearest_marked_ancestor


def _reference(ancestor_ids, mask):
    result = []
    for idx in range(len(ancestor_ids)):
        node = idx
        while not mask[node] and ancestor_ids[node] != node:
            node = ancestor_ids[node]
        result.append(node if mask[node] else -1)
    return np.array(result)


def test_simple():
    # 0 -> 1 -> 2, 0 -> 3; 4 is a separate root
    ancestor_ids = np.array([0, 0, 1, 0, 4])
    mask = np.array([False, True, False, False, False])
    np.testing.assert_array_equal(
        nearest_marked_ancestor(ancestor_ids, mask), [-1, 1, 1, -1, -1]
    )


def test_nested():
    ancestor_ids = np.array([0, 0, 1, 2])
    mask = np.array([True, False, True, False])
    np.testing.assert_array_equal(
        nearest_marked_ancestor(ancestor_ids, mask), [0, 0, 2, 2]
    )


def test_unsorted():
    with pytest.raises(ValueError):
        nearest_marked_ancestor(np.array([1, 1]), np.array([True, False]))


@pytest.mark.parametrize("seed", range(50))
def test_matches_reference_random(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 200))
    ancestor_ids = np.array(
        [0] + [rng.integers(0, i) for i in range(1, n)], dtype=np.int64
    )
    roots = rng.random(n) < 0.05
    ancestor_ids[roots] = np.flatnonzero(roots)
    mask = rng.random(n) < rng.random()
    np.testing.assert_array_equal(
        nearest_marked_ancestor(ancestor_ids, mask),
        _reference(ancestor_ids, mask),
    )
