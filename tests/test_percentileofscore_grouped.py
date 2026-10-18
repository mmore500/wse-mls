import numpy as np
from scipy import stats as scipy_stats

from pylib import percentileofscore_grouped


def _reference(values, groups, is_comparator):
    result = np.full(len(values), np.nan)
    for group in np.unique(groups):
        in_group = groups == group
        comps = values[in_group & is_comparator]
        if len(comps):
            result[in_group] = scipy_stats.percentileofscore(
                comps, values[in_group], kind="mean"
            )
    return result


def test_single_group():
    values = np.array([1, 2, 2, 3, 5])
    is_comparator = np.array([True, True, True, False, True])
    np.testing.assert_allclose(
        percentileofscore_grouped(values, np.zeros(5), is_comparator),
        [12.5, 50.0, 50.0, 75.0, 87.5],
    )


def test_group_without_comparators():
    result = percentileofscore_grouped(
        np.array([1.0, 2.0, 3.0]),
        np.array([0, 0, 1]),
        np.array([True, False, False]),
    )
    np.testing.assert_allclose(result, [50.0, 100.0, np.nan])


def test_empty():
    result = percentileofscore_grouped(
        np.array([], dtype=float),
        np.array([], dtype=int),
        np.array([], dtype=bool),
    )
    assert result.shape == (0,)


def test_matches_scipy_random():
    rng = np.random.default_rng(1)
    for __ in range(100):
        n = rng.integers(1, 200)
        values = rng.integers(0, 10, n).astype(float) * rng.choice([1, 0.5])
        groups = rng.integers(0, 5, n)
        is_comparator = rng.random(n) < rng.random()
        np.testing.assert_allclose(
            percentileofscore_grouped(values, groups, is_comparator),
            _reference(values, groups, is_comparator),
        )
