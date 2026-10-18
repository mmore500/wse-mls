import numpy as np


def percentileofscore_grouped(
    values: np.ndarray,
    groups: np.ndarray,
    is_comparator: np.ndarray,
) -> np.ndarray:
    """Percentile-rank each value against comparator values in its group.

    Equivalent to `scipy.stats.percentileofscore(comps, v, kind="mean")`
    for each value `v`, with `comps` the values at `is_comparator` sharing
    `v`'s group, but vectorized over all groups at once. Entries in groups
    with no comparators are NaN.
    """
    values = np.asarray(values)
    is_comparator = np.asarray(is_comparator, dtype=bool)
    __, group_codes = np.unique(np.asarray(groups), return_inverse=True)
    __, value_codes = np.unique(values, return_inverse=True)

    # sortable composite keys: group major, value minor
    stride = np.int64(value_codes.max(initial=0) + 1)
    keys = group_codes.astype(np.int64) * stride + value_codes
    comp_keys = np.sort(keys[is_comparator])
    group_starts = group_codes.astype(np.int64) * stride

    first = np.searchsorted(comp_keys, group_starts, side="left")
    num_comps = np.searchsorted(comp_keys, group_starts + stride) - first
    num_below = np.searchsorted(comp_keys, keys, side="left") - first
    num_at_or_below = np.searchsorted(comp_keys, keys, side="right") - first

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 50.0 * (num_below + num_at_or_below) / num_comps
    return np.where(num_comps > 0, result, np.nan)
