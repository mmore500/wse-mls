import numba
import numpy as np


@numba.njit(cache=True)
def _nearest_marked_ancestor_jit(
    ancestor_ids: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    result = np.full(len(ancestor_ids), -1, dtype=np.int64)
    for idx in range(len(ancestor_ids)):
        if mask[idx]:
            result[idx] = idx
        elif ancestor_ids[idx] != idx:
            result[idx] = result[ancestor_ids[idx]]
    return result


def nearest_marked_ancestor(
    ancestor_ids: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Find each node's nearest marked ancestor, counting the node itself.

    Requires contiguous ids (`ancestor_ids[i]` is node `i`'s ancestor, with
    roots their own ancestor) and topological sorting (`ancestor_ids[i] <=
    i`), as in phyloframe working format.

    Returns an array of the nearest marked ancestor's id for each node, or
    -1 where no ancestor (nor the node itself) is marked.
    """
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    if (ancestor_ids > np.arange(len(ancestor_ids))).any():
        raise ValueError("phylogeny must be topologically sorted")
    return _nearest_marked_ancestor_jit(ancestor_ids, mask)
