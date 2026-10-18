import numba
import numpy as np


@numba.njit(cache=True)
def _screen_mdcs_fitch_jit(
    ancestor_ids: np.ndarray,
    trait_sets: np.ndarray,
) -> np.ndarray:
    n = len(ancestor_ids)
    num_absent = np.zeros(n, dtype=np.int64)
    num_present = np.zeros(n, dtype=np.int64)
    intersect = np.full(n, 3, dtype=np.uint8)
    union = trait_sets.copy()

    # bottom-up pass: children have higher ids than parents, so are
    # resolved before them in reverse id order
    for idx in range(n - 1, -1, -1):
        if trait_sets[idx] == 0:
            if num_absent[idx] == num_present[idx]:
                if intersect[idx] != 0:
                    trait_sets[idx] = intersect[idx]
                else:
                    trait_sets[idx] = union[idx]
            elif num_absent[idx] > num_present[idx]:
                trait_sets[idx] = 1
            else:
                trait_sets[idx] = 2

        ancestor_id = ancestor_ids[idx]
        if ancestor_id != idx:
            num_absent[ancestor_id] += trait_sets[idx] & 1
            num_present[ancestor_id] += trait_sets[idx] >> 1
            intersect[ancestor_id] &= trait_sets[idx]
            union[ancestor_id] |= trait_sets[idx]

    # top-down pass: resolve ambiguity in favor of parent state, with
    # ambiguous roots assumed trait absent
    is_mdc = np.zeros(n, dtype=np.bool_)
    for idx in range(n):
        ancestor_id = ancestor_ids[idx]
        if ancestor_id == idx:
            if trait_sets[idx] == 3:
                trait_sets[idx] = 1
            continue

        common = trait_sets[idx] & trait_sets[ancestor_id]
        if common != 0 or trait_sets[idx] == 0:
            trait_sets[idx] = common
        is_mdc[idx] = trait_sets[idx] == 2 and trait_sets[ancestor_id] != 2

    return is_mdc


def screen_mdcs_fitch(
    ancestor_ids: np.ndarray,
    mask_trait_absent: np.ndarray,
    mask_trait_present: np.ndarray,
) -> np.ndarray:
    """Mark mutation-defined clades (MDCs) by Fitch maximum parsimony.

    Equivalent to phyloframe's
    `alifestd_screen_trait_defined_clades_fitch_asexual` (including its
    majority-vote resolution at polytomies and assumed trait-absent roots),
    but runs in time linear in the number of nodes rather than in nodes
    times tree depth.

    Requires contiguous ids (`ancestor_ids[i]` is node `i`'s ancestor, with
    roots their own ancestor) and topological sorting (`ancestor_ids[i] <=
    i`), as in phyloframe working format.

    Returns a boolean array marking nodes where the trait is gained, i.e.,
    nodes inferred to have the trait whose ancestor is inferred not to.
    """
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    mask_trait_absent = np.asarray(mask_trait_absent, dtype=bool)
    mask_trait_present = np.asarray(mask_trait_present, dtype=bool)
    if (mask_trait_absent & mask_trait_present).any():
        raise ValueError("conflicting trait presence/absence masks")
    if (ancestor_ids > np.arange(len(ancestor_ids))).any():
        raise ValueError("phylogeny must be topologically sorted")

    trait_sets = mask_trait_absent.astype(
        np.uint8
    ) + 2 * mask_trait_present.astype(np.uint8)
    return _screen_mdcs_fitch_jit(ancestor_ids, trait_sets)
