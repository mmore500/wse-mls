import numpy as np


def screen_mdcs_fitch(
    ancestor_ids: np.ndarray,
    node_depths: np.ndarray,
    mask_trait_absent: np.ndarray,
    mask_trait_present: np.ndarray,
) -> np.ndarray:
    """Identify mutation-defined clade (MDC) roots by Fitch parsimony.

    Nodes are processed one depth level at a time, deepest first, so each
    node's state set is computed only after all of its children are
    resolved. Multifurcations use the generalized Fitch rule, taking the
    state(s) most common among children. Nodes in neither mask (e.g., inner
    nodes) are inferred; leaves in neither mask do not constrain their
    parent. Ambiguous roots are assigned trait-absent, and a root carrying
    the trait is not counted as an MDC.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Ancestor row index of each node, with contiguous ids (i.e., id equals
        row index) and roots as self-loops.
    node_depths : np.ndarray
        Depth of each node, with roots at depth 0.
    mask_trait_absent, mask_trait_present : np.ndarray
        Boolean masks of nodes observed without and with the trait.

    Returns
    -------
    np.ndarray
        Boolean mask of nodes where the trait is gained, i.e., nodes with
        trait present whose parent has trait absent.
    """
    ancestor_ids = np.asarray(ancestor_ids)
    node_depths = np.asarray(node_depths)
    mask_trait_absent = np.asarray(mask_trait_absent, dtype=bool)
    mask_trait_present = np.asarray(mask_trait_present, dtype=bool)
    if (mask_trait_absent & mask_trait_present).any():
        raise ValueError("conflicting trait presence/absence masks")

    n = len(ancestor_ids)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # state sets as bit flags: 1 is trait absent, 2 is trait present
    sets = mask_trait_absent.astype(np.uint8) + 2 * mask_trait_present.astype(
        np.uint8
    )
    observed = sets != 0
    is_root = ancestor_ids == np.arange(n)

    order = np.argsort(node_depths, kind="stable")
    boundaries = np.searchsorted(
        node_depths[order], np.arange(node_depths.max() + 2)
    )
    levels = [
        order[begin:end] for begin, end in zip(boundaries, boundaries[1:])
    ]

    # bottom-up pass, deepest level first
    for children in reversed(levels[1:]):
        parents, inverse = np.unique(
            ancestor_ids[children], return_inverse=True
        )
        child_sets = sets[children]
        num_absent = np.bincount(inverse, weights=child_sets & 1)
        num_present = np.bincount(inverse, weights=(child_sets & 2) >> 1)
        proposed = (num_absent >= num_present).astype(np.uint8) + 2 * (
            num_present >= num_absent
        ).astype(np.uint8)
        unobserved = ~observed[parents]
        sets[parents[unobserved]] = proposed[unobserved]

    # top-down pass, shallowest level first
    states = np.zeros(n, dtype=bool)  # True is trait present
    roots = np.flatnonzero(is_root)
    states[roots] = (sets[roots] & 1) == 0
    for children in levels[1:]:
        parent_states = states[ancestor_ids[children]]
        parent_bits = np.where(parent_states, 2, 1).astype(np.uint8)
        child_sets = sets[children]
        keeps_parent = (child_sets & parent_bits) != 0
        states[children] = np.where(
            keeps_parent, parent_states, (child_sets & 2) != 0
        )

    return states & ~states[ancestor_ids] & ~is_root
