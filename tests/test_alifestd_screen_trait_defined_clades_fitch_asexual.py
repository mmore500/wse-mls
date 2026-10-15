"""Regression tests for phyloframe's Fitch MDC screen.

phyloframe < 0.10.3 locked each node's Fitch state set as soon as any child
was resolved, missing some MDCs and placing others too high. These tests
check the library against a recursive textbook Fitch implementation.
"""

import numpy as np
import pandas as pd
from phyloframe import legacy as pfl
import pytest


def _screen(ancestor_ids, absent, present):
    phylogeny_df = pd.DataFrame(
        {"id": np.arange(len(ancestor_ids)), "ancestor_id": ancestor_ids},
    )
    return np.flatnonzero(
        pfl.alifestd_screen_trait_defined_clades_fitch_asexual(
            phylogeny_df,
            mask_trait_absent=np.asarray(absent, dtype=bool),
            mask_trait_present=np.asarray(present, dtype=bool),
        )
    ).tolist()


def _reference(ancestor_ids, absent, present):
    """Recursive textbook Fitch, for comparison."""
    n = len(ancestor_ids)
    children = [[] for __ in range(n)]
    for i, a in enumerate(ancestor_ids):
        if a != i:
            children[a].append(i)

    sets = [None] * n

    def up(i):
        if absent[i]:
            sets[i] = {0}
        elif present[i]:
            sets[i] = {1}
        else:
            counts = {0: 0, 1: 0}
            for c in children[i]:
                up(c)
                for s in sets[c]:
                    counts[s] += 1
            best = max(counts.values())
            sets[i] = {s for s, k in counts.items() if k == best}

    states = [None] * n

    def down(i, parent_state):
        if parent_state is None:
            states[i] = 0 if 0 in sets[i] else 1
        else:
            states[i] = (
                parent_state
                if parent_state in sets[i]
                else (next(iter(sets[i])))
            )
        for c in children[i]:
            down(c, states[i])

    for i in range(n):
        if ancestor_ids[i] == i:
            up(i)
            down(i, None)
    return [
        i
        for i in range(n)
        if ancestor_ids[i] != i and states[i] and not states[ancestor_ids[i]]
    ]


#        0
#       / \
#      1   2
#     / \
#    3   4
#   / \
#  5   6
TREE = [0, 0, 0, 1, 1, 3, 3]
LEAVES = np.array([0, 0, 1, 0, 1, 1, 1], dtype=bool)


@pytest.mark.parametrize(
    "present, expected",
    [
        ([0, 0, 0, 0, 1, 1, 1], [1]),
        ([0, 0, 1, 0, 0, 0, 0], [2]),
        ([0, 0, 0, 0, 1, 0, 0], [4]),
        ([0, 0, 0, 0, 0, 1, 1], [3]),
        ([0, 0, 0, 0, 0, 0, 0], []),
    ],
)
def test_hand_cases(present, expected):
    present = LEAVES & np.array(present, dtype=bool)
    assert _screen(TREE, LEAVES & ~present, present) == expected


def test_multifurcation_majority():
    # root with three leaf children, two present: root is assigned present
    # by majority (root gains are not counted) and the absent leaf is a loss
    tree = [0, 0, 0, 0]
    leaves = np.array([0, 1, 1, 1], dtype=bool)
    present = np.array([0, 1, 1, 0], dtype=bool)
    assert _screen(tree, leaves & ~present, present) == []
    # nested: inner node with two present of three leaves under absent root
    tree = [0, 0, 0, 1, 1, 1]
    leaves = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
    present = np.array([0, 0, 0, 1, 1, 0], dtype=bool)
    assert _screen(tree, leaves & ~present, present) == [1]


def test_unobserved_leaf_does_not_constrain():
    #   0
    #  / \
    # 1   2 (unobserved leaf)
    tree = [0, 0, 0]
    absent = np.array([0, 0, 0], dtype=bool)
    present = np.array([0, 1, 0], dtype=bool)
    assert _screen(tree, absent, present) == []  # root gain not counted


def test_forest():
    tree = [0, 0, 0, 3, 3, 3]
    leaves = np.array([0, 1, 1, 0, 1, 1], dtype=bool)
    present = np.array([0, 1, 0, 0, 0, 1], dtype=bool)
    assert _screen(tree, leaves & ~present, present) == [1, 5]


def test_empty():
    assert _screen([], [], []) == []


def test_conflicting_masks_raise():
    with pytest.raises(ValueError):
        _screen([0, 0], [0, 1], [0, 1])


@pytest.mark.parametrize("seed", range(200))
def test_matches_reference_on_random_trees(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    roots = rng.random(n) < 0.05
    roots[0] = True
    tree = [i if roots[i] else int(rng.integers(0, i)) for i in range(n)]
    has_child = np.zeros(n, dtype=bool)
    for i, a in enumerate(tree):
        if a != i:
            has_child[a] = True
    leaves = ~has_child
    present = leaves & (rng.random(n) < rng.random())
    observed = leaves & (rng.random(n) < 0.95)
    absent = observed & ~present
    present &= observed
    assert _screen(tree, absent, present) == _reference(tree, absent, present)
