import typing

import numpy as np


def focal_sites_from_masks(mask_bytes: typing.Iterable[int]) -> list[int]:
    """Decode `flag_is_focal_mask_byte{B}` values into trait site indices.

    Bit `b` of byte `B` corresponds to trait site `B * 8 + b`, matching the
    `trait_num{B * 8 + b}` columns assigned during wse-async-ga phylogeny
    build.
    """
    bits = np.unpackbits(
        np.asarray(list(mask_bytes), dtype=np.uint8), bitorder="little"
    )
    return np.flatnonzero(bits).tolist()
