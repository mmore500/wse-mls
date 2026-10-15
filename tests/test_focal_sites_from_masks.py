from pylib import focal_sites_from_masks


def test_no_focal_sites():
    assert focal_sites_from_masks([0] * 8) == []


def test_single_focal_site():
    assert focal_sites_from_masks([0, 0, 0, 1, 0, 0, 0, 0]) == [24]


def test_multiple_focal_sites():
    assert focal_sites_from_masks([0b101, 0, 0, 0, 0, 0, 0, 0x80]) == [
        0,
        2,
        63,
    ]


def test_float_mask_values():
    assert focal_sites_from_masks([0.0, 2.0]) == [9]
