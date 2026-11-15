import numpy as np

from training.dataset import create_spherical_mask, extract_patch_2_5d


def test_reported_right_boundary_annotation_is_centered_and_zero_padded():
    # Broadcasted coordinates keep the exact reported volume geometry without
    # allocating a 116 MB dense array.
    volume = np.broadcast_to(
        np.arange(310, dtype=np.float32)[None, None, :],
        (301, 310, 310),
    )
    center_xyz = np.array([289, 156, 81])

    patch = extract_patch_2_5d(volume, center_xyz, patch_size=64)

    assert patch is not None
    assert patch.shape == (3, 64, 64)
    assert patch[1, 32, 32] == volume[81, 156, 289]
    assert patch[1, 32, 31] == volume[81, 156, 288]
    assert patch[1, 32, 52] == volume[81, 156, 309]
    assert np.all(patch[:, :, 53:] == 0)  # the 11 out-of-volume X columns

    mask = create_spherical_mask(64, center_xyz, diameter_mm=4, spacing=np.ones(3))
    assert mask[32, 32] == 1


def test_left_boundary_pads_without_shifting_valid_pixels():
    volume = np.broadcast_to(
        np.arange(7, dtype=np.float32)[None, None, :], (3, 5, 7)
    )
    patch = extract_patch_2_5d(volume, np.array([2, 2, 1]), patch_size=6)

    assert patch is not None
    assert patch.shape == (3, 6, 6)
    assert np.all(patch[:, :, 0] == 0)
    np.testing.assert_array_equal(patch[1, 3, 1:6], np.arange(5, dtype=np.float32))
    assert patch[1, 3, 3] == volume[1, 2, 2]


def test_top_and_bottom_y_boundaries_are_zero_padded():
    volume = np.broadcast_to(
        np.arange(5, dtype=np.float32)[None, :, None], (3, 5, 7)
    )

    top = extract_patch_2_5d(volume, np.array([3, 0, 1]), patch_size=6)
    assert top is not None
    assert np.all(top[:, :3, :] == 0)
    np.testing.assert_array_equal(top[1, 3:, 3], np.arange(3, dtype=np.float32))
    assert top[1, 3, 3] == volume[1, 0, 3]

    bottom = extract_patch_2_5d(volume, np.array([3, 4, 1]), patch_size=6)
    assert bottom is not None
    assert np.all(bottom[:, 4:, :] == 0)
    np.testing.assert_array_equal(bottom[1, :4, 3], np.arange(1, 5, dtype=np.float32))
    assert bottom[1, 3, 3] == volume[1, 4, 3]


def test_first_and_last_z_slice_pad_only_the_outside_neighbor():
    volume = np.broadcast_to(
        np.arange(3, dtype=np.float32)[:, None, None], (3, 5, 5)
    )

    first = extract_patch_2_5d(volume, np.array([2, 2, 0]), patch_size=4)
    assert first is not None
    assert np.all(first[0] == 0)
    assert np.all(first[1] == 0)
    assert np.all(first[2] == 1)

    last = extract_patch_2_5d(volume, np.array([2, 2, 2]), patch_size=4)
    assert last is not None
    assert np.all(last[0] == 1)
    assert np.all(last[1] == 2)
    assert np.all(last[2] == 0)


def test_center_outside_volume_returns_none():
    volume = np.ones((3, 5, 7), dtype=np.float32)
    for center in (
        np.array([-1, 2, 1]),
        np.array([7, 2, 1]),
        np.array([3, -1, 1]),
        np.array([3, 5, 1]),
        np.array([3, 2, -1]),
        np.array([3, 2, 3]),
    ):
        assert extract_patch_2_5d(volume, center, patch_size=4) is None


def test_interior_patch_matches_previous_extraction_numerically():
    volume = np.arange(5 * 9 * 11, dtype=np.float32).reshape(5, 9, 11)
    center = np.array([5, 4, 2])
    expected = np.stack(
        [volume[z, 1:7, 2:8] for z in (1, 2, 3)], axis=0
    )

    actual = extract_patch_2_5d(volume, center, patch_size=6)

    np.testing.assert_array_equal(actual, expected)
