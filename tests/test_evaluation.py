import h5py
import numpy as np
import pytest

from scripts.evaluate_common import (
    dice_score,
    keep_largest_connected_component,
    read_spacing_or_default,
)


def test_dice_score_for_partial_overlap():
    prediction = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [1, 0]])
    assert dice_score(prediction, target, class_id=1) == pytest.approx(0.5)


def test_largest_connected_component_removes_island():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[2:4, 2:4] = True
    cleaned = keep_largest_connected_component(mask)
    assert cleaned.sum() == 4
    assert not cleaned[0, 0]


def test_spacing_is_read_in_array_axis_order(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["spacing_zyx"] = [5.0, 1.37, 1.37]
        assert np.allclose(read_spacing_or_default(h5_file, ndim=3), [5.0, 1.37, 1.37])
        assert np.allclose(read_spacing_or_default(h5_file, ndim=2), [1.37, 1.37])


def test_missing_spacing_is_rejected(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5_file:
        with pytest.raises(ValueError, match="Physical spacing metadata"):
            read_spacing_or_default(h5_file, ndim=3)
