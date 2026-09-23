import h5py
import numpy as np
import pytest

from scripts.preprocess_acdc_2d import preprocess_slice
from scripts.preprocess_acdc_3d import preprocess_volume


def _write_source(path, image, label, spacing):
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("image", data=image)
        h5_file.create_dataset("label", data=label)
        h5_file.attrs["spacing_zyx"] = spacing
        h5_file.attrs["diagnosis"] = "NOR"


def test_2d_preprocessing_preserves_target_spacing_and_label_classes(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "output.h5"
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    label = np.zeros((4, 4), dtype=np.uint8)
    label[1:3, 1:3] = 3
    _write_source(source, image, label, [10.0, 2.0, 2.0])

    preprocess_slice(source, output, 8, 8, np.asarray([1.0, 1.0], dtype=np.float32))
    with h5py.File(output, "r") as h5_file:
        assert h5_file["image"].shape == (8, 8)
        assert h5_file["label"].shape == (8, 8)
        assert set(np.unique(h5_file["label"][:])) <= {0, 3}
        assert np.allclose(h5_file.attrs["source_spacing_zyx"], [10.0, 2.0, 2.0])
        assert np.allclose(h5_file.attrs["target_spacing_zyx"], [10.0, 1.0, 1.0])


def test_3d_preprocessing_preserves_target_spacing_and_label_classes(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "output.h5"
    image = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 2
    _write_source(source, image, label, [10.0, 2.0, 2.0])

    preprocess_volume(source, output, 8, 8, 8, np.asarray([5.0, 1.0, 1.0], dtype=np.float32))
    with h5py.File(output, "r") as h5_file:
        assert h5_file["image"].shape == (8, 8, 8)
        assert h5_file["label"].shape == (8, 8, 8)
        assert set(np.unique(h5_file["label"][:])) <= {0, 2}
        assert np.allclose(h5_file.attrs["target_spacing_zyx"], [5.0, 1.0, 1.0])


def test_preprocessing_rejects_invalid_spacing(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "output.h5"
    _write_source(source, np.zeros((4, 4)), np.zeros((4, 4)), [10.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="Invalid spacing"):
        preprocess_slice(source, output, 4, 4, np.asarray([1.0, 1.0], dtype=np.float32))
