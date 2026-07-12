from pathlib import Path

import h5py
import numpy as np


SPACING_ATTR = "spacing_zyx"
SOURCE_SPACING_ATTR = "source_spacing_zyx"
TARGET_SPACING_ATTR = "target_spacing_zyx"


def as_float_list(value):
    return [float(item) for item in np.asarray(value).reshape(-1)]


def read_spacing_zyx(h5_file):
    for key in (SPACING_ATTR, TARGET_SPACING_ATTR, SOURCE_SPACING_ATTR):
        if key in h5_file.attrs:
            return np.asarray(h5_file.attrs[key], dtype=np.float32)
    raise KeyError(
        "HDF5 file is missing spacing metadata. Expected one of: "
        f"{SPACING_ATTR}, {TARGET_SPACING_ATTR}, {SOURCE_SPACING_ATTR}"
    )


def copy_attrs(source_attrs, target_attrs):
    for key, value in source_attrs.items():
        target_attrs[key] = value


def write_common_attrs(attrs, source_path, image_shape, spacing_zyx, extra_attrs=None):
    attrs["source_path"] = str(source_path)
    attrs["shape_zyx"] = np.asarray(image_shape, dtype=np.int32)
    attrs[SPACING_ATTR] = np.asarray(spacing_zyx, dtype=np.float32)
    if extra_attrs:
        for key, value in extra_attrs.items():
            attrs[key] = value


def inspect_h5_spacing(files, limit=20):
    rows = []
    for path in list(files)[:limit]:
        with h5py.File(path, "r") as h5_file:
            try:
                spacing = read_spacing_zyx(h5_file)
            except KeyError:
                spacing = None
            rows.append(
                {
                    "path": Path(path),
                    "shape": h5_file["image"].shape,
                    "spacing_zyx": None if spacing is None else as_float_list(spacing),
                }
            )
    return rows
