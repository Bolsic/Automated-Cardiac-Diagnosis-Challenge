import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

from acdc_h5 import write_common_attrs


def parse_info_cfg(path):
    info = {}
    if not path.exists():
        return info
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def load_nifti_zyx(path, dtype):
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=dtype)
    # NIfTI is X x Y x Z; project HDF5 volumes are Z x Y x X.
    data = np.transpose(data, (2, 1, 0))
    spacing_xyz = np.asarray(image.header.get_zooms()[:3], dtype=np.float32)
    spacing_zyx = spacing_xyz[::-1]
    return image, data, spacing_xyz, spacing_zyx


def find_frame_images(patient_dir):
    return sorted(
        path
        for path in patient_dir.glob("*_frame*.nii.gz")
        if not path.name.endswith("_gt.nii.gz")
    )


def frame_number(path):
    return int(path.name.split("_frame", 1)[1].split(".", 1)[0])


def nifti_stem(path):
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def split_name(split_dir_name):
    return "training" if split_dir_name == "training" else "testing"


def save_volume(output_path, image, label, attrs):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("image", data=image, compression="gzip")
        if label is not None:
            h5_file.create_dataset("label", data=label, compression="gzip")
        for key, value in attrs.items():
            h5_file.attrs[key] = value


def save_slices(output_dir, stem, image, label, attrs):
    output_dir.mkdir(parents=True, exist_ok=True)
    spacing_zyx = np.asarray(attrs["spacing_zyx"], dtype=np.float32)
    for slice_index in range(image.shape[0]):
        output_path = output_dir / f"{stem}_slice_{slice_index}.h5"
        with h5py.File(output_path, "w") as h5_file:
            h5_file.create_dataset("image", data=image[slice_index], compression="gzip")
            if label is not None:
                h5_file.create_dataset("label", data=label[slice_index], compression="gzip")
            for key, value in attrs.items():
                h5_file.attrs[key] = value
            h5_file.attrs["slice_index"] = slice_index
            h5_file.attrs["slice_location_mm"] = float(slice_index * spacing_zyx[0])


def convert_frame(frame_path, output_root, source_root, export_slices):
    patient_dir = frame_path.parent
    patient_id = patient_dir.name
    split = split_name(patient_dir.parent.name)
    info = parse_info_cfg(patient_dir / "Info.cfg")

    gt_path = frame_path.with_name(frame_path.name.replace(".nii.gz", "_gt.nii.gz"))
    image_nii, image, spacing_xyz, spacing_zyx = load_nifti_zyx(frame_path, np.float32)
    label = None
    if gt_path.exists():
        _, label, gt_spacing_xyz, _ = load_nifti_zyx(gt_path, np.uint8)
        if not np.allclose(spacing_xyz, gt_spacing_xyz):
            raise ValueError(f"Image/label spacing mismatch for {frame_path}")

    frame = frame_number(frame_path)
    phase = "ED" if info.get("ED") == str(frame) else "ES" if info.get("ES") == str(frame) else "unknown"
    relative_source = frame_path.relative_to(source_root)
    extra_attrs = {
        "patient_id": patient_id,
        "frame": frame,
        "phase": phase,
        "split": split,
        "spacing_xyz": spacing_xyz,
        "affine": image_nii.affine.astype(np.float32),
        "pixdim": np.asarray(image_nii.header["pixdim"], dtype=np.float32),
    }
    if "Group" in info:
        extra_attrs["diagnosis"] = info["Group"]

    attrs = {}
    write_common_attrs(attrs, relative_source, image.shape, spacing_zyx, extra_attrs)

    collection = f"ACDC_{split}_volumes"
    stem = nifti_stem(frame_path)
    output_path = output_root / collection / f"{stem}.h5"
    save_volume(output_path, image, label, attrs)

    if export_slices:
        save_slices(output_root / f"ACDC_{split}_slices", stem, image, label, attrs)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ACDC NIfTI files to HDF5 while preserving spacing metadata.")
    parser.add_argument("--input-root", type=Path, default=Path("ACDC/database"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/acdc_h5_with_metadata"))
    parser.add_argument("--no-slices", action="store_true", help="Only write volume HDF5 files.")
    parser.add_argument("--max-patients", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    patient_dirs = []
    for split_dir in (args.input_root / "training", args.input_root / "testing"):
        patient_dirs.extend(sorted(path for path in split_dir.glob("patient*") if path.is_dir()))
    if args.max_patients is not None:
        patient_dirs = patient_dirs[: args.max_patients]
    if not patient_dirs:
        raise FileNotFoundError(f"No ACDC patient folders found under {args.input_root}")

    frames = []
    for patient_dir in patient_dirs:
        frames.extend(find_frame_images(patient_dir))
    for frame_path in tqdm(frames, desc="convert nifti to h5"):
        convert_frame(frame_path, args.output_root, args.input_root, export_slices=not args.no_slices)

    print(f"Converted {len(frames)} frame volumes to {args.output_root}")


if __name__ == "__main__":
    main()
