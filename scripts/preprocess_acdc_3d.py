import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from acdc_h5 import SOURCE_SPACING_ATTR, TARGET_SPACING_ATTR, copy_attrs, read_spacing_zyx


def resample_to_spacing(volume, source_spacing_zyx, target_spacing_zyx, order):
    zoom_factors = np.asarray(source_spacing_zyx, dtype=np.float32) / np.asarray(target_spacing_zyx, dtype=np.float32)
    return zoom(volume, zoom_factors, order=order)


def center_crop_or_pad(volume, target_depth, target_height, target_width, pad_value):
    # First crop centrally if the source is larger than the target.
    depth, height, width = volume.shape
    start_d = max((depth - target_depth) // 2, 0)
    start_h = max((height - target_height) // 2, 0)
    start_w = max((width - target_width) // 2, 0)
    volume = volume[
        start_d : start_d + min(depth, target_depth),
        start_h : start_h + min(height, target_height),
        start_w : start_w + min(width, target_width),
    ]

    # Then pad centrally if the source is smaller than the target.
    depth, height, width = volume.shape
    output = np.full((target_depth, target_height, target_width), pad_value, dtype=volume.dtype)
    start_d = max((target_depth - depth) // 2, 0)
    start_h = max((target_height - height) // 2, 0)
    start_w = max((target_width - width) // 2, 0)
    output[start_d : start_d + depth, start_h : start_h + height, start_w : start_w + width] = volume
    return output


def normalize_image(image):
    # Normalize only the resized image content before padding so padded areas
    # remain exactly zero.
    std = image.std()
    if std < 1e-8:
        return image - image.mean()
    return (image - image.mean()) / std


def preprocess_volume(source_path, output_path, target_depth, target_height, target_width, target_spacing_zyx):
    with h5py.File(source_path, "r") as source:
        image = source["image"][:].astype(np.float32)
        label = source["label"][:].astype(np.uint8) if "label" in source else None
        source_spacing_zyx = read_spacing_zyx(source)

        image = resample_to_spacing(image, source_spacing_zyx, target_spacing_zyx, order=1)
        image = normalize_image(image)
        image = center_crop_or_pad(image, target_depth, target_height, target_width, pad_value=0).astype(np.float32)

        if label is not None:
            label = resample_to_spacing(label, source_spacing_zyx, target_spacing_zyx, order=0)
            label = center_crop_or_pad(label, target_depth, target_height, target_width, pad_value=0).astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as output:
            output.create_dataset("image", data=image, compression="gzip")
            if label is not None:
                output.create_dataset("label", data=label, compression="gzip")
            copy_attrs(source.attrs, output.attrs)
            output.attrs[SOURCE_SPACING_ATTR] = source_spacing_zyx.astype(np.float32)
            output.attrs[TARGET_SPACING_ATTR] = np.asarray(target_spacing_zyx, dtype=np.float32)
            output.attrs["spacing_zyx"] = np.asarray(target_spacing_zyx, dtype=np.float32)
            output.attrs["target_depth"] = target_depth
            output.attrs["target_height"] = target_height
            output.attrs["target_width"] = target_width
            output.attrs["preprocessing"] = "3d spacing resample, center pad/crop, zero-mean unit-variance image"


def parse_args():
    parser = argparse.ArgumentParser(description="Export ACDC preprocessed volumes for 3D U-Net training.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/acdc_h5_with_metadata/ACDC_training_volumes"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/acdc_preprocessed_3d_spacing/ACDC_training_volumes"))
    parser.add_argument("--target-depth", type=int, default=60)
    parser.add_argument("--target-height", type=int, default=204)
    parser.add_argument("--target-width", type=int, default=204)
    parser.add_argument("--target-spacing-z", type=float, default=5.0)
    parser.add_argument("--target-spacing-y", type=float, default=2.5)
    parser.add_argument("--target-spacing-x", type=float, default=2.5)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(args.input_dir.glob("*.h5"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {args.input_dir}")

    target_spacing_zyx = np.asarray(
        [args.target_spacing_z, args.target_spacing_y, args.target_spacing_x],
        dtype=np.float32,
    )
    for source_path in tqdm(files, desc="preprocess 3d"):
        output_path = args.output_dir / source_path.name
        preprocess_volume(
            source_path,
            output_path,
            target_depth=args.target_depth,
            target_height=args.target_height,
            target_width=args.target_width,
            target_spacing_zyx=target_spacing_zyx,
        )

    print(f"Exported {len(files)} volumes to {args.output_dir}")


if __name__ == "__main__":
    main()
