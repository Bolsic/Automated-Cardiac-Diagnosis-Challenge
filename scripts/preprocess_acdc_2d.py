import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from acdc_h5 import SOURCE_SPACING_ATTR, TARGET_SPACING_ATTR, copy_attrs, read_spacing_zyx


ARCHITECTURE_DEFAULTS = {
    "fcn8": {"height": 224, "width": 224},
    "unet2d": {"height": 396, "width": 396},
}


def resample_in_plane(array, source_spacing_yx, target_spacing_yx, order):
    zoom_y = float(source_spacing_yx[0] / target_spacing_yx[0])
    zoom_x = float(source_spacing_yx[1] / target_spacing_yx[1])
    return zoom(array, (zoom_y, zoom_x), order=order)


def center_crop_or_pad_2d(array, target_height, target_width, pad_value):
    height, width = array.shape
    start_h = max((height - target_height) // 2, 0)
    start_w = max((width - target_width) // 2, 0)
    array = array[
        start_h : start_h + min(height, target_height),
        start_w : start_w + min(width, target_width),
    ]

    height, width = array.shape
    output = np.full((target_height, target_width), pad_value, dtype=array.dtype)
    start_h = max((target_height - height) // 2, 0)
    start_w = max((target_width - width) // 2, 0)
    output[start_h : start_h + height, start_w : start_w + width] = array
    return output


def normalize_image(image):
    image = image.astype(np.float32)
    std = image.std()
    if std < 1e-8:
        normalized = image - image.mean()
    else:
        normalized = (image - image.mean()) / std
    return normalized.astype(np.float32), {"mean": float(image.mean()), "std": float(std)}


def preprocess_slice(source_path, output_path, target_height, target_width, target_spacing_yx):
    with h5py.File(source_path, "r") as source:
        image = source["image"][:].astype(np.float32)
        label = source["label"][:].astype(np.uint8) if "label" in source else None
        spacing_zyx = read_spacing_zyx(source)
        source_spacing_yx = spacing_zyx[1:]

        image = resample_in_plane(image, source_spacing_yx, target_spacing_yx, order=1)
        image, stats = normalize_image(image)
        image = center_crop_or_pad_2d(image, target_height, target_width, pad_value=0).astype(np.float32)

        if label is not None:
            label = resample_in_plane(label, source_spacing_yx, target_spacing_yx, order=0)
            label = center_crop_or_pad_2d(label, target_height, target_width, pad_value=0).astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as output:
            output.create_dataset("image", data=image, compression="gzip")
            if label is not None:
                output.create_dataset("label", data=label, compression="gzip")
            copy_attrs(source.attrs, output.attrs)
            output.attrs[SOURCE_SPACING_ATTR] = spacing_zyx.astype(np.float32)
            output.attrs[TARGET_SPACING_ATTR] = np.asarray(
                [spacing_zyx[0], target_spacing_yx[0], target_spacing_yx[1]],
                dtype=np.float32,
            )
            output.attrs["spacing_zyx"] = output.attrs[TARGET_SPACING_ATTR]
            output.attrs["target_height"] = target_height
            output.attrs["target_width"] = target_width
            output.attrs["normalization_mean"] = stats["mean"]
            output.attrs["normalization_std"] = stats["std"]
            output.attrs["preprocessing"] = "in-plane spacing resample, center pad/crop, zero-mean unit-variance image"


def parse_args():
    parser = argparse.ArgumentParser(description="Create spacing-aware 2D ACDC HDF5 training slices.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/acdc_h5_with_metadata/ACDC_training_slices"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/acdc_preprocessed_2d_spacing"))
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURE_DEFAULTS), default="fcn8")
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--target-spacing-y", type=float, default=1.37)
    parser.add_argument("--target-spacing-x", type=float, default=1.37)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = ARCHITECTURE_DEFAULTS[args.architecture]
    target_height = args.target_height or defaults["height"]
    target_width = args.target_width or defaults["width"]
    output_dir = args.output_root / args.architecture / "ACDC_training_slices"

    files = sorted(args.input_dir.glob("*.h5"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {args.input_dir}")

    target_spacing_yx = np.asarray([args.target_spacing_y, args.target_spacing_x], dtype=np.float32)
    for source_path in tqdm(files, desc=f"preprocess 2d {args.architecture}"):
        preprocess_slice(
            source_path,
            output_dir / source_path.name,
            target_height,
            target_width,
            target_spacing_yx,
        )

    print(f"Exported {len(files)} slices to {output_dir}")


if __name__ == "__main__":
    main()
