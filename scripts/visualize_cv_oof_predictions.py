"""Export a deterministic qualitative figure from out-of-fold 2D U-Net predictions.

Run this on the CUDA/checkpoint machine after spacing-aware 2D preprocessing.
Every displayed volume is predicted only by the fold for which its patient was
in validation.  The final example is the median-Dice ED volume, not a manually
selected best case.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_2d import group_slices, predict_slices
from scripts.evaluate_common import dice_score, keep_largest_connected_components_per_class, load_model


MASK_COLORS = ["#000000", "#2ca02c", "#ffbf00", "#d62728"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/unet2d_cv"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/acdc_preprocessed_2d_spacing/unet2d/ACDC_training_slices"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("report/figures_cv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-slices", type=int, default=5)
    return parser.parse_args()


def macro_dice(prediction, target):
    scores = [dice_score(prediction, target, class_id) for class_id in (1, 2, 3)]
    scores = [score for score in scores if np.isfinite(score)]
    return float(np.mean(scores))


def load_images(slice_paths):
    images = []
    indices = []
    for path in slice_paths:
        with h5py.File(path, "r") as h5_file:
            images.append(h5_file["image"][:].astype(np.float32))
            indices.append(int(h5_file.attrs.get("slice_index", len(indices))))
    return np.stack(images), indices


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on the checkpoint machine for this export.")
    files = sorted(args.data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No spacing-preprocessed slices found in {args.data_dir}")
    groups = group_slices(files, patient_filter=None)
    assignments = json.loads((args.run_dir / "fold_assignments.json").read_text(encoding="utf-8"))
    candidates = []

    for fold_spec in assignments["folds"]:
        fold = int(fold_spec["fold"])
        validation_patients = set(map(int, fold_spec["val_patients"]))
        model, _, _, checkpoint, device = load_model(args.run_dir / f"fold_{fold}", model_name="unet2d")
        for (patient, frame), slice_paths in groups.items():
            if patient not in validation_patients:
                continue
            prediction, target, _, phase = predict_slices(model, slice_paths, args.batch_size, device)
            prediction = keep_largest_connected_components_per_class(prediction)
            if phase == "ED":
                candidates.append(
                    {
                        "fold": fold,
                        "patient": patient,
                        "frame": frame,
                        "phase": phase,
                        "checkpoint": str(checkpoint),
                        "dice": macro_dice(prediction, target),
                        "prediction": prediction,
                        "target": target,
                        "slice_paths": slice_paths,
                    }
                )
        model.to("cpu")
        torch.cuda.empty_cache()

    if not candidates:
        raise RuntimeError("No ED out-of-fold volumes were found.")
    candidates.sort(key=lambda candidate: (candidate["dice"], candidate["patient"], candidate["frame"]))
    selected = candidates[len(candidates) // 2]
    images, slice_indices = load_images(selected["slice_paths"])
    foreground = np.flatnonzero(np.any(selected["target"] > 0, axis=(1, 2)))
    pool = foreground if len(foreground) else np.arange(len(images))
    positions = np.unique(np.linspace(0, len(pool) - 1, min(args.num_slices, len(pool))).round().astype(int))
    positions = pool[positions]

    cmap = ListedColormap(MASK_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(len(positions), 3, figsize=(10, 3.2 * len(positions)), squeeze=False)
    for row, position in enumerate(positions):
        image = images[position]
        low, high = np.percentile(image, [1, 99])
        axes[row, 0].imshow(image, cmap="gray", vmin=low, vmax=high)
        axes[row, 1].imshow(selected["prediction"][position], cmap=cmap, norm=norm, interpolation="nearest")
        axes[row, 2].imshow(selected["target"][position], cmap=cmap, norm=norm, interpolation="nearest")
        axes[row, 0].set_ylabel(f"Presek {slice_indices[position]}")
    for axis, title in zip(axes[0], ("MR slika", "OOF predikcija", "Ekspertska maska")):
        axis.set_title(title, fontweight="bold")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.legend(
        handles=[Patch(color=MASK_COLORS[index], label=label) for index, label in ((1, "RV"), (2, "Miokard"), (3, "LV"))],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"2D U-Net, out-of-fold | pacijent {selected['patient']:03d}, ED | volumetrijski Dice {selected['dice']:.3f}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "unet2d_oof_prediction_comparison.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")

    metadata = {key: value for key, value in selected.items() if key not in {"prediction", "target", "slice_paths"}}
    metadata.update({"selection": "median foreground Dice among all out-of-fold ED volumes", "slice_indices": [slice_indices[index] for index in positions]})
    metadata_path = args.output_dir / "unet2d_oof_prediction_comparison.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {figure_path} and {metadata_path}")


if __name__ == "__main__":
    main()
