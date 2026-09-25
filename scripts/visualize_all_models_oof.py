"""Export a single-case out-of-fold qualitative comparison for all four models.

The default case is patient 002 in ED (frame 1), the deterministic median-Dice
case previously selected from standard 2D U-Net out-of-fold predictions.  Each
model uses only the checkpoint of the fold where that patient was held out.
Run this script on the CUDA machine that contains all CV checkpoints and the
spacing-aware 2D and 3D preprocessing outputs.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_2d import group_slices, predict_slices
from scripts.evaluate_common import dice_score, keep_largest_connected_components_per_class, load_model, read_phase_or_unknown


MASK_COLORS = ["#000000", "#2ca02c", "#ffbf00", "#d62728"]
MODEL_SPECS = (
    ("fcn8", "FCN-8", "fcn8_cv", "fcn8"),
    ("unet2d", "2D U-Net", "unet2d_cv", "unet2d"),
    ("unet2d_modified", "Modifikovani 2D U-Net", "unet2d_modified_cv", "unet2d"),
    ("unet3d", "Anizotropni 3D U-Net", "unet3d_cv", None),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--fcn8-data-dir",
        type=Path,
        default=Path("outputs/acdc_preprocessed_2d_spacing/fcn8/ACDC_training_slices"),
    )
    parser.add_argument(
        "--unet2d-data-dir",
        type=Path,
        default=Path("outputs/acdc_preprocessed_2d_spacing/unet2d/ACDC_training_slices"),
    )
    parser.add_argument(
        "--unet3d-data-dir",
        type=Path,
        default=Path("outputs/acdc_preprocessed_3d_spacing/ACDC_training_volumes"),
    )
    parser.add_argument("--patient", type=int, default=2)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("report/figures_cv"))
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def macro_dice(prediction, target):
    scores = [dice_score(prediction, target, class_id) for class_id in (1, 2, 3)]
    return float(np.nanmean(scores))


def fold_for_patient(run_dir, patient):
    assignments = json.loads((run_dir / "fold_assignments.json").read_text(encoding="utf-8"))
    matches = [int(spec["fold"]) for spec in assignments["folds"] if patient in set(map(int, spec["val_patients"]))]
    if len(matches) != 1:
        raise ValueError(f"Expected patient {patient:03d} in exactly one validation fold of {run_dir}, found {matches}.")
    return matches[0]


def predict_2d(run_dir, model_name, data_dir, patient, frame, batch_size):
    files = sorted(data_dir.glob("*.h5"))
    groups = group_slices(files, patient_filter={patient})
    try:
        slice_paths = groups[(patient, frame)]
    except KeyError as error:
        raise FileNotFoundError(f"No slices for patient {patient:03d}, frame {frame:02d} in {data_dir}.") from error
    fold = fold_for_patient(run_dir, patient)
    model, _, _, checkpoint, device = load_model(run_dir / f"fold_{fold}", model_name=model_name)
    prediction, target, _, phase = predict_slices(model, slice_paths, batch_size, device)
    model.to("cpu")
    torch.cuda.empty_cache()
    images = []
    slice_indices = []
    for path in slice_paths:
        with h5py.File(path, "r") as h5_file:
            images.append(h5_file["image"][:].astype(np.float32))
            slice_indices.append(int(h5_file.attrs.get("slice_index", len(slice_indices))))
    return {
        "fold": fold,
        "checkpoint": str(checkpoint),
        "phase": phase,
        "images": np.stack(images),
        "prediction": keep_largest_connected_components_per_class(prediction),
        "target": target,
        "slice_indices": slice_indices,
    }


def predict_3d(run_dir, data_dir, patient, frame):
    paths = sorted(data_dir.glob(f"patient{patient:03d}_frame{frame:02d}*.h5"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one 3D volume for patient {patient:03d}, frame {frame:02d} in {data_dir}, found {len(paths)}.")
    fold = fold_for_patient(run_dir, patient)
    model, _, _, checkpoint, device = load_model(run_dir / f"fold_{fold}", model_name="unet3d")
    with h5py.File(paths[0], "r") as h5_file:
        image = h5_file["image"][:].astype(np.float32)
        target = h5_file["label"][:].astype(np.uint8)
        phase = read_phase_or_unknown(h5_file)
    with torch.no_grad():
        logits = model(torch.from_numpy(image[None, None]).to(device))
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    model.to("cpu")
    torch.cuda.empty_cache()
    return {
        "fold": fold,
        "checkpoint": str(checkpoint),
        "phase": phase,
        "images": image,
        "prediction": keep_largest_connected_components_per_class(prediction),
        "target": target,
        "slice_indices": list(range(image.shape[0])),
    }


def metrics(record):
    return {
        "foreground_dice": macro_dice(record["prediction"], record["target"]),
        "class_dice": {name: dice_score(record["prediction"], record["target"], class_id) for class_id, name in ((1, "RV"), (2, "Myo"), (3, "LV"))},
    }


def representative_position(target):
    foreground = (target > 0).sum(axis=(1, 2))
    return int(np.argmax(foreground))


def plot(records, patient, frame, output_path):
    cmap = ListedColormap(MASK_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(len(records), 3, figsize=(9.6, 3.0 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        position = representative_position(record["target"])
        image = record["images"][position]
        low, high = np.percentile(image, [1, 99])
        axes[row, 0].imshow(image, cmap="gray", vmin=low, vmax=high)
        axes[row, 1].imshow(record["prediction"][position], cmap=cmap, norm=norm, interpolation="nearest")
        axes[row, 2].imshow(record["target"][position], cmap=cmap, norm=norm, interpolation="nearest")
        axes[row, 0].set_ylabel(
            f"{record['label']}\\nDice {record['metrics']['foreground_dice']:.3f}\\npresek {record['slice_indices'][position]}",
            fontweight="bold",
        )
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
    fig.suptitle(f"Out-of-fold poređenje modela | pacijent {patient:03d}, frame {frame:02d}", y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on the checkpoint machine for this export.")
    data_dirs = {"fcn8": args.fcn8_data_dir, "unet2d": args.unet2d_data_dir, "unet3d": args.unet3d_data_dir}
    records = []
    for model_name, label, run_name, data_key in MODEL_SPECS:
        run_dir = args.runs_dir / run_name
        if model_name == "unet3d":
            record = predict_3d(run_dir, data_dirs["unet3d"], args.patient, args.frame)
        else:
            record = predict_2d(run_dir, model_name, data_dirs[data_key], args.patient, args.frame, args.batch_size)
        record["model"] = model_name
        record["label"] = label
        record["metrics"] = metrics(record)
        records.append(record)
    phases = {record["phase"] for record in records}
    folds = {record["fold"] for record in records}
    if len(phases) != 1 or len(folds) != 1:
        raise ValueError(f"Models disagree on phase {phases} or validation fold {folds}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "all_models_oof_prediction_comparison.png"
    plot(records, args.patient, args.frame, figure_path)
    metadata = {
        "selection": "patient 002 ED is the deterministic median-Dice standard-2D-U-Net OOF case",
        "patient": args.patient,
        "frame": args.frame,
        "phase": phases.pop(),
        "fold": folds.pop(),
        "models": [
            {
                "model": record["model"],
                "checkpoint": record["checkpoint"],
                "foreground_dice": record["metrics"]["foreground_dice"],
                "class_dice": record["metrics"]["class_dice"],
                "slice_index": record["slice_indices"][representative_position(record["target"])],
            }
            for record in records
        ],
        "postprocessing": "largest_connected_component_per_class",
    }
    metadata_path = args.output_dir / "all_models_oof_prediction_comparison.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {figure_path} and {metadata_path}")


if __name__ == "__main__":
    main()
