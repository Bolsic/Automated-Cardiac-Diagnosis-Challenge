import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    from evaluate_common import (
        LARGEST_COMPONENT_POSTPROCESSING,
        find_checkpoint,
        get_frame_number,
        get_patient_id,
        load_model,
        read_phase_or_unknown,
        read_spacing_or_default,
        save_evaluation,
        summarize_rows,
        volume_metrics,
    )
    from evaluate_2d import evaluate_run as evaluate_2d_run, group_slices
    from evaluate_3d import evaluate_run as evaluate_3d_run
except ModuleNotFoundError:
    from scripts.evaluate_common import (
        LARGEST_COMPONENT_POSTPROCESSING,
        find_checkpoint,
        get_frame_number,
        get_patient_id,
        load_model,
        read_phase_or_unknown,
        read_spacing_or_default,
        save_evaluation,
        summarize_rows,
        volume_metrics,
    )
    from scripts.evaluate_2d import evaluate_run as evaluate_2d_run, group_slices
    from scripts.evaluate_3d import evaluate_run as evaluate_3d_run


METRIC_KEYS = ("dice", "assd_mm", "hd_mm")


def evaluate_validation_fold(fold_dir, model_name, dimension, device, batch_size=8):
    if dimension == 2:
        return evaluate_2d_run(
            fold_dir,
            model_name=model_name,
            split="val",
            batch_size=batch_size,
            device=device,
        )
    return evaluate_3d_run(fold_dir, model_name=model_name, split="val", device=device)


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def aggregate_cross_validation(run_dir, fold_rows):
    """Write fold/class metrics and mean/std summaries across completed folds."""
    run_dir = Path(run_dir)
    records = []
    fold_macro = []
    for fold, rows in sorted(fold_rows.items()):
        macro = {"fold": fold, "class_id": "foreground", "class_name": "foreground_macro"}
        for metric in METRIC_KEYS:
            macro[metric] = _finite_mean([row[metric] for row in rows])
        records.append(macro)
        fold_macro.append(macro)
        for class_id in (1, 2, 3):
            class_rows = [row for row in rows if int(row["class_id"]) == class_id]
            record = {"fold": fold, "class_id": class_id, "class_name": class_rows[0]["class_name"]}
            for metric in METRIC_KEYS:
                record[metric] = _finite_mean([row[metric] for row in class_rows])
            records.append(record)

    metrics_path = run_dir / "cross_validation_metrics.csv"
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["fold", "class_id", "class_name", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(records)

    summary = {"num_folds": len(fold_rows), "folds": sorted(fold_rows), "foreground_macro": {}}
    for metric in METRIC_KEYS:
        values = np.asarray([row[metric] for row in fold_macro], dtype=float)
        values = values[np.isfinite(values)]
        summary["foreground_macro"][f"mean_{metric}"] = float(values.mean()) if values.size else math.nan
        summary["foreground_macro"][f"std_{metric}"] = float(values.std(ddof=0)) if values.size else math.nan
    summary["by_class"] = {}
    for class_id in (1, 2, 3):
        class_records = [record for record in records if record["class_id"] == class_id]
        name = class_records[0]["class_name"]
        summary["by_class"][name] = {}
        for metric in METRIC_KEYS:
            values = np.asarray([record[metric] for record in class_records], dtype=float)
            values = values[np.isfinite(values)]
            summary["by_class"][name][f"mean_{metric}"] = float(values.mean()) if values.size else math.nan
            summary["by_class"][name][f"std_{metric}"] = float(values.std(ddof=0)) if values.size else math.nan

    summary_path = run_dir / "cross_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True))
    return metrics_path, summary_path


def _load_fold_models(fold_dirs, model_name):
    models = []
    checkpoints = []
    for fold_dir in fold_dirs:
        model, _, _, checkpoint, _ = load_model(
            fold_dir,
            checkpoint=find_checkpoint(fold_dir),
            model_name=model_name,
            device=torch.device("cpu"),
        )
        models.append(model)
        checkpoints.append(str(checkpoint))
    return models, checkpoints


def _ensemble_probabilities_2d(models, slice_paths, batch_size, device):
    probability_sum = None
    labels = []
    spacing = None
    phase = "unknown"
    for start in range(0, len(slice_paths), batch_size):
        batch_paths = slice_paths[start : start + batch_size]
        images = []
        batch_labels = []
        for path in batch_paths:
            with h5py.File(path, "r") as h5_file:
                images.append(h5_file["image"][:].astype(np.float32))
                batch_labels.append(h5_file["label"][:].astype(np.uint8))
                if spacing is None:
                    spacing = read_spacing_or_default(h5_file, ndim=3)
                    phase = read_phase_or_unknown(h5_file)
        tensor = torch.from_numpy(np.stack(images)[:, None]).to(device)
        batch_sum = None
        for model in models:
            model.to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1).cpu()
            model.to("cpu")
            batch_sum = probabilities if batch_sum is None else batch_sum + probabilities
        batch_sum = batch_sum.numpy()
        probability_sum = batch_sum if probability_sum is None else np.concatenate([probability_sum, batch_sum], axis=0)
        labels.extend(batch_labels)
    prediction = np.argmax(probability_sum / len(models), axis=1).astype(np.uint8)
    return prediction, np.stack(labels), spacing, phase


def evaluate_test_ensemble_2d(run_dir, fold_dirs, model_name, data_dir, device, batch_size=8):
    files = sorted(Path(data_dir).glob("*.h5"))
    groups = group_slices(files, patient_filter=None)
    if not groups:
        raise FileNotFoundError(f"No test slices found in {data_dir}")
    models, checkpoints = _load_fold_models(fold_dirs, model_name)
    rows = []
    for (patient, frame), slice_paths in groups.items():
        prediction, target, spacing, phase = _ensemble_probabilities_2d(models, slice_paths, batch_size, device)
        rows.extend(volume_metrics(prediction, target, spacing, patient, frame, phase, "five_fold_ensemble"))
    output_dir = Path(run_dir) / "test_ensemble"
    metadata = {
        "model": model_name,
        "checkpoints": checkpoints,
        "data_dir": str(data_dir),
        "patients": len({patient for patient, _ in groups}),
        "volumes_evaluated": len(groups),
        "ensemble": "mean_softmax_probability",
        "postprocessing": LARGEST_COMPONENT_POSTPROCESSING,
        "distance_units": "mm",
        "device": str(device),
    }
    return save_evaluation(output_dir, rows, summarize_rows(rows), metadata)


def evaluate_test_ensemble_3d(run_dir, fold_dirs, data_dir, device):
    files = sorted(Path(data_dir).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No test volumes found in {data_dir}")
    models, checkpoints = _load_fold_models(fold_dirs, "unet3d")
    rows = []
    for path in files:
        with h5py.File(path, "r") as h5_file:
            image = h5_file["image"][:].astype(np.float32)
            target = h5_file["label"][:].astype(np.uint8)
            spacing = read_spacing_or_default(h5_file, ndim=3)
            phase = read_phase_or_unknown(h5_file)
        tensor = torch.from_numpy(image[None, None]).to(device)
        probability_sum = None
        for model in models:
            model.to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1).cpu()
            model.to("cpu")
            probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
        prediction = torch.argmax(probability_sum, dim=1).squeeze(0).numpy().astype(np.uint8)
        rows.extend(
            volume_metrics(
                prediction,
                target,
                spacing,
                get_patient_id(path),
                get_frame_number(path),
                phase,
                "five_fold_ensemble",
            )
        )
    output_dir = Path(run_dir) / "test_ensemble"
    metadata = {
        "model": "unet3d",
        "checkpoints": checkpoints,
        "data_dir": str(data_dir),
        "patients": len({get_patient_id(path) for path in files}),
        "volumes_evaluated": len(files),
        "ensemble": "mean_softmax_probability",
        "postprocessing": LARGEST_COMPONENT_POSTPROCESSING,
        "distance_units": "mm",
        "device": str(device),
    }
    return save_evaluation(output_dir, rows, summarize_rows(rows), metadata)
