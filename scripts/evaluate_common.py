import json
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import FCN8, ModifiedUNet2D, UNet2D, UNet3D
from scripts.acdc_h5 import read_spacing_zyx


CLASS_NAMES = {
    0: "background",
    1: "RV",
    2: "Myo",
    3: "LV",
}


def get_patient_id(path):
    match = re.search(r"patient(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse patient id from {path}")
    return int(match.group(1))


def get_frame_number(path):
    match = re.search(r"frame(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse frame number from {path}")
    return int(match.group(1))


def get_slice_index(path):
    match = re.search(r"slice_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    with h5py.File(path, "r") as h5_file:
        if "slice_index" in h5_file.attrs:
            return int(h5_file.attrs["slice_index"])
    raise ValueError(f"Could not parse slice index from {path}")


def read_config(run_dir):
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def get_config_args(config):
    return config.get("args", {})


def find_checkpoint(run_dir, checkpoint=None, prefer="best"):
    if checkpoint is not None:
        return Path(checkpoint)
    run_dir = Path(run_dir)
    patterns = [f"{prefer}_epoch_*.pt", "best_epoch_*.pt", "latest_epoch_*.pt"]
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def infer_model_name(run_dir, config, explicit_model=None):
    if explicit_model:
        return explicit_model
    run_name = Path(run_dir).name.lower()
    args = get_config_args(config)
    script_hint = str(args.get("run_dir", "")).lower()
    combined = f"{run_name} {script_hint}"
    if "fcn8" in combined:
        return "fcn8"
    if "unet3d" in combined:
        return "unet3d"
    if "modified" in combined:
        return "unet2d_modified"
    if "unet2d" in combined:
        return "unet2d"
    raise ValueError("Could not infer model type. Pass --model explicitly.")


def build_model(model_name, config):
    args = get_config_args(config)
    in_channels = int(args.get("in_channels", 1))
    num_classes = int(args.get("num_classes", 4))

    if model_name == "fcn8":
        return FCN8(
            in_channels=in_channels,
            num_classes=num_classes,
            classifier_channels=int(args.get("classifier_channels", 1024)),
            use_batch_norm=True,
        )
    if model_name == "unet2d":
        return UNet2D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=int(args.get("base_channels", 64)),
            use_batch_norm=True,
        )
    if model_name == "unet2d_modified":
        return ModifiedUNet2D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=int(args.get("base_channels", 64)),
            use_batch_norm=True,
        )
    if model_name == "unet3d":
        return UNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=int(args.get("base_channels", 16)),
            use_batch_norm=True,
        )
    raise ValueError(f"Unknown model: {model_name}")


def load_model(run_dir, checkpoint=None, model_name=None, device=None):
    config = read_config(run_dir)
    model_name = infer_model_name(run_dir, config, model_name)
    checkpoint_path = find_checkpoint(run_dir, checkpoint)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, config)
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name, config, checkpoint_path, device


def patient_filter_from_config(config, split):
    if split == "all":
        return None
    key = "val_patients" if split == "val" else "train_patients"
    patients = config.get(key)
    return None if patients is None else set(int(patient) for patient in patients)


def dice_score(prediction, target, class_id):
    pred = prediction == class_id
    true = target == class_id
    denom = pred.sum() + true.sum()
    if denom == 0:
        return math.nan
    return float(2.0 * np.logical_and(pred, true).sum() / denom)


def surface_mask(mask):
    if not mask.any():
        return mask.astype(bool)
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def surface_distances_mm(pred_mask, true_mask, spacing):
    pred_surface = surface_mask(pred_mask)
    true_surface = surface_mask(true_mask)
    if not pred_surface.any() and not true_surface.any():
        return np.array([], dtype=np.float32)
    if not pred_surface.any() or not true_surface.any():
        return None

    true_distance = ndimage.distance_transform_edt(~true_surface, sampling=spacing)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances_pred_to_true = true_distance[pred_surface]
    distances_true_to_pred = pred_distance[true_surface]
    return np.concatenate([distances_pred_to_true, distances_true_to_pred])


def class_metrics(prediction, target, spacing, class_id):
    dice = dice_score(prediction, target, class_id)
    pred_mask = prediction == class_id
    true_mask = target == class_id
    distances = surface_distances_mm(pred_mask, true_mask, spacing)

    if distances is None:
        assd = math.inf
        hd = math.inf
    elif distances.size == 0:
        assd = math.nan
        hd = math.nan
    else:
        assd = float(distances.mean())
        hd = float(distances.max())

    return {
        "dice": dice,
        "assd_mm": assd,
        "hd_mm": hd,
        "pred_voxels": int(pred_mask.sum()),
        "target_voxels": int(true_mask.sum()),
    }


def volume_metrics(prediction, target, spacing, patient, frame=None, source=None):
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == 0:
            continue
        metrics = class_metrics(prediction, target, spacing, class_id)
        rows.append(
            {
                "patient": patient,
                "frame": frame,
                "source": source,
                "class_id": class_id,
                "class_name": class_name,
                **metrics,
            }
        )
    return rows


def summarize_rows(rows):
    if not rows:
        return {}
    finite = {}
    for key in ("dice", "assd_mm", "hd_mm"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        finite[f"mean_{key}"] = float(values.mean()) if values.size else math.nan
    finite["volumes"] = len({(row["patient"], row.get("frame")) for row in rows})
    finite["class_rows"] = len(rows)
    return finite


def read_spacing_or_default(h5_file, ndim):
    try:
        spacing = read_spacing_zyx(h5_file)
    except KeyError:
        spacing = np.ones(3, dtype=np.float32)
    if ndim == 2:
        return np.asarray(spacing[-2:], dtype=np.float32)
    return np.asarray(spacing, dtype=np.float32)


def save_evaluation(output_dir, rows, summary, metadata):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_by_class.csv"
    summary_path = output_dir / "summary.json"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    summary_path.write_text(json.dumps({"summary": summary, "metadata": metadata}, indent=2, default=str))
    return metrics_path, summary_path
