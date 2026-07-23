import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from evaluate_common import (
    LARGEST_COMPONENT_POSTPROCESSING,
    get_frame_number,
    get_patient_id,
    get_slice_index,
    load_model,
    patient_filter_from_config,
    read_phase_or_unknown,
    read_spacing_or_default,
    save_evaluation,
    summarize_rows,
    volume_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 2D model by reconstructing 3D patient/frame volumes.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing config/checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to best checkpoint.")
    parser.add_argument("--model", choices=["fcn8", "unet2d", "unet2d_modified"], default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder of preprocessed 2D HDF5 slices.")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def group_slices(files, patient_filter):
    groups = defaultdict(list)
    for path in files:
        patient = get_patient_id(path)
        if patient_filter is not None and patient not in patient_filter:
            continue
        frame = get_frame_number(path)
        groups[(patient, frame)].append(path)
    return {
        key: sorted(paths, key=get_slice_index)
        for key, paths in sorted(groups.items())
    }


def predict_slices(model, slice_paths, batch_size, device):
    predictions = []
    labels = []
    spacing = None
    phase = "unknown"
    for start in range(0, len(slice_paths), batch_size):
        batch_paths = slice_paths[start : start + batch_size]
        images = []
        batch_labels = []
        for path in batch_paths:
            with h5py.File(path, "r") as h5_file:
                image = h5_file["image"][:].astype(np.float32)
                label = h5_file["label"][:].astype(np.int64)
                if spacing is None:
                    spacing = read_spacing_or_default(h5_file, ndim=3)
                    phase = read_phase_or_unknown(h5_file)
            images.append(image)
            batch_labels.append(label)

        tensor = torch.from_numpy(np.stack(images)[:, None, :, :]).to(device)
        with torch.no_grad():
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        predictions.extend(pred)
        labels.extend(batch_labels)

    return np.stack(predictions), np.stack(labels).astype(np.uint8), spacing, phase


def main():
    args = parse_args()
    model, model_name, config, checkpoint_path, device = load_model(
        args.run_dir,
        checkpoint=args.checkpoint,
        model_name=args.model,
    )

    config_args = config.get("args", {})
    data_dir = args.data_dir or Path(config_args.get("data_dir", ""))
    if not data_dir:
        raise ValueError("No data directory provided and none found in run config.")
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    patient_filter = patient_filter_from_config(config, args.split)
    groups = group_slices(files, patient_filter)
    if args.max_volumes is not None:
        groups = dict(list(groups.items())[: args.max_volumes])
    if not groups:
        raise RuntimeError(f"No slice groups found for split={args.split} in {data_dir}")

    rows = []
    for (patient, frame), slice_paths in tqdm(groups.items(), desc="evaluate 2d volumes"):
        prediction, target, spacing, phase = predict_slices(model, slice_paths, args.batch_size, device)
        rows.extend(volume_metrics(
            prediction,
            target,
            spacing,
            patient=patient,
            frame=frame,
            phase=phase,
            source="2d_reconstruction",
        ))

    summary = summarize_rows(rows)
    output_dir = args.output_dir or (Path(args.run_dir) / f"evaluation_2d_{args.split}")
    metadata = {
        "run_dir": args.run_dir,
        "checkpoint": checkpoint_path,
        "model": model_name,
        "postprocessing": LARGEST_COMPONENT_POSTPROCESSING,
        "data_dir": data_dir,
        "split": args.split,
        "device": str(device),
        "volumes_evaluated": len(groups),
        "distance_units": "mm",
        "spacing_source": "HDF5 spacing_zyx metadata",
    }
    metrics_path, summary_path = save_evaluation(output_dir, rows, summary, metadata)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
