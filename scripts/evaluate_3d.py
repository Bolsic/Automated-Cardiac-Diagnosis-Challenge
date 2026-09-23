import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

try:
    from evaluate_common import (
        LARGEST_COMPONENT_POSTPROCESSING,
        get_frame_number,
        get_patient_id,
        load_model,
        patient_filter_from_config,
        read_phase_or_unknown,
        read_spacing_or_default,
        save_evaluation,
        summarize_rows,
        volume_metrics,
    )
except ModuleNotFoundError:
    from scripts.evaluate_common import (
    LARGEST_COMPONENT_POSTPROCESSING,
    get_frame_number,
    get_patient_id,
    load_model,
    patient_filter_from_config,
    read_phase_or_unknown,
    read_spacing_or_default,
    save_evaluation,
    summarize_rows,
    volume_metrics,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 3D U-Net model on 3D HDF5 volumes.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing config/checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to best checkpoint.")
    parser.add_argument("--model", choices=["unet3d"], default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder of preprocessed 3D HDF5 volumes.")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def evaluate_run(
    run_dir,
    checkpoint=None,
    model_name="unet3d",
    data_dir=None,
    split="val",
    max_volumes=None,
    output_dir=None,
    device=None,
):
    model, model_name, config, checkpoint_path, device = load_model(
        run_dir,
        checkpoint=checkpoint,
        model_name=model_name,
        device=device,
    )

    config_args = config.get("args", {})
    data_dir = Path(data_dir) if data_dir is not None else Path(config_args.get("data_dir", ""))
    if not data_dir:
        raise ValueError("No data directory provided and none found in run config.")
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    patient_filter = patient_filter_from_config(config, split)
    if patient_filter is not None:
        files = [path for path in files if get_patient_id(path) in patient_filter]
    if max_volumes is not None:
        files = files[:max_volumes]
    if not files:
        raise RuntimeError(f"No volume files found for split={split} in {data_dir}")

    rows = []
    for path in tqdm(files, desc="evaluate 3d volumes"):
        patient = get_patient_id(path)
        frame = get_frame_number(path)
        with h5py.File(path, "r") as h5_file:
            image = h5_file["image"][:].astype(np.float32)
            target = h5_file["label"][:].astype(np.uint8)
            spacing = read_spacing_or_default(h5_file, ndim=3)
            phase = read_phase_or_unknown(h5_file)

        tensor = torch.from_numpy(image[None, None, :, :, :]).to(device)
        with torch.no_grad():
            logits = model(tensor)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        rows.extend(volume_metrics(prediction, target, spacing, patient=patient, frame=frame, phase=phase, source=path.name))

    summary = summarize_rows(rows)
    output_dir = output_dir or (Path(run_dir) / f"evaluation_3d_{split}")
    metadata = {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "model": model_name,
        "postprocessing": LARGEST_COMPONENT_POSTPROCESSING,
        "data_dir": data_dir,
        "split": split,
        "device": str(device),
        "volumes_evaluated": len(files),
        "distance_units": "mm",
        "spacing_source": "HDF5 spacing_zyx metadata",
    }
    metrics_path, summary_path = save_evaluation(output_dir, rows, summary, metadata)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(summary)
    return rows, summary, metadata


def main():
    args = parse_args()
    evaluate_run(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        model_name=args.model or "unet3d",
        data_dir=args.data_dir,
        split=args.split,
        max_volumes=args.max_volumes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
