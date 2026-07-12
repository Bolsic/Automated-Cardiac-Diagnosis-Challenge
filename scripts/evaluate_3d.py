import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from evaluate_common import (
    get_frame_number,
    get_patient_id,
    load_model,
    patient_filter_from_config,
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


def main():
    args = parse_args()
    model, model_name, config, checkpoint_path, device = load_model(
        args.run_dir,
        checkpoint=args.checkpoint,
        model_name=args.model or "unet3d",
    )

    config_args = config.get("args", {})
    data_dir = args.data_dir or Path(config_args.get("data_dir", ""))
    if not data_dir:
        raise ValueError("No data directory provided and none found in run config.")
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    patient_filter = patient_filter_from_config(config, args.split)
    if patient_filter is not None:
        files = [path for path in files if get_patient_id(path) in patient_filter]
    if args.max_volumes is not None:
        files = files[: args.max_volumes]
    if not files:
        raise RuntimeError(f"No volume files found for split={args.split} in {data_dir}")

    rows = []
    for path in tqdm(files, desc="evaluate 3d volumes"):
        patient = get_patient_id(path)
        frame = get_frame_number(path)
        with h5py.File(path, "r") as h5_file:
            image = h5_file["image"][:].astype(np.float32)
            target = h5_file["label"][:].astype(np.uint8)
            spacing = read_spacing_or_default(h5_file, ndim=3)

        tensor = torch.from_numpy(image[None, None, :, :, :]).to(device)
        with torch.no_grad():
            logits = model(tensor)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        rows.extend(volume_metrics(prediction, target, spacing, patient=patient, frame=frame, source=path.name))

    summary = summarize_rows(rows)
    output_dir = args.output_dir or (Path(args.run_dir) / f"evaluation_3d_{args.split}")
    metadata = {
        "run_dir": args.run_dir,
        "checkpoint": checkpoint_path,
        "model": model_name,
        "data_dir": data_dir,
        "split": args.split,
        "device": str(device),
        "volumes_evaluated": len(files),
    }
    metrics_path, summary_path = save_evaluation(output_dir, rows, summary, metadata)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
