import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow the script to be run as `python scripts/train_unet2d_modified.py`
# while still importing project modules from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import ModifiedUNet2D
from training_losses import add_loss_arguments, build_loss
from train_unet2d import (
    ACDCSliceDataset,
    append_metrics,
    args_to_dict,
    format_seconds,
    load_starting_weights,
    run_epoch,
    save_single_epoch_checkpoint,
    split_by_patient,
)


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train modified 2D U-Net on ACDC preprocessed 2D slices.")

    # Data and output locations.
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/acdc_preprocessed_2d/ACDC_training_slices"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/unet2d_modified"))
    parser.add_argument("--weights", type=Path, default=None, help="Optional model weights to load before training.")

    # Training settings. The optimizer defaults follow the paper.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    add_loss_arguments(parser)

    # Validation split and reproducibility.
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Debug options for quick smoke tests.
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    training_start_time = time.perf_counter()

    # Make the patient split and weight initialization repeatable.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load all exported 2D HDF5 slices and split them by patient.
    files = sorted(args.data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {args.data_dir}")

    train_files, val_files, train_patients, val_patients = split_by_patient(
        files,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    if args.max_train_samples is not None:
        train_files = train_files[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_files = val_files[: args.max_val_samples]

    # DataLoader turns HDF5 files into mini-batches of tensors.
    train_loader = DataLoader(
        ACDCSliceDataset(train_files),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ACDCSliceDataset(val_files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Prefer GPU automatically, but keep CPU training available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedUNet2D(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        use_batch_norm=True,
    ).to(device)

    if args.weights is not None:
        load_starting_weights(model, args.weights, device)

    criterion = build_loss(
        args.loss,
        args.num_classes,
        device,
        background_weight=args.background_class_weight,
        foreground_weight=args.foreground_class_weight,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Save run configuration and split details before training starts.
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "config.json").open("w") as file:
        json.dump(
            {
                "args": args_to_dict(args),
                "device": str(device),
                "train_patients": train_patients,
                "val_patients": val_patients,
                "num_train_files": len(train_files),
                "num_val_files": len(val_files),
                "training_started_at_unix": time.time(),
            },
            file,
            indent=2,
            default=str,
        )

    print(f"Device: {device}")
    print(f"Train files: {len(train_files)} from {len(train_patients)} patients")
    print(f"Validation files: {len(val_files)} from {len(val_patients)} patients")
    print(f"Loss: {args.loss}")
    print(f"Run directory: {args.run_dir}")

    best_val_dice = -1.0
    metrics_path = args.run_dir / "metrics.csv"

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}")

        # One full pass over the training split updates model weights.
        train_loss, train_accuracy, train_dice, train_dice_per_class = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.num_classes,
            train=True,
        )

        # One full pass over validation measures generalization without updates.
        val_loss, val_accuracy, val_dice, val_dice_per_class = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            args.num_classes,
            train=False,
        )

        epoch_seconds = time.perf_counter() - epoch_start_time
        elapsed_seconds = time.perf_counter() - training_start_time

        # Store scalar metrics in CSV form for later plotting.
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_pixel_accuracy": train_accuracy,
            "train_mean_foreground_dice": train_dice,
            "val_loss": val_loss,
            "val_pixel_accuracy": val_accuracy,
            "val_mean_foreground_dice": val_dice,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_time": format_seconds(elapsed_seconds),
        }
        row.update({f"train_dice_class_{i}": train_dice_per_class[i] for i in range(args.num_classes)})
        row.update({f"val_dice_class_{i}": val_dice_per_class[i] for i in range(args.num_classes)})
        append_metrics(metrics_path, row, args.num_classes)

        checkpoint_metrics = {
            "train_loss": train_loss,
            "train_pixel_accuracy": train_accuracy,
            "train_mean_foreground_dice": train_dice,
            "val_loss": val_loss,
            "val_pixel_accuracy": val_accuracy,
            "val_mean_foreground_dice": val_dice,
        }

        # Keep only the newest checkpoint, named with the epoch that produced it.
        latest_checkpoint_path = save_single_epoch_checkpoint(
            args.run_dir,
            "latest",
            model,
            optimizer,
            epoch,
            checkpoint_metrics,
            args,
            train_patients,
            val_patients,
            elapsed_seconds,
        )

        # Keep a separate copy of the model with best validation foreground Dice.
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_checkpoint_path = save_single_epoch_checkpoint(
                args.run_dir,
                "best",
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
                args,
                train_patients,
                val_patients,
                elapsed_seconds,
            )
        else:
            best_checkpoint_path = None

        print(
            "train_loss={:.4f} train_dice={:.4f} val_loss={:.4f} val_dice={:.4f} epoch_time={} elapsed={}".format(
                train_loss,
                train_dice,
                val_loss,
                val_dice,
                format_seconds(epoch_seconds),
                format_seconds(elapsed_seconds),
            )
        )
        print(f"Saved latest checkpoint: {latest_checkpoint_path}")
        if best_checkpoint_path is not None:
            print(f"Saved best checkpoint: {best_checkpoint_path}")

    total_seconds = time.perf_counter() - training_start_time
    print(f"\nTraining finished in {format_seconds(total_seconds)} ({total_seconds:.1f} seconds)")


if __name__ == "__main__":
    main()
