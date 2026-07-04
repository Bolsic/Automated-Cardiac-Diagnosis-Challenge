import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow the script to be run as `python scripts/train_fcn8.py` while still
# importing project modules from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import FCN8
from training_losses import add_loss_arguments, build_loss


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def get_patient_id(path):
    # File names look like patient001_frame01_slice_0.h5.
    name = path.stem
    patient_part = name.split("_")[0]
    return int(patient_part.replace("patient", ""))


def split_by_patient(files, val_fraction, seed):
    # Split by patient instead of by slice so slices from the same patient do
    # not appear in both training and validation sets.
    patients = sorted({get_patient_id(path) for path in files})
    rng = random.Random(seed)
    rng.shuffle(patients)

    num_val = max(1, round(len(patients) * val_fraction))
    val_patients = set(patients[:num_val])
    train_patients = set(patients[num_val:])

    train_files = [path for path in files if get_patient_id(path) in train_patients]
    val_files = [path for path in files if get_patient_id(path) in val_patients]
    return train_files, val_files, sorted(train_patients), sorted(val_patients)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class ACDCSliceDataset(Dataset):
    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with h5py.File(path, "r") as h5_file:
            # Images are stored as H x W float arrays; labels are H x W class IDs.
            image = h5_file["image"][:].astype(np.float32)
            label = h5_file["label"][:].astype(np.int64)

        # PyTorch convolution layers expect channel-first tensors: C x H x W.
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label)
        return image, label


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def update_confusion_matrix(confusion_matrix, logits, labels, num_classes):
    # Convert logits to predicted class IDs and accumulate a num_classes x
    # num_classes confusion matrix. Rows are true labels, columns are predictions.
    predictions = torch.argmax(logits, dim=1)
    valid = (labels >= 0) & (labels < num_classes)
    encoded = labels[valid] * num_classes + predictions[valid]
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion_matrix += counts.reshape(num_classes, num_classes).cpu()


def metrics_from_confusion(confusion_matrix):
    # Dice is computed per class from the accumulated confusion matrix:
    # Dice = 2TP / (ground_truth_pixels + predicted_pixels).
    confusion_matrix = confusion_matrix.float()
    true_positive = torch.diag(confusion_matrix)
    label_count = confusion_matrix.sum(dim=1)
    prediction_count = confusion_matrix.sum(dim=0)
    denominator = label_count + prediction_count

    dice_per_class = torch.where(
        denominator > 0,
        2 * true_positive / denominator.clamp_min(1),
        torch.ones_like(true_positive),
    )
    accuracy = true_positive.sum() / confusion_matrix.sum().clamp_min(1)

    foreground_dice = dice_per_class[1:].mean()
    return accuracy.item(), foreground_dice.item(), dice_per_class.tolist()


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------


def run_epoch(model, loader, criterion, optimizer, device, num_classes, train):
    # The same function handles training and validation. The `train` flag
    # controls gradient tracking, optimizer updates, and BatchNorm/Dropout mode.
    model.train(train)
    total_loss = 0.0
    total_examples = 0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    description = "train" if train else "val"
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            # Clear old gradients before computing this batch's gradient.
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            # FCN-8 returns raw logits with shape B x num_classes x H x W.
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                # Backpropagate and update model weights only during training.
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        update_confusion_matrix(confusion_matrix, logits.detach(), labels.detach(), num_classes)

    average_loss = total_loss / max(total_examples, 1)
    accuracy, mean_dice, dice_per_class = metrics_from_confusion(confusion_matrix)
    return average_loss, accuracy, mean_dice, dice_per_class


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------


def format_seconds(seconds):
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def args_to_dict(args):
    # Convert Path values to strings so checkpoint metadata is easy to inspect.
    clean_args = {}
    for key, value in vars(args).items():
        clean_args[key] = str(value) if isinstance(value, Path) else value
    return clean_args


def save_checkpoint(path, model, optimizer, epoch, metrics, args, train_patients, val_patients, elapsed_seconds):
    # Save model weights, optimizer state, metrics, and run metadata.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": args_to_dict(args),
            "train_patients": train_patients,
            "val_patients": val_patients,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_time": format_seconds(elapsed_seconds),
        },
        path,
    )


def save_single_epoch_checkpoint(
    run_dir,
    checkpoint_kind,
    model,
    optimizer,
    epoch,
    metrics,
    args,
    train_patients,
    val_patients,
    elapsed_seconds,
):
    # Keep only one checkpoint for each kind: latest and best.
    for old_checkpoint in run_dir.glob(f"{checkpoint_kind}_epoch_*.pt"):
        old_checkpoint.unlink()

    checkpoint_path = run_dir / f"{checkpoint_kind}_epoch_{epoch:03d}.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch,
        metrics,
        args,
        train_patients,
        val_patients,
        elapsed_seconds,
    )
    return checkpoint_path


def load_starting_weights(model, weights_path, device):
    # Accept either a full checkpoint from this script or a plain model state dict.
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        source_epoch = checkpoint.get("epoch", "unknown")
    else:
        state_dict = checkpoint
        source_epoch = "unknown"

    model.load_state_dict(state_dict)
    print(f"Loaded starting weights from {weights_path} (source epoch: {source_epoch})")


def append_metrics(metrics_path, row, num_classes):
    # Append one row per epoch so training curves can be plotted later.
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    exists = metrics_path.exists()

    fieldnames = [
        "epoch",
        "train_loss",
        "train_pixel_accuracy",
        "train_mean_foreground_dice",
        "val_loss",
        "val_pixel_accuracy",
        "val_mean_foreground_dice",
        "learning_rate",
        "epoch_seconds",
        "elapsed_seconds",
        "elapsed_time",
    ]
    fieldnames += [f"train_dice_class_{i}" for i in range(num_classes)]
    fieldnames += [f"val_dice_class_{i}" for i in range(num_classes)]

    with metrics_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train FCN-8 on ACDC preprocessed 2D slices.")

    # Data and output locations.
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/acdc_preprocessed_2d/ACDC_training_slices"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/fcn8"))
    parser.add_argument("--weights", type=Path, default=None, help="Optional model weights to load before training.")

    # Training settings. The optimizer defaults follow the paper.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--classifier-channels", type=int, default=1024)
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
    model = FCN8(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        classifier_channels=args.classifier_channels,
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
