import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from cv_reporting import (
        aggregate_cross_validation,
        evaluate_test_ensemble_2d,
        evaluate_test_ensemble_3d,
        evaluate_validation_fold,
    )
    from training_losses import build_loss
    from training_utils import (
        build_scheduler,
        build_stratified_folds,
        diagnosis_counts,
        require_cuda,
        validate_spacing_metadata,
        verify_model_on_cuda,
    )
except ModuleNotFoundError:
    from scripts.cv_reporting import (
        aggregate_cross_validation,
        evaluate_test_ensemble_2d,
        evaluate_test_ensemble_3d,
        evaluate_validation_fold,
    )
    from scripts.training_losses import build_loss
    from scripts.training_utils import (
        build_scheduler,
        build_stratified_folds,
        diagnosis_counts,
        require_cuda,
        validate_spacing_metadata,
        verify_model_on_cuda,
    )


def _args_dict(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _write_fold_assignments(run_dir, folds, seed, num_folds):
    payload = {
        "seed": seed,
        "num_folds": num_folds,
        "folds": [
            {
                "fold": fold["fold"],
                "train_patients": fold["train_patients"],
                "val_patients": fold["val_patients"],
            }
            for fold in folds
        ],
    }
    (Path(run_dir) / "fold_assignments.json").write_text(json.dumps(payload, indent=2))


def run_cross_validation(
    args,
    model_name,
    dimension,
    model_factory,
    dataset_factory,
    run_epoch,
    append_metrics,
    save_single_epoch_checkpoint,
    format_seconds,
    load_starting_weights,
    patch_shape=None,
):
    device = require_cuda()
    files = sorted(args.data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {args.data_dir}")
    spacing_metadata = validate_spacing_metadata(files, expected_ndim=3)
    folds = build_stratified_folds(files, num_folds=args.num_folds, seed=args.seed)
    selected_folds = folds if args.fold is None else [folds[args.fold]]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_fold_assignments(args.run_dir, folds, args.seed, args.num_folds)
    fold_rows = {}

    for fold in selected_folds:
        fold_index = fold["fold"]
        fold_seed = args.seed + fold_index
        _seed_everything(fold_seed)
        fold_dir = args.run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        training_start = time.perf_counter()

        train_files = fold["train_files"]
        val_files = fold["val_files"]
        if args.max_train_samples is not None:
            train_files = train_files[: args.max_train_samples]
        if args.max_val_samples is not None:
            val_files = val_files[: args.max_val_samples]

        train_loader = DataLoader(
            dataset_factory(train_files, True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            dataset_factory(val_files, False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        model = model_factory().to(device)
        verify_model_on_cuda(model)
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
        scheduler = build_scheduler(optimizer, args)

        config = {
            "args": _args_dict(args),
            "model": model_name,
            "fold": fold_index,
            "fold_seed": fold_seed,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "patch_shape": patch_shape,
            "train_patients": fold["train_patients"],
            "val_patients": fold["val_patients"],
            "train_diagnosis_counts": diagnosis_counts(files, fold["train_patients"]),
            "val_diagnosis_counts": diagnosis_counts(files, fold["val_patients"]),
            "num_train_files": len(train_files),
            "num_val_files": len(val_files),
            "spacing_metadata": spacing_metadata,
            "training_started_at_unix": time.time(),
        }
        (fold_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        print(f"\nFold {fold_index}/{args.num_folds - 1}: {len(fold['train_patients'])} train, "
              f"{len(fold['val_patients'])} validation patients")

        best_val_dice = -1.0
        metrics_path = fold_dir / "metrics.csv"
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy, train_dice, train_per_class = run_epoch(
                model, train_loader, criterion, optimizer, device, args.num_classes, train=True
            )
            val_loss, val_accuracy, val_dice, val_per_class = run_epoch(
                model, val_loader, criterion, optimizer, device, args.num_classes, train=False
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step(val_loss)
            epoch_seconds = time.perf_counter() - epoch_start
            elapsed_seconds = time.perf_counter() - training_start
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pixel_accuracy": train_accuracy,
                "train_mean_foreground_dice": train_dice,
                "val_loss": val_loss,
                "val_pixel_accuracy": val_accuracy,
                "val_mean_foreground_dice": val_dice,
                "learning_rate": learning_rate,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "elapsed_time": format_seconds(elapsed_seconds),
            }
            row.update({f"train_dice_class_{i}": train_per_class[i] for i in range(args.num_classes)})
            row.update({f"val_dice_class_{i}": val_per_class[i] for i in range(args.num_classes)})
            append_metrics(metrics_path, row, args.num_classes)
            checkpoint_metrics = {
                "train_loss": train_loss,
                "train_pixel_accuracy": train_accuracy,
                "train_mean_foreground_dice": train_dice,
                "val_loss": val_loss,
                "val_pixel_accuracy": val_accuracy,
                "val_mean_foreground_dice": val_dice,
            }
            save_single_epoch_checkpoint(
                fold_dir, "latest", model, optimizer, epoch, checkpoint_metrics, args,
                fold["train_patients"], fold["val_patients"], elapsed_seconds
            )
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                save_single_epoch_checkpoint(
                    fold_dir, "best", model, optimizer, epoch, checkpoint_metrics, args,
                    fold["train_patients"], fold["val_patients"], elapsed_seconds
                )
            print(
                f"Fold {fold_index} epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} "
                f"train_dice={train_dice:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
                f"time={format_seconds(epoch_seconds)}"
            )

        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()
        rows, _, _ = evaluate_validation_fold(
            fold_dir, model_name, dimension, device, batch_size=args.batch_size
        )
        fold_rows[fold_index] = rows
        torch.cuda.empty_cache()

    if args.fold is None:
        aggregate_cross_validation(args.run_dir, fold_rows)
        fold_dirs = [args.run_dir / f"fold_{index}" for index in range(args.num_folds)]
        if dimension == 2:
            evaluate_test_ensemble_2d(
                args.run_dir, fold_dirs, model_name, args.test_data_dir, device, args.batch_size
            )
        else:
            evaluate_test_ensemble_3d(args.run_dir, fold_dirs, args.test_data_dir, device)
