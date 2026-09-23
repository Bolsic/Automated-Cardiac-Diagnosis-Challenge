import random
from collections import defaultdict

import h5py
import numpy as np
import torch


def get_patient_id(path):
    patient_part = path.stem.split("_")[0]
    return int(patient_part.replace("patient", ""))


def _read_diagnosis(path):
    with h5py.File(path, "r") as h5_file:
        if "diagnosis" not in h5_file.attrs:
            raise ValueError(
                f"{path} has no 'diagnosis' attribute. Recreate the dataset with "
                "convert_acdc_nifti_to_h5.py and the spacing-aware preprocessor."
            )
        diagnosis = h5_file.attrs["diagnosis"]
    if isinstance(diagnosis, bytes):
        diagnosis = diagnosis.decode("utf-8")
    return str(diagnosis).strip()


def split_by_patient(files, val_fraction, seed):
    """Make a patient-level split stratified by ACDC diagnosis."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    representative_file = {}
    for path in files:
        representative_file.setdefault(get_patient_id(path), path)

    patients_by_diagnosis = {}
    for patient_id, path in representative_file.items():
        diagnosis = _read_diagnosis(path)
        patients_by_diagnosis.setdefault(diagnosis, []).append(patient_id)

    rng = random.Random(seed)
    val_patients = set()
    for diagnosis in sorted(patients_by_diagnosis):
        patients = sorted(patients_by_diagnosis[diagnosis])
        rng.shuffle(patients)
        num_val = max(1, round(len(patients) * val_fraction))
        if num_val >= len(patients):
            raise ValueError(
                f"Diagnosis {diagnosis!r} has only {len(patients)} patients; "
                "the requested validation fraction leaves none for training."
            )
        val_patients.update(patients[:num_val])

    all_patients = set(representative_file)
    train_patients = all_patients - val_patients
    train_files = [path for path in files if get_patient_id(path) in train_patients]
    val_files = [path for path in files if get_patient_id(path) in val_patients]
    return train_files, val_files, sorted(train_patients), sorted(val_patients)


def build_stratified_folds(files, num_folds=5, seed=42):
    """Return deterministic, diagnosis-stratified patient folds.

    Each item is a dictionary containing train/validation files and patient IDs.
    Files belonging to the same patient can never cross fold boundaries.
    """
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    representative_file = {}
    for path in files:
        representative_file.setdefault(get_patient_id(path), path)
    if not representative_file:
        raise ValueError("Cannot build folds from an empty file list")

    patients_by_diagnosis = defaultdict(list)
    for patient_id, path in representative_file.items():
        patients_by_diagnosis[_read_diagnosis(path)].append(patient_id)

    fold_patients = [set() for _ in range(num_folds)]
    rng = random.Random(seed)
    for diagnosis in sorted(patients_by_diagnosis):
        patients = sorted(patients_by_diagnosis[diagnosis])
        if len(patients) < num_folds:
            raise ValueError(
                f"Diagnosis {diagnosis!r} has {len(patients)} patients, fewer than {num_folds} folds"
            )
        rng.shuffle(patients)
        for index, patient_id in enumerate(patients):
            fold_patients[index % num_folds].add(patient_id)

    all_patients = set(representative_file)
    validation_union = set().union(*fold_patients)
    if validation_union != all_patients or sum(len(group) for group in fold_patients) != len(all_patients):
        raise RuntimeError("Invalid folds: validation sets are not disjoint and exhaustive")

    folds = []
    for fold_index, val_patients_set in enumerate(fold_patients):
        train_patients_set = all_patients - val_patients_set
        train_files = [path for path in files if get_patient_id(path) in train_patients_set]
        val_files = [path for path in files if get_patient_id(path) in val_patients_set]
        if not train_files or not val_files:
            raise RuntimeError(f"Fold {fold_index} has an empty training or validation set")
        folds.append(
            {
                "fold": fold_index,
                "train_files": train_files,
                "val_files": val_files,
                "train_patients": sorted(train_patients_set),
                "val_patients": sorted(val_patients_set),
            }
        )
    return folds


def require_cuda():
    """Require a usable CUDA device; training is intentionally GPU-only."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA GPU training is required, but PyTorch cannot access a CUDA device. "
            "Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available() before retrying."
        )
    device = torch.device("cuda:0")
    # Force CUDA initialization here so driver/runtime failures happen before outputs are created.
    torch.empty(1, device=device)
    print(f"CUDA device: {torch.cuda.get_device_name(device)} ({device})")
    return device


def verify_model_on_cuda(model):
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise RuntimeError("Cannot verify CUDA placement for a model with no parameters") from error
    if not parameter.is_cuda:
        raise RuntimeError("Training model is not on CUDA; refusing to continue")


def validate_spacing_metadata(files, expected_ndim=3):
    """Validate and summarize physical spacing stored in preprocessed HDF5 files."""
    spacings = []
    for path in files:
        with h5py.File(path, "r") as h5_file:
            for dataset_name in ("image", "label"):
                if dataset_name not in h5_file:
                    raise ValueError(f"{path} is missing required dataset {dataset_name!r}")
            if h5_file["image"].shape != h5_file["label"].shape:
                raise ValueError(f"Image/label shape mismatch in {path}")
            try:
                from acdc_h5 import read_spacing_zyx
            except ModuleNotFoundError:
                from scripts.acdc_h5 import read_spacing_zyx

            spacing = np.asarray(read_spacing_zyx(h5_file), dtype=np.float32).reshape(-1)
            if spacing.size != expected_ndim or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
                raise ValueError(f"Invalid {expected_ndim}D spacing metadata in {path}: {spacing.tolist()}")
            spacings.append(spacing)
    spacing_array = np.stack(spacings)
    return {
        "spacing_zyx_min": spacing_array.min(axis=0).astype(float).tolist(),
        "spacing_zyx_max": spacing_array.max(axis=0).astype(float).tolist(),
        "spacing_zyx_mean": spacing_array.mean(axis=0).astype(float).tolist(),
    }


def diagnosis_counts(files, patient_ids):
    wanted = set(patient_ids)
    seen = set()
    counts = {}
    for path in files:
        patient_id = get_patient_id(path)
        if patient_id in wanted and patient_id not in seen:
            diagnosis = _read_diagnosis(path)
            counts[diagnosis] = counts.get(diagnosis, 0) + 1
            seen.add(patient_id)
    return dict(sorted(counts.items()))


def add_scheduler_arguments(parser):
    parser.add_argument(
        "--lr-scheduler",
        choices=["reduce_on_plateau", "none"],
        default="reduce_on_plateau",
        help="Learning-rate schedule; reduce_on_plateau monitors validation loss.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)


def build_scheduler(optimizer, args):
    if args.lr_scheduler == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )
