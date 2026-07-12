import random

import h5py
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
