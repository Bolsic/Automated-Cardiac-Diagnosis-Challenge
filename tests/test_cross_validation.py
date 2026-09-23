import h5py
import numpy as np
import pytest
import torch

from scripts.cv_reporting import _ensemble_probabilities_2d, aggregate_cross_validation
from scripts.training_utils import build_stratified_folds, require_cuda


def _write_patient_file(path, diagnosis):
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("image", data=np.zeros((4, 4), dtype=np.float32))
        h5_file.create_dataset("label", data=np.zeros((4, 4), dtype=np.uint8))
        h5_file.attrs["diagnosis"] = diagnosis
        h5_file.attrs["spacing_zyx"] = [10.0, 1.37, 1.37]


def test_five_folds_are_stratified_disjoint_and_exhaustive(tmp_path):
    files = []
    diagnoses = ["DCM", "HCM", "MINF", "NOR", "RV"]
    for patient_id in range(1, 101):
        diagnosis = diagnoses[(patient_id - 1) // 20]
        for frame in (1, 2):
            path = tmp_path / f"patient{patient_id:03d}_frame{frame:02d}.h5"
            _write_patient_file(path, diagnosis)
            files.append(path)

    folds = build_stratified_folds(files, num_folds=5, seed=42)
    validation_sets = [set(fold["val_patients"]) for fold in folds]
    assert all(len(fold["train_patients"]) == 80 for fold in folds)
    assert all(len(fold["val_patients"]) == 20 for fold in folds)
    assert len(set().union(*validation_sets)) == 100
    assert sum(len(group) for group in validation_sets) == 100

    for fold in folds:
        assert len(fold["train_files"]) == 160
        assert len(fold["val_files"]) == 40
        counts = {diagnosis: 0 for diagnosis in diagnoses}
        for patient_id in fold["val_patients"]:
            counts[diagnoses[(patient_id - 1) // 20]] += 1
        assert set(counts.values()) == {4}

    repeated = build_stratified_folds(files, num_folds=5, seed=42)
    assert [fold["val_patients"] for fold in repeated] == [fold["val_patients"] for fold in folds]


def test_folds_reject_diagnosis_with_too_few_patients(tmp_path):
    files = []
    for patient_id in range(1, 5):
        path = tmp_path / f"patient{patient_id:03d}_frame01.h5"
        _write_patient_file(path, "NOR")
        files.append(path)
    with pytest.raises(ValueError, match="fewer than 5 folds"):
        build_stratified_folds(files, num_folds=5, seed=42)


def test_gpu_guard_rejects_cpu_only_environment(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="CUDA GPU training is required"):
        require_cuda()


def test_cross_validation_summary_contains_fold_mean_and_std(tmp_path):
    fold_rows = {}
    for fold in range(5):
        rows = []
        for class_id, class_name in ((1, "RV"), (2, "Myo"), (3, "LV")):
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "dice": 0.5 + fold * 0.1,
                    "assd_mm": 2.0 + fold,
                    "hd_mm": 10.0 + fold,
                }
            )
        fold_rows[fold] = rows

    metrics_path, summary_path = aggregate_cross_validation(tmp_path, fold_rows)
    assert metrics_path.exists()
    summary = __import__("json").loads(summary_path.read_text())
    assert summary["num_folds"] == 5
    assert summary["foreground_macro"]["mean_dice"] == pytest.approx(0.7)
    assert summary["foreground_macro"]["std_dice"] == pytest.approx(np.std([0.5, 0.6, 0.7, 0.8, 0.9]))


def test_2d_ensemble_averages_softmax_probabilities(tmp_path):
    path = tmp_path / "patient101_frame01_slice_0.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("image", data=np.zeros((2, 2), dtype=np.float32))
        h5_file.create_dataset("label", data=np.zeros((2, 2), dtype=np.uint8))
        h5_file.attrs["spacing_zyx"] = [10.0, 1.37, 1.37]
        h5_file.attrs["phase"] = "ED"

    class ConstantModel(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("constant_logits", torch.tensor(logits, dtype=torch.float32))

        def forward(self, images):
            values = self.constant_logits[None, :, None, None]
            return values.expand(images.shape[0], -1, images.shape[2], images.shape[3])

    models = [ConstantModel([10.0, 0.0, 0.0, 0.0]), ConstantModel([0.0, 1.0, 0.0, 0.0])]
    prediction, target, spacing, phase = _ensemble_probabilities_2d(
        models, [path], batch_size=1, device=torch.device("cpu")
    )
    assert np.all(prediction == 0)
    assert target.shape == (1, 2, 2)
    assert np.allclose(spacing, [10.0, 1.37, 1.37])
    assert phase == "ED"
