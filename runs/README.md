# Experiment records

This directory contains lightweight, versioned provenance for both the final
five-fold experiments and earlier single-split runs. Model checkpoints are
intentionally excluded from Git because of their size.

## Current five-fold runs

| Directory | Model | Loss | Reported CV Dice | Local ensemble Dice |
|---|---|---|---:|---:|
| `fcn8_cv/` | FCN-8 | Cross-entropy | 0.8810 ± 0.0105 | 0.8934 |
| `unet2d_cv/` | Standard 2D U-Net | Cross-entropy | **0.9025 ± 0.0064** | **0.9109** |
| `unet2d_modified_cv/` | Modified 2D U-Net | Weighted cross-entropy | 0.9006 ± 0.0043 | 0.9085 |
| `unet3d_cv/` | Anisotropic 3D U-Net | Cross-entropy | 0.8407 ± 0.0176 | 0.8583 |

These four directories are the machine-readable source for the current report.
For each architecture, patients 001–100 are divided into five deterministic,
diagnosis-stratified folds. Every fold trains on 80 patients and validates on
20, with four validation patients from each ACDC diagnostic group. The final
ensemble evaluation covers 100 ED/ES volumes from patients 101–150.

Each `*_cv/` directory has this structure:

```text
model_cv/
├── fold_assignments.json
├── cross_validation_metrics.csv
├── cross_validation_summary.json
├── fold_0/
│   ├── config.json
│   ├── metrics.csv
│   └── evaluation_2d_val/ or evaluation_3d_val/
├── fold_1/ ... fold_4/
└── test_ensemble/
    ├── metrics_by_class.csv
    └── summary.json
```

The fold configuration records hyperparameters, train/validation patients,
diagnosis counts, spatial metadata, GPU information, and the random seed.
`metrics.csv` contains epoch-level slice or volume training metrics. Evaluation
directories contain per-volume, per-class Dice, ASSD, and Hausdorff distance,
plus an aggregate summary.

Checkpoint names referenced by the JSON files are not present in a clean clone.
Recreate them by training, or copy them separately before rerunning inference
or qualitative prediction scripts.

## Earlier single-split runs

The following directories predate the final five-fold protocol and are kept as
historical records:

| Directory | Purpose |
|---|---|
| `fcn8_100epoch_gpu_bs16_lr1e-3/` | Earlier FCN-8 baseline |
| `unet2d_200epoch_gpu_bs4_lr1e-4/` | Earlier standard 2D U-Net baseline |
| `unet2d_modified_100epoch_weighted_ce/` | Earlier modified U-Net weighted-CE run |
| `unet2d_modified_100epoch_dice_loss/` | Earlier modified U-Net Dice-loss ablation |
| `unet3d/` | Earlier anisotropic 3D U-Net run |

Each historical run contains its `config.json`, `metrics.csv`, and cached
validation evaluation. These runs feed the legacy CSVs under `results/` and
the notebooks, but they should not be used to describe the final model ranking.

## Metric interpretation

- Cross-validation summaries report the mean and population standard deviation
  of fold-level macro metrics.
- A validation fold contains 40 reconstructed ED/ES volumes from 20 patients.
- Final ensemble summaries cover 100 volumes from 50 held-out patients.
- ASSD and Hausdorff distance are measured in millimetres using saved physical
  spacing.
- Predictions are post-processed by keeping the largest connected component
  independently for RV, myocardium, and LV.
- The final ensemble averages the five models' softmax probabilities before
  `argmax` and post-processing.
