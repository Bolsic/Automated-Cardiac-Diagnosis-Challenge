# Project Handoff: Spacing-Aware ACDC Five-Fold Validation

## Goal and dataset

This repository reproduces cardiac MRI segmentation experiments for FCN-8,
2D U-Net, modified 2D U-Net, and modified 3D U-Net. The target classes are
background, right ventricle (RV), myocardium, and left ventricle (LV).

The downloaded ACDC dataset is under `database/`:

- `database/training`: patients 001–100, with 20 patients in each diagnosis
  group (DCM, HCM, MINF, NOR, and RV).
- `database/testing`: patients 101–150, also fully labeled and balanced.
- Every patient has labeled ED and ES volumes.
- Native in-plane spacing ranges from approximately 0.703 to 1.953 mm,
  through-plane spacing from 5 to 10 mm, and volume depth from 6 to 21 slices.
- Raw medical data and generated HDF5 datasets are ignored by Git.

Patients 001–100 are used for five-fold cross-validation. Patients 101–150
remain held out for final ensemble evaluation.

## Implemented pipeline

The project now provides spacing-aware conversion, preprocessing, training,
validation, and testing:

- NIfTI conversion preserves patient ID, diagnosis, cardiac phase, affine,
  image shape, and spacing in `Z × Y × X` order.
- The 2D models resample images and masks to 1.37 × 1.37 mm in-plane spacing.
  FCN-8 uses 224 × 224 inputs; both U-Net variants use 396 × 396 inputs.
- The 3D model resamples volumes to 5.0 × 2.5 × 2.5 mm and pads/crops to
  60 × 204 × 204.
- Images use linear interpolation and per-image standardization. Labels use
  nearest-neighbor interpolation.
- Spacing is used to normalize physical scale before training; it is not fed
  to the networks as an extra input channel.
- All training scripts require CUDA and terminate before creating outputs if
  no CUDA device is available.
- Five folds are deterministic and diagnosis-stratified. Each fold has 80
  training and 20 validation patients, including exactly 16/4 patients from
  every diagnosis group.
- All slices and both cardiac phases from one patient remain in the same fold.
- Each fold saves configuration, epoch metrics, and best/latest checkpoints.
- Best checkpoints are selected by validation foreground Dice.
- Final 2D Dice, ASSD, and Hausdorff distance are calculated on reconstructed
  3D patient/frame volumes, not on independent slices.
- Evaluation applies the largest connected component per foreground class.
- ASSD and HD use physical `Z/Y/X` spacing and are reported in millimetres.
- Held-out testing averages softmax probabilities from the five best fold
  checkpoints, followed by argmax and volume-level evaluation.

The main shared implementation is in:

- `scripts/cross_validation_training.py`
- `scripts/cv_reporting.py`
- `scripts/training_utils.py`

The four entry points are:

- `scripts/train_fcn8.py`
- `scripts/train_unet2d.py`
- `scripts/train_unet2d_modified.py`
- `scripts/train_unet3d.py`

## Reproduction commands

Run from the repository root:

```powershell
python scripts\convert_acdc_nifti_to_h5.py

python scripts\preprocess_acdc_2d.py --architecture fcn8 --split training
python scripts\preprocess_acdc_2d.py --architecture fcn8 --split testing
python scripts\preprocess_acdc_2d.py --architecture unet2d --split training
python scripts\preprocess_acdc_2d.py --architecture unet2d --split testing

python scripts\preprocess_acdc_3d.py --split training
python scripts\preprocess_acdc_3d.py --split testing
```

Train all five folds and evaluate the test ensemble:

```powershell
python scripts\train_fcn8.py --epochs 100 --batch-size 4 --run-dir runs\fcn8_cv
python scripts\train_unet2d.py --epochs 100 --batch-size 2 --run-dir runs\unet2d_cv
python scripts\train_unet2d_modified.py --epochs 100 --batch-size 2 --loss weighted_cross_entropy --run-dir runs\unet2d_modified_cv
python scripts\train_unet3d.py --epochs 100 --batch-size 1 --run-dir runs\unet3d_cv
```

Use `--fold 0` through `--fold 4` to train one fold. Add `--num-workers 0`
if Windows multiprocessing causes trouble.

## Completed results

All four five-fold experiments and held-out ensembles completed successfully.

| Model | CV Dice (mean ± SD) | CV ASSD (mm) | CV HD (mm) | Test Dice | Test ASSD (mm) | Test HD (mm) |
|---|---:|---:|---:|---:|---:|---:|
| FCN-8 | 0.8810 ± 0.0105 | 0.9210 | 11.8831 | 0.8934 | 0.8937 | 10.5639 |
| 2D U-Net | 0.9025 ± 0.0064 | 0.7532 | 10.5616 | 0.9109 | 0.7613 | 9.8436 |
| Modified 2D U-Net | 0.9006 ± 0.0043 | 0.7807 | 10.5887 | 0.9085 | 0.8096 | 9.8810 |
| Modified 3D U-Net | 0.8407 ± 0.0176 | 1.8843 | 13.2831 | 0.8583 | 1.4895 | 11.4728 |

Run artifacts are stored under `runs/*_cv/`. Each run contains fold
assignments, per-fold configs and histories, validation metrics, cross-fold
summaries, and held-out test-ensemble metrics.

## Checkpoints and storage

Local `.pt` files are PyTorch checkpoints containing model weights, Adam
optimizer state, epoch, metrics, arguments, patient split, and elapsed time.
Each fold has:

- `best_epoch_XXX.pt`: best validation foreground Dice; used for evaluation.
- `latest_epoch_100.pt`: final epoch; useful for inspection/resuming.

The checkpoints total approximately 11.54 GB and individual files are roughly
289–475 MB. They remain local and are ignored by `*.pt`; they were not pushed
because they exceed normal GitHub file limits and would require Git LFS quota.
All lightweight run data (configs, CSV histories, and JSON summaries) is
tracked and pushed.

## Verification and Git state

- Test suite: 16 tests passing with `python -m pytest`.
- Lint: `python -m ruff check scripts tests` passes.
- CUDA hardware used: NVIDIA GeForce RTX 3050 with 6 GB VRAM.
- `k-fold-validation` commit: `1ed93dd`.
- Merge commit on `main`: `5bc4f93`.
- Both `origin/k-fold-validation` and `origin/main` were pushed successfully.
- The repository was on `main`, synchronized with `origin/main`, immediately
  before this handoff file was created.

## Operational notes

- Standard 2D U-Net took about 42.4 hours for all five folds on the RTX 3050;
  FCN-8 took about 8.2 hours. This difference was expected from the 396 × 396
  U-Net inputs and high-resolution feature maps, not a data-loading problem.
- Mixed-precision training and early stopping are not implemented. They are
  reasonable future improvements if runtime matters.
- Training epoch Dice is accumulated over 2D pixels/batches for 2D models.
  Published validation and test metrics are computed from reconstructed 3D
  volumes and should be treated as authoritative.
- Do not commit `database/`, `outputs/`, or the local `.pt` checkpoints without
  explicitly planning external artifact storage or Git LFS capacity.
