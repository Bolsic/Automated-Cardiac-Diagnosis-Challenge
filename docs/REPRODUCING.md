# Reproducing the current five-fold pipeline

Run all commands from the repository root. Generated datasets, checkpoints,
and general outputs are intentionally ignored by Git. The lightweight metrics
already committed under `runs/` are sufficient to inspect the reported
results, but rerunning inference or training requires the ACDC data and model
checkpoints.

## 1. Create the environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Install the appropriate PyTorch build from
[pytorch.org](https://pytorch.org/get-started/locally/) first if the default
package does not match the local CUDA runtime.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pytest
```

Tests and report generation can run on CPU. All training entry points require
a working CUDA device and exit before writing run artifacts if CUDA is not
available.

## 2. Arrange the ACDC dataset

Download ACDC from the
[official challenge page](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).
The converter automatically recognizes either of these layouts:

```text
ACDC/database/
├── training/
│   ├── patient001/
│   └── ...
└── testing/
    ├── patient101/
    └── ...
```

or:

```text
database/
├── training/
└── testing/
```

Each patient directory is expected to contain `Info.cfg`, frame NIfTI images,
and, for evaluated data, matching `_gt.nii.gz` masks. Raw data is never
committed.

## 3. Convert NIfTI to metadata-preserving HDF5

```bash
python scripts/convert_acdc_nifti_to_h5.py --input-root ACDC/database
```

The converter changes array order from NIfTI `X × Y × Z` to project order
`Z × Y × X` and retains the affine, source spacing, phase, diagnosis, patient,
frame, and split metadata. By default it writes both volumes and slices:

```text
outputs/acdc_h5_with_metadata/
├── ACDC_training_volumes/
├── ACDC_training_slices/
├── ACDC_testing_volumes/
└── ACDC_testing_slices/
```

Use `--no-slices` only when the 2D pipeline is not needed.

## 4. Create spacing-aware model inputs

### 2D models

FCN-8 uses `224 × 224` slices. Both U-Net variants use the same `396 × 396`
inputs. All 2D inputs are resampled in-plane to `1.37 × 1.37 mm` before
normalization and center cropping/padding.

```bash
python scripts/preprocess_acdc_2d.py --architecture fcn8
python scripts/preprocess_acdc_2d.py --architecture unet2d
python scripts/preprocess_acdc_2d.py --architecture fcn8 --split testing
python scripts/preprocess_acdc_2d.py --architecture unet2d --split testing
```

Output:

```text
outputs/acdc_preprocessed_2d_spacing/
├── fcn8/
│   ├── ACDC_training_slices/
│   └── ACDC_testing_slices/
└── unet2d/
    ├── ACDC_training_slices/
    └── ACDC_testing_slices/
```

The modified U-Net uses the `unet2d` inputs because its required shape and
spacing are identical to those of the standard 2D U-Net.

### 3D model

The 3D workflow resamples to `5.0 × 2.5 × 2.5 mm` in `Z × Y × X` order and
crops or pads every volume to `60 × 204 × 204`.

```bash
python scripts/preprocess_acdc_3d.py
python scripts/preprocess_acdc_3d.py --split testing
```

Output:

```text
outputs/acdc_preprocessed_3d_spacing/
├── ACDC_training_volumes/
└── ACDC_testing_volumes/
```

Images are normalized to zero mean and unit variance before padding. Labels
are resampled with nearest-neighbour interpolation; images use linear
interpolation.

## 5. Train the five folds

The reported runs used 100 epochs, Adam with learning rate `0.01`,
`ReduceLROnPlateau`, seed `42`, and no weight decay. FCN-8, standard 2D U-Net,
and 3D U-Net use cross-entropy. The modified 2D U-Net uses weighted
cross-entropy with background weight `0.1` and foreground-class weight `0.3`.

```bash
python scripts/train_fcn8.py \
  --epochs 100 --batch-size 4 --learning-rate 0.01 \
  --run-dir runs/fcn8_cv

python scripts/train_unet2d.py \
  --epochs 100 --batch-size 2 --learning-rate 0.01 \
  --run-dir runs/unet2d_cv

python scripts/train_unet2d_modified.py \
  --epochs 100 --batch-size 2 --learning-rate 0.01 \
  --loss weighted_cross_entropy \
  --run-dir runs/unet2d_modified_cv

python scripts/train_unet3d.py \
  --epochs 100 --batch-size 1 --learning-rate 0.01 \
  --run-dir runs/unet3d_cv
```

Without `--fold`, a trainer performs all five folds, evaluates each best
checkpoint on its validation patients, aggregates the CV metrics, and finally
evaluates the five-model softmax ensemble on the testing directory. A complete
four-architecture experiment trains 20 models.

The patient split is deterministic and diagnosis-stratified. In every fold,
80 of patients 001–100 are used for training and 20 for validation. Every
diagnostic group contributes 16 training and four validation patients.
Patients 101–150 are reserved for final ensemble evaluation.

Each fold directory contains:

- `config.json`, including arguments, patient IDs, diagnosis counts, spacing,
  device, and start time;
- `metrics.csv`, containing epoch-level training and validation history;
- `best_epoch_*.pt` and `latest_epoch_*.pt` checkpoints; and
- `evaluation_2d_val/` or `evaluation_3d_val/` volume-level metrics.

The run root contains `fold_assignments.json`,
`cross_validation_metrics.csv`, `cross_validation_summary.json`, and
`test_ensemble/`. Checkpoints are intentionally excluded from Git.

To train only one fold, for example for separate GPU scheduling:

```bash
python scripts/train_unet2d.py --fold 2 --run-dir runs/unet2d_cv
```

A single-fold invocation does not create the aggregate CV summary or final
five-model ensemble. Run all folds together, or aggregate them after all fold
checkpoints are present.

### Smoke tests

Quick 2D smoke test:

```bash
python scripts/train_unet2d_modified.py --fold 0 \
  --epochs 1 --batch-size 1 --num-workers 0 --base-channels 8 \
  --max-train-samples 2 --max-val-samples 2 \
  --run-dir /tmp/acdc_unet2d_smoke
```

Quick 3D smoke test:

```bash
python scripts/train_unet3d.py --fold 0 \
  --epochs 1 --batch-size 1 --num-workers 0 --base-channels 4 \
  --patch-depth 16 --patch-height 32 --patch-width 32 \
  --max-train-samples 1 --max-val-samples 1 \
  --run-dir /tmp/acdc_unet3d_smoke
```

## 6. Evaluate complete validation volumes

The trainers perform this evaluation automatically. To repeat it for a saved
fold, supply a run directory containing its configuration and checkpoint.

For a 2D model, slices are predicted in batches and reconstructed into one
volume per patient and cardiac phase before metrics are calculated:

```bash
python scripts/evaluate_2d.py \
  --run-dir runs/unet2d_cv/fold_0 \
  --model unet2d --split val
```

Evaluate a 3D model directly:

```bash
python scripts/evaluate_3d.py \
  --run-dir runs/unet3d_cv/fold_0 \
  --model unet3d --split val
```

Evaluation reports Dice, average symmetric surface distance (ASSD), and full
Hausdorff distance (HD) for RV, myocardium, and LV. ASSD and HD use physical
`spacing_zyx` metadata and are reported in millimetres. Each foreground class
is reduced to its largest six-connected component before scoring.

## 7. Regenerate report figures

The main CV plots need only the committed lightweight run records. The spacing
plot additionally reads converted training volumes:

```bash
python scripts/generate_cv_report_figures.py
```

This writes the report plots to `report/figures_cv/`.

The qualitative out-of-fold visualizations require the ignored checkpoints,
preprocessed datasets, and CUDA:

```bash
python scripts/visualize_all_models_oof.py
python scripts/visualize_cv_oof_predictions.py
```

The notebooks under `notebooks/` predate the final five-fold analysis and are
retained for the earlier selected-run evaluation and visualization workflow.
The authoritative current results are the `*_cv` records under `runs/` and the
tables in `report/izvestaj_cv.pdf`.
