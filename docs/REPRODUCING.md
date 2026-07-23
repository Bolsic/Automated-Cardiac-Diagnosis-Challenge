# Reproducing the pipeline

Run every command from the repository root. Generated datasets, checkpoints,
and outputs are intentionally ignored by Git.

## 1. Environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Install the appropriate PyTorch build from
[pytorch.org](https://pytorch.org/get-started/locally/) first if the default
package does not match your CUDA runtime.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pytest
```

## 2. Dataset layout

Download ACDC from the
[official challenge page](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).
The expected input layout is:

```text
ACDC/database/
├── training/
│   ├── patient001/
│   └── ...
└── testing/
```

The raw data is never committed.

## 3. Preserve spatial metadata

Convert NIfTI images to HDF5 while retaining spacing, affine, cardiac phase,
diagnosis, and patient identifiers:

```bash
python scripts/convert_acdc_nifti_to_h5.py
```

Output:

```text
outputs/acdc_h5_with_metadata/
├── ACDC_training_volumes/
├── ACDC_training_slices/
├── ACDC_testing_volumes/
└── ACDC_testing_slices/
```

## 4. Preprocess

FCN-8 uses 224 × 224 slices. Both 2D U-Net variants use 396 × 396 slices.
In-plane spacing is resampled to 1.37 × 1.37 mm.

```bash
python scripts/preprocess_acdc_2d.py --architecture fcn8
python scripts/preprocess_acdc_2d.py --architecture unet2d
```

The 3D workflow resamples to 5.0 × 2.5 × 2.5 mm in `Z × Y × X` order and
crops/pads to 60 × 204 × 204:

```bash
python scripts/preprocess_acdc_3d.py
```

Images are normalized to zero mean and unit variance. Labels use nearest
neighbour interpolation. Training scripts make a seeded, diagnosis-stratified
split at patient level.

## 5. Train

Representative commands:

```bash
python scripts/train_fcn8.py \
  --epochs 100 --batch-size 8 --run-dir runs/fcn8

python scripts/train_unet2d.py \
  --epochs 100 --batch-size 8 --run-dir runs/unet2d

python scripts/train_unet2d_modified.py \
  --epochs 100 --batch-size 8 --loss weighted_cross_entropy \
  --run-dir runs/unet2d_modified_weighted_ce

python scripts/train_unet3d.py \
  --epochs 100 --batch-size 1 --run-dir runs/unet3d
```

Every run stores its arguments and patient split in `config.json`, an
epoch-level `metrics.csv`, and best/latest checkpoints. Use `--num-workers 0`
if multiprocessing is unavailable.

Quick 2D smoke test:

```bash
python scripts/train_unet2d_modified.py \
  --epochs 1 --batch-size 1 --num-workers 0 --base-channels 8 \
  --max-train-samples 2 --max-val-samples 2 \
  --run-dir /tmp/acdc_unet2d_smoke
```

Quick 3D smoke test:

```bash
python scripts/train_unet3d.py \
  --epochs 1 --batch-size 1 --num-workers 0 --base-channels 4 \
  --patch-depth 16 --patch-height 32 --patch-width 32 \
  --max-train-samples 1 --max-val-samples 1 \
  --run-dir /tmp/acdc_unet3d_smoke
```

## 6. Evaluate complete volumes

For a 2D network, slices are predicted in batches and reconstructed into one
volume per patient/frame before metrics are calculated:

```bash
python scripts/evaluate_2d.py \
  --run-dir runs/unet2d_modified_weighted_ce \
  --model unet2d_modified --split val
```

Evaluate the 3D network directly:

```bash
python scripts/evaluate_3d.py --run-dir runs/unet3d --split val
```

The evaluator reports Dice, average symmetric surface distance (ASSD), and
Hausdorff distance for RV, myocardium, and LV. ASSD and Hausdorff are reported
in millimetres and require `spacing_zyx` metadata. Each class prediction is
reduced to its largest connected component before scoring.

## 7. Analyze

- `notebooks/evaluate_all_runs_volume_reconstruction.ipynb` discovers selected
  runs, evaluates checkpoints when available, and compares cached results.
- `notebooks/plot_best_runs_training.ipynb` selects the best evaluated run per
  architecture and plots training histories.

The compact published tables are in [`results/`](../results/README.md).
