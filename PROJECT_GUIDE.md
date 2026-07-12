# ACDC Segmentation Project Guide

This project trains PyTorch FCN-8, 2D U-Net, modified 2D U-Net, and 3D U-Net segmentation models on the ACDC preprocessed dataset.

## 1. Environment Setup

Run commands from the repository root:

```bash
cd /home/basic/Faks/Automated-Cardiac-Diagnosis-Challenge
source .venv/bin/activate
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Check whether PyTorch can see a GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The training script automatically uses CUDA when available. If CUDA is not available, it runs on CPU.

## 2. Spacing-Aware Data Preparation

The original ACDC NIfTI files contain voxel spacing in their headers. The older
`ACDC_preprocessed` HDF5 files do not preserve that metadata, so the current
workflow starts by converting the NIfTI files to HDF5 again while keeping the
spacing, affine, phase, and diagnosis metadata.

Create metadata-preserving HDF5 volumes and slices:

```bash
source .venv/bin/activate && python scripts/convert_acdc_nifti_to_h5.py
```

This writes:

```text
outputs/acdc_h5_with_metadata/ACDC_training_volumes/
outputs/acdc_h5_with_metadata/ACDC_training_slices/
outputs/acdc_h5_with_metadata/ACDC_testing_volumes/
outputs/acdc_h5_with_metadata/ACDC_testing_slices/
```

Each converted HDF5 file stores:

- `image`: image array in `Z x Y x X` order for volumes, or `Y x X` for slices
- `label`: segmentation label map when ground truth is available
- `spacing_zyx`: physical voxel spacing in millimetres in the same axis order as the stored array
- `spacing_xyz`, `affine`, `pixdim`: original NIfTI spatial metadata
- `phase`, `diagnosis`, `patient_id`, `frame`: ACDC metadata from the NIfTI/`Info.cfg` files

The 2D paper preprocessing resamples only the in-plane axes to `1.37 x 1.37 mm`
and preserves the original through-plane spacing. Export FCN-8 inputs at
`224 x 224`:

```text
source .venv/bin/activate && python scripts/preprocess_acdc_2d.py --architecture fcn8
```

Export 2D U-Net and modified 2D U-Net inputs at `396 x 396`:

```bash
source .venv/bin/activate && python scripts/preprocess_acdc_2d.py --architecture unet2d
```

These commands write:

```text
outputs/acdc_preprocessed_2d_spacing/fcn8/ACDC_training_slices/
outputs/acdc_preprocessed_2d_spacing/unet2d/ACDC_training_slices/
```

The 3D paper preprocessing resamples volumes to `5.0 x 2.5 x 2.5 mm`
(`Z x Y x X`) and then pads/crops to `60 x 204 x 204` voxels:

```bash
source .venv/bin/activate && python scripts/preprocess_acdc_3d.py
```

This writes:

```text
outputs/acdc_preprocessed_3d_spacing/ACDC_training_volumes/
```

Training scripts now default to these spacing-aware output folders and record a
`spacing_metadata` summary in each run's `config.json`.

## 3. Start FCN-8 Training

Basic FCN-8 3-epoch local run:

```bash
source .venv/bin/activate && python scripts/train_fcn8.py --epochs 3 --batch-size 8 --num-workers 2 --run-dir runs/fcn8_3epoch_local
```

Longer default-style run:

```bash
source .venv/bin/activate && python scripts/train_fcn8.py --epochs 100 --batch-size 8 --num-workers 2 --run-dir runs/fcn8
```

If `num-workers 2` causes a multiprocessing/DataLoader error, use:

```bash
source .venv/bin/activate && python scripts/train_fcn8.py --epochs 3 --batch-size 8 --num-workers 0 --run-dir runs/fcn8_3epoch_local
```

## 4. Start 2D U-Net Training

Basic 2D U-Net 3-epoch local run:

```bash
source .venv/bin/activate && python scripts/train_unet2d.py --epochs 3 --batch-size 8 --num-workers 2 --run-dir runs/unet2d_3epoch_local
```

Longer default-style run:

```bash
source .venv/bin/activate && python scripts/train_unet2d.py --epochs 100 --batch-size 8 --num-workers 2 --run-dir runs/unet2d
```

If `num-workers 2` causes a multiprocessing/DataLoader error, use:

```bash
source .venv/bin/activate && python scripts/train_unet2d.py --epochs 3 --batch-size 8 --num-workers 0 --run-dir runs/unet2d_3epoch_local
```

## 5. Start Modified 2D U-Net Training

The modified 2D U-Net follows the paper's lighter decoder idea: each transposed-convolution upsampling layer outputs `num_classes` channels instead of a wide decoder feature map.

Basic modified 2D U-Net 3-epoch local run:

```bash
source .venv/bin/activate && python scripts/train_unet2d_modified.py --epochs 3 --batch-size 8 --num-workers 2 --run-dir runs/unet2d_modified_3epoch_local
```

Longer default-style run:

```bash
source .venv/bin/activate && python scripts/train_unet2d_modified.py --epochs 100 --batch-size 8 --num-workers 2 --run-dir runs/unet2d_modified
```

If `num-workers 2` causes a multiprocessing/DataLoader error, use:

```bash
source .venv/bin/activate && python scripts/train_unet2d_modified.py --epochs 3 --batch-size 8 --num-workers 0 --run-dir runs/unet2d_modified_3epoch_local
```

## 6. Start 3D U-Net Training

First export the 3D volumes:

```bash
source .venv/bin/activate && python scripts/preprocess_acdc_3d.py
```

Basic 3D U-Net 3-epoch local run:

```bash
source .venv/bin/activate && python scripts/train_unet3d.py --epochs 3 --batch-size 1 --num-workers 2 --run-dir runs/unet3d_3epoch_local
```

Lower-memory patch-based run:

```bash
source .venv/bin/activate && python scripts/train_unet3d.py --epochs 3 --batch-size 1 --num-workers 2 --patch-depth 32 --patch-height 128 --patch-width 128 --run-dir runs/unet3d_patch_3epoch_local
```

If `num-workers 2` causes a multiprocessing/DataLoader error, use:

```bash
source .venv/bin/activate && python scripts/train_unet3d.py --epochs 3 --batch-size 1 --num-workers 0 --run-dir runs/unet3d_3epoch_local
```

## 7. Useful Training Options

The most useful command-line options are:

```text
--data-dir              Folder containing preprocessed .h5 slices
--run-dir               Folder where checkpoints and metrics are saved
--weights               Optional saved weights to load before training
--epochs                Number of training epochs
--batch-size            Number of slices per batch
--num-workers           DataLoader worker processes
--learning-rate         Adam learning rate, default 0.01
--base-channels         U-Net channel width, default 64 for 2D and 16 for 3D
--classifier-channels   FCN-8 classifier width, default 1024
--patch-depth           Optional 3D patch depth
--patch-height          Optional 3D patch height
--patch-width           Optional 3D patch width
--val-fraction          Fraction of patients used for validation, default 0.2
--seed                  Random seed for reproducibility
--max-train-samples     Limit training samples for a quick test
--max-val-samples       Limit validation samples for a quick test
```

Very small FCN-8 smoke test:

```bash
source .venv/bin/activate && python scripts/train_fcn8.py --epochs 1 --batch-size 1 --num-workers 0 --classifier-channels 64 --max-train-samples 2 --max-val-samples 2 --run-dir /tmp/fcn8_smoke_test
```

Very small 2D U-Net smoke test:

```bash
source .venv/bin/activate && python scripts/train_unet2d.py --epochs 1 --batch-size 1 --num-workers 0 --base-channels 8 --max-train-samples 2 --max-val-samples 2 --run-dir /tmp/unet2d_smoke_test
```

Very small modified 2D U-Net smoke test:

```bash
source .venv/bin/activate && python scripts/train_unet2d_modified.py --epochs 1 --batch-size 1 --num-workers 0 --base-channels 8 --max-train-samples 2 --max-val-samples 2 --run-dir /tmp/unet2d_modified_smoke_test
```

Very small 3D preprocessing and training smoke test:

```bash
source .venv/bin/activate && python scripts/convert_acdc_nifti_to_h5.py --output-root /tmp/acdc_h5_smoke --max-patients 2
source .venv/bin/activate && python scripts/preprocess_acdc_3d.py --input-dir /tmp/acdc_h5_smoke/ACDC_training_volumes --output-dir /tmp/acdc_3d_smoke --max-files 4 --target-depth 16 --target-height 32 --target-width 32
source .venv/bin/activate && python scripts/train_unet3d.py --data-dir /tmp/acdc_3d_smoke --epochs 1 --batch-size 1 --num-workers 0 --base-channels 4 --patch-depth 16 --patch-height 32 --patch-width 32 --max-train-samples 1 --max-val-samples 1 --run-dir /tmp/unet3d_smoke_test
```

## 8. Volume-Level Evaluation

Training logs report Dice over slices or volumes seen by the validation loader.
For paper-style evaluation of 2D networks, reconstruct patient/frame volumes
from predicted 2D slices first:

```bash
source .venv/bin/activate && python scripts/evaluate_2d.py --run-dir runs/unet2d_modified_spacing_100epoch_ce --split val
```

For FCN-8 or standard 2D U-Net, pass the model name if it cannot be inferred
from the run folder:

```bash
source .venv/bin/activate && python scripts/evaluate_2d.py --run-dir runs/fcn8 --model fcn8 --split val
source .venv/bin/activate && python scripts/evaluate_2d.py --run-dir runs/unet2d --model unet2d --split val
```

Evaluate the 3D U-Net directly on 3D volumes:

```bash
source .venv/bin/activate && python scripts/evaluate_3d.py --run-dir runs/unet3d --split val
```

Both evaluators save:

```text
metrics_by_class.csv
summary.json
```

Metrics are computed per reconstructed patient/frame volume and per foreground
class (`RV`, `Myo`, `LV`). Dice is unitless; ASSD and Hausdorff distance use the
HDF5 spacing metadata and are reported in millimetres. Use `--split all` to
evaluate every available labeled file, `--checkpoint` to select a specific
checkpoint, and `--output-dir` to choose where results are written.

## 9. Stop Training

To stop a foreground training run, press:

```text
Ctrl+C
```

Checkpoints are saved at the end of each completed epoch. If you stop in the middle of an epoch, the work from that partial epoch is not saved.

The script can start a new training run from saved model weights with `--weights`. This loads the model parameters only; the optimizer starts fresh.

## 10. Files Saved During Training

For a run directory such as:

```text
runs/unet3d_3epoch_local/
```

the script saves:

```text
config.json
metrics.csv
latest_epoch_003.pt
best_epoch_002.pt
```

`config.json` stores the run settings, device, train/validation patient split, and dataset counts.

`metrics.csv` stores one row per epoch.

`latest_epoch_XXX.pt` stores the newest model checkpoint. The epoch number tells which epoch produced the weights.

`best_epoch_XXX.pt` stores the model checkpoint with the best validation foreground Dice so far. The epoch number tells which epoch produced the weights.

Only one `latest_epoch_XXX.pt` and one `best_epoch_XXX.pt` are kept. Older latest/best checkpoint files are removed automatically.

## 10. Start From Saved Weights

Start a new run from the best weights of a previous run:

```bash
source .venv/bin/activate && python scripts/train_unet3d.py --epochs 20 --batch-size 1 --num-workers 2 --weights runs/unet3d_3epoch_local/best_epoch_003.pt --run-dir runs/unet3d_from_best
```

Start a new run from the latest weights of a previous run:

```bash
source .venv/bin/activate && python scripts/train_unet3d.py --epochs 20 --batch-size 1 --num-workers 2 --weights runs/unet3d_3epoch_local/latest_epoch_003.pt --run-dir runs/unet3d_from_latest
```

Use the exact filename that exists in your run folder. For example, if the best validation Dice happened at epoch 2, the file will be named `best_epoch_002.pt`.

For FCN-8, regular 2D U-Net, or modified 2D U-Net, use the same pattern with the matching training script and checkpoint. Do not load weights across different architectures because the layer names and shapes are different.

## 11. Read Training Progress

During training, the script prints a summary after each epoch:

```text
train_loss=... train_dice=... val_loss=... val_dice=... epoch_time=... elapsed=...
```

The most important values are:

- `train_loss`: loss on the training split
- `train_dice`: mean Dice over foreground classes on the training split
- `val_loss`: loss on the validation split
- `val_dice`: mean Dice over foreground classes on the validation split
- `epoch_time`: time taken by that epoch
- `elapsed`: total training time so far

For segmentation, `val_dice` is usually more informative than pixel accuracy because the background class is very large.

## 12. Read The Metrics CSV

Show the saved metrics:

```bash
column -s, -t < runs/unet3d_3epoch_local/metrics.csv | less -S
```

Print only the main columns:

```bash
python - <<'PY'
import pandas as pd

metrics = pd.read_csv("runs/unet3d_3epoch_local/metrics.csv")
print(metrics[[
    "epoch",
    "train_loss",
    "train_mean_foreground_dice",
    "val_loss",
    "val_mean_foreground_dice",
    "epoch_seconds",
    "elapsed_time",
]])
PY
```

Plot training curves:

```bash
python - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt

metrics = pd.read_csv("runs/unet3d_3epoch_local/metrics.csv")

plt.figure()
plt.plot(metrics["epoch"], metrics["train_loss"], label="train loss")
plt.plot(metrics["epoch"], metrics["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(metrics["epoch"], metrics["train_mean_foreground_dice"], label="train Dice")
plt.plot(metrics["epoch"], metrics["val_mean_foreground_dice"], label="val Dice")
plt.xlabel("Epoch")
plt.ylabel("Mean foreground Dice")
plt.legend()
plt.tight_layout()
plt.show()
PY
```

## 13. Inspect A Saved Checkpoint

Print basic checkpoint information:

```bash
python - <<'PY'
import torch

checkpoint = torch.load("runs/unet3d_3epoch_local/latest_epoch_003.pt", map_location="cpu")
print("epoch:", checkpoint["epoch"])
print("elapsed_time:", checkpoint["elapsed_time"])
print("metrics:", checkpoint["metrics"])
print("args:", checkpoint["args"])
PY
```

Use `best_epoch_XXX.pt` when you want the checkpoint with the best validation foreground Dice so far.

## 14. Interpreting Results

A healthy early run should usually show:

- training loss decreasing over time
- validation loss staying finite
- foreground Dice slowly increasing from a low initial value
- no `nan` or `inf` values

Very low Dice in the first few epochs is not automatically a failure because the network starts from random weights. A useful sign is that the validation Dice improves across epochs or at least that loss decreases without numerical instability.

If CPU training is too slow, use a CUDA-enabled PyTorch install and rerun the same command. The script will automatically choose the GPU when `torch.cuda.is_available()` returns `True`.
