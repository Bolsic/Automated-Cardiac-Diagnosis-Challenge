# ACDC FCN-8 Project Guide

This project trains a PyTorch FCN-8 segmentation model on the 2D ACDC preprocessed dataset exported by `notebooks/acdc_paper_preprocessing.ipynb`.

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

## 2. Dataset Used For Training

The current training workflow uses the exported 2D slices:

```text
outputs/acdc_preprocessed_2d/ACDC_training_slices/
```

Each `.h5` file should contain:

- `image`: a preprocessed 2D image slice
- `label`: the segmentation label map for that slice

If this folder does not exist, first run `notebooks/acdc_paper_preprocessing.ipynb` to export the 2D preprocessed slices.

## 3. Start Training

Basic 3-epoch local run:

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

## 4. Useful Training Options

The most useful command-line options are:

```text
--data-dir              Folder containing preprocessed .h5 slices
--run-dir               Folder where checkpoints and metrics are saved
--weights               Optional saved weights to load before training
--epochs                Number of training epochs
--batch-size            Number of slices per batch
--num-workers           DataLoader worker processes
--learning-rate         Adam learning rate, default 0.01
--val-fraction          Fraction of patients used for validation, default 0.2
--seed                  Random seed for reproducibility
--max-train-samples     Limit training samples for a quick test
--max-val-samples       Limit validation samples for a quick test
```

Very small smoke test:

```bash
source .venv/bin/activate && python scripts/train_fcn8.py --epochs 1 --batch-size 1 --num-workers 0 --classifier-channels 64 --max-train-samples 2 --max-val-samples 2 --run-dir /tmp/fcn8_smoke_test
```

## 5. Stop Training

To stop a foreground training run, press:

```text
Ctrl+C
```

Checkpoints are saved at the end of each completed epoch. If you stop in the middle of an epoch, the work from that partial epoch is not saved.

The script can start a new training run from saved model weights with `--weights`. This loads the model parameters only; the optimizer starts fresh.

## 6. Files Saved During Training

For a run directory such as:

```text
runs/fcn8_3epoch_local/
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

## 7. Start From Saved Weights

Start a new run from the best weights of a previous run:

```bash
source .venv/bin/activate && python scripts/train_fcn8.py --epochs 20 --batch-size 8 --num-workers 2 --weights runs/fcn8_3epoch_local/best_epoch_003.pt --run-dir runs/fcn8_from_best
```

Start a new run from the latest weights of a previous run:

```bash
source .venv/bin/activate && python scripts/train_fcn8.py --epochs 20 --batch-size 8 --num-workers 2 --weights runs/fcn8_3epoch_local/latest_epoch_003.pt --run-dir runs/fcn8_from_latest
```

Use the exact filename that exists in your run folder. For example, if the best validation Dice happened at epoch 2, the file will be named `best_epoch_002.pt`.

## 8. Read Training Progress

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

## 9. Read The Metrics CSV

Show the saved metrics:

```bash
column -s, -t < runs/fcn8_3epoch_local/metrics.csv | less -S
```

Print only the main columns:

```bash
python - <<'PY'
import pandas as pd

metrics = pd.read_csv("runs/fcn8_3epoch_local/metrics.csv")
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

metrics = pd.read_csv("runs/fcn8_3epoch_local/metrics.csv")

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

## 10. Inspect A Saved Checkpoint

Print basic checkpoint information:

```bash
python - <<'PY'
import torch

checkpoint = torch.load("runs/fcn8_3epoch_local/latest_epoch_003.pt", map_location="cpu")
print("epoch:", checkpoint["epoch"])
print("elapsed_time:", checkpoint["elapsed_time"])
print("metrics:", checkpoint["metrics"])
print("args:", checkpoint["args"])
PY
```

Use `best_epoch_XXX.pt` when you want the checkpoint with the best validation foreground Dice so far.

## 11. Interpreting Results

A healthy early run should usually show:

- training loss decreasing over time
- validation loss staying finite
- foreground Dice slowly increasing from a low initial value
- no `nan` or `inf` values

Very low Dice in the first few epochs is not automatically a failure because the network starts from random weights. A useful sign is that the validation Dice improves across epochs or at least that loss decreases without numerical instability.

If CPU training is too slow, use a CUDA-enabled PyTorch install and rerun the same command. The script will automatically choose the GPU when `torch.cuda.is_available()` returns `True`.
