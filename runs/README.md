# Selected experiment records

The repository keeps a small set of representative runs:

| Directory | Purpose |
|---|---|
| `fcn8_100epoch_gpu_bs16_lr1e-3` | FCN-8 baseline |
| `unet2d_200epoch_gpu_bs4_lr1e-4` | Standard 2D U-Net baseline |
| `unet2d_modified_100epoch_weighted_ce` | Best overall Dice |
| `unet2d_modified_100epoch_dice_loss` | Loss-function ablation |
| `unet3d` | Anisotropic 3D U-Net |

Each run includes:

- `config.json`: hyperparameters and patient split
- `metrics.csv`: epoch-level training and validation history
- `evaluation_*_val/metrics_by_class.csv`: per-volume, per-class metrics
- `evaluation_*_val/summary.json`: aggregate metrics and evaluation provenance

Model checkpoints are intentionally excluded from Git because of their size.
The cached historical evaluation files are evidence for the published Dice
results, but their distance columns used unit spacing rather than physical
millimetres. New evaluations require spacing-aware HDF5 data.
