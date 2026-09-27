# Legacy single-split result tables

This directory contains compact CSV summaries from the project's earlier
single-split experiments. They are retained for provenance and for the older
analysis notebooks, but they are not the source of the current report's
headline results.

- `model_comparison.csv` compares one selected run for each architecture.
- `loss_ablation.csv` compares weighted cross-entropy and Dice loss for the
  modified 2D U-Net.

Those CSV values were calculated on 40 reconstructed validation volumes from
one 20-patient split. In that earlier experiment, the modified 2D U-Net with
weighted cross-entropy had the best selected-run Dice (`0.875`).

## Current authoritative results

The final project uses diagnosis-stratified five-fold cross-validation. The
standard 2D U-Net is now the strongest model:

| Evaluation | Standard 2D U-Net Dice |
|---|---:|
| Five-fold validation, mean ± SD | **0.9025 ± 0.0064** |
| Five-model local test ensemble | **0.9109** |

Current machine-readable data is stored per architecture in:

- `runs/fcn8_cv/`;
- `runs/unet2d_cv/`;
- `runs/unet2d_modified_cv/`; and
- `runs/unet3d_cv/`.

Each directory contains `cross_validation_metrics.csv`,
`cross_validation_summary.json`, fold-level evaluation records, and a
`test_ensemble/` summary. See [`runs/README.md`](../runs/README.md) for the full
layout and [`report/izvestaj_cv.pdf`](../report/izvestaj_cv.pdf) for the final
reported analysis.

## Historical figures

The following figures also belong to the earlier selected-run analysis:

- [`best_models_validation_dice.png`](../docs/assets/best_models_validation_dice.png)
- [`fcn8_training_dashboard.png`](../docs/assets/fcn8_training_dashboard.png)
- [`unet2d_training_dashboard.png`](../docs/assets/unet2d_training_dashboard.png)
- [`unet2d_modified_training_dashboard.png`](../docs/assets/unet2d_modified_training_dashboard.png)
- [`unet3d_training_dashboard.png`](../docs/assets/unet3d_training_dashboard.png)

Current cross-validation figures are under
[`report/figures_cv/`](../report/figures_cv/).
