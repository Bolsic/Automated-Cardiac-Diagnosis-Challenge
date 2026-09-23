"""Generate reproducible figures for the spacing-aware five-fold CV report.

The script deliberately reads only saved datasets and lightweight run artifacts;
it never requires model checkpoints.  It is safe to run on a reporting machine.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("fcn8", "unet2d", "unet2d_modified", "unet3d")
MODEL_LABELS = {
    "fcn8": "FCN-8",
    "unet2d": "2D U-Net",
    "unet2d_modified": "Modifikovani 2D U-Net",
    "unet3d": "Modifikovani 3D U-Net",
}
MODEL_COLORS = {
    "fcn8": "#4C78A8",
    "unet2d": "#59A14F",
    "unet2d_modified": "#F28E2B",
    "unet3d": "#B07AA1",
}
CLASS_ORDER = ("RV", "Myo", "LV")
CLASS_LABELS = {"RV": "RV", "Myo": "Miokard", "LV": "LV"}
MASK_COLORS = {1: "#2ca02c", 2: "#ffbf00", 3: "#d62728"}
DIAGNOSIS_ORDER = ("DCM", "HCM", "MINF", "NOR", "RV")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--training-volumes-dir",
        type=Path,
        default=Path("outputs/acdc_h5_with_metadata/ACDC_training_volumes"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("report/figures_cv"))
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def numeric(row, key):
    return float(row[key])


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )


def save(fig, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def patient_volume_paths(training_volumes_dir):
    by_patient = defaultdict(list)
    for path in sorted(training_volumes_dir.glob("*.h5")):
        with h5py.File(path, "r") as h5_file:
            patient = str(h5_file.attrs["patient_id"])
        by_patient[patient].append(path)
    if len(by_patient) != 100:
        raise ValueError(f"Expected 100 training patients, found {len(by_patient)} in {training_volumes_dir}.")
    return by_patient


def plot_dataset_reference(training_volumes_dir, output_dir):
    path = training_volumes_dir / "patient001_frame01.h5"
    if not path.exists():
        path = sorted(training_volumes_dir.glob("*.h5"))[0]
    with h5py.File(path, "r") as h5_file:
        image = h5_file["image"][:]
        label = h5_file["label"][:]
        patient = str(h5_file.attrs["patient_id"])
        phase = str(h5_file.attrs["phase"])
    slice_index = int(np.argmax((label > 0).sum(axis=(1, 2))))
    image_slice = image[slice_index]
    label_slice = label[slice_index]
    low, high = np.percentile(image_slice, [1, 99])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    axes[0].imshow(image_slice, cmap="gray", vmin=low, vmax=high)
    axes[0].set_title("MR presek")
    axes[1].imshow(image_slice, cmap="gray", vmin=low, vmax=high)
    for class_id, color in MASK_COLORS.items():
        axes[1].contour(label_slice == class_id, levels=[0.5], colors=[color], linewidths=1.6)
    axes[1].set_title("Ekspertska segmentacija")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    handles = [plt.Line2D([0], [0], color=color, lw=2, label=label) for label, color in zip(("RV", "Miokard", "LV"), MASK_COLORS.values())]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{patient}, {phase}, presek {slice_index}", y=1.03)
    save(fig, output_dir, "dataset_reference_example.png")


def plot_spacing_distribution(training_volumes_dir, output_dir):
    spacings = []
    for paths in patient_volume_paths(training_volumes_dir).values():
        with h5py.File(paths[0], "r") as h5_file:
            spacings.append(np.asarray(h5_file.attrs["spacing_zyx"], dtype=float))
    spacing = np.stack(spacings)
    axis_info = ((0, "Z (kroz ravan)", (5.0,)), (1, "Y (u ravni)", (1.37, 2.5)), (2, "X (u ravni)", (1.37, 2.5)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for axis, (index, label, targets) in zip(axes, axis_info):
        axis.hist(spacing[:, index], bins="auto", color="#4C78A8", edgecolor="white")
        for target in targets:
            style = "--" if target == 1.37 else ":"
            axis.axvline(target, color="#D62728", linestyle=style, linewidth=1.6)
        axis.set(title=label, xlabel="Razmak voksela [mm]", ylabel="Broj pacijenata")
    fig.legend(
        [plt.Line2D([0], [0], color="#D62728", ls="--"), plt.Line2D([0], [0], color="#D62728", ls=":")],
        ["2D cilj: 1,37 mm", "3D cilj: 5,0 / 2,5 mm"],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    save(fig, output_dir, "spacing_distribution.png")


def artifact_paths(runs_dir, model):
    root = runs_dir / f"{model}_cv"
    return root, root / "cross_validation_summary.json", root / "cross_validation_metrics.csv"


def load_artifacts(runs_dir):
    artifacts = {}
    for model in MODEL_ORDER:
        root, summary_path, metrics_path = artifact_paths(runs_dir, model)
        summary = load_json(summary_path)
        metrics = load_csv(metrics_path)
        if summary["num_folds"] != 5 or len({int(row["fold"]) for row in metrics}) != 5:
            raise ValueError(f"{root} does not contain exactly five folds.")
        artifacts[model] = {"root": root, "summary": summary, "metrics": metrics}
    return artifacts


def validate_artifacts(artifacts):
    """Fail early if saved summaries differ from the values plotted below."""
    for model, artifact in artifacts.items():
        rows = [row for row in artifact["metrics"] if row["class_name"] == "foreground_macro"]
        for metric in ("dice", "assd_mm", "hd_mm"):
            values = np.asarray([numeric(row, metric) for row in rows])
            summary = artifact["summary"]["foreground_macro"]
            if not np.isclose(values.mean(), summary[f"mean_{metric}"]):
                raise ValueError(f"{model}: saved mean {metric} disagrees with cross-validation rows.")
            if not np.isclose(values.std(ddof=0), summary[f"std_{metric}"]):
                raise ValueError(f"{model}: saved SD {metric} disagrees with cross-validation rows.")


def plot_fold_composition(artifacts, output_dir):
    configs = [load_json(artifacts["fcn8"]["root"] / f"fold_{fold}" / "config.json") for fold in range(5)]
    matrix = np.asarray([[config["val_diagnosis_counts"][diagnosis] for diagnosis in DIAGNOSIS_ORDER] for config in configs])
    if not np.all(matrix == 4):
        raise ValueError("Validation folds are not balanced at four patients per diagnosis.")
    fig, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=4, cmap="Blues")
    axis.set(xticks=np.arange(5), xticklabels=DIAGNOSIS_ORDER, yticks=np.arange(5), yticklabels=[f"Fold {i}" for i in range(5)])
    for row in range(5):
        for column in range(5):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontweight="bold")
    fig.colorbar(image, ax=axis, label="Broj validacionih pacijenata")
    axis.set_title("Dijagnostički stratifikovana petostruka validacija")
    save(fig, output_dir, "cv_fold_composition.png")


def class_values(artifact, class_name, metric="dice"):
    return np.asarray([numeric(row, metric) for row in artifact["metrics"] if row["class_name"] == class_name])


def plot_model_comparison(artifacts, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    means = [artifacts[model]["summary"]["foreground_macro"]["mean_dice"] for model in MODEL_ORDER]
    deviations = [artifacts[model]["summary"]["foreground_macro"]["std_dice"] for model in MODEL_ORDER]
    positions = np.arange(len(MODEL_ORDER))
    axes[0].bar(positions, means, yerr=deviations, capsize=4, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    for position, model in enumerate(MODEL_ORDER):
        fold_values = class_values(artifacts[model], "foreground_macro")
        axes[0].scatter(np.full(5, position), fold_values, color="black", s=18, zorder=3)
    axes[0].set(xticks=positions, xticklabels=[MODEL_LABELS[m] for m in MODEL_ORDER], ylim=(0.78, 0.93), ylabel="Dice", title="Srednji foreground Dice po foldovima")
    axes[0].tick_params(axis="x", rotation=24)

    width = 0.19
    for offset, model in enumerate(MODEL_ORDER):
        means = [artifacts[model]["summary"]["by_class"][name]["mean_dice"] for name in CLASS_ORDER]
        deviations = [artifacts[model]["summary"]["by_class"][name]["std_dice"] for name in CLASS_ORDER]
        axes[1].bar(np.arange(3) + (offset - 1.5) * width, means, width, yerr=deviations, capsize=3, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    axes[1].set(xticks=np.arange(3), xticklabels=[CLASS_LABELS[name] for name in CLASS_ORDER], ylim=(0.75, 0.97), ylabel="Dice", title="Dice po anatomskim strukturama")
    axes[1].legend(fontsize=8, loc="lower right")
    save(fig, output_dir, "cv_model_comparison.png")


def plot_metric_spread(artifacts, output_dir):
    metrics = (("dice", "Dice"), ("assd_mm", "ASSD [mm]"), ("hd_mm", "HD [mm]"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, (metric, label) in zip(axes, metrics):
        values = [class_values(artifacts[model], "foreground_macro", metric) for model in MODEL_ORDER]
        box = axis.boxplot(
            values,
            patch_artist=True,
            tick_labels=[MODEL_LABELS[m] for m in MODEL_ORDER],
            showfliers=False,
        )
        for patch, model in zip(box["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.75)
        for position, model_values in enumerate(values, start=1):
            axis.scatter(np.full(len(model_values), position), model_values, color="black", s=18, zorder=3)
        axis.set(title=label, ylabel=label)
        axis.tick_params(axis="x", rotation=24, labelsize=8)
    save(fig, output_dir, "cv_fold_metric_spread.png")


def plot_learning_curves(artifacts, output_dir):
    fig, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    for model in MODEL_ORDER:
        histories = []
        for fold in range(5):
            rows = load_csv(artifacts[model]["root"] / f"fold_{fold}" / "metrics.csv")
            histories.append(np.asarray([numeric(row, "val_mean_foreground_dice") for row in rows]))
        epochs = min(len(history) for history in histories)
        values = np.stack([history[:epochs] for history in histories])
        x = np.arange(1, epochs + 1)
        mean, std = values.mean(axis=0), values.std(axis=0)
        axis.plot(x, mean, label=MODEL_LABELS[model], color=MODEL_COLORS[model], linewidth=2)
        axis.fill_between(x, mean - std, mean + std, color=MODEL_COLORS[model], alpha=0.14)
    axis.set(xlabel="Epoha", ylabel="Validacioni foreground Dice po presecima", ylim=(0, 1), xlim=(1, 100), title="Tok obučavanja: srednja vrednost ± SD preko pet foldova")
    axis.legend(fontsize=9, loc="lower right")
    save(fig, output_dir, "cv_learning_curves.png")


def main():
    args = parse_args()
    configure_style()
    artifacts = load_artifacts(args.runs_dir)
    validate_artifacts(artifacts)
    plot_dataset_reference(args.training_volumes_dir, args.output_dir)
    plot_spacing_distribution(args.training_volumes_dir, args.output_dir)
    plot_fold_composition(artifacts, args.output_dir)
    plot_model_comparison(artifacts, args.output_dir)
    plot_metric_spread(artifacts, args.output_dir)
    plot_learning_curves(artifacts, args.output_dir)
    print(f"Wrote six report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
