#!/usr/bin/env python3
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


def read_loss(dataset, optimizer):
    path = RESULTS_DIR / f"{dataset.lower()}_{optimizer.lower()}_loss.csv"
    epochs = []
    losses = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
    return epochs, losses


def read_summary():
    path = RESULTS_DIR / "summary.csv"
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_dataset(name):
    path = DATA_DIR / f"{name.lower()}.csv"
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    feature_names = [name for name in reader.fieldnames if name != "target"]
    features = [[float(row[feature]) for feature in feature_names] for row in rows]
    labels = [int(float(row["target"])) for row in rows]
    return feature_names, features, labels


def read_confusion(dataset, optimizer):
    path = RESULTS_DIR / f"{dataset.lower()}_{optimizer.lower()}_confusion.csv"
    matrix = []
    with path.open(newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append([int(value) for value in row])
    return matrix


def plot_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, dataset in zip(axes, ["D1", "D2"]):
        for optimizer, color in [("SGD", "#2563eb"), ("Adam", "#dc2626")]:
            epochs, losses = read_loss(dataset, optimizer)
            axis.plot(epochs, losses, label=optimizer, linewidth=2.0, color=color)
        axis.set_title(f"{dataset}: convergence")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.suptitle("Loss convergence on available datasets", fontsize=14, fontweight="bold")
    fig.savefig(PLOTS_DIR / "loss_convergence.png", dpi=180)
    plt.close(fig)


def plot_metrics_summary():
    rows = read_summary()
    labels = [f"{row['dataset']} {row['optimizer']}" for row in rows]
    accuracy = [float(row["test_accuracy"]) for row in rows]
    f1 = [float(row["test_macro_f1"]) for row in rows]
    x = range(len(rows))

    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.bar([value - 0.18 for value in x], accuracy, width=0.36, label="Accuracy", color="#16a34a")
    axis.bar([value + 0.18 for value in x], f1, width=0.36, label="Macro-F1", color="#7c3aed")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Test quality by dataset and optimizer")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")

    for i, value in enumerate(f1):
        axis.text(i + 0.18, min(value + 0.02, 1.04), f"{value:.2f}", ha="center", fontsize=9)

    fig.savefig(PLOTS_DIR / "metrics_summary.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for axis, dataset, optimizer in zip(
        axes.flat,
        ["D1", "D1", "D2", "D2"],
        ["SGD", "Adam", "SGD", "Adam"],
    ):
        matrix = read_confusion(dataset, optimizer)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{dataset} {optimizer}")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(len(matrix)))
        axis.set_yticks(range(len(matrix)))
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                axis.text(j, i, str(value), ha="center", va="center", color="#111827")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices on test splits", fontsize=14, fontweight="bold")
    fig.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def plot_dataset_views():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, dataset in zip(axes, ["D1", "D2"]):
        feature_names, features, labels = read_dataset(dataset)
        x_index = 0
        y_index = 1 if len(feature_names) > 1 else 0
        colors = ["#2563eb" if label == 0 else "#dc2626" for label in labels]
        axis.scatter(
            [row[x_index] for row in features],
            [row[y_index] for row in features],
            c=colors,
            s=18,
            alpha=0.75,
            edgecolors="none",
        )
        axis.set_title(f"{dataset}: class layout")
        axis.set_xlabel(feature_names[x_index])
        axis.set_ylabel(feature_names[y_index])
        axis.grid(True, alpha=0.25)
    fig.suptitle("Available datasets by class", fontsize=14, fontweight="bold")
    fig.savefig(PLOTS_DIR / "dataset_views.png", dpi=180)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_loss_curves()
    plot_metrics_summary()
    plot_confusion_matrices()
    plot_dataset_views()
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
