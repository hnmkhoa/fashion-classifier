"""Metric calculation and report artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

# Training commands must render plots in headless shells and containers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from backend.app.domain import CLASS_NAMES


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    """Calculate aggregate and per-class metrics using a stable label order."""
    labels = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(
            precision_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Persist JSON using a human-reviewable representation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_confusion_matrix(matrix: list[list[int]], path: Path, title: str) -> None:
    """Render a labeled confusion matrix without requiring an interactive display."""
    array = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(array, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = array.max() / 2.0 if array.size else 0
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            axis.text(
                column,
                row,
                f"{array[row, column]:d}",
                ha="center",
                va="center",
                color="white" if array[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_learning_curves(history: list[dict[str, float]], path: Path) -> None:
    """Render loss and accuracy curves from epoch records."""
    epochs = [int(item["epoch"]) for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [item["train_accuracy"] for item in history], label="Train")
    axes[1].plot(epochs, [item["validation_accuracy"] for item in history], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0.0, 1.0))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
