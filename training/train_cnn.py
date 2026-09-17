"""Train, select, evaluate, and package the primary CNN model."""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from backend.app.domain import (
    CHECKPOINT_SCHEMA_VERSION,
    CLASS_NAMES,
    INPUT_SIZE,
    MODEL_ARCHITECTURE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)
from backend.app.model import FashionCNN
from training.data import DataLoaders, create_data_loaders, seed_everything
from training.metrics import (
    classification_metrics,
    save_confusion_matrix,
    save_json,
    save_learning_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/fashion_cnn.pt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/cnn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--augmentation", choices=("none", "light"), default="light")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    """Run one train or evaluation epoch and return aggregate outputs."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_examples += batch_size
        targets_all.extend(targets.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        targets_all,
        predictions_all,
    )


def estimate_inference_ms(model: nn.Module, device: torch.device, repeats: int = 100) -> float:
    """Estimate warm single-image model-forward latency on the selected device."""
    example = torch.zeros((1, *INPUT_SIZE), device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return ((time.perf_counter() - started) * 1000.0) / repeats


def build_checkpoint(
    model_state: dict[str, torch.Tensor],
    model_version: str,
    args: argparse.Namespace,
    loaders: DataLoaders,
    best_epoch: int,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": MODEL_ARCHITECTURE,
        "model_version": model_version,
        "model_state_dict": model_state,
        "class_names": list(CLASS_NAMES),
        "normalization": {"mean": NORMALIZATION_MEAN, "std": NORMALIZATION_STD},
        "input_size": list(INPUT_SIZE),
        "training": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "augmentation": args.augmentation,
            "validation_fraction": args.validation_fraction,
            "train_size": loaders.train_size,
            "validation_size": loaders.validation_size,
            "test_size": loaders.test_size,
        },
        "metrics": {"validation": validation_metrics, "test": test_metrics},
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("Epochs, batch size, and patience must be positive.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        augmentation=args.augmentation,
        num_workers=args.num_workers,
    )
    model = FashionCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(
            model, loaders.train, criterion, device, optimizer
        )
        validation_loss, validation_accuracy, _, _ = run_epoch(
            model, loaders.validation, criterion, device
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"validation loss {validation_loss:.4f}, acc {validation_accuracy:.4f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)

    _, _, validation_targets, validation_predictions = run_epoch(
        model, loaders.validation, criterion, device
    )
    _, _, test_targets, test_predictions = run_epoch(model, loaders.test, criterion, device)
    validation_metrics = classification_metrics(validation_targets, validation_predictions)
    test_metrics = classification_metrics(test_targets, test_predictions)
    test_metrics["inference_ms_single_image"] = estimate_inference_ms(model, device)

    model_version = datetime.now(UTC).strftime("cnn-%Y%m%dT%H%M%SZ")
    checkpoint = build_checkpoint(
        model_state={key: value.detach().cpu() for key, value in best_state.items()},
        model_version=model_version,
        args=args,
        loaders=loaders,
        best_epoch=best_epoch,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.artifact)
    test_metrics["checkpoint_bytes"] = args.artifact.stat().st_size

    summary = {
        "model": MODEL_ARCHITECTURE,
        "model_version": model_version,
        "device": str(device),
        "best_epoch": best_epoch,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "arguments": vars(args),
    }
    summary["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in summary["arguments"].items()
    }
    save_json(summary, args.results_dir / "metrics.json")
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        args.results_dir / "confusion_matrix.png",
        "FashionCNN test confusion matrix",
    )
    save_learning_curves(history, args.results_dir / "learning_curves.png")
    (args.results_dir / "run-summary.txt").write_text(
        json.dumps(
            {
                "model_version": model_version,
                "best_epoch": best_epoch,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "checkpoint": str(args.artifact),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Saved {args.artifact}. Test accuracy={test_metrics['accuracy']:.4f}, "
        f"macro F1={test_metrics['macro_f1']:.4f}."
    )


if __name__ == "__main__":
    main()

