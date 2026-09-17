"""Train and evaluate a multinomial logistic-regression baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import FashionMNIST

from training.data import stratified_indices
from training.metrics import classification_metrics, save_confusion_matrix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/logistic_regression.joblib")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument(
        "--regularization-c",
        type=float,
        default=0.01,
        help="Inverse L2 regularization strength passed to LogisticRegression.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def flatten_and_scale(images: np.ndarray) -> np.ndarray:
    """Convert uint8 image batches into float32 feature rows in the unit interval."""
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def main() -> None:
    args = parse_args()
    training_data = FashionMNIST(root=args.data_dir, train=True, download=True)
    test_data = FashionMNIST(root=args.data_dir, train=False, download=True)
    train_indices, validation_indices = stratified_indices(
        training_data.targets.numpy(), args.validation_fraction, args.seed
    )

    features = flatten_and_scale(training_data.data.numpy())
    targets = training_data.targets.numpy()
    test_features = flatten_and_scale(test_data.data.numpy())
    test_targets = test_data.targets.numpy()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            C=args.regularization_c,
            max_iter=args.max_iterations,
            tol=1e-4,
            random_state=args.seed,
        ),
    )
    started = time.perf_counter()
    model.fit(features[train_indices], targets[train_indices])
    training_seconds = time.perf_counter() - started

    validation_predictions = model.predict(features[validation_indices]).tolist()
    test_started = time.perf_counter()
    test_predictions = model.predict(test_features).tolist()
    test_seconds = time.perf_counter() - test_started
    validation_metrics = classification_metrics(
        targets[validation_indices].tolist(), validation_predictions
    )
    test_metrics = classification_metrics(test_targets.tolist(), test_predictions)
    test_metrics["inference_ms_per_image"] = (test_seconds * 1000.0) / len(test_targets)

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_scaling": "uint8 / 255.0 followed by train-fitted StandardScaler",
            "input_features": 28 * 28,
            "seed": args.seed,
        },
        args.artifact,
    )
    test_metrics["checkpoint_bytes"] = args.artifact.stat().st_size
    summary = {
        "model": "multinomial_logistic_regression",
        "training_seconds": training_seconds,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "test_size": len(test_targets),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    save_json(summary, args.results_dir / "metrics.json")
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        args.results_dir / "confusion_matrix.png",
        "Logistic regression test confusion matrix",
    )
    print(
        f"Saved {args.artifact}. Test accuracy={test_metrics['accuracy']:.4f}, "
        f"macro F1={test_metrics['macro_f1']:.4f}."
    )


if __name__ == "__main__":
    main()
