"""Create a concise Markdown comparison from completed experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline/metrics.json"))
    parser.add_argument("--cnn", type=Path, default=Path("results/cnn/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("results/model-comparison.md"))
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def row(name: str, report: dict[str, Any]) -> str:
    test = report["test"]
    latency = test.get("inference_ms_single_image", test.get("inference_ms_per_image", 0.0))
    size_mb = test.get("checkpoint_bytes", 0) / (1024 * 1024)
    return (
        f"| {name} | {test['accuracy']:.4f} | {test['macro_precision']:.4f} | "
        f"{test['macro_recall']:.4f} | {test['macro_f1']:.4f} | {latency:.3f} | {size_mb:.2f} |"
    )


def main() -> None:
    args = parse_args()
    baseline = read_metrics(args.baseline)
    cnn = read_metrics(args.cnn)
    content = "\n".join(
        [
            "# Model comparison",
            "",
            "Generated from held-out Fashion-MNIST test results.",
            "",
            "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | "
            "Inference ms/image | Size MB |",
            "|---|---:|---:|---:|---:|---:|---:|",
            row("Logistic regression", baseline),
            row("FashionCNN", cnn),
            "",
            "> Latency values are environment-specific and should only be compared when "
            "measured on the same machine.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
