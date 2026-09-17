"""Neural-network definitions used for training and serving."""

from __future__ import annotations

import torch
from torch import nn

from .domain import INPUT_SIZE


class FashionCNN(nn.Module):
    """Compact CNN designed for one-channel, 28-by-28 Fashion-MNIST images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits for a batch of images."""
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != INPUT_SIZE:
            raise ValueError(
                f"Expected input shape [N, {INPUT_SIZE[0]}, {INPUT_SIZE[1]}, "
                f"{INPUT_SIZE[2]}], received {tuple(inputs.shape)}."
            )
        logits = self.classifier(self.features(inputs))
        if not isinstance(logits, torch.Tensor):
            raise TypeError("The classifier must return a torch.Tensor.")
        return logits
