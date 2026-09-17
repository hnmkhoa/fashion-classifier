# FashionCNN Model Card

This model card describes the `fashion_cnn_v1` architecture and the verified reference checkpoint
trained on 2026-08-05. Update the results and environment for every newly promoted checkpoint.

## Model details

- Task: single-label image classification.
- Architecture: compact convolutional neural network defined in `backend/app/model.py`.
- Input: one grayscale image resized and padded to `[1, 28, 28]`.
- Output: logits for ten ordered Fashion-MNIST classes.
- Training framework: PyTorch.
- Artifact format: weights-only-compatible PyTorch checkpoint with schema metadata.
- License: no model license is asserted by this scaffold.

## Intended use

- Educational comparison of classical and convolutional classifiers.
- Demonstration of reproducible ML training and web deployment.
- Classification of inputs resembling the Fashion-MNIST visual domain.

## Out-of-scope use

- Automated catalog decisions without human review.
- Safety-critical or legally consequential decisions.
- General recognition of real-world fashion photography without representative retraining.
- Recognition of classes outside the ten-label taxonomy.
- Multiple objects, detection, segmentation, attributes, brands, color, size, or material prediction.

## Training data

Fashion-MNIST contains 70,000 grayscale images in ten balanced classes: 60,000 official training
examples and 10,000 official test examples. The training command reserves a stratified 20% of the
official training data for validation by default.

Dataset paper: <https://arxiv.org/abs/1708.07747>  
Official repository: <https://github.com/zalandoresearch/fashion-mnist>

## Preprocessing and augmentation

Pixel tensors are normalized with mean `0.2860405969887955` and standard deviation
`0.35302424451492237`. The `light` training preset applies small random affine rotation,
translation, and scaling. Validation and test examples are never augmented.

User uploads additionally receive EXIF orientation correction and grayscale conversion. The
inference adapter estimates a uniform background from the image border, maps foreground contrast
to a bright Fashion-MNIST-like silhouette, crops it with a margin, preserves its aspect ratio, and
centers it on a black 28-by-28 canvas. Native 28-by-28 dark-background samples remain unchanged.
This heuristic improves simple catalog photos but cannot eliminate domain shift or isolate one
item reliably in complex, multi-object scenes.

## Evaluation protocol

- Select the best training epoch using validation loss.
- Restore that state before evaluating the official test set.
- Report accuracy, macro precision, macro recall, macro F1, per-class metrics, and confusion matrix.
- Record the split seed and all relevant hyperparameters.
- Compare latency only on identical hardware and software environments.

Generated metrics location: `results/cnn/metrics.json`  
Generated confusion matrix: `results/cnn/confusion_matrix.png`  
Generated learning curves: `results/cnn/learning_curves.png`

## Promoted checkpoint results

| Field | Value |
|---|---|
| Model version | `cnn-20260805T145058Z` |
| Training seed | 42 |
| Training configuration | 12 epochs maximum, batch 128, AdamW, light affine augmentation |
| Best epoch | 12 |
| Validation accuracy | 92.42% |
| Test accuracy | 91.93% |
| Test macro F1 | 91.88% |
| Checkpoint size | 1,882,684 bytes (1.80 MiB) |
| Inference environment | Python 3.11.15, PyTorch 2.5.1 CPU, Windows, 14 PyTorch threads |
| Single-image forward latency | 1.51 ms, warm mean of 100 forwards; decoding excluded |

The weakest class was Shirt with F1 75.34%. The largest directed confusions were T-shirt/top to
Shirt (103 images), Shirt to T-shirt/top (99), Shirt to Coat (81), and Shirt to Pullover (63).
These results match the expected ambiguity among low-resolution upper-body categories and should
guide error analysis or future targeted augmentation.

With the serving score rules applied post hoc to the 10,000-image official test set, 95.13% of
samples are accepted and accuracy within that accepted subset is 94.12%; 4.87% are returned as
`Unknown`. This evaluates in-domain selectivity only. The thresholds have not been calibrated
against a representative real-photo OOD dataset.

## Known limitations

- Low spatial resolution removes texture and fine structural detail.
- Visually similar upper-body classes are commonly confused.
- The dataset does not represent modern, varied e-commerce imagery.
- Softmax confidence is not calibrated proof that an input belongs to the training distribution.
- The serving reject option uses image-complexity and score heuristics; it can still falsely accept
  out-of-domain scenes or reject valid garments.
- Grayscale conversion discards color, which may be informative in a real catalog.
- The taxonomy omits many garment and accessory types.

## Ethical and operational considerations

The model predicts a product-like visual class, not a personal attribute. Nevertheless, incorrect
automation can create catalog errors and downstream search or recommendation failures. Production
use should include representative evaluation, monitoring, a correction workflow, and human review
for uncertain or high-impact cases.

Do not retain user images without a clear purpose, consent basis, access policy, and retention
limit. This implementation does not persist uploads.
