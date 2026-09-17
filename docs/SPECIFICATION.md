# Fashion Image Classifier — Project Specification

Status: Implementation baseline  
Last updated: 2026-08-05  
Primary language for code and documentation: English

## 1. Purpose

This project is an educational, end-to-end image classification system. It trains and compares
a classical logistic-regression baseline and a convolutional neural network (CNN) on the
Fashion-MNIST dataset, then serves the selected CNN through a FastAPI API and a React web client.

The application accepts one garment image and returns a ranked prediction over the ten
Fashion-MNIST classes. The project demonstrates a reproducible machine-learning workflow,
model evaluation, API deployment, and browser integration.

## 2. Scope and non-goals

### In scope

- Download and inspect Fashion-MNIST.
- Use the official 60,000-image training split and 10,000-image test split.
- Create a deterministic, stratified validation split from the official training split.
- Train a logistic-regression baseline.
- Train a compact CNN, optionally with light image augmentation.
- Report accuracy, macro precision, macro recall, macro F1, per-class metrics, confusion matrix,
  learning curves, model size, and approximate inference latency.
- Persist a versioned CNN checkpoint with preprocessing and class metadata.
- Expose health, class-list, and prediction endpoints through FastAPI.
- Provide an accessible React upload interface with image preview and top-k probabilities.
- Support local development and Docker-based deployment.

### Explicit non-goals

- Production-ready classification of arbitrary e-commerce photography.
- Multi-label classification, object detection, segmentation, or multiple garments in one image.
- Authentication, user accounts, a prediction-history database, or model retraining from the UI.
- Guaranteed out-of-distribution detection.

Fashion-MNIST contains centered, 28-by-28 grayscale catalog-like images. Real product photographs
may include color, people, complex backgrounds, multiple objects, and viewpoints absent from the
training data. A model trained only on Fashion-MNIST can be confidently wrong on such inputs.

## 3. Dataset contract

The canonical label mapping is immutable unless a deliberately versioned migration is made:

| ID | Canonical name | Display name |
|---:|---|---|
| 0 | T-shirt/top | T-shirt / Top |
| 1 | Trouser | Trouser |
| 2 | Pullover | Pullover |
| 3 | Dress | Dress |
| 4 | Coat | Coat |
| 5 | Sandal | Sandal |
| 6 | Shirt | Shirt |
| 7 | Sneaker | Sneaker |
| 8 | Bag | Bag |
| 9 | Ankle boot | Ankle boot |

Required split policy:

- Official training data: 60,000 samples.
- Training subset: 80% of the official training data.
- Validation subset: 20% of the official training data.
- Official test data: 10,000 samples and used only after model selection.
- The split is stratified and controlled by a recorded random seed.

## 4. Functional requirements

### ML pipeline

- `FR-ML-001`: A command shall train the logistic-regression baseline and save metrics.
- `FR-ML-002`: A command shall train the CNN and save the best validation checkpoint.
- `FR-ML-003`: Training shall record the seed, hyperparameters, class names, normalization values,
  epoch history, and final metrics.
- `FR-ML-004`: Evaluation shall generate machine-readable JSON and human-readable plots.
- `FR-ML-005`: Augmentation shall affect training samples only, never validation or test samples.
- `FR-ML-006`: The official test set shall be evaluated once after selecting the best validation
  checkpoint in a training run.

### API

- `FR-API-001`: `GET /health` shall report service and model readiness.
- `FR-API-002`: `GET /api/v1/classes` shall return the canonical label mapping.
- `FR-API-003`: `POST /api/v1/predict` shall accept one PNG, JPEG, or WebP image as multipart data.
- `FR-API-004`: Prediction output shall include either an accepted class or `Unknown`, the rejection
  state and reason, top predictions, model version, and inference duration.
- `FR-API-005`: Invalid, oversized, or undecodable files shall produce a structured 4xx response.
- `FR-API-006`: If no compatible checkpoint is available, prediction shall return HTTP 503 while
  health and class-list endpoints remain available.

### Web client

- `FR-WEB-001`: The user shall be able to browse for or drag and drop one image.
- `FR-WEB-002`: The client shall validate file type and size before upload.
- `FR-WEB-003`: The client shall show a preview, pending state, result, and actionable error state.
- `FR-WEB-004`: The client shall show the accepted class or `Unknown` and the top-three raw model
  probabilities.
- `FR-WEB-005`: The interface shall disclose the Fashion-MNIST domain limitation.

## 5. Non-functional requirements

- `NFR-001 Reproducibility`: The same seed and dependencies should reproduce the data split and a
  statistically comparable training result. Exact GPU floating-point equality is not required.
- `NFR-002 Maintainability`: Training, preprocessing, model definition, serving, and presentation
  responsibilities remain in separate modules.
- `NFR-003 Testability`: Core image decoding, checkpoint validation, inference, and API errors are
  testable without downloading the dataset.
- `NFR-004 Security`: Upload type, byte count, decoded dimensions, and image validity are checked.
  Original uploads are not persisted.
- `NFR-005 Performance`: The model is loaded once during application startup, not per request.
- `NFR-006 Accessibility`: Main controls are keyboard accessible and status messages use appropriate
  live-region semantics.
- `NFR-007 Observability`: Startup state, prediction failures, and request IDs are logged without
  logging image contents.

## 6. Preprocessing contract

Training and inference must share these constants:

- Color space: one-channel grayscale.
- Tensor range before normalization: `[0.0, 1.0]`.
- Spatial size: `28 × 28`.
- Mean: `0.2860405969887955`.
- Standard deviation: `0.35302424451492237`.

Uploaded-image preprocessing additionally performs EXIF orientation correction, safe decoding,
grayscale conversion, border-based background estimation, contrast foreground extraction,
aspect-ratio-preserving resize, and black padding to 28 by 28. Native 28-by-28 dark-background
inputs remain pixel-identical to the evaluation path.

The serving layer returns `Unknown` when the background or foreground coverage is incompatible with
this preprocessing contract, when the maximum softmax probability is below `0.55`, or when the
top-one/top-two margin is below `0.15`. These rules are a conservative reject option, not guaranteed
out-of-distribution detection.

The constants live in `backend/app/domain.py`. Training and serving code must import that module;
they must not duplicate the values.

## 7. CNN architecture contract

The checkpoint architecture identifier is `fashion_cnn_v1`:

```text
Input [N, 1, 28, 28]
  -> Conv(1, 32, 3, padding=1) -> BatchNorm -> ReLU
  -> Conv(32, 32, 3, padding=1) -> ReLU -> MaxPool(2) -> Dropout
  -> Conv(32, 64, 3, padding=1) -> BatchNorm -> ReLU
  -> Conv(64, 64, 3, padding=1) -> ReLU -> MaxPool(2) -> Dropout
  -> Flatten -> Linear(3136, 128) -> ReLU -> Dropout -> Linear(128, 10)
```

The training loss consumes raw logits through cross entropy. Softmax is applied only for
human-facing inference probabilities.

## 8. Checkpoint contract

A compatible PyTorch checkpoint is a dictionary containing:

```text
schema_version: integer
architecture: string
model_version: string
model_state_dict: mapping
class_names: ordered list of 10 strings
normalization: {mean: float, std: float}
input_size: [1, 28, 28]
training: mapping of seed and hyperparameters
metrics: mapping of validation/test metrics
```

The API rejects a checkpoint when the architecture, ordered labels, input size, or normalization
metadata conflicts with the running code. This fail-fast behavior prevents silent label or
preprocessing drift.

## 9. API contract

### `GET /health`

Successful response is always HTTP 200 so an operator can inspect degraded state:

```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "cnn-20260805T120000Z",
  "detail": "Model is ready."
}
```

### `GET /api/v1/classes`

Returns an ordered array of `{id, name}` objects.

### `POST /api/v1/predict`

Request content type: `multipart/form-data`; field name: `file`.

```json
{
  "request_id": "9b5c1e8a-6ab6-4d98-9f8e-9e4a3ab471fb",
  "class_id": 7,
  "class_name": "Sneaker",
  "confidence": 0.9462,
  "is_unknown": false,
  "rejection_reason": null,
  "top_predictions": [
    {"class_id": 7, "class_name": "Sneaker", "probability": 0.9462}
  ],
  "model_version": "cnn-20260805T120000Z",
  "inference_ms": 8.4
}
```

For a rejected input, `class_id` and `confidence` are `null`, `class_name` is `Unknown`, and
`rejection_reason` explains the decision. `top_predictions` still contains raw softmax scores for
diagnostics; those scores are not accepted classifications.

Errors follow this shape:

```json
{
  "error": {
    "code": "invalid_image",
    "message": "The uploaded file could not be decoded as an image.",
    "request_id": "9b5c1e8a-6ab6-4d98-9f8e-9e4a3ab471fb"
  }
}
```

## 10. Module boundaries and refactoring invariants

- `backend/app/domain.py` owns labels, normalization, input shape, and checkpoint schema constants.
- `backend/app/model.py` owns neural-network structure only. It does not open files or know HTTP.
- `backend/app/preprocessing.py` converts validated image bytes to a model tensor.
- `backend/app/inference.py` owns checkpoint validation and tensor-to-prediction behavior.
- `backend/app/main.py` owns HTTP concerns and dependency wiring.
- `training/` may import backend model/domain modules to guarantee parity. Backend code must never
  import training modules.
- `frontend/src/api.js` is the only client module that knows endpoint paths or transport details.
- UI components consume plain result objects and do not parse backend internals.

Any refactor must preserve:

1. Ordered label identity between dataset, checkpoint, API, and UI.
2. Exact normalization parity between training and inference.
3. Validation/test transforms without augmentation.
4. One-time model loading at application startup.
5. Structured error responses and existing public endpoint paths.
6. No persistence of uploaded image contents.

## 11. Acceptance criteria

- All automated backend tests pass.
- Frontend component tests pass and the production bundle builds.
- The baseline and CNN commands run from the repository root.
- A trained checkpoint makes `/health` report `model_ready: true`.
- A valid image returns a top-three ranking with probabilities in `[0, 1]`, sorted descending.
- The generated report includes test accuracy and macro F1 plus a confusion matrix.
- As a target rather than a hard guarantee, the compact CNN should reach at least 90% test accuracy
  and outperform the logistic-regression baseline under the recorded split.
- README instructions reproduce local setup, training, serving, testing, and Docker workflows.

## 12. Safe extension points

- Add another architecture through a model registry keyed by checkpoint architecture.
- Add ONNX inference behind the existing prediction-service interface.
- Add a real e-commerce dataset without changing the API only if its taxonomy remains identical;
  otherwise version the endpoint and label schema.
- Replace or calibrate the current reject heuristics using a representative in-domain/OOD
  validation set; do not reinterpret softmax probability as an out-of-distribution guarantee.
- Add prediction history only with an explicit privacy and retention policy.
