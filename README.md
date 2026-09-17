# ThreadSense — Fashion Image Classifier

ThreadSense is an end-to-end educational machine-learning project that classifies one garment
image into the ten Fashion-MNIST categories. It compares a multinomial logistic-regression
baseline with a compact convolutional neural network (CNN), packages the selected CNN as a
versioned PyTorch checkpoint, serves predictions through FastAPI, and presents them in a React
web interface.

This repository is intentionally organized for review and refactoring. Start with the Vietnamese
onboarding guide [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md), then use
[`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) for the behavioral contract and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for dependency boundaries before changing model,
preprocessing, API, or label code.

## What the project does

1. Downloads the official Fashion-MNIST dataset: 60,000 training images and 10,000 test images.
2. Creates a deterministic, stratified training/validation split from the 60,000 training images.
3. Trains and evaluates a logistic-regression baseline.
4. Trains a purpose-built CNN with optional light augmentation and early stopping.
5. Saves evaluation metrics, confusion matrices, learning curves, and compatible model artifacts.
6. Loads the best CNN once when FastAPI starts.
7. Accepts a PNG, JPEG, or WebP upload and returns either an accepted class or `Unknown`, plus the
   top-three raw model probabilities.
8. Displays the image and ranked result in a responsive React interface.

## Important scope limitation

Fashion-MNIST images are centered, grayscale, and only 28 by 28 pixels. A model trained on this
dataset is suitable for a benchmark and system-integration study. It is **not** a production model
for arbitrary e-commerce photography. Photos containing models, complex backgrounds, several
objects, unusual lighting, or unseen garment types are outside the training distribution, and the
model can be confidently wrong.

The serving layer includes a conservative reject option for obvious domain mismatch and ambiguous
scores. This reduces misleading forced classifications, but it is a heuristic rather than a
guaranteed out-of-distribution detector.

A production extension should use labeled product photography that matches the intended catalog,
split the data by product identity, and fine-tune a pretrained image model such as ResNet or
MobileNet. Merely enlarging 28-by-28 Fashion-MNIST images to 224 by 224 does not add visual detail.

## Canonical classes

| ID | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The order is an API and checkpoint invariant. Change it only through an explicit versioned
migration.

## Architecture

```text
Fashion-MNIST ──> training/data.py ──> baseline and CNN experiments
                                             │
                                             ├──> results/*.json and plots
                                             └──> artifacts/fashion_cnn.pt
                                                            │
Browser ──> React/Vite ──multipart upload──> FastAPI ──> preprocessing ──> CNN
   ▲                                                   │                      │
   └────────────── ranked JSON result ─────────────────┴──────────────────────┘
```

Training imports the shared model and domain constants from `backend/app`. The backend never
imports training code. This one-way dependency guarantees that serving uses the same class order,
input shape, and normalization values as training without coupling production startup to dataset
or plotting packages.

## Repository structure

```text
.
├── artifacts/                  Generated model files; not committed
├── backend/
│   ├── app/
│   │   ├── config.py           Environment-backed runtime configuration
│   │   ├── domain.py           Shared labels and preprocessing constants
│   │   ├── errors.py           Stable application error definitions
│   │   ├── inference.py        Checkpoint validation and prediction service
│   │   ├── main.py             FastAPI factory, lifecycle, and endpoints
│   │   ├── model.py            FashionCNN architecture
│   │   ├── preprocessing.py    Safe decode and model tensor conversion
│   │   └── schemas.py          Public API response models
│   ├── tests/                  Backend unit and API tests
│   ├── Dockerfile
│   ├── requirements.txt        API runtime dependencies
│   └── requirements-dev.txt    Training, test, and quality dependencies
├── docs/
│   ├── ARCHITECTURE.md         System design and refactoring guide
│   ├── MODEL_CARD.md           Intended use, evaluation, and limitations
│   ├── PROJECT_GUIDE.md        Vietnamese codebase and workflow walkthrough
│   └── SPECIFICATION.md        Normative behavioral contract
├── frontend/
│   ├── src/
│   │   ├── components/         Upload and result presentation components
│   │   ├── api.js              HTTP boundary
│   │   ├── App.jsx             Page state and workflow orchestration
│   │   └── styles.css          Responsive visual system
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── training/
│   ├── data.py                 Deterministic split and data loaders
│   ├── metrics.py              Metrics and plots
│   ├── train_baseline.py       Logistic-regression experiment
│   ├── train_cnn.py            CNN training and checkpoint packaging
│   └── compare_models.py       Markdown comparison generator
├── .env.example
├── docker-compose.yml
└── pyproject.toml              Test, lint, and type-check configuration
```

## Prerequisites

Recommended local tools:

- Python 3.11 or 3.12. The pinned PyTorch release is not intended for Python 3.14.
- Node.js 20 LTS or newer and npm 10 or newer.
- Approximately 2 GB of free disk space for environments, data, artifacts, and package caches.
- A CUDA-capable GPU is optional. CPU training works but takes longer.
- Docker Desktop is optional for container deployment.

All commands below start at the repository root unless a step explicitly changes directory.

## Quick start

The application requires a trained `artifacts/fashion_cnn.pt` checkpoint before it can predict.
The API intentionally starts in a degraded, inspectable state when the checkpoint is absent.

### 1. Create the Python environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements-dev.txt
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
```

PyTorch installation can vary for CUDA environments. If GPU-specific wheels are required, use
the command generated by the official PyTorch installer, then install the remaining requirements.

### 2. Train the primary CNN

```bash
python -m training.train_cnn --epochs 20 --augmentation light
```

The command downloads Fashion-MNIST into `data/`, trains with seed 42, evaluates the best
validation checkpoint on the official test set, and creates:

- `artifacts/fashion_cnn.pt`
- `results/cnn/metrics.json`
- `results/cnn/confusion_matrix.png`
- `results/cnn/learning_curves.png`
- `results/cnn/run-summary.txt`

For a short pipeline check, use fewer epochs. The result is not expected to meet the quality target:

```bash
python -m training.train_cnn --epochs 1 --patience 1 --augmentation none
```

Useful training options:

```text
--device auto|cpu|cuda|mps
--batch-size 128
--learning-rate 0.001
--weight-decay 0.0001
--validation-fraction 0.2
--patience 5
--augmentation none|light
--seed 42
--num-workers 0
```

`num-workers=0` is the safest default on Windows. Increasing it may improve loading speed on Linux.

### 3. Start the FastAPI server

```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8765
```

Open the interactive API documentation at <http://127.0.0.1:8765/docs> and check readiness at
<http://127.0.0.1:8765/health>.

Expected ready health response:

```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "cnn-20260805T120000Z",
  "detail": "Model is ready."
}
```

If `model_ready` is false, inspect `detail`, confirm that `artifacts/fashion_cnn.pt` exists, and
restart the API after training.

### 4. Start the React client

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The default client expects the API at <http://127.0.0.1:8765>.

## Train and compare the baseline

Train multinomial logistic regression using flattened pixels:

```bash
python -m training.train_baseline --max-iterations 1000 --regularization-c 0.01
```

After both experiments finish, generate a compact comparison:

```bash
python -m training.compare_models
```

The output is `results/model-comparison.md`. Compare accuracy and macro F1 first. Model size and
latency are secondary and must be measured in the same environment for a fair comparison.

Typical project targets, not guaranteed results, are approximately 83–86% test accuracy for the
baseline and at least 90% for the compact CNN. Record the actual output rather than replacing it
with target values.

### Verified reference run

The included result files were generated on 2026-08-05 with seed 42, an 80/20 stratified split,
PyTorch 2.5.1 CPU, and the documented commands:

| Model | Validation accuracy | Test accuracy | Test macro F1 | Artifact size |
|---|---:|---:|---:|---:|
| Logistic regression (`C=0.01`) | 85.54% | 84.51% | 84.42% | 0.08 MiB |
| FashionCNN (light augmentation) | 92.42% | 91.93% | 91.88% | 1.80 MiB |

FashionCNN improved held-out test accuracy by 7.42 percentage points. Its measured single-image
model-forward latency was 1.51 ms in this environment; this excludes HTTP and image decoding and
must not be treated as a cross-device benchmark. See `results/model-comparison.md` and the detailed
JSON files for reproducible values.

## API usage

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Report API and model readiness |
| `GET` | `/api/v1/classes` | Return canonical ordered labels |
| `POST` | `/api/v1/predict` | Classify one multipart image |

Accepted upload types are PNG, JPEG, and WebP. The default byte limit is 5 MB, and the default
decoded-image limit is 40 million pixels.

Bash or macOS:

```bash
curl -X POST http://localhost:8765/api/v1/predict \
  -H "Accept: application/json" \
  -F "file=@/absolute/path/to/garment.png"
```

Windows PowerShell using `curl.exe`:

```powershell
curl.exe -X POST http://localhost:8765/api/v1/predict `
  -H "Accept: application/json" `
  -F "file=@C:\absolute\path\to\garment.png"
```

Example response:

```json
{
  "request_id": "9b5c1e8a-6ab6-4d98-9f8e-9e4a3ab471fb",
  "class_id": 7,
  "class_name": "Sneaker",
  "confidence": 0.9462,
  "is_unknown": false,
  "rejection_reason": null,
  "top_predictions": [
    {"class_id": 7, "class_name": "Sneaker", "probability": 0.9462},
    {"class_id": 9, "class_name": "Ankle boot", "probability": 0.0314},
    {"class_id": 5, "class_name": "Sandal", "probability": 0.0128}
  ],
  "model_version": "cnn-20260805T120000Z",
  "inference_ms": 8.4
}
```

For an image that does not resemble the Fashion-MNIST input domain, or whose scores are too
ambiguous, the primary result is rejected while the raw guesses remain available for inspection:

```json
{
  "class_id": null,
  "class_name": "Unknown",
  "confidence": null,
  "is_unknown": true,
  "rejection_reason": "The image background is too complex for this Fashion-MNIST model.",
  "top_predictions": [
    {"class_id": 8, "class_name": "Bag", "probability": 0.9998}
  ]
}
```

The API echoes or generates an `X-Request-ID` response header. Error responses include the same ID
for troubleshooting. Uploaded bytes are processed in memory and are not persisted.

## Image preprocessing

The serving pipeline performs these deterministic steps:

1. Validate declared media type and compressed byte count.
2. Decode with Pillow and reject invalid or excessively large images.
3. Apply EXIF orientation.
4. Convert to one-channel grayscale and estimate a uniform background from the image border.
5. Extract the contrast foreground, resize it within a 24-by-24 area, and center it on a black
   28-by-28 canvas. Native dark-background 28-by-28 inputs remain unchanged.
6. Mark inputs with a complex background or implausible foreground coverage as out of domain.
7. Convert pixels to `[0, 1]`.
8. Normalize using Fashion-MNIST mean `0.2860405969887955` and standard deviation
   `0.35302424451492237`.

The canonical values live only in `backend/app/domain.py`. Do not copy them into a new module.

The foreground extraction is intentionally lightweight, not semantic garment segmentation. For
the most representative manual test, use one centered item on a plain, uncluttered background.

## Configuration

Copy `.env.example` values into the relevant shell or deployment configuration. The application
does not automatically read a `.env` file, avoiding a hidden runtime dependency.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `artifacts/fashion_cnn.pt` | Checkpoint path, relative to repository root or absolute |
| `MAX_UPLOAD_BYTES` | `5242880` | Maximum compressed upload size |
| `MAX_IMAGE_PIXELS` | `40000000` | Maximum decoded width multiplied by height |
| `CORS_ORIGINS` | local ports 5173 and 3000 | Comma-separated allowed browser origins |
| `VITE_API_URL` | `http://localhost:8765` | API URL embedded in the frontend build |

Example PowerShell override:

```powershell
$env:MODEL_PATH = "C:\models\approved-fashion-cnn.pt"
$env:CORS_ORIGINS = "https://classifier.example.com"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8765
```

## Run with Docker Compose

Train or copy a compatible checkpoint to `artifacts/fashion_cnn.pt` first, then run:

```bash
docker compose up --build
```

- Web interface: <http://localhost:3000>
- API: <http://127.0.0.1:8765>
- OpenAPI UI: <http://127.0.0.1:8765/docs>

The checkpoint directory is mounted read-only into the API container. The web container waits for
the API health check to report that a model is ready. Stop the stack with:

```bash
docker compose down
```

## Testing and quality checks

Backend tests do not download Fashion-MNIST. They create a temporary random-weight checkpoint to
test the checkpoint and API contracts:

```bash
python -m pytest
python -m pytest --cov=backend.app --cov=training --cov-report=term-missing
```

Lint and type-check:

```bash
python -m ruff check backend training
python -m mypy backend/app
```

Frontend tests and production build:

```bash
cd frontend
npm test
npm run build
```

The random checkpoint used in tests validates data flow, not classification quality. Only metrics
from a trained checkpoint may be cited as model performance.

## Experiment protocol

For a defensible report, run and preserve at least these experiments:

| Experiment | Model | Augmentation | Purpose |
|---|---|---|---|
| E1 | Logistic regression | None | Classical baseline |
| E2 | FashionCNN | None | CNN architecture comparison |
| E3 | FashionCNN | Light affine | Augmentation ablation |

Use the same split seed and validation fraction. Select hyperparameters with validation results,
not the official test set. Run test evaluation after selecting the best validation checkpoint.
Discuss class-level errors, especially visually similar upper-body categories such as Shirt,
T-shirt/top, Pullover, and Coat.

For a rigorous augmentation comparison, save E2 and E3 to different artifact and result paths:

```bash
python -m training.train_cnn --augmentation none \
  --artifact artifacts/fashion_cnn_no_aug.pt \
  --results-dir results/cnn-no-augmentation

python -m training.train_cnn --augmentation light \
  --artifact artifacts/fashion_cnn.pt \
  --results-dir results/cnn-light-augmentation
```

## Checkpoint safety and compatibility

The API uses PyTorch's weights-only loader and treats the local checkpoint as a deployment
artifact. It verifies schema version, architecture ID, ordered labels, input size, and normalization
metadata before loading weights. A mismatch leaves the API in degraded state instead of silently
returning incorrectly mapped predictions.

Do not accept checkpoint uploads from web users. Model approval and artifact provenance remain
deployment responsibilities.

## Refactoring guide

Read the specification before refactoring. The highest-risk changes are label order, preprocessing,
and checkpoint structure because errors may produce plausible but semantically wrong predictions.

- Add an architecture through a registry keyed by the checkpoint `architecture` value.
- Keep HTTP behavior in `main.py`; keep tensor and model behavior outside it.
- Keep endpoint details inside `frontend/src/api.js` rather than presentation components.
- Version the API if the taxonomy or response semantics change.
- Add a checkpoint migration rather than weakening compatibility validation.
- Add real-world data as a separate experiment and document its licensing and split strategy.
- Preserve structured errors and request IDs when introducing middleware or routers.

The normative invariants and acceptance criteria are in `docs/SPECIFICATION.md`.

## Troubleshooting

### Health reports `degraded`

- Confirm `artifacts/fashion_cnn.pt` exists.
- Confirm the API process starts from the repository root or set an absolute `MODEL_PATH`.
- Read the health `detail` for checkpoint compatibility errors.
- Restart the API after creating or replacing a checkpoint.

### `No module named backend`

Run commands from the repository root. The documented module commands assume that location.

### PyTorch cannot be installed

Use Python 3.11 or 3.12. For CUDA, select the correct installation command from the official
PyTorch website. Do not force an incompatible wheel into the environment.

### Browser shows a network or CORS error

- Confirm the API is reachable at the URL configured by `VITE_API_URL`.
- Confirm the exact frontend origin is included in `CORS_ORIGINS`.
- Restart Vite after changing `VITE_API_URL`; it is read when Vite starts/builds.

### The API port is already in use

Inspect port 8765 first, stop a stale process if appropriate, or choose another local port and point
Vite to the same value before starting both processes:

```powershell
Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
$env:VITE_API_URL = "http://127.0.0.1:8766"
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8766
```

Restart the Vite development server after changing the variable.

### Predictions on phone or catalog photos look unreasonable

This is expected domain shift. Crop to one item on a simple dark background for a demonstration,
or retrain on representative product photography for a real application.

## Deliverables produced by this repository

- Reproducible baseline and CNN training commands.
- Versioned CNN checkpoint contract.
- Machine-readable metrics and report plots.
- FastAPI service with OpenAPI documentation and upload protections.
- Tested React client with responsive result visualization.
- Docker definitions for local deployment.
- Specification, architecture notes, model card, and refactoring invariants.

Generated result reports are suitable for source control. Dataset files and binary model artifacts
are ignored by Git; publish an approved checkpoint through a release or model registry when other
users should run inference without training it themselves.

## References

- Xiao, H., Rasul, K., and Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for
  Benchmarking Machine Learning Algorithms*. <https://arxiv.org/abs/1708.07747>
- Official Fashion-MNIST repository and label definitions:
  <https://github.com/zalandoresearch/fashion-mnist>
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*.
  <https://arxiv.org/abs/1512.03385>

## License note

No project license is asserted by this scaffold. Add an appropriate repository license before
redistributing the application. Review the dataset and dependency licenses independently.
