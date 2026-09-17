# Hướng dẫn nắm nhanh dự án ThreadSense

Tài liệu này là bản đồ kỹ thuật của toàn bộ repository. Mục tiêu là giúp một người mới:

- hiểu bài toán machine learning mà dự án đang giải quyết;
- biết mỗi thư mục và file chịu trách nhiệm gì;
- theo được luồng chạy từ huấn luyện đến dự đoán trên trình duyệt;
- biết từng class, function và test chính đang làm gì;
- xác định đúng nơi cần sửa khi muốn mở rộng dự án.

Nội dung mô tả source hiện tại, bao gồm cơ chế trả về `Unknown` cho ảnh không phù hợp với miền
Fashion-MNIST.

## 1. Mô hình tư duy trong 90 giây

ThreadSense có ba chương trình độc lập nhưng liên kết với nhau:

```text
TRAINING (chạy offline)
Fashion-MNIST -> train Logistic Regression/CNN -> artifacts + metrics + biểu đồ
                                                    |
                                                    v
BACKEND (chạy liên tục)
FastAPI startup -> đọc artifacts/fashion_cnn.pt -> nhận ảnh -> CNN -> JSON
                                                                    |
                                                                    v
FRONTEND (chạy trong browser)
React -> chọn ảnh -> POST /api/v1/predict -> hiển thị class hoặc Unknown
```

Ba ý quan trọng nhất:

1. Frontend không chứa model và không tự phân loại ảnh. Nó chỉ gửi file cho backend.
2. Backend không huấn luyện model. Nó chỉ load checkpoint do `training/train_cnn.py` tạo ra.
3. `CLASS_NAMES`, input size và normalization là hợp đồng dùng chung. Sai label order hoặc sai
   normalization có thể tạo dự đoán trông hợp lệ nhưng mang ý nghĩa sai.

## 2. Bài toán và phạm vi

### Input và output

- Input: một ảnh PNG, JPEG hoặc WebP, tối đa 5 MB theo cấu hình mặc định.
- Output hợp lệ: một trong 10 lớp Fashion-MNIST và top 3 softmax scores.
- Output bị từ chối: `class_name = "Unknown"`, kèm `rejection_reason` và top 3 dự đoán thô.

### Mười lớp cố định

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

### Dữ liệu thực tế

Fashion-MNIST có tổng cộng 70.000 ảnh grayscale 28 x 28:

- 60.000 ảnh thuộc official training split;
- 10.000 ảnh thuộc official test split;
- mặc định project tách 20% của 60.000 ảnh làm validation;
- do đó một run mặc định có 48.000 train, 12.000 validation và 10.000 test.

Test set chỉ được dùng sau khi đã chọn epoch tốt nhất bằng validation loss. Nó không được dùng để
chọn model hoặc điều chỉnh hyperparameter trong cùng một experiment.

### Giới hạn cần nhớ

Model học trên ảnh nhỏ, grayscale, vật thể nằm giữa và nền tối. Ảnh sản phẩm ngoài đời có màu sắc,
nền phức tạp, nhiều vật thể hoặc góc chụp lạ là domain khác. Cơ chế `Unknown` hiện tại giảm các kết
quả ép chọn sai, nhưng chỉ là heuristic chứ không phải OOD detector có bảo đảm.

## 3. Thứ tự đọc source được khuyến nghị

1. `backend/app/domain.py`: label, input shape và normalization contract.
2. `backend/app/model.py`: CNN nhận tensor gì và trả ra gì.
3. `training/data.py`: dữ liệu được split và transform như thế nào.
4. `training/train_cnn.py`: vòng lặp train, validation, early stopping và save model.
5. `backend/app/preprocessing.py`: ảnh upload được biến thành tensor ra sao.
6. `backend/app/inference.py`: load checkpoint, softmax, top-k và `Unknown`.
7. `backend/app/main.py`: HTTP API và error handling.
8. `frontend/src/App.jsx`, `api.js` và hai component: UI gửi request và render kết quả.
9. Các file test: những behavior nào đang được bảo vệ khi refactor.

Sau đó đọc `SPECIFICATION.md`, `ARCHITECTURE.md` và `MODEL_CARD.md` để hiểu yêu cầu, quyết định
thiết kế, số liệu và giới hạn của model.

## 4. Cây thư mục

Các cache và dependency folder như `.venv`, `node_modules`, `.pytest_cache`, `.mypy_cache`,
`.ruff_cache` và `frontend/dist` được rút gọn vì chúng được sinh tự động.

```text
CNN/
|-- .env.example
|-- .gitignore
|-- docker-compose.yml
|-- pyproject.toml
|-- README.md
|-- artifacts/
|   |-- .gitkeep
|   |-- fashion_cnn.pt
|   `-- logistic_regression.joblib
|-- backend/
|   |-- __init__.py
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- requirements-dev.txt
|   |-- app/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- domain.py
|   |   |-- errors.py
|   |   |-- schemas.py
|   |   |-- model.py
|   |   |-- preprocessing.py
|   |   |-- inference.py
|   |   `-- main.py
|   `-- tests/
|       |-- __init__.py
|       |-- conftest.py
|       |-- test_api.py
|       |-- test_inference.py
|       |-- test_model.py
|       `-- test_preprocessing.py
|-- data/
|   `-- FashionMNIST/raw/...
|-- docs/
|   |-- SPECIFICATION.md
|   |-- ARCHITECTURE.md
|   |-- MODEL_CARD.md
|   `-- PROJECT_GUIDE.md
|-- frontend/
|   |-- Dockerfile
|   |-- nginx.conf
|   |-- index.html
|   |-- package.json
|   |-- package-lock.json
|   |-- vite.config.js
|   `-- src/
|       |-- main.jsx
|       |-- App.jsx
|       |-- api.js
|       |-- styles.css
|       |-- App.test.jsx
|       |-- components/
|       |   |-- UploadPanel.jsx
|       |   `-- PredictionPanel.jsx
|       `-- test/setup.js
|-- results/
|   |-- .gitkeep
|   |-- model-comparison.md
|   |-- baseline/
|   |   |-- metrics.json
|   |   `-- confusion_matrix.png
|   `-- cnn/
|       |-- metrics.json
|       |-- run-summary.txt
|       |-- confusion_matrix.png
|       `-- learning_curves.png
`-- training/
    |-- __init__.py
    |-- data.py
    |-- metrics.py
    |-- train_baseline.py
    |-- train_cnn.py
    |-- compare_models.py
    `-- tests/test_data.py
```

## 5. Các file ở project root

| File | Nhiệm vụ |
|---|---|
| `README.md` | Hướng dẫn setup, train, chạy local, Docker, API, test và troubleshooting. |
| `pyproject.toml` | Cấu hình pytest, Ruff và mypy cho Python. Không phải danh sách dependency. |
| `.env.example` | Mẫu các biến môi trường. App không tự động đọc file `.env`. |
| `.gitignore` | Loại dataset, model binary, environment, cache và build output khỏi Git. |
| `docker-compose.yml` | Build/chạy service `api` và `web`, mount checkpoint read-only. |

### Các thư mục ẩn hoặc được sinh tự động

| Thư mục | Ý nghĩa | Có nên sửa tay? |
|---|---|---|
| `.git/` | Lịch sử và metadata Git. | Không. |
| `.venv/` | Python virtual environment. | Không; tạo lại từ requirements. |
| `.tools/` | Tool cục bộ phục vụ môi trường phát triển. | Không. |
| `.pytest_cache/` | Cache của pytest. | Không. |
| `.mypy_cache/` | Cache type checking. | Không. |
| `.ruff_cache/` | Cache lint. | Không. |
| `.matplotlib/` | Cache/config runtime của Matplotlib. | Không. |

## 6. Backend: FastAPI và inference

Dependency direction:

```text
main.py
|-- config.py
|-- domain.py
|-- errors.py
|-- schemas.py
`-- inference.py
    |-- domain.py
    |-- model.py
    `-- preprocessing.py
```

`main.py` biết HTTP nhưng model và preprocessing không biết FastAPI. Vì vậy có thể test model mà
không mở server và thay HTTP layer mà không sửa logic ML.

### 6.1 `backend/app/domain.py`

Đây là nguồn sự thật duy nhất cho các hằng số nghiệp vụ và model contract.

| Thành phần | Ý nghĩa |
|---|---|
| `CLASS_NAMES` | Tuple 10 class theo đúng index Fashion-MNIST. Không được tự ý reorder. |
| `INPUT_CHANNELS/HEIGHT/WIDTH` | Model nhận 1 channel, cao 28, rộng 28. |
| `INPUT_SIZE` | Shape không có batch dimension: `(1, 28, 28)`. |
| `NORMALIZATION_MEAN` | Mean Fashion-MNIST dùng cho cả train và inference. |
| `NORMALIZATION_STD` | Standard deviation dùng cho cả train và inference. |
| `MODEL_ARCHITECTURE` | ID `fashion_cnn_v1` để checkpoint và code nhận diện nhau. |
| `CHECKPOINT_SCHEMA_VERSION` | Version cấu trúc dictionary trong file `.pt`. |
| `SUPPORTED_MEDIA_TYPES` | MIME types backend cho phép: JPEG, PNG và WebP. |

File này không có function và cố ý không phụ thuộc FastAPI, torchvision hoặc training code.

### 6.2 `backend/app/config.py`

| Function/class | Input -> output | Việc thực hiện |
|---|---|---|
| `_positive_int_from_env(name, default)` | tên biến + default -> `int` | Dùng default nếu env thiếu, ép kiểu và từ chối số không/âm. |
| `_path_from_env(name, default)` | tên biến + path -> `Path` | Mở rộng `~`; path tương đối được neo vào project root. |
| `Settings` | dataclass immutable | Gom model path, giới hạn byte/pixel và CORS origins. |
| `Settings.from_env()` | environment -> `Settings` | Parse bốn biến cấu hình runtime. |

`PROJECT_ROOT = Path(__file__).resolve().parents[2]` trỏ tới root repository, nên model path mặc định
là `artifacts/fashion_cnn.pt` kể cả khi logic config nằm sâu trong package.

### 6.3 `backend/app/errors.py`

| Function/class | Nhiệm vụ |
|---|---|
| `AppError` | Exception cho lỗi request dự kiến; chứa HTTP status, machine code và message. |
| `AppError.__init__()` | Gán ba trường trên và truyền message cho base `Exception`. |

`main.py` bắt `AppError` tập trung để mọi lỗi có cùng JSON shape.

### 6.4 `backend/app/schemas.py`

Các Pydantic model định nghĩa response contract và validate trước khi serialize JSON.

| Schema | Nội dung |
|---|---|
| `ClassItem` | Một cặp `id`, `name`. |
| `ClassListResponse` | Danh sách 10 `ClassItem`. |
| `HealthResponse` | Trạng thái server, model readiness, version và detail. |
| `RankedPrediction` | Một class dự đoán thô; probability bị giới hạn trong `[0, 1]`. |
| `PredictionResponse` | Request ID, class chính/Unknown, top-k, model version và latency. |
| `ErrorBody` | Error code, message và request ID phục vụ trace. |
| `ErrorResponse` | Bọc `ErrorBody` trong key `error`. |

Khi `is_unknown = true`, contract chính là:

```json
{
  "class_id": null,
  "class_name": "Unknown",
  "confidence": null,
  "is_unknown": true,
  "rejection_reason": "...",
  "top_predictions": ["các softmax guesses thô vẫn được giữ lại"]
}
```

### 6.5 `backend/app/model.py`

#### `FashionCNN.__init__(num_classes=10)`

Tạo hai block:

- `features`: convolution, batch normalization, ReLU, max pooling và dropout để học đặc trưng;
- `classifier`: flatten feature map, dense 128 units, dropout rồi trả 10 logits.

Shape đi qua mạng:

| Bước | Output shape cho batch N |
|---|---|
| Input | `[N, 1, 28, 28]` |
| Conv 1 -> Conv 2 | `[N, 32, 28, 28]` |
| MaxPool đầu | `[N, 32, 14, 14]` |
| Conv 3 -> Conv 4 | `[N, 64, 14, 14]` |
| MaxPool sau | `[N, 64, 7, 7]` |
| Flatten | `[N, 3136]` |
| Linear 128 | `[N, 128]` |
| Linear cuối | `[N, 10]` logits |

#### `FashionCNN.forward(inputs)`

1. Kiểm tra tensor có 4 chiều và phần sau batch là `(1, 28, 28)`.
2. Chạy `features` rồi `classifier`.
3. Trả raw logits, chưa chạy softmax.

Training dùng logits với `CrossEntropyLoss`. Backend chỉ chạy softmax khi cần scores hiển thị. Không
đặt softmax trong model vì `CrossEntropyLoss` đã thực hiện phép biến đổi ổn định số học cần thiết.

### 6.6 `backend/app/preprocessing.py`

File này biến ảnh người dùng thành tensor gần distribution Fashion-MNIST hơn và đưa ra tín hiệu
domain rejection.

#### Hằng số private

| Hằng số | Vai trò |
|---|---|
| `_FASHION_MNIST_BORDER_MAX = 32` | Contrast tối thiểu để xem pixel là foreground. |
| `_FOREGROUND_CANVAS_SIZE = 24` | Foreground co vào tối đa 24 x 24 để chừa padding. |
| `_MAX_BACKGROUND_VARIATION = 28` | Border biến thiên cao hơn mức này là nền phức tạp. |
| `_MIN_FOREGROUND_FRACTION = 0.01` | Foreground dưới 1% bị xem là quá nhỏ. |
| `_MAX_FOREGROUND_FRACTION = 0.70` | Foreground trên 70% là quá nhiều chi tiết. |

#### Class và function

| Function/class | Input -> output | Tóm tắt logic |
|---|---|---|
| `ImageDecodingError` | exception | Báo ảnh rỗng, không decode được hoặc quá nhiều pixel. |
| `PreparedImage` | dataclass | Tensor `[1,1,28,28]` và optional `rejection_reason`. |
| `decode_image(bytes, max_pixels)` | bytes -> PIL image | Decode, giới hạn pixel, sửa EXIF orientation và convert grayscale. |
| `_border_pixels(pixels)` | array 2D -> array 1D | Ghép bốn cạnh để ước lượng background. |
| `_foreground_canvas(image)` | image -> `(canvas, reason)` | Ước lượng nền, tạo mask, kiểm tra domain, crop, scale và center. |
| `prepare_image(image)` | PIL image -> `PreparedImage` | Adapter, scale `[0,1]`, normalize và thêm batch/channel. |
| `image_to_tensor(image)` | PIL image -> tensor | Wrapper cho nơi chỉ cần tensor. |
| `prepare_image_bytes(bytes, max_pixels)` | bytes -> `PreparedImage` | Nối decode và prepare; entry point inference chính. |
| `preprocess_image_bytes(bytes, max_pixels)` | bytes -> tensor | Wrapper tương thích trả riêng tensor. |

Chi tiết `_foreground_canvas()`:

1. Lấy median của border làm mức sáng background.
2. Dùng percentile 90 của độ lệch border để đo độ phức tạp nền.
3. Dùng absolute difference để hỗ trợ vật tối/nền sáng và vật sáng/nền tối.
4. Nếu gần như không có contrast, trả canvas đen và lý do không có foreground.
5. Tạo mask với threshold `max(32, 10% peak contrast)`.
6. Đo tỷ lệ foreground và gán reason nếu nền/coverage không hợp lệ.
7. Ảnh Fashion-MNIST gốc 28 x 28 nền tối được giữ nguyên pixel.
8. Ảnh khác được crop bounding box, thêm margin 8%, resize tối đa 24 x 24 và center.
9. Canvas vẫn được trả khi reject để CNN tạo top-k thô phục vụ debug.

### 6.7 `backend/app/inference.py`

Đây là orchestration giữa checkpoint, preprocessing và CNN.

#### Ngưỡng reject

- `_MIN_ACCEPTED_PROBABILITY = 0.55`: top-1 phải đạt ít nhất 55%.
- `_MIN_ACCEPTED_MARGIN = 0.15`: top-1 phải hơn top-2 ít nhất 15 điểm phần trăm.
- `_UNKNOWN_CLASS_NAME = "Unknown"`: nhãn serving; không phải class thứ 11 của CNN.

#### Class và function

| Function/class | Nhiệm vụ |
|---|---|
| `CheckpointCompatibilityError` | Báo checkpoint không khớp source đang chạy. |
| `RankedPredictionData` | TypedDict nội bộ cho một phần tử top-k. |
| `Prediction` | Dataclass nội bộ trước khi Pydantic serialize. |
| `_require_checkpoint_contract(checkpoint)` | Kiểm tra keys, schema, architecture, labels, input size, mean/std. |
| `PredictionService.__init__()` | Giữ model ở eval mode, model version và pixel limit. |
| `PredictionService.from_checkpoint()` | Load weights-only ở CPU, validate, dựng CNN và strict-load state. |
| `PredictionService.predict_bytes()` | Preprocess, forward, softmax, top-k, reject rules và trả kết quả. |

Thứ tự rejection:

1. Lý do từ preprocessing, ví dụ background phức tạp.
2. Nếu domain gate qua nhưng top-1 dưới 0.55, reject vì score thấp.
3. Nếu top-1 đủ cao nhưng margin top-1/top-2 dưới 0.15, reject vì hai class quá sát.
4. Nếu không có reason, chấp nhận class top-1.

Softmax luôn cộng thành 1 trên 10 class. Ảnh lâu đài vẫn có thể có raw guess `Bag = 99.9%`, nhưng
domain gate có quyền trả kết quả chính là `Unknown`.

### 6.8 `backend/app/main.py`

Đây là composition root: tạo FastAPI app, nối các module và đổi exception thành HTTP response.

| Function/nested function | Nhiệm vụ |
|---|---|
| `_request_id(request)` | Lấy request ID đã gán; fallback bằng UUID mới. |
| `create_app(settings=None)` | Tạo app từ Settings; cho phép inject settings trong test. |
| `lifespan(app)` | Startup load model đúng một lần; lỗi thì giữ trạng thái degraded. |
| `add_request_id(request, call_next)` | Middleware nhận/tạo `X-Request-ID` và trả lại qua header. |
| `handle_app_error(request, exc)` | Chuyển `AppError` thành JSON error ổn định. |
| `handle_validation_error(request, exc)` | Chuyển lỗi FastAPI/Pydantic thành HTTP 422. |
| `health(request)` | `GET /health`; trả `ok` hoặc `degraded`. |
| `list_classes()` | `GET /api/v1/classes`; enumerate label đúng order. |
| `predict(request, file)` | Validate upload, gọi inference trong thread pool và serialize. |

`app = create_app()` là object Uvicorn import bằng `backend.app.main:app`.

Luồng lỗi của `predict()`:

| Điều kiện | HTTP | Code |
|---|---:|---|
| Chưa có checkpoint hợp lệ | 503 | `model_unavailable` |
| MIME type không hỗ trợ | 400 | `unsupported_media_type` |
| File vượt byte limit | 413 | `file_too_large` |
| File không phải ảnh | 400 | `invalid_image` |
| Request sai schema | 422 | `invalid_request` |
| Lỗi inference bất ngờ | 500 | `prediction_failed` |

`run_in_threadpool()` được dùng vì Pillow và PyTorch là CPU-bound; endpoint async không nên chặn
event loop trong lúc decode hoặc forward.

### 6.9 Các file backend còn lại

| File | Nhiệm vụ |
|---|---|
| `backend/__init__.py` | Đánh dấu `backend` là package, giúp import/mypy ổn định. |
| `backend/app/__init__.py` | Đánh dấu package ứng dụng. |
| `backend/tests/__init__.py` | Đánh dấu test folder là package để import helper ổn định. |
| `backend/requirements.txt` | Runtime dependencies. |
| `backend/requirements-dev.txt` | Runtime deps cộng test, lint, typing, sklearn và plotting. |
| `backend/Dockerfile` | Python 3.11 image, cài deps, copy source/artifact và chạy port 8765. |

## 7. Training pipeline

Training import `FashionCNN` và domain constants từ backend để model lúc train và serve không lệch
kiến trúc, label hoặc normalization.

### 7.1 `training/data.py`

| Function/class | Nhiệm vụ |
|---|---|
| `DataLoaders` | Gom ba DataLoader và kích thước các split. |
| `seed_everything(seed)` | Seed Python, NumPy, Torch, CUDA và bật CuDNN deterministic. |
| `stratified_indices(...)` | Chia index train/validation riêng theo class, deterministic, không overlap. |
| `_training_transform(augmentation)` | Với `light`, thêm RandomAffine; sau đó ToTensor/Normalize. |
| `_evaluation_transform()` | Chỉ ToTensor/Normalize, không augmentation. |
| `_worker_seed(worker_id)` | Seed NumPy/Python trong từng DataLoader worker. |
| `create_data_loaders(...)` | Download/read data, split 60k và tạo ba DataLoader. |

Light augmentation: xoay tối đa 10 độ, dịch tối đa 8%, scale 0.95 đến 1.05, vùng trống fill 0.
Train và validation là hai dataset object khác nhau nên augmentation không rò sang validation.

### 7.2 `training/metrics.py`

Matplotlib dùng backend `Agg` để chạy trong terminal/container không có GUI.

| Function | Nhiệm vụ |
|---|---|
| `classification_metrics(...)` | Accuracy, macro precision/recall/F1, per-class report và confusion matrix. |
| `save_json(payload, path)` | Tạo parent folder và ghi JSON dễ review. |
| `save_confusion_matrix(...)` | Vẽ heatmap 10 x 10 có label/số đếm rồi lưu PNG. |
| `save_learning_curves(...)` | Vẽ train/validation loss và accuracy theo epoch. |

Macro metrics cho mỗi class trọng số bằng nhau, giúp không che lấp class yếu bằng class mạnh.

### 7.3 `training/train_cnn.py`

| Function | Nhiệm vụ |
|---|---|
| `parse_args()` | CLI cho path, epochs, batch, LR, decay, split, patience, augmentation, seed, worker, device. |
| `resolve_device(requested)` | `auto` ưu tiên CUDA, MPS, rồi CPU. |
| `run_epoch(...)` | Chạy một epoch train/eval; trả mean loss, accuracy, targets và predictions. |
| `estimate_inference_ms(...)` | Warm-up rồi đo forward latency một ảnh. |
| `build_checkpoint(...)` | Đóng gói weights cùng metadata, hyperparameters và metrics. |
| `main()` | Điều phối toàn bộ experiment CNN và ghi outputs. |

Chi tiết `run_epoch()`:

1. Có optimizer nghĩa là train mode; không có optimizer là evaluation mode.
2. Chuyển ảnh và label sang device.
3. Khi train, clear gradient.
4. Forward CNN và tính cross-entropy.
5. Khi train, backpropagation và optimizer step.
6. Lấy `argmax` làm predicted class.
7. Cộng dồn loss, số đúng, targets và predictions.
8. Trả giá trị tổng hợp cuối epoch.

Chi tiết `main()`:

1. Validate args, seed RNG và chọn device.
2. Tạo data loaders, CNN, CrossEntropyLoss, AdamW và ReduceLROnPlateau.
3. Mỗi epoch chạy train rồi validation.
4. Scheduler giảm LR khi validation loss không cải thiện.
5. Deep-copy state có validation loss tốt nhất.
6. Early stop khi số epoch không cải thiện đạt `patience`.
7. Restore best state rồi mới evaluate validation và official test.
8. Tính metric, latency và model version theo UTC timestamp.
9. Save checkpoint, JSON, confusion matrix, learning curves và run summary.

### 7.4 `training/train_baseline.py`

| Function | Nhiệm vụ |
|---|---|
| `parse_args()` | CLI cho path, split, số iteration, regularization C và seed. |
| `flatten_and_scale(images)` | `[N,28,28]` thành `[N,784]`, float32 và chia 255. |
| `main()` | Download, split, fit sklearn pipeline, evaluate và ghi artifact/reports. |

Pipeline là `StandardScaler -> LogisticRegression(lbfgs)`. Baseline không được API serve; nó là mốc
so sánh để chứng minh CNN tận dụng spatial structure tốt hơn.

### 7.5 `training/compare_models.py`

| Function | Nhiệm vụ |
|---|---|
| `parse_args()` | Path hai metrics JSON và Markdown output. |
| `read_metrics(path)` | Kiểm tra tồn tại rồi parse JSON. |
| `row(name, report)` | Biến metric một model thành dòng Markdown. |
| `main()` | Đọc hai report và ghi bảng `results/model-comparison.md`. |

### 7.6 Các file training còn lại

| File | Nhiệm vụ |
|---|---|
| `training/__init__.py` | Đánh dấu package để chạy `python -m training...`. |
| `training/tests/test_data.py` | Test stratified split deterministic, disjoint và cân bằng. |

## 8. Artifacts, data và results

### `artifacts/`

| File | Nguồn tạo | Người dùng |
|---|---|---|
| `fashion_cnn.pt` | `training.train_cnn` | FastAPI load khi startup. |
| `logistic_regression.joblib` | `training.train_baseline` | Lưu baseline, API hiện tại không serve. |
| `.gitkeep` | Repository | Giữ folder rỗng trong Git. |

Checkpoint CNN hiện khoảng 1.80 MiB, version `cnn-20260805T145058Z`.

### Cấu trúc checkpoint CNN

```text
schema_version       format checkpoint
architecture         fashion_cnn_v1
model_version        ID theo timestamp
model_state_dict     weights và buffers
class_names          ordered list 10 labels
normalization        mean/std
input_size           [1, 28, 28]
training             seed và hyperparameters
metrics              validation/test reports
```

### `results/`

| File | Nội dung |
|---|---|
| `baseline/metrics.json` | Args, timing, validation/test metrics baseline. |
| `baseline/confusion_matrix.png` | Ma trận nhầm lẫn baseline. |
| `cnn/metrics.json` | Epoch history, args, version và metrics CNN. |
| `cnn/confusion_matrix.png` | Ma trận nhầm lẫn CNN. |
| `cnn/learning_curves.png` | Loss và accuracy theo epoch. |
| `cnn/run-summary.txt` | Summary ngắn của run được promote. |
| `model-comparison.md` | Bảng generate từ hai metrics JSON. |

Kết quả tham chiếu:

| Model | Test accuracy | Macro F1 | Size |
|---|---:|---:|---:|
| Logistic Regression | 84.51% | 84.42% | 0.08 MB |
| FashionCNN | 91.93% | 91.88% | 1.80 MB |

Không so sánh latency nếu hai model không được đo trong cùng hardware/software run.

### `data/`

`torchvision.datasets.FashionMNIST` quản lý `data/FashionMNIST/raw/`. File `.gz` là bản nén tải về;
file không có `.gz` là dữ liệu đã giải nén gồm image bytes và labels. Folder được Git ignore vì có
thể download lại và không nên commit binary dataset.

| Nhóm file | Nội dung |
|---|---|
| `train-images-idx3-ubyte(.gz)` | 60.000 ảnh của official training split. |
| `train-labels-idx1-ubyte(.gz)` | 60.000 class IDs tương ứng. |
| `t10k-images-idx3-ubyte(.gz)` | 10.000 ảnh official test. |
| `t10k-labels-idx1-ubyte(.gz)` | 10.000 test class IDs tương ứng. |

## 9. Frontend: React và Vite

Frontend chỉ biết public API contract. Nó không import Python và không đọc checkpoint.

### 9.1 `frontend/src/main.jsx`

Entry point tìm `#root` trong `index.html`, import CSS và render `<App />` trong React StrictMode.

### 9.2 `frontend/src/api.js`

| Function/class | Nhiệm vụ |
|---|---|
| `API_BASE_URL` | Lấy `VITE_API_URL`, fallback port 8765 và bỏ slash cuối. |
| `ApiError` | Error client có message, code và optional request ID. |
| `ApiError.constructor()` | Gán metadata để UI hiển thị lỗi nhất quán. |
| `classifyImage(file, signal)` | Tạo FormData, fetch, parse JSON và chuẩn hóa network/API errors. |

Không set `Content-Type` bằng tay khi gửi FormData; browser tự thêm multipart boundary.

### 9.3 `frontend/src/App.jsx`

| State/ref | Ý nghĩa |
|---|---|
| `file` | File đang chọn. |
| `previewUrl` | Blob URL để preview ảnh local. |
| `result` | JSON prediction gần nhất. |
| `error` | ApiError gần nhất. |
| `busy` | Đang chờ request. |
| `activeRequest` | AbortController của request hiện tại. |

| Function/effect | Nhiệm vụ |
|---|---|
| Effect unmount | Abort request còn chạy khi App bị tháo. |
| Effect theo `previewUrl` | Revoke blob URL cũ, tránh memory leak. |
| `selectFile(nextFile)` | Abort request cũ, lưu file/preview và reset result/error. |
| `submit()` | Guard, tạo controller, gọi API, cập nhật state và cleanup busy. |
| `App()` return | Ghép header, tiêu đề công cụ, upload panel và prediction panel. |

Check `activeRequest.current === controller` ngăn request cũ kết thúc muộn làm sai state request mới.

### 9.4 `frontend/src/components/UploadPanel.jsx`

| Function/state | Nhiệm vụ |
|---|---|
| `ACCEPTED_TYPES` | Client allowlist trùng backend. Backend vẫn validate lại. |
| `MAX_FILE_BYTES` | Client limit 5 MB. Backend vẫn là trust boundary thực. |
| `validateFile(file)` | Trả message nếu MIME sai/quá lớn; hợp lệ trả `null`. |
| `UploadPanel(props)` | Render drop zone, input, preview, error và submit button. |
| `acceptFile(candidate)` | Validate rồi gọi `onFile` nếu hợp lệ. |
| `handleDrop(event)` | Chặn browser mở file và lấy file đầu tiên. |
| `dragActive` | Visual state khi kéo file. |
| `localError` | Lỗi trước khi gọi API. |

### 9.5 `frontend/src/components/PredictionPanel.jsx`

| Function/component | Nhiệm vụ |
|---|---|
| `formatPercentage(value)` | Probability 0..1 thành phần trăm một chữ số thập phân. |
| `PredictionPanel({result,error})` | Render error, result hoặc empty state. |

Accepted result hiện class và model probability. Unknown result hiện reason và đổi tiêu đề top-k
thành `Raw model guesses`, tránh gọi raw 100% là class được chấp nhận.

### 9.6 Các file frontend còn lại

| File | Nhiệm vụ |
|---|---|
| `src/styles.css` | Layout, typography, responsive states, spinner và result styling. |
| `index.html` | HTML shell, metadata, `#root` và entry script. |
| `vite.config.js` | React plugin, dev port 5173 và Vitest jsdom. |
| `package.json` | Scripts và direct dependencies. |
| `package-lock.json` | Lock dependency tree cho `npm ci`; không sửa tay. |
| `src/test/setup.js` | Nạp matcher jest-dom vào Vitest. |
| `Dockerfile` | Build Vite rồi dùng Nginx serve static files. |
| `nginx.conf` | SPA fallback và cache static assets 7 ngày. |

## 10. Workflow dự đoán end-to-end

```text
UploadPanel.acceptFile()
  -> App.selectFile()
  -> URL.createObjectURL() tạo preview

Người dùng bấm Classify image
  -> App.submit()
  -> api.classifyImage(file, AbortSignal)
  -> POST http://127.0.0.1:8765/api/v1/predict
  -> middleware add_request_id()
  -> endpoint predict()
  -> validate model, MIME type và byte size
  -> run_in_threadpool(PredictionService.predict_bytes)
  -> prepare_image_bytes()
  -> decode_image()
  -> prepare_image()
  -> _foreground_canvas()
  -> FashionCNN.forward()
  -> torch.softmax() + torch.topk()
  -> domain/score rejection rules
  -> PredictionResponse JSON
  -> App.setResult()
  -> PredictionPanel render class hoặc Unknown
```

Một request hợp lệ đi qua các bước:

1. Browser gửi multipart field `file`.
2. Backend đọc tối đa `MAX_UPLOAD_BYTES + 1` để phát hiện file quá lớn.
3. Pillow decode và giới hạn số pixel giải nén.
4. Ảnh thành grayscale tensor `[1,1,28,28]`.
5. CNN trả 10 logits; softmax đổi thành 10 scores tổng bằng 1.
6. Top 3 được sort giảm dần.
7. Nếu qua reject rules, top 1 là class chính; nếu không, class chính là Unknown.
8. Pydantic validate, FastAPI serialize và React render JSON.

Ví dụ Unknown:

```text
raw model winner = Bag, probability = 0.9999
domain check = background too complex
final API class = Unknown
```

`Unknown` là serving policy, không phải output neuron thứ 11.

## 11. Workflow startup backend

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8765
```

```text
Settings.from_env()
  -> tìm artifacts/fashion_cnn.pt
  -> PredictionService.from_checkpoint()
  -> torch.load(weights_only=True, map_location="cpu")
  -> _require_checkpoint_contract()
  -> FashionCNN()
  -> load_state_dict(strict=True)
  -> model.eval()
  -> app.state.predictor = service
```

Checkpoint thiếu/sai vẫn cho server khởi động để `/health` trả `degraded`; predict trả 503. Flag
`--reload` chỉ dành cho development.

## 12. Workflow huấn luyện CNN

```powershell
python -m training.train_cnn --epochs 20 --augmentation light
```

```text
FashionMNIST raw files
  -> deterministic stratified split
  -> train transform có augmentation
  -> validation/test transform không augmentation
  -> DataLoader batches
  -> FashionCNN logits
  -> CrossEntropyLoss
  -> backward + AdamW step
  -> validation loss
  -> scheduler + early stopping + best_state
  -> restore best_state
  -> official test evaluation
  -> checkpoint + metrics JSON + plots
```

Best model chọn bằng validation loss; dùng test để chọn epoch sẽ làm kết quả test bị optimistic.

## 13. Workflow baseline và so sánh

```powershell
python -m training.train_baseline --max-iterations 1000 --regularization-c 0.01
python -m training.compare_models
```

Baseline xem ảnh như vector 784 pixels, không tận dụng locality như CNN. Script compare không chạy
model; nó chỉ đọc hai `metrics.json` và generate Markdown table.

## 14. Các test đang bảo vệ điều gì

### `backend/tests/conftest.py`

| Helper/fixture | Behavior |
|---|---|
| `checkpoint_payload(**overrides)` | Tạo CNN/checkpoint giả; cố định output class 7 để API known path deterministic; cho phép override metadata để test checkpoint lỗi. |
| `valid_checkpoint(tmp_path)` | Ghi payload hợp lệ vào temp path và trả path cho test cần model. |
| `make_settings(model_path, **overrides)` | Tạo `Settings` cô lập với environment thật và cho phép override giới hạn. |

### `backend/tests/test_api.py`

| Helper/test function | Behavior được bảo vệ |
|---|---|
| `png_bytes()` | Tạo PNG 28 x 28 nền đen, foreground sáng cho request hợp lệ. |
| `complex_scene_bytes()` | Tạo checkerboard nhiều chi tiết để mô phỏng ảnh out-of-domain. |
| `test_health_is_degraded_when_checkpoint_is_missing()` | Thiếu model không làm app crash; health báo degraded. |
| `test_classes_follow_canonical_order()` | Endpoint class giữ đúng thứ tự `CLASS_NAMES`. |
| `test_valid_prediction_returns_ranked_output()` | Response known có schema, version, confidence và top 3 sort đúng. |
| `test_complex_scene_is_returned_as_unknown()` | Cảnh phức tạp trả class ID/confidence null và `Unknown`. |
| `test_invalid_image_has_structured_error()` | Bytes không phải ảnh trả 400 `invalid_image` có request ID. |
| `test_upload_larger_than_limit_is_rejected_before_decoding()` | File quá byte limit trả 413 trước decode. |
| `test_prediction_is_unavailable_without_checkpoint()` | Predict trả 503 `model_unavailable` khi predictor chưa ready. |

### `backend/tests/test_inference.py`

| Helper/test function | Behavior được bảo vệ |
|---|---|
| `fashion_like_image_bytes()` | Tạo input đơn giản không bị domain gate reject. |
| `service_with_probabilities(probabilities)` | Zero weights và dùng output bias để tạo chính xác distribution cần test. |
| `test_checkpoint_rejects_changed_label_order()` | Checkpoint có label order sai bị từ chối. |
| `test_checkpoint_rejects_changed_normalization()` | Checkpoint mean/std sai bị từ chối. |
| `test_prediction_rejects_low_maximum_probability()` | Top-1 dưới 0.55 thành Unknown. |
| `test_prediction_rejects_small_top_two_margin()` | Margin top-1/top-2 dưới 0.15 thành Unknown. |

### `backend/tests/test_model.py`

| Test function | Behavior được bảo vệ |
|---|---|
| `test_model_returns_ten_logits()` | Batch bốn ảnh tạo output shape `[4,10]`. |
| `test_model_rejects_an_incompatible_shape()` | Tensor RGB/sai shape bị từ chối rõ ràng. |

### `backend/tests/test_preprocessing.py`

| Helper/test function | Behavior được bảo vệ |
|---|---|
| `image_bytes(size)` | Tạo ảnh PNG in-memory với size tùy chọn. |
| `object_image_bytes(background, foreground)` | Tạo một ellipse để so sánh nền sáng và nền tối. |
| `test_preprocessing_returns_expected_shape_and_finite_values()` | Tensor output là float32 `[1,1,28,28]`, không NaN/Inf. |
| `test_preprocessing_rejects_non_image_bytes()` | Nội dung text phát sinh `ImageDecodingError`. |
| `test_preprocessing_rejects_too_many_pixels()` | Ảnh vượt decoded-pixel limit bị từ chối. |
| `test_light_and_dark_backgrounds_produce_equivalent_foregrounds()` | Hai polarity nền/vật tạo tensor tương đương. |
| `test_native_fashion_mnist_input_is_not_recropped()` | Ảnh native 28 x 28 nền tối giữ đúng giá trị pixel. |
| `test_complex_scene_is_marked_out_of_domain()` | Checkerboard nhận rejection reason nền phức tạp. |

### `training/tests/test_data.py`

| Test function | Behavior được bảo vệ |
|---|---|
| `test_stratified_indices_are_deterministic_disjoint_and_balanced()` | Cùng seed cho cùng split, hai split không overlap và mỗi class có số validation bằng nhau. |

### Frontend

| Test | Behavior được bảo vệ |
|---|---|
| accepted image | Upload, gọi API với AbortSignal và render ranking. |
| unknown result | Hiện Unknown/reason, không gọi raw 100% là confidence chính. |
| unsupported file | Chặn file text trước API call. |

## 15. Cấu hình và port

| Variable | Default | Ý nghĩa |
|---|---|---|
| `MODEL_PATH` | `artifacts/fashion_cnn.pt` | Checkpoint CNN được serve. |
| `MAX_UPLOAD_BYTES` | `5242880` | Compressed upload limit, 5 MiB. |
| `MAX_IMAGE_PIXELS` | `40000000` | Decoded width x height limit. |
| `CORS_ORIGINS` | localhost và 127.0.0.1 trên 5173, 3000 | Browser origins được gọi API. |
| `VITE_API_URL` | `http://127.0.0.1:8765` | URL nhúng vào frontend lúc start/build. |

Đổi frontend env cần restart Vite; đổi backend env/model path cần restart Uvicorn.

Port local chuẩn:

- FastAPI: `8765`;
- Vite: `5173`;
- Nginx frontend bằng Compose: `3000`.

## 16. Cách chạy local hoàn chỉnh

### Terminal 1: backend

```powershell
cd C:\Project\CNN
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8765
```

- Health: <http://127.0.0.1:8765/health>
- Swagger: <http://127.0.0.1:8765/docs>

### Terminal 2: frontend

```powershell
cd C:\Project\CNN\frontend
npm run dev
```

Mở <http://localhost:5173>. Không cần train lại nếu checkpoint đã tồn tại và tương thích.

## 17. Docker workflow

```powershell
docker compose up --build
```

Compose build backend, chạy healthcheck, build frontend với API URL 8765 và dùng Nginx serve web.

- Web: <http://localhost:3000>
- API: <http://127.0.0.1:8765>
- API docs: <http://127.0.0.1:8765/docs>

## 18. Quality commands

```powershell
python -m pytest
python -m ruff check backend training
python -m mypy backend/app
```

```powershell
cd frontend
npm test
npm run build
```

pytest/Vitest kiểm tra behavior; Ruff kiểm tra style/import; mypy kiểm tra type contract; Vite build
kiểm tra production bundling.

## 19. Bản đồ thay đổi

| Muốn thay đổi | File chính | Cần kiểm tra cùng |
|---|---|---|
| Đổi/tăng class | `domain.py` | dataset, output layer, checkpoint, API, UI, tests; phải retrain. |
| Đổi CNN | `model.py` | architecture ID, training, inference, tests; phải retrain. |
| Đổi normalization | `domain.py` | transforms, checkpoint, preprocessing, tests; phải retrain. |
| Đổi augmentation | `training/data.py` | experiment args/results; không áp dụng val/test. |
| Đổi optimizer | `training/train_cnn.py` | model card và experiment report. |
| Đổi xử lý upload | `preprocessing.py` | tests, contract, có thể cần retrain/calibrate. |
| Đổi ngưỡng Unknown | `inference.py` | in-domain/OOD evaluation, tests, model card. |
| Thêm endpoint | `main.py`, `schemas.py` | api.js và API tests. |
| Đổi backend URL | `VITE_API_URL` | CORS và restart Vite. |
| Đổi upload UI | `UploadPanel.jsx` | App, CSS và frontend tests. |
| Đổi result UI | `PredictionPanel.jsx` | API schema, CSS và tests. |

## 20. Những invariant không nên phá

1. Label index giống nhau giữa dataset, checkpoint, backend và UI.
2. Training và inference dùng cùng mean/std.
3. Validation/test không augmentation.
4. Test set không dùng để chọn model trong cùng experiment.
5. Backend load model một lần lúc startup.
6. Upload bị giới hạn cả compressed bytes và decoded pixels.
7. Checkpoint là deployment artifact tin cậy, không phải user upload.
8. Ảnh upload chỉ xử lý trong memory.
9. Raw softmax không phải bảo đảm đúng hoặc in-domain.
10. Contract đổi thì checkpoint cũ không được load âm thầm.

## 21. Từ điển ML của project

| Khái niệm | Ý nghĩa trong source này |
|---|---|
| Logit | 10 số chưa normalize từ layer cuối. |
| Softmax | Đổi logits thành scores tổng 1; có thể overconfident trên ảnh lạ. |
| Cross entropy | Loss dùng logits và true class. |
| Backpropagation | `loss.backward()` tính gradient. |
| Optimizer | AdamW cập nhật weights. |
| Epoch | Một lượt qua training subset. |
| Batch | Nhóm ảnh forward/backward cùng lúc. |
| Validation | Theo dõi generalization, scheduler, early stopping và chọn state. |
| Test | Đánh giá cuối sau model selection. |
| Augmentation | Biến đổi ngẫu nhiên ảnh train. |
| Dropout | Tắt activation ngẫu nhiên khi train; tắt trong eval mode. |
| BatchNorm | Có behavior train/eval khác nhau nên `model.eval()` quan trọng. |
| Confusion matrix | Hàng là true class, cột là predicted class. |
| Macro F1 | F1 từng class rồi trung bình đều. |
| Checkpoint | Weights cộng metadata tái tạo/xác minh model. |
| Domain shift | Runtime distribution khác training distribution. |
| OOD | Out-of-distribution, ví dụ ảnh lâu đài. |

## 22. Debug theo triệu chứng

### `/health` trả `degraded`

Kiểm tra checkpoint, `MODEL_PATH`, trường `detail`, sau đó restart backend khi thay artifact.

### Frontend không kết nối API

Mở `/health`, kiểm tra `VITE_API_URL`, `CORS_ORIGINS` và restart Vite sau khi đổi env.

### Ảnh trang phục thật bị sai

Fashion-MNIST không đại diện ảnh e-commerce. Kiểm tra ảnh có một vật thể, nền đơn giản, vật đủ lớn;
xem reason/raw top-k. Giải pháp production là dataset ảnh thật và fine-tune pretrained model.

### Ảnh không liên quan vẫn được chấp nhận

Thêm ảnh vào OOD evaluation set, đo false accept/reject trên cả hai miền rồi mới chỉnh threshold.
Có thể nghiên cứu calibration, energy score, embedding distance hoặc OOD model riêng.

### Uvicorn báo WinError 10013

Kiểm tra port bị process/policy giữ. Port chuẩn hiện là `127.0.0.1:8765`; xem listener bằng
`Get-NetTCPConnection -LocalPort 8765`.

## 23. Vai trò các tài liệu

| File | Nên đọc khi nào |
|---|---|
| `README.md` | Cần command setup, train, run, test và Docker. |
| `docs/SPECIFICATION.md` | Cần behavior chính thức và acceptance criteria. |
| `docs/ARCHITECTURE.md` | Cần refactor hoặc hiểu dependency/design decisions. |
| `docs/MODEL_CARD.md` | Cần số liệu, intended use, limitation và risk. |
| `docs/PROJECT_GUIDE.md` | Cần onboarding và walkthrough toàn project. |

## 24. Checklist tự kiểm tra

Bạn đã nắm project khi trả lời được:

1. Tại sao label order nằm trong checkpoint và được backend validate?
2. Tại sao `CrossEntropyLoss` nhận logits thay vì softmax?
3. Tại sao validation có dataset object riêng?
4. Tại sao best model chọn bằng validation loss?
5. Tại sao backend dùng `run_in_threadpool()`?
6. Tại sao model có thể đoán Bag 99.9% cho ảnh lâu đài?
7. `Unknown` là neuron mới hay serving policy?
8. File nào cần sửa khi đổi preprocessing, taxonomy hoặc API URL?
9. Checkpoint chứa gì ngoài weights?
10. Request đi qua function nào từ click React đến JSON response?

Theo được call chain ở mục 10 và 12 là bạn đã có mental model đủ tốt để bắt đầu sửa hoặc mở rộng
project mà không phải đọc source một cách ngẫu nhiên.
