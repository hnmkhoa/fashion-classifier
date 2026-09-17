# Architecture and Refactoring Guide

## 1. System context

ThreadSense has three independently executable concerns:

1. Offline experiment code downloads Fashion-MNIST, trains models, and writes versioned artifacts.
2. The FastAPI process validates one approved artifact and performs stateless image inference.
3. The React client sends one selected image and presents the API response.

There is no database. Uploaded image bytes live only for the request lifetime.

## 2. Dependency direction

```text
training scripts ──────> backend.app.domain
       │                backend.app.model
       │
       └──> data, metrics, generated artifacts

backend.app.main ──────> config, errors, inference, schemas
backend.app.inference ─> domain, model, preprocessing

frontend App ──────────> frontend API adapter
frontend components ───> plain props only
```

Rules:

- `backend/app/domain.py` has no framework or training dependency.
- `backend/app/model.py` knows PyTorch but not HTTP, files, or Fashion-MNIST download behavior.
- `backend/app/preprocessing.py` knows image bytes and tensors but not FastAPI.
- `backend/app/inference.py` knows checkpoint format and model execution but not request objects.
- `backend/app/main.py` wires dependencies and translates domain failures into HTTP responses.
- `training/` can depend on shared backend domain/model code; dependency direction never reverses.
- React components do not construct endpoint URLs or interpret error payload structure.

## 3. Runtime sequence

### API startup

```text
Resolve environment settings
  -> locate checkpoint
  -> load weights-only checkpoint
  -> validate metadata contract
  -> construct FashionCNN
  -> load state dictionary strictly
  -> set evaluation mode
  -> expose ready health state
```

If any checkpoint step fails, startup continues in degraded state. This allows `/health` and API
documentation to explain the issue. Prediction returns HTTP 503 until a compatible artifact is
available and the process restarts.

### Prediction request

```text
Assign request ID
  -> check model readiness
  -> check declared media type
  -> read at most limit + 1 bytes
  -> decode and check pixel count
  -> orient, grayscale, estimate background, extract foreground, resize, pad, normalize
  -> execute inference without gradients
  -> softmax and top-k
  -> apply domain, probability, and top-two-margin rejection rules
  -> serialize public schema
```

CPU-bound decoding and inference run in FastAPI's thread pool so the event loop can continue
serving lightweight requests.

## 4. Training lifecycle

The official training split is deterministically divided per class. Separate dataset instances are
used so augmentation cannot leak into validation. Validation loss controls early stopping and
checkpoint selection. The official test split is evaluated only after the best state is restored.

The saved checkpoint is more than weights: it carries the semantic metadata required to prove it
matches the API. The JSON result is the detailed experiment record; the checkpoint contains the
minimum deployment-relevant subset.

## 5. Trust boundaries

- Browser input is untrusted. Validate type, compressed size, decoded size, and decodability.
- Checkpoints are trusted deployment inputs, never user uploads. The weights-only loader reduces
  Python object-deserialization risk but does not replace artifact provenance controls.
- Environment configuration is operator-controlled. Invalid positive-integer limits fail at startup.
- CORS is a browser policy, not authentication. The API is intentionally unauthenticated for the
  local educational scope.

## 6. Design decisions

### Compact CNN instead of ResNet for the primary model

Fashion-MNIST is one-channel and 28 by 28. A compact CNN preserves enough spatial hierarchy while
keeping training and serving approachable. Resizing the dataset for an ImageNet-pretrained ResNet
adds computation without recovering lost detail. ResNet remains a sensible extension when the
training domain changes to higher-resolution product photography.

### Checkpoint fail-fast validation

Incorrect label order or normalization can produce syntactically valid yet semantically wrong
responses. The serving layer therefore rejects metadata drift rather than loading weights loosely.

### API startup without a model

An absent artifact is a common first-run condition because generated binary weights are not stored
in source control. A degraded health endpoint makes the condition diagnosable while prediction
fails explicitly.

### Conservative reject option

The service returns `Unknown` for obvious preprocessing-domain mismatch, a maximum softmax
probability below `0.55`, or a top-one/top-two margin below `0.15`. It returns `null` for the primary
class ID and confidence while retaining raw top-k scores for diagnostics. These heuristics reduce
misleading forced choices but are not calibrated or guaranteed out-of-distribution detection.

## 7. Common changes

### Add another PyTorch architecture

1. Give it a new immutable architecture identifier.
2. Add a constructor registry rather than branching throughout HTTP code.
3. Update the checkpoint producer and compatibility tests.
4. Preserve the public API unless prediction semantics change.

### Change preprocessing

1. Treat it as a model-contract change.
2. Create a new schema or preprocessing version.
3. Retrain the model.
4. Add parity tests using fixed example tensors.
5. Do not silently accept old checkpoints.

### Change the taxonomy

1. Version the label schema and API.
2. Retrain every model.
3. Update dataset mapping, response schema, UI, tests, and documentation together.
4. Never reorder labels only for display purposes; localization belongs in a separate mapping.

### Add ONNX or another inference runtime

Preserve a small prediction-service interface that accepts image bytes and returns the existing
plain prediction record. Keep runtime selection out of route handlers.

### Add a database

Define retention, privacy, and deletion requirements first. Store metadata by default, not original
image bytes. Persistence must not make prediction availability dependent on a history feature.

## 8. Review checklist

- Does the canonical ordered label mapping remain identical everywhere?
- Does training import, rather than copy, normalization constants?
- Are validation and test transforms augmentation-free?
- Is test data excluded from model and hyperparameter selection?
- Is the checkpoint loaded once rather than per request?
- Do invalid uploads still have bounded memory and decoded dimensions?
- Are errors still structured and correlated by request ID?
- Does the frontend API adapter remain the only transport-aware client module?
- Are new claims about model quality backed by generated held-out metrics?
- Are domain limitations still visible to end users?
