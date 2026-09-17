import { useRef, useState } from "react";

const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const MAX_FILE_BYTES = 5 * 1024 * 1024;

function validateFile(file) {
  if (!ACCEPTED_TYPES.has(file.type)) {
    return "Choose a PNG, JPEG, or WebP image.";
  }
  if (file.size > MAX_FILE_BYTES) {
    return "Choose an image smaller than 5 MB.";
  }
  return null;
}

export default function UploadPanel({ file, previewUrl, busy, onFile, onSubmit }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [localError, setLocalError] = useState("");

  function acceptFile(candidate) {
    if (!candidate) return;
    const validationError = validateFile(candidate);
    setLocalError(validationError || "");
    if (!validationError) onFile(candidate);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  return (
    <section className="upload-card" aria-labelledby="upload-heading">
      <div className="card-heading">
        <span className="step-number">01</span>
        <div>
          <p className="eyebrow">Input</p>
          <h2 id="upload-heading">Upload garment</h2>
        </div>
      </div>

      <div
        className={`drop-zone ${dragActive ? "drop-zone--active" : ""} ${previewUrl ? "drop-zone--filled" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <div className="preview-wrap">
            <img src={previewUrl} alt="Selected garment preview" />
            <div className="file-meta">
              <span>{file.name}</span>
              <span>{(file.size / 1024).toFixed(1)} KB</span>
            </div>
          </div>
        ) : (
          <div className="drop-copy">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 13v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5" />
            </svg>
            <strong>Drop image here</strong>
            <span>or choose a file</span>
          </div>
        )}
        <input
          ref={inputRef}
          className="visually-hidden"
          id="image-upload"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(event) => acceptFile(event.target.files?.[0])}
          disabled={busy}
        />
        <label className="secondary-button" htmlFor="image-upload">
          {previewUrl ? "Choose another image" : "Choose image"}
        </label>
      </div>

      {localError && (
        <p className="inline-error" role="alert">
          {localError}
        </p>
      )}

      <button className="primary-button" onClick={onSubmit} disabled={!file || busy} type="button">
        {busy ? (
          <>
            <span className="spinner" aria-hidden="true" />
            Classifying…
          </>
        ) : (
          <>
            Classify image
            <span aria-hidden="true">→</span>
          </>
        )}
      </button>
      <p className="upload-hint">PNG, JPEG or WebP / Maximum 5 MB</p>
    </section>
  );
}
