import { useEffect, useRef, useState } from "react";

import { classifyImage } from "./api";
import PredictionPanel from "./components/PredictionPanel";
import UploadPanel from "./components/UploadPanel";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const activeRequest = useRef(null);

  useEffect(() => {
    return () => activeRequest.current?.abort();
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function selectFile(nextFile) {
    activeRequest.current?.abort();
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setError(null);
    setBusy(false);
  }

  async function submit() {
    if (!file || busy) return;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setBusy(true);
    setResult(null);
    setError(null);
    try {
      setResult(await classifyImage(file, controller.signal));
    } catch (requestError) {
      if (requestError.name !== "AbortError") setError(requestError);
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setBusy(false);
      }
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="ThreadSense home">
          <span className="brand-mark" aria-hidden="true">TS</span>
          <span>ThreadSense</span>
        </a>
        <span className="status-pill"><i /> Fashion-MNIST / 10 classes</span>
      </header>

      <main id="top">
        <section className="page-intro" aria-labelledby="page-title">
          <p>Image classification</p>
          <h1 id="page-title">Garment classifier</h1>
        </section>

        <div className="workspace-grid">
          <UploadPanel
            file={file}
            previewUrl={previewUrl}
            busy={busy}
            onFile={selectFile}
            onSubmit={submit}
          />
          <PredictionPanel result={result} error={error} />
        </div>

      </main>
    </div>
  );
}
