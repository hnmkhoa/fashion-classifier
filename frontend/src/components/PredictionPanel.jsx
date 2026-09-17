function formatPercentage(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function PredictionPanel({ result, error }) {
  return (
    <section className="result-card" aria-labelledby="result-heading" aria-live="polite">
      <div className="card-heading">
        <span className="step-number">02</span>
        <div>
          <p className="eyebrow">Model output</p>
          <h2 id="result-heading">Prediction</h2>
        </div>
      </div>

      {error ? (
        <div className="error-state" role="alert">
          <span className="error-mark" aria-hidden="true">!</span>
          <div>
            <strong>Classification failed</strong>
            <p>{error.message}</p>
            {error.requestId && <small>Request ID: {error.requestId}</small>}
          </div>
        </div>
      ) : result ? (
        <div className="prediction-content">
          <div className="winner">
            <span className="winner-label">
              {result.is_unknown ? "Classification status" : "Most likely class"}
            </span>
            <strong>{result.class_name}</strong>
            {result.is_unknown ? (
              <span className="rejection-reason">{result.rejection_reason}</span>
            ) : (
              <span className="confidence">
                {formatPercentage(result.confidence)} model probability
              </span>
            )}
          </div>

          <div className="rankings">
            <div className="rankings-heading">
              <h3>{result.is_unknown ? "Raw model guesses" : "Top predictions"}</h3>
              <span>Softmax score</span>
            </div>
            <ol>
              {result.top_predictions.map((prediction) => (
                <li key={prediction.class_id}>
                  <div className="rank-label">
                    <span>{prediction.class_name}</span>
                    <strong>{formatPercentage(prediction.probability)}</strong>
                  </div>
                  <div className="probability-track" aria-hidden="true">
                    <span style={{ width: formatPercentage(prediction.probability) }} />
                  </div>
                </li>
              ))}
            </ol>
          </div>

          <div className="model-meta">
            <span>Model {result.model_version}</span>
            <span>{result.inference_ms.toFixed(1)} ms inference</span>
          </div>
        </div>
      ) : (
        <div className="empty-state">
          <div className="scan-illustration" aria-hidden="true">
            <span />
          </div>
          <strong>Awaiting image</strong>
        </div>
      )}
    </section>
  );
}
