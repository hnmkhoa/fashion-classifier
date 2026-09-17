const configuredBaseUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8765"; // Default to local FastAPI server if not set
const API_BASE_URL = configuredBaseUrl.replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, code = "request_failed", requestId = null) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.requestId = requestId;
  }
}

export async function classifyImage(file, signal) {
  const body = new FormData();
  body.append("file", file);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
      method: "POST",
      body,
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      "The API could not be reached. Confirm that the FastAPI server is running.",
      "network_error",
    );
  }

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const apiError = payload?.error;
    throw new ApiError(
      apiError?.message || `The API returned HTTP ${response.status}.`,
      apiError?.code,
      apiError?.request_id,
    );
  }
  return payload;
}

