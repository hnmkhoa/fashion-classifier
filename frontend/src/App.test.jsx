import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { classifyImage } from "./api";

vi.mock("./api", () => ({ classifyImage: vi.fn() }));

describe("App", () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.clearAllMocks();
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("uploads an accepted image and renders the ranked prediction", async () => {
    classifyImage.mockResolvedValue({
      class_id: 7,
      class_name: "Sneaker",
      confidence: 0.91,
      is_unknown: false,
      rejection_reason: null,
      top_predictions: [
        { class_id: 7, class_name: "Sneaker", probability: 0.91 },
        { class_id: 9, class_name: "Ankle boot", probability: 0.06 },
        { class_id: 5, class_name: "Sandal", probability: 0.02 },
      ],
      model_version: "test-model",
      inference_ms: 4.2,
    });
    const user = userEvent.setup();
    render(<App />);
    const image = new File(["image"], "shoe.png", { type: "image/png" });

    await user.upload(screen.getByLabelText("Choose image"), image);
    await user.click(screen.getByRole("button", { name: /classify image/i }));

    await waitFor(() =>
      expect(screen.getByText("91.0% model probability")).toBeInTheDocument()
    );
    expect(screen.getAllByText("Sneaker").length).toBeGreaterThan(0);
    expect(classifyImage).toHaveBeenCalledWith(image, expect.any(AbortSignal));
  });

  it("renders an unknown result without presenting a misleading confidence", async () => {
    classifyImage.mockResolvedValue({
      class_id: null,
      class_name: "Unknown",
      confidence: null,
      is_unknown: true,
      rejection_reason: "The image background is too complex for this Fashion-MNIST model.",
      top_predictions: [
        { class_id: 8, class_name: "Bag", probability: 1.0 },
        { class_id: 0, class_name: "T-shirt/top", probability: 0.0 },
        { class_id: 5, class_name: "Sandal", probability: 0.0 },
      ],
      model_version: "test-model",
      inference_ms: 4.2,
    });
    const user = userEvent.setup();
    render(<App />);
    const image = new File(["image"], "scene.png", { type: "image/png" });

    await user.upload(screen.getByLabelText("Choose image"), image);
    await user.click(screen.getByRole("button", { name: /classify image/i }));

    expect(await screen.findByText("Unknown")).toBeInTheDocument();
    expect(screen.getByText(/background is too complex/i)).toBeInTheDocument();
    expect(screen.queryByText(/100.0% model probability/i)).not.toBeInTheDocument();
    expect(screen.getByText("Raw model guesses")).toBeInTheDocument();
  });

  it("rejects unsupported files before an API call", () => {
    render(<App />);
    const textFile = new File(["not an image"], "notes.txt", { type: "text/plain" });

    fireEvent.change(screen.getByLabelText("Choose image"), { target: { files: [textFile] } });

    expect(screen.getByRole("alert")).toHaveTextContent("Choose a PNG, JPEG, or WebP image.");
    expect(classifyImage).not.toHaveBeenCalled();
  });
});
