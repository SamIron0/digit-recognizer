# QA Test Plan — Handwritten Digit Recognizer

## 1. Functional Requirements (F1–F5)
- [ ] **F1 — Drawing & Image Input:** User can draw a digit (0–9) smoothly on the 280x280 canvas using mouse or touch input.
- [ ] **F2 — Image Preprocessing:** `preprocess()` converts input images into `(1, 28, 28, 1)` float32 arrays (grayscale, inverted/normalized to `[0, 1]`, cropped, and centered).
- [ ] **F3 — Model Inference & API:** `POST /predict` on `uvicorn api.main:app` accepts a base64 PNG and returns a predicted `digit` (0–9), `confidence`, and 10 `probabilities`.
- [ ] **F4 — Frontend Prediction & Reset Flow:** Web UI displays the predicted digit, confidence score, loading/error states, and resets cleanly when **Clear** is clicked.
- [ ] **F5 — Model Accuracy & QA Evaluation:** Trained CNN (`models/digit_cnn.keras`) achieves ≥ 98% on MNIST test set and `tests/real_samples_eval.py` reports ≥ 80% on the 50+ team handwriting samples.

## 2. Edge Case Checklist
- [ ] **Blank Canvas:** Submitting an empty canvas without drawing returns a clear validation error (`4xx`) or warning without crashing the server.
- [ ] **Random Scribble:** Drawing non-digit scribbles does not crash the backend (`500` error) and surfaces low confidence / unsure feedback.
- [ ] **Tiny Digit in a Corner:** Drawing a very small digit near the top-left or bottom-right corner is properly cropped/centered by `preprocess()` and recognized.
- [ ] **Large / Off-Center Digit:** Drawing a digit touching the edge of the canvas still preprocesses cleanly without clipping crucial strokes.
- [ ] **Drawing on a Phone:** Touch drawing on a mobile viewport works without scrolling the page and submits accurately.

## 3. Non-Functional Requirement (N3)
- [ ] **Fresh Clone in Under 10 Minutes:** Verified on a clean machine/folder using only `README.md` (`git clone` → `.venv` → `pip install -r requirements.txt` → `python -m src.train` → `uvicorn api.main:app` → open `web/index.html`).
