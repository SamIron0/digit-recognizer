# Team Playbook: who does what, starting now

Companion to the PRD. Every work package gets a **real input** (what it eventually consumes) and a **stand-in input** (what you use on day 1), so nobody waits on anybody.

Every section follows the same shape: **your file(s)**, **hour one**, **the full job**, and **you're done when**.

## Day-1 shared kit (everyone uses these)

Four stand-ins unblock almost everyone. The repo skeleton (folders and placeholder files) already exists; each owner below commits the piece named after them.

- `fake_data(n=100)` returns random arrays with the exact real shapes (owner: #2).
- **A placeholder model**: an untrained `build_model()` saved as `models/digit_cnn.keras`. Its predictions are garbage but its input and output shapes are correct (owner: #4).
- **Sample images** in `samples/`: a few hand-drawn digits as 280×280 PNGs (owner: #11, with help from anyone).
- **A mock API response**: `{"digit": 7, "confidence": 0.97, "probabilities": [...]}` (defined in the PRD, used by #10).

---

## #2 — Data pipeline → `src/data.py`

**Hour one:** Open `src/data.py` and write `fake_data(n=100)`: return `(x_train, y_train), (x_val, y_val), (x_test, y_test)` where each `x` is `np.random.rand(n, 28, 28, 1).astype("float32")` and each `y` is `np.random.randint(0, 10, n)`. Push this as PR #1 of the project — four other people are waiting on it.

**The full job:** Add `load_data()` with the same return signature, backed by `keras.datasets.mnist.load_data()`:

1. Reshape `x` from `(N, 28, 28)` to `(N, 28, 28, 1)` and convert to float32 divided by 255.
2. Split the 60,000 training images 90/10 into train/validation (use a fixed `random_state` or slice so it's reproducible).
3. Add a docstring stating the exact shapes and the 54,000 / 6,000 / 10,000 split.

**Done when:** `python -c "from src.data import load_data; ..."` prints the three shapes correctly, dtypes are float32, and `x.max() <= 1.0`. Your output feeds #3, #5, and #6.

## #3 — Baseline model → `src/baseline.py` + `docs/baseline.md`

**Hour one:** Import `fake_data` from `src.data`, flatten images to `(N, 784)`, and fit `sklearn.linear_model.LogisticRegression` on it. Accuracy will be ~10% on random data — that's fine, you're proving the plumbing.

**The full job:** A `train_baseline()` function that loads real data via `load_data()`, trains logistic regression (`max_iter=1000`, consider `solver="saga"` for speed), and returns test accuracy. Write `docs/baseline.md` (new file): one paragraph on the setup, the accuracy number (expect ~92%), and one sentence on why we expect the CNN to beat it.

**Done when:** `python -m src.baseline` prints a test accuracy near 92% and `docs/baseline.md` is merged. Nobody depends on you — you're the comparison point, so you can polish independently.

## #4 — CNN definition → `src/model.py`

**Hour one:** You need literally nothing from anyone. Write `build_model()` returning a compiled Keras model:

```python
Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1))
MaxPooling2D()
Conv2D(64, 3, activation="relu")
MaxPooling2D()
Flatten()
Dropout(0.5)
Dense(10, activation="softmax")
```

Compile with `adam`, `sparse_categorical_crossentropy`, `metrics=["accuracy"]`.

**Then immediately:** run `build_model().save("models/digit_cnn.keras")` and commit that file. This untrained model is the placeholder #6 and #8 build against — shipping it is as important as the architecture itself.

**Done when:** `model.predict(np.zeros((1, 28, 28, 1)))` returns shape `(1, 10)` summing to ~1.0, and `models/digit_cnn.keras` exists in the repo. Put that predict check at the bottom of the file under `if __name__ == "__main__":` so reviewers can verify in one command.

## #5 — Training script → `src/train.py`

**Hour one:** Write the script against stand-ins: `from src.data import fake_data` and `from src.model import build_model` (coordinate with #4 on timing; if their PR isn't merged yet, write your own 5-line throwaway model inline). Run `model.fit` for 1 epoch, save to `models/digit_cnn.keras`. The goal of hour one is only "the script runs end to end and produces a file."

**The full job:**

1. Switch to `load_data()`.
2. Train with `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`, around 15 max epochs.
3. Save three artifacts: `models/digit_cnn.keras`, a training-curves PNG in `docs/` (accuracy and loss vs. epoch, matplotlib), and `models/metrics.json` containing `{"val_accuracy": ..., "test_accuracy": ...}`.

**Done when:** `models/metrics.json` shows test accuracy ≥ 0.98 (the PRD's N2 requirement). When you merge, post in the team channel — #6 and #8 swap in your model the moment it lands.

## #6 — Evaluation → `src/evaluate.py`

**Hour one:** Load `models/digit_cnn.keras` (the untrained placeholder #4 committed) and `fake_data()`. Build the whole report pipeline now: predictions, confusion matrix, misclassified grid. The charts will look like noise; that's expected.

**The full job:** A script that loads the model and real test data and writes three files:

- `docs/confusion_matrix.png` — use `sklearn.metrics.confusion_matrix` + matplotlib.
- `docs/misclassified.png` — a 4×4 grid of 16 wrong predictions, each titled "true 4 / pred 9" style.
- `docs/evaluation.md` — test accuracy plus 3–4 sentences on which digit pairs confuse the model (classically 4/9 and 3/5).

**Done when:** Running `python -m src.evaluate` after #5's real model lands regenerates all three files with real numbers and you've re-merged. Your structural work is done days before that; the final run is a ten-minute swap.

## #7 — Preprocessing → `src/preprocess.py` + `tests/test_preprocess.py`

You own the project's highest-risk code — the PRD's top risk is "works on MNIST, fails on canvas drawings," and your module is the mitigation. You need no model and no teammates.

**Hour one:** Write the tests first in `tests/test_preprocess.py`:

- output shape `(1, 28, 28, 1)`, dtype float32, values in 0–1;
- a blank image raises a clean `ValueError` (never crashes);
- a tiny digit and a near-full-frame digit both end up ~20 px tall;
- center of mass of the output lands near pixel (14, 14).

**The full job,** implementing `preprocess(image: np.ndarray) -> np.ndarray` in order:

1. If the mean pixel is bright, invert — MNIST is white-on-black, the canvas is black-on-white.
2. Threshold and crop to the digit's bounding box (reject if empty → `ValueError`).
3. Scale the longer side to 20 px, preserving aspect ratio (Pillow is in `requirements.txt`).
4. Paste into a 28×28 black canvas, shifted so the center of mass sits at (14, 14) (scipy-free version: compute the weighted mean of pixel coordinates yourself).
5. Normalize to 0–1 float32, reshape to `(1, 28, 28, 1)`.

**Done when:** `pytest tests/test_preprocess.py` passes, and visually: save a before/after image of one file from `samples/` in your PR description so reviewers can see a messy drawing becoming a clean MNIST-style digit.

## #8 — Backend API → `api/main.py` + `tests/test_api.py`

**Hour one:** Build the full FastAPI app with every dependency stubbed: load the placeholder `models/digit_cnn.keras` (or return fixed probabilities if even that isn't merged yet), use a `preprocess` stub that returns `np.zeros((1, 28, 28, 1))`, and make a test base64 string from any file in `samples/` (`base64.b64encode(open(path, "rb").read())`).

**The full job:**

- `GET /health` → `{"status": "ok"}`.
- `POST /predict`: decode base64 → PNG via Pillow → grayscale NumPy array → `preprocess()` → `model.predict()` → `{"digit": int, "confidence": float, "probabilities": [10 floats]}`. Cast NumPy types to Python types or FastAPI's JSON encoder will choke.
- Errors: malformed base64, non-image data, and blank canvas (#7's `ValueError`) each return `{"error": "..."}` with a 400 status — never a 500.
- Load the model **once** at startup (module level or lifespan handler), not per request.
- **Enable CORS** with `CORSMiddleware` allowing all origins; without it, #10's browser calls fail and will cost an evening of confused debugging.

**Done when:** `tests/test_api.py` (use FastAPI's `TestClient`, `httpx` is already pinned) passes with tests for health, a valid predict, and bad input; and `uvicorn api.main:app` + a `curl` returns the contract response. Swapping in #7's real preprocess and #5's real model is one import and zero logic changes.

## #9 — Frontend canvas → `web/index.html`, `web/style.css`, `web/canvas.js`

**Hour one:** In `index.html`, a 280×280 `<canvas>`, a Clear button, a Predict button (wired by #10, not you), and a result area. In `canvas.js`, mouse drawing: `mousedown`/`mousemove`/`mouseup` with `lineWidth` around 18, `lineCap: "round"`, black stroke, and the canvas pre-filled white (important: an unfilled canvas exports as transparent, which breaks preprocessing).

**The full job:**

1. Touch support: `touchstart`/`touchmove`/`touchend`, with `preventDefault()` so the page doesn't scroll while drawing, and coordinates adjusted by `getBoundingClientRect()`.
2. Clear button refills the canvas white.
3. Export function, the piece #10 consumes:
   ```js
    function getCanvasPNG() { return canvas.toDataURL("image/png"); }
   ```

    Agree with #10 whether the `data:image/png;base64,` prefix is stripped on your side or theirs — settle it in your PR description.

**Done when:** You can draw and clear with mouse and with a phone/trackpad touch, and a debug `<img>` next to the canvas (set its `src` to `getCanvasPNG()`) shows exactly what you drew — black ink, white background, 280×280. Test with `python -m http.server` from `web/`.

## #10 — Frontend integration → `web/app.js`

**Hour one:** You don't need #8 or #9. At the top of `app.js`:

```js
const USE_MOCK = true;
const API_URL = "http://localhost:8000";
```

When `USE_MOCK` is true, your predict function resolves (after a fake 500 ms delay) to `{"digit": 7, "confidence": 0.97, "probabilities": [...]}`. Use a hardcoded base64 image string until #9's `getCanvasPNG()` exists. Build the entire UI flow against this.

**The full job,** on Predict click:

1. Loading state — disable the button, show "Thinking…".
2. `fetch(API_URL + "/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({image: getCanvasPNG()})})`.
3. Success: "Prediction: 7 (confidence 97%)". If confidence < 0.5, show an "I'm not sure — try redrawing?" state instead.
4. Failure (network error, non-200, `{"error": ...}` body): a readable message, never a frozen spinner.

**Done when:** All four states (loading, result, unsure, error) are demonstrable with the mock, and then flipping `USE_MOCK = false` against #8's running server gives a working end-to-end demo. That flip is the project's integration moment — pair with #8 for 30 minutes when you do it.

## #11 — QA, samples, docs → `samples/`, `tests/real_samples_eval.py`, `README.md`

**Hour one:** Message every teammate: "Write digits 0–9 on white paper with a dark pen, one digit per photo, and send them to me." Crop each to roughly square and save as `samples/<label>_<person>_<n>.png` (e.g. `samples/7_sam_1.png`) — the filename **is** the ground-truth label, which is what makes your eval script possible. With 11 people at 10 digits each you get 110 samples, well above the 50 target.

**The full job:**

1. `tests/real_samples_eval.py` (new file): loop over `samples/`, parse the label from each filename, run `preprocess()` + the model, and print per-digit and overall accuracy. Build and run it against the placeholder model now (~10% accuracy expected) so it's one command when the real model arrives.
2. A test plan in `docs/`: a checklist covering the F1–F5 requirements plus edge cases (blank canvas, a scribble, a tiny digit in a corner, drawing on a phone).
3. Rewrite `README.md`: setup (clone, venv, `pip install -r requirements.txt`), train (`python -m src.train`), serve (`uvicorn api.main:app`), open frontend — and verify it on a machine or fresh folder that isn't yours, because "fresh clone in under 10 minutes" is requirement N3.

**Done when:** `real_samples_eval.py` reports ≥ 80% on 50+ samples with the real model (the Definition of Done's hardest line item), and someone else has reproduced the project from your README alone.

---

## The handoff order

The stand-ins are replaced as real pieces land: #2 replaces `fake_data`, #5 replaces the placeholder model, #7 replaces the preprocessing stub, #8 replaces the mock API, and #9 replaces the hardcoded image. Because everyone coded against the same shapes, each swap should be a one-line change.

---
