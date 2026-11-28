# Handwritten Digit Recognizer

A local web application for recognizing handwritten digits from 0 to 9. The MVP goal is to draw a digit, send it to a FastAPI backend, and show the predicted digit with a confidence score.

The project follows the MVP design in [PRD Handwritten Digit Recognizer (MVP).md](PRD%20Handwritten%20Digit%20Recognizer%20%28MVP%29.md).

This README matches the branch as it is today. You can load MNIST, start the API, draw with the mouse, and run the sample-evaluation script. `preprocess()` is still a placeholder, training still uses stand-in random data, and the web page does not send drawings to the API yet.

## What runs today

1. **MNIST data pipeline** (`src/data.py`) — Loads MNIST, scales pixels to float32 values in 0–1, reshapes images to `(28, 28, 1)`, and returns 54,000 training, 6,000 validation, and 10,000 test images.
2. **Baseline model** (`src/baseline.py`) — Logistic regression. `python -m src.baseline` fits stand-in random data and writes `docs/baseline_results.json`.
3. **CNN model** (`src/model.py`) — Keras CNN with two convolution layers, two pooling layers, dropout, and a 10-class softmax output.
4. **Model training** (`src/train.py`) — Fits that CNN for one epoch on stand-in random data and saves `models/digit_cnn.keras`. The file is already in the repo. Accuracy stays near 10% until training calls `load_data()`.
5. **Model evaluation** (`src/evaluate.py`) — Scores the stand-in model on random data and writes `docs/evaluation.md`, `docs/confusion_matrix.png`, and `docs/misclassified.png`. The checked-in report is 9.00% and is not MNIST accuracy.
6. **Image preprocessing** (`src/preprocess.py`) — Placeholder. The module does not define `preprocess()` yet. The API uses a stub that returns a blank `(1, 28, 28, 1)` tensor for every image.
7. **Prediction API** (`api/main.py`) — FastAPI `GET /health` and `POST /predict`. Loads `models/digit_cnn.keras` once at startup, validates the image payload, and returns 10 class probabilities. With the stub preprocess, those probabilities are not a reading of the picture you sent.
8. **Drawing canvas** (`web/index.html`, `web/canvas.js`) — A 400×400 canvas, black mouse strokes, and a Clear button. Touch drawing is not implemented. The PRD canvas size is 280×280.
9. **Frontend prediction** (`web/app.js`) — Placeholder, and `index.html` does not load it. Predict stores a PNG data URL in a local variable. The page has no digit, confidence, loading, or error display.
10. **Team handwriting QA** — 66 labeled photos in `samples/`, named `<digit>_<person>_1.jpeg`, plus `tests/real_samples_eval.py` and the checklist in `docs/test_plan.md`.

## Project Structure

```text
digit-recognizer/
├── api/main.py                 # FastAPI /health and /predict
├── docs/
│   ├── test_plan.md            # QA checklist
│   ├── evaluation.md           # Stand-in evaluation report (9.00%)
│   ├── confusion_matrix.png    # Confusion matrix from stand-in data
│   └── misclassified.png       # Misclassified grid from stand-in data
├── models/digit_cnn.keras      # CNN trained on stand-in random data
├── samples/                    # 66 team handwriting JPEGs
├── src/
│   ├── data.py                 # MNIST loading, normalize, reshape, split
│   ├── baseline.py             # Logistic regression on stand-in data
│   ├── model.py                # CNN architecture
│   ├── train.py                # Stand-in training run
│   ├── evaluate.py             # Stand-in accuracy report
│   └── preprocess.py           # Placeholder until preprocess() lands
├── tests/
│   ├── real_samples_eval.py    # Scores samples/ through the API path
│   ├── test_api.py             # Backend API tests
│   └── test_preprocess.py      # Placeholder, no tests yet
└── web/
    ├── index.html              # 400×400 canvas page
    ├── style.css               # Placeholder, not linked from the page
    ├── canvas.js               # Mouse drawing and Clear
    └── app.js                  # Placeholder, not loaded by the page
```

## Requirements

- Python 3.10, 3.11, or 3.12. `tensorflow==2.16.2` does not install on Python 3.13.
- Git
- A terminal
- A modern browser
- Internet access for cloning the repo and for the first MNIST download

Installing TensorFlow is the slow step. On a clean machine, `pip install -r requirements.txt` can take several minutes.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SamIron0/digit-recognizer.git
cd digit-recognizer
```

### 2. Create a Virtual Environment and Install Dependencies

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is not available, install Python 3.12 and try again. `py -3` is unsafe here when it points at Python 3.13.

If PowerShell blocks script activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.12` is not on your `PATH`, use the `python3` binary only after `python3 --version` prints 3.10, 3.11, or 3.12.

## Load and Prepare the Data

From the repository root:

```bash
python -c "from src.data import load_data; (xt,_),(xv,_),(xs,_)=load_data(); print(xt.shape, xv.shape, xs.shape)"
```

The result should contain:

- Training images: `(54000, 28, 28, 1)`
- Validation images: `(6000, 28, 28, 1)`
- Test images: `(10000, 28, 28, 1)`

The first run downloads MNIST.

## Train the Model

`models/digit_cnn.keras` is already in the repo. To replace it with a fresh stand-in run:

```bash
python -m src.train
```

The script calls `fake_data()`, trains for one epoch, and overwrites `models/digit_cnn.keras`. Real MNIST training is required for the 98% MVP target.

`python -m src.model` saves an untrained CNN to that same path. Run `python -m src.train` afterward if you used that command.

## Evaluate the Model

```bash
python -m src.evaluate
```

This needs `models/digit_cnn.keras` and regenerates:

- `docs/confusion_matrix.png`
- `docs/misclassified.png`
- `docs/evaluation.md`

The script evaluates stand-in random images. The report it writes is not the model's MNIST accuracy. The copy currently in the repo says 9.00%.

## Run the API

From the repository root:

```bash
uvicorn api.main:app --reload
```

The API listens at `http://localhost:8000`.

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Health Check

```bash
curl http://localhost:8000/health
```

With the checked-in model file and the current placeholder preprocess module, the response is:

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocess": "stub"
}
```

`model_loaded` is true because `models/digit_cnn.keras` loaded. That file was trained on random stand-in data. `preprocess` stays `"stub"` until `src/preprocess.py` defines `preprocess()`. If the model file is missing or TensorFlow cannot load it, `model_loaded` is false and the API uses a dummy model that always returns digit 7 with confidence 0.973.

### Prediction

From Windows PowerShell, with a PNG named `image.png` in the current directory:

```powershell
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes(".\image.png"))
$body = @{ image = $base64 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

The body may be raw base64 or a canvas data URL (`data:image/png;base64,...`).

A successful response has this shape:

```json
{
  "digit": 7,
  "confidence": 0.973,
  "probabilities": [0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.973, 0.003, 0.003]
}
```

Those exact numbers are the dummy model, used only when `models/digit_cnn.keras` does not load. A loaded stand-in CNN returns the same three fields with different numbers. While `preprocess` is `"stub"`, every image is replaced with a blank tensor before prediction, so the digit is not a reading of the PNG.

These inputs return `{"error": "..."}` with status 400:

- an empty string
- invalid base64
- bytes that are not an image

A missing `image` field, a non-string `image`, or invalid JSON returns status 422.

A blank or all-white image currently returns 200. The preprocess stub does not reject it. When `preprocess()` raises `ValueError`, the API returns 400 and uses that message as `error`.

## Run the Web Page

The page is a drawing canvas. Predict does not call the API, so the API does not need to be running for this step.

From the repository root:

```bash
cd web
python -m http.server 5500
```

Open `http://localhost:5500`.

- Draw with the mouse. Strokes are black. The canvas is 400×400.
- **Clear** erases the canvas.
- **Predict** stores a PNG data URL in a local variable and leaves the page unchanged.

`web/style.css` is a placeholder and the page does not link it. `web/app.js` is a placeholder and the page does not load it. Touch drawing is not implemented.

The API already allows cross-origin requests (`allow_origins=["*"]`) for when the page starts posting drawings to `http://localhost:8000/predict`.

## Run Tests and the Sample Script

API tests, from the repository root:

```bash
python -m pytest tests/test_api.py -v
```

`tests/test_api.py` checks `/health`, valid PNGs, canvas data URLs, transparent PNGs, CORS, malformed input, and the 400 response used when preprocess raises `ValueError` for a blank canvas. That blank-canvas case is simulated inside the test. The live stub returns 200 for a blank image.

`tests/test_preprocess.py` is still a placeholder and contains no tests.

Sample script, also from the repository root:

```bash
python -m tests.real_samples_eval
```

The script lists image files directly under `samples/`, reads the label from the first character of each filename, and runs `decode_image`, `load_preprocess`, and `load_model` from `api/main.py`.

While preprocess is the stub, every photo becomes the same blank tensor, so the table is a wiring check. If the Keras file fails to load, the dummy model predicts 7 for every file. Seven of the 66 photos are labeled 7, so that run prints 10.61% (7/66). Use an accuracy of at least 80% as the MVP sample gate only after `preprocess()` exists and the CNN has been trained on MNIST.

## Real-Handwriting Samples

The 66 photos are files in `samples/`, one digit per file:

```text
samples/<digit>_<person>_1.jpeg
```

Example: `samples/7_king_1.jpeg` is a photo of the digit 7.

Every digit from 0 through 9 is present. Four person/digit pairs are absent: `0_chisom`, `4_declan`, `6_declan`, and `8_declan`.

The QA checklist is [docs/test_plan.md](docs/test_plan.md).

PRD release targets, still ahead of this branch:

- at least 98% accuracy on the MNIST test set
- at least 80% accuracy on these 66 photos

## Troubleshooting

- **`python` is not recognized (Windows):** Use `py -3.12` instead of `python`.
- **TensorFlow or another package fails to install:** Use Python 3.10, 3.11, or 3.12, and activate the virtual environment before `pip install`. Python 3.13 cannot install `tensorflow==2.16.2`.
- **`/health` reports `model_loaded: false`:** The API is on the dummy model and will return digit 7. Put `models/digit_cnn.keras` back, or run `python -m src.train` from the repository root.
- **`/health` reports `preprocess: "stub"`:** Expected on this branch. `src/preprocess.py` does not define `preprocess()` yet.
- **MNIST download fails:** Retry with internet access. After one successful download, Keras reuses the cache.
- **The web page shows no prediction:** Expected on this branch. `web/app.js` is not loaded. Confirm the API itself with `curl http://localhost:8000/health`.
- **`/predict` returns 400:** The body was empty, not valid base64, or not an image. A blank drawing still returns 200 until `preprocess()` rejects it.

## API Contract

| Item | Contract |
| --- | --- |
| Health endpoint | `GET /health` |
| Prediction endpoint | `POST /predict` |
| Request body | `{"image": "<base64 PNG string>"}` |
| Success response | `{"digit": 7, "confidence": 0.97, "probabilities": [10 float values]}` |
| Error response | `{"error": "message"}` |
| Model input | `(1, 28, 28, 1)`, float32, values from 0 to 1 |
