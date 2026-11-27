# Handwritten Digit Recognizer

A local web application for recognizing handwritten digits from 0 to 9. Users draw a digit on a canvas, send it to a FastAPI backend, and receive a predicted digit with a confidence score.

The project follows the MVP design in [PRD Handwritten Digit Recognizer (MVP).md](PRD%20Handwritten%20Digit%20Recognizer%20%28MVP%29.md).

## Features

1. **MNIST data pipeline** - Loads MNIST, converts pixels to float32 values in the 0-1 range, reshapes images to `(28, 28, 1)`, and creates 54,000 training, 6,000 validation, and 10,000 test examples.
2. **Baseline model** - Uses logistic regression to provide a baseline accuracy for comparison with the CNN.
3. **CNN model** - Defines a Keras CNN with two convolution layers, two pooling layers, dropout, and a 10-class softmax output.
4. **Model training** - Fits the CNN and saves `models/digit_cnn.keras`. The training package also supports stand-in random data for verifying the training flow.
5. **Model evaluation** - Calculates test accuracy and generates a confusion matrix and a grid of misclassified digits under `docs/`.
6. **Image preprocessing** - Converts an uploaded or drawn image into the MNIST format: grayscale, normalized, resized, cropped, and centered as `(1, 28, 28, 1)`.
7. **Prediction API** - Exposes FastAPI `/health` and `/predict` endpoints, validates image input, loads the model once at startup, and returns 10 class probabilities.
8. **Drawing canvas** - Provides a 280-pixel canvas with mouse and touch drawing, black ink, a white background, and a Clear control.
9. **Frontend prediction flow** - Exports the canvas as a PNG, calls `/predict`, and displays the predicted digit, confidence, loading status, unsure state, and errors.
10. **Team handwriting QA** - Includes 50+ team-collected handwriting samples under `samples/` and an automated accuracy evaluation script (`tests/real_samples_eval.py`) for real-world testing beyond MNIST.

## Project Structure

```text
digit-recognizer/
├── api/main.py                 # FastAPI health and prediction endpoints
├── docs/
│   ├── test_plan.md            # Functional (F1-F5) and edge-case QA checklist
│   ├── evaluation.md           # Model evaluation report
│   ├── confusion_matrix.png    # Confusion matrix chart
│   └── misclassified.png       # Misclassified digits grid
├── models/digit_cnn.keras      # Trained or stand-in CNN model
├── samples/                    # Team-collected handwriting images (50+ samples)
├── src/
│   ├── data.py                 # MNIST loading and splitting
│   ├── baseline.py             # Logistic regression baseline
│   ├── model.py                # CNN architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Accuracy and error-analysis report
│   └── preprocess.py           # Canvas image preprocessing
├── tests/
│   ├── real_samples_eval.py    # Real-world handwriting accuracy evaluation
│   ├── test_api.py             # Backend API tests
│   └── test_preprocess.py      # Preprocessing tests
└── web/
    ├── index.html              # Page layout
    ├── style.css               # Page styles
    ├── canvas.js               # Mouse/touch drawing
    └── app.js                  # API integration and result display
```

## Requirements

- Python 3.10 or newer
- Git
- A terminal
- A modern browser
- Internet access for cloning and the first MNIST download

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd digit-recognizer
```

### 2. Create a Virtual Environment & Install Dependencies

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** If `python` is not recognized on Windows, replace `python` with `py -3` (for example: `py -3 -m venv .venv`).

If PowerShell blocks script activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Load and Prepare the Data

```bash
python -c "from src.data import load_data; (xt,_),(xv,_),(xs,_)=load_data(); print(xt.shape, xv.shape, xs.shape)"
```

The result should contain:

- Training images: `(54000, 28, 28, 1)`
- Validation images: `(6000, 28, 28, 1)`
- Test images: `(10000, 28, 28, 1)`

## Train the Model

Create the CNN and a first model file:

```bash
python -m src.model
```

Run the training script:

```bash
python -m src.train
```

The script writes `models/digit_cnn.keras`. It currently uses stand-in random data unless the team upgrades `src/train.py` to call `load_data()`; real MNIST training is required for the project's 98% MVP accuracy target.

## Evaluate the Model

```bash
python -m src.evaluate
```

This regenerates:

- `docs/confusion_matrix.png`
- `docs/misclassified.png`
- `docs/evaluation.md`

The existing evaluation report is based on placeholder data and must not be interpreted as the real model's MNIST accuracy.

## Run the API

```bash
uvicorn api.main:app --reload
```

The API is available at `http://localhost:8000`.

Open the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Health Check Example

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocess": "real"
}
```

`model_loaded` is `false` and `preprocess` is `"stub"` until the corresponding team deliverables are available.

### Prediction Example

From Windows PowerShell:

```powershell
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes(".\image.png"))
$body = @{ image = $base64 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

Save the PNG as `image.png`, or send the canvas `data:image/png;base64,...` value directly from the browser.

The endpoint returns:

```json
{
  "digit": 7,
  "confidence": 0.973,
  "probabilities": [0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.973, 0.003, 0.003]
}
```

Invalid base64, non-image data, empty input, and blank canvases produce a JSON error with a 4xx status:

```json
{
  "error": "Image is empty"
}
```

## Run the Web Page

Keep the API running in one terminal. In a second terminal, from the repository root:

```bash
cd web
python -m http.server 5500
```

Open `http://localhost:5500` in a browser.

1. Draw one digit with the mouse or a touch device.
2. Select **Predict**.
3. Read the digit and confidence percentage.
4. Select **Clear** to try another digit.

The frontend accepts the API URL as `http://localhost:8000`. If a browser blocks a local request, verify that the API is running and that CORS is enabled.

## Run Tests & QA Evaluation

Run the unit and API test suite:

```bash
python -m pytest tests/test_api.py tests/test_preprocess.py -v
```

Run the real-world handwriting evaluation across all 50+ team samples:

```bash
python -m tests.real_samples_eval
```

The API tests cover the health route, valid PNG predictions, canvas data URLs, transparent PNGs, CORS, malformed input, and blank canvases.

## Real-Handwriting QA

Team samples are stored under `samples/Group digit Samples/`. The image folders contain one handwriting sample per digit for several team members.

See the complete functional (F1–F5) and edge-case QA checklist in [docs/test_plan.md](docs/test_plan.md).

When evaluating the real model, test:

- Every digit from 0 through 9
- Blank canvases
- Tiny drawings near an image edge
- Large or off-center drawings
- Scribbles
- Mouse and touch input
- Mobile and desktop browser layouts

Before release, the MVP target is at least 98% MNIST test accuracy and at least 80% accuracy across the team's 50+ real handwriting samples.

## Troubleshooting

- **`python` is not recognized (Windows):** Use `py -3` instead of `python` (e.g., `py -3 -m venv .venv`).
- **TensorFlow or package installation fails:** Confirm that Python is 3.10 or newer and that the virtual environment is activated.
- **The API cannot find the model:** A stand-in model may be used only for API testing. A real `models/digit_cnn.keras` is needed for meaningful predictions.
- **MNIST download fails:** Retry with internet access enabled. The dataset cache will be reused after a successful download.
- **The browser cannot reach the API:** Start FastAPI on port 8000 and keep the frontend server on port 5500.
- **A blank canvas returns an error:** Draw visible, dark ink over the white canvas before predicting.

## API Contract

| Item | Contract |
| --- | --- |
| Health endpoint | `GET /health` |
| Prediction endpoint | `POST /predict` |
| Request body | `{"image": "<base64 PNG string>"}` |
| Success response | `{"digit": 7, "confidence": 0.97, "probabilities": [10 float values]}` |
| Error response | `{"error": "message"}` |
| Model input | `(1, 28, 28, 1)`, float32, values from 0 to 1 |
