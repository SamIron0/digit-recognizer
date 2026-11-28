# Handwritten Digit Recognizer

A local web app that recognizes handwritten digits from 0 to 9. Draw a digit on the canvas, click **Predict**, and read the predicted digit with a confidence score.

The project follows the MVP design in [PRD Handwritten Digit Recognizer (MVP).md](PRD%20Handwritten%20Digit%20Recognizer%20%28MVP%29.md).

## Features

1. **MNIST data pipeline** — Loads MNIST, scales pixels to float32 values in 0–1, reshapes images to `(28, 28, 1)`, and returns 54,000 training, 6,000 validation, and 10,000 test images.
2. **Baseline model** — Logistic regression on the same data, as a comparison point for the CNN (about 92% test accuracy).
3. **CNN model** — Two convolution layers, two pooling layers, dropout, and a 10-class softmax output.
4. **Training** — Trains the CNN on MNIST with early stopping and saves `models/digit_cnn.keras`. Target test accuracy is at least 98%.
5. **Evaluation** — Writes test accuracy, a confusion matrix, and a grid of misclassified digits under `docs/`.
6. **Preprocessing** — Turns a drawn or uploaded image into MNIST format: grayscale, inverted to white-on-black, cropped, scaled so the long side is 20 pixels, centered by center of mass in a 28×28 canvas, and returned as `(1, 28, 28, 1)` float32 in 0–1. A blank image raises `ValueError`.
7. **Prediction API** — FastAPI `GET /health` and `POST /predict`. The model loads once at startup. A prediction returns the digit, confidence, and 10 class probabilities.
8. **Drawing canvas** — 280×280 canvas, black ink on a white background, mouse and touch drawing, and a Clear button.
9. **Frontend prediction** — Exports the canvas as a PNG, calls `/predict`, and shows the digit, confidence as a percentage, a loading state, an unsure state when confidence is below 0.5, and a readable error if the request fails.
10. **Team handwriting QA** — 66 labeled photos in `samples/`, plus `tests/real_samples_eval.py`. The release target is at least 80% accuracy on these photos.

## Project Structure

```text
digit-recognizer/
├── api/main.py                 # FastAPI /health and /predict
├── docs/
│   ├── test_plan.md            # QA checklist
│   ├── evaluation.md           # MNIST evaluation write-up
│   ├── confusion_matrix.png    # Confusion matrix
│   └── misclassified.png       # Grid of misclassified digits
├── models/digit_cnn.keras      # Trained CNN
├── samples/                    # 66 team handwriting images
├── src/
│   ├── data.py                 # MNIST load, normalize, reshape, split
│   ├── baseline.py             # Logistic regression baseline
│   ├── model.py                # CNN architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Accuracy and error analysis
│   └── preprocess.py           # Canvas image to MNIST tensor
├── tests/
│   ├── real_samples_eval.py    # Accuracy on samples/
│   ├── test_api.py             # API tests
│   └── test_preprocess.py      # Preprocessing tests
└── web/
    ├── index.html              # Page
    ├── style.css               # Styles
    ├── canvas.js               # Mouse and touch drawing
    └── app.js                  # Predict flow
```

## Requirements

- Python 3.10, 3.11, or 3.12 (`tensorflow==2.16.2` does not install on Python 3.13)
- Git
- A terminal
- A modern browser
- Internet access for the clone and the first MNIST download

The first `pip install` and the first training run are the slow steps.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SamIron0/digit-recognizer.git
cd digit-recognizer
```

### 2. Create a virtual environment and install dependencies

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

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

Use `python3.10` or `python3.11` instead when that is the interpreter you have. Check with `python3 --version` before creating the environment.

## Check the data

From the repository root:

```bash
python -c "from src.data import load_data; (xt,_),(xv,_),(xs,_)=load_data(); print(xt.shape, xv.shape, xs.shape)"
```

Expected shapes:

- Training images: `(54000, 28, 28, 1)`
- Validation images: `(6000, 28, 28, 1)`
- Test images: `(10000, 28, 28, 1)`

The first run downloads MNIST.

## Train the model

```bash
python -m src.train
```

This trains on MNIST, uses early stopping, and writes `models/digit_cnn.keras`. Test accuracy should be at least 98%.

## Evaluate the model

```bash
python -m src.evaluate
```

This regenerates:

- `docs/confusion_matrix.png`
- `docs/misclassified.png`
- `docs/evaluation.md`

## Run the API

From the repository root, in one terminal:

```bash
uvicorn api.main:app --reload
```

The API listens at `http://localhost:8000`.

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocess": "real"
}
```

### Prediction

From Windows PowerShell, with a PNG named `image.png` in the current directory:

```powershell
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes(".\image.png"))
$body = @{ image = $base64 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

The `image` field accepts raw base64 or a canvas data URL (`data:image/png;base64,...`).

Example response:

```json
{
  "digit": 7,
  "confidence": 0.97,
  "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.97, 0.0, 0.0]
}
```

An empty body, invalid base64, non-image bytes, and a blank canvas return `{"error": "message"}` with a 4xx status. A missing or mistyped `image` field returns 422.

## Run the web page

Keep the API running. In a second terminal, from the repository root:

```bash
cd web
python -m http.server 5500
```

Open `http://localhost:5500`.

1. Draw one digit with the mouse or a touch device. Ink is black on a white 280×280 canvas.
2. Click **Predict**.
3. Read the digit and the confidence percentage.
4. Click **Clear** and try another digit.

The page calls `http://localhost:8000`. The API allows cross-origin requests from the page.

## Tests and handwriting accuracy

From the repository root:

```bash
python -m pytest tests/test_api.py tests/test_preprocess.py -v
```

```bash
python -m tests.real_samples_eval
```

The sample script reads every image directly under `samples/`, takes the label from the first character of the filename, and runs the same preprocess and predict path as the API. It prints per-digit accuracy and overall accuracy. The release bar is at least 80%.

The API tests cover `/health`, valid PNGs, canvas data URLs, transparent PNGs, CORS, malformed input, and a blank canvas.

## Handwriting samples

Team photos live in `samples/`, one digit per file:

```text
samples/<digit>_<person>_1.jpeg
```

Example: `samples/7_king_1.jpeg` is a photo of the digit 7. There are 66 images, covering digits 0 through 9.

The manual checklist for drawing, prediction, edge cases, and a fresh clone is [docs/test_plan.md](docs/test_plan.md).

Release targets:

- at least 98% accuracy on the MNIST test set
- at least 80% accuracy on these 66 photos
- a fresh clone runs from this README in under 10 minutes

## Troubleshooting

- **`python` is not recognized (Windows):** Use `py -3.12` to create the virtual environment.
- **TensorFlow fails to install:** Use Python 3.10, 3.11, or 3.12, and activate the virtual environment before `pip install`. Python 3.13 cannot install `tensorflow==2.16.2`.
- **The API cannot load the model:** Run `python -m src.train` from the repository root so `models/digit_cnn.keras` exists, then start `uvicorn` again.
- **MNIST download fails:** Retry with internet access. Later runs reuse the Keras cache.
- **The page cannot reach the API:** Keep FastAPI on port 8000 and the page server on port 5500.
- **A blank canvas returns an error:** Draw a digit in dark ink on the white canvas, then predict again.

## API Contract

| Item | Contract |
| --- | --- |
| Health endpoint | `GET /health` |
| Prediction endpoint | `POST /predict` |
| Request body | `{"image": "<base64 PNG string>"}` |
| Success response | `{"digit": 7, "confidence": 0.97, "probabilities": [10 float values]}` |
| Error response | `{"error": "message"}` |
| Model input | `(1, 28, 28, 1)`, float32, values from 0 to 1, white digit on black |
