# Handwritten Digit Recognizer

Skeleton only. No implementation is in this repository yet.

Each file below is a placeholder. The teammate assigned to that work package replaces it with the real work. Interface contracts, dependencies, and the definition of done are in `PRD Handwritten Digit Recognizer (MVP).md`.

The trained model file `models/digit_cnn.keras` is produced by `src/train.py`. It is not checked in until package 5 saves it.

| Path | Package | What the owner adds |
| --- | --- | --- |
| `.gitignore` | 1. Repo and project setup | Ignore rules |
| `requirements.txt` | 1. Repo and project setup | Pinned dependencies |
| `.github/PULL_REQUEST_TEMPLATE.md` | 1. Repo and project setup | Pull request template |
| `src/data.py` | 2. Data pipeline | MNIST load, normalize, reshape, 90/10 split |
| `src/baseline.py` | 3. Baseline model | Logistic regression or dense net, plus reported test accuracy |
| `src/model.py` | 4. CNN model definition | `build_model()` |
| `src/train.py` | 5. Training script | Train the CNN, early stopping, save the model and training curves |
| `src/evaluate.py` | 6. Evaluation | Test accuracy, confusion matrix, misclassified digits, written analysis |
| `docs/evaluation.md` | 6. Evaluation | Short written analysis |
| `src/preprocess.py` | 7. Preprocessing module | `preprocess()` per the interface contract |
| `tests/test_preprocess.py` | 7. Preprocessing module | Unit tests on sample images |
| `api/main.py` | 8. Backend API | FastAPI `/predict` and `/health` |
| `tests/test_api.py` | 8. Backend API | API tests |
| `web/index.html` | 9. Frontend canvas | Page shell |
| `web/style.css` | 9. Frontend canvas | Page styles |
| `web/canvas.js` | 9. Frontend canvas | Mouse and touch drawing, Clear |
| `web/app.js` | 10. Frontend integration | Export PNG, call `/predict`, show result, loading, and errors |
| `samples/` | 11. QA, testing, and docs | 50+ real handwriting images from the team |
| `README.md` | 11. QA, testing, and docs | Setup and demo instructions (replace this skeleton note) |
