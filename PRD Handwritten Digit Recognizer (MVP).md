# PRD: Handwritten Digit Recognizer (MVP)

**Team size:** 11 contributors 

1 Overview

We are building the simplest possible working handwritten digit recognizer: a user draws a digit (0-9) on a web canvas, clicks **Predict**, and sees the model's guess with a confidence score.

The project is deliberately small in technical scope.

## 2. Goals and Non-Goals

### Goals

1. A trained model that reaches **at least 98% accuracy** on the MNIST test set.
2. A web page with a drawing canvas that returns a prediction for a digit the user draws.
3. A reproducible repository: anyone can clone it, run two or three commands, and see it work.
4. **Every team member ships at least one reviewed, merged contribution.**

## 3. Users and Use Case

**Primary user:** Our lecturer.

**Core flow:**

1. Open the web page.
2. Draw a digit with mouse or finger.
3. Click **Predict**.
4. See "Prediction: 7 (confidence 97%)".
5. Click **Clear** and try again.

## 4. Scope (MVP)

| Area | Decision (keep it simple) |
| --- | --- |
| Dataset | MNIST (60,000 train / 10,000 test), loaded via Keras |
| Baseline model | Logistic regression or a simple dense network, to compare against |
| Main model | Small CNN: 2 conv layers, 2 pooling layers, dropout, softmax |
| Language | Python 3.10+ for ML and API; plain HTML/CSS/JavaScript for the frontend |
| Serving | FastAPI with one endpoint: `POST /predict` |
| Frontend | Single page with canvas, Predict button, Clear button, result display |
| Hosting | Runs locally |

## 5. System Architecture

```
[Canvas UI] --(base64 PNG)--> [FastAPI /predict] --> [Preprocess] --> [CNN model] --> {digit, confidence}
```

Training happens offline in a separate script and produces one file: `models/digit_cnn.keras`. The API loads that file at startup.

## 6. Interface Contracts

These contracts let people work **in parallel without waiting on each other**. Changing one requires team agreement.

**Model file:** `models/digit_cnn.keras`. Input shape `(1, 28, 28, 1)`, float32 values in 0-1, white digit on black background. Output: 10 probabilities.

**Preprocessing function:** `preprocess(image: np.ndarray) -> np.ndarray`. Takes a grayscale image (any size, dark ink on light background or the reverse, handled per the spec) and returns a `(1, 28, 28, 1)` float32 array that matches MNIST format: cropped to the digit, scaled to fit 20x20, centered in 28x28 by center of mass, normalized to 0-1.

**API:**

- Request: `POST /predict` with JSON `{ "image": "<base64 PNG string>" }`
- Response: `{ "digit": 7, "confidence": 0.97, "probabilities": [0.0, ... 10 values] }`
- Error response: `{ "error": "message" }` with an appropriate HTTP status

**Canvas export:** the frontend sends the canvas as a PNG, black ink on white background, 280x280 pixels.

## 7. Work Packages (one owner each)

There are **11 work packages for 11 people**, each with a clear deliverable. Size is intentionally balanced: each package is roughly 4-8 hours of work.

| # | Work package | Deliverable | Depends on |
| --- | --- | --- | --- |
| 1 | **Repo and project setup** | GitHub repo, folder structure, `.gitignore`, `requirements.txt`, branch protection, PR template, issue board | None |
| 2 | **Data pipeline** | `src/data.py`: loads MNIST, normalizes, reshapes, creates 90/10 train/validation split | 1 |
| 3 | **Baseline model** | `src/baseline.py`: logistic regression or dense net, reports test accuracy, documented results | 2 |
| 4 | **CNN model definition** | `src/model.py`: the CNN architecture as a function `build_model()` | 2 |
| 5 | **Training script** | `src/train.py`: trains the CNN, uses early stopping, saves `models/digit_cnn.keras`, saves training curves as PNG | 2, 4 |
| 6 | **Evaluation** | `src/evaluate.py`: test accuracy, confusion matrix image, grid of 16 misclassified digits, short written analysis | 5 |
| 7 | **Preprocessing module** | `src/preprocess.py` implementing the contract above, with unit tests using sample images | 1 |
| 8 | **Backend API** | `api/main.py`: FastAPI app with `/predict` and `/health`, loads the model once at startup, validates input | 5 (can mock the model until ready), 7 |
| 9 | **Frontend canvas** | `web/index.html`, `web/style.css`, `web/canvas.js`: drawing with mouse and touch, Clear button | 1 |
| 10 | **Frontend integration** | `web/app.js`: exports canvas as PNG, calls `/predict`, shows digit, confidence, and loading and error states | 8, 9 |
| 11 | **QA, testing, and docs** | Test plan, 50+ real handwritten samples collected from teammates, accuracy report on them, `README.md` with setup and demo instructions | 8, 10 |

### How dependencies are handled

People whose package depends on another can start immediately by working against the **interface contracts**:

- 8 builds the API with a dummy model that returns a fixed prediction, then swaps in the real model.
- 10 builds against a mocked API response.
- 7 tests preprocessing on sample images with no model required.

## 8. Fair Contribution Policy

The goal is that everyone contributes meaningfully and that this is visible.

1. **One/Two**** owner(s)**** per work package.**
2. **Everyone commits under their own account.** All work goes through pull requests. No direct pushes to `main`.
3. **Minimum bar for everyone:**
   - At least **1 merged PR** for their work package
   - Attendance at every sync meeting (or an async written update)
4. **Pair up for support.** Each work package has a "buddy" (a second person who reviews and can step in). Pairs are assigned so that stronger and newer members work together.
5. **Stretch tasks are shared out.** When someone finishes early, they pick from the stretch list (section 11) rather than taking over someone else's package.
6. **Contribution is tracked by the repo.** Commit history, PRs, reviews, and the issue board are the source of truth. Non-code work (documentation, testing, data collection) counts equally and is recorded as issues and PRs.
7. **Blocked? Say so early.** If someone is stuck for more than a day, post in the team channel. The buddy or lead helps. Work is reassigned only after discussion with the owner.

## 9. Milestones

| Phase | Target | Outcome |
| --- | --- | --- |
| **0. Kickoff** | Day 1 | Roles assigned, contracts agreed, repo created (#1) |
| **1. Foundations** | Days 2-3 | Data pipeline (#2), preprocessing (#7), canvas UI (#9), CNN definition (#4), baseline (#3) |
| **2. Core build** | Days 3-4 | Training (#5), API (#8), evaluation (#6) |
| **3. Integration** | Days 4-5 | Frontend wired to API (#10), end-to-end demo works |
| **4. Testing and polish** | Days 5-6 | QA on real handwriting (#11), bug fixes, README, demo rehearsal |

Adjust day counts to the team's real deadline, keeping the order.

## 10. Requirements

### Functional

| ID | Requirement | Priority |
| --- | --- | --- |
| F1 | User can draw on a canvas with mouse or touch | Must |
| F2 | User can clear the canvas | Must |
| F3 | Clicking Predict sends the drawing to the API and displays the predicted digit | Must |
| F4 | Display confidence as a percentage | Must |
| F5 | Show a loading state while waiting and a readable error if the API fails | Should |

### Non-functional

| ID | Requirement |
| --- | --- |
| N1 | Prediction response time under 1 second locally |
| N2 | Model test accuracy of 98% or higher on MNIST |
| N3 | Fresh clone runs end-to-end by following the README in under 10 minutes |
| N4 | Code is readable: consistent formatting, docstrings on public functions |

## 11. Stretch Goals (only after MVP works)

- Data augmentation (rotation, shift, zoom) and measuring the effect on real handwriting
- Run the model in-browser with TensorFlow.js (no backend)
- Multi-digit recognition with contour segmentation
- Deploy to a free hosting service
- Fine-tune on the team's own collected samples
- Dockerfile for one-command setup

## 12. Definition of Done

The MVP is complete when:

- All Must requirements (F1-F4) work end-to-end
- Model test accuracy is at least 98% and documented in the README
- The model correctly recognizes at least **80% of the team's 50+ real handwriting samples** (this tests the MNIST-versus-real-world gap)
- Every team member has at least one merged PR and two reviews
- README lets a new person run the project from a fresh clone
- The team has run a live demo successfully at least once

## 13. Risks and Mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Model does well on MNIST but poorly on canvas drawings | High | Preprocessing (#7) matches MNIST format exactly; QA (#11) tests early with real drawings |
| Merge conflicts with 11 people | High | Small PRs, one file or folder per owner, pull from `main` often |
| Package depends on a late teammate | Medium | Interface contracts plus mocks so work proceeds in parallel |
| Uneven skill levels | Medium | Buddy system, balanced package sizes, shared stretch tasks |
| Uneven participation | Medium | Minimum contribution bar, visible issue board, weekly check-ins |
| Environment differences (OS, Python versions) | Medium | Pinned `requirements.txt`, setup steps tested by #11 on a clean machine |

## 14. Suggested Repository Structure

```
digit-recognizer/
  README.md
  requirements.txt
  src/
    data.py
    baseline.py
    model.py
    train.py
    evaluate.py
    preprocess.py
  api/
    main.py
  web/
    index.html
    style.css
    canvas.js
    app.js
  models/
    digit_cnn.keras
  tests/
    test_preprocess.py
    test_api.py
  samples/
    (team handwriting images)
  docs/
    evaluation.md
```

## 15. Working Agreements

- **Branch naming:** `feature/<package-number>-<short-name>`, for example `feature/07-preprocessing`
- **PRs:** small, linked to an issue, reviewed by at least one person other than the author before merging
