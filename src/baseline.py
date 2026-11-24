"""
Work package 3 — Baseline model
Owner: teammate assigned to this package.

Deliverable: logistic regression or a simple dense network, with reported test accuracy.
Depends on: package 2.

Replace this placeholder with the implementation.
"""

"""Baseline models for MNIST: logistic regression and a dense network."""


import json
import time
from pathlib import Path

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data import load_mnist


# Our project directories
# This will be where the trained Logistic Regression model will be saved
MODEL_PATH = Path("models/logistic_regression.joblib")
# This is path where evaluation results will be saved.
RESULTS_PATH = Path("docs/baseline_results.json")
# Where the confusion matrix will be saved
CONFUSION_PATH = Path("docs/confusion_matrix.csv")


def flatten(x: np.ndarray) -> np.ndarray:
    """I am working on an assumption that each MNSIT image has 28 * 28 piexels in height and weight """
    """so i am converting (N, 28, 28) images into (N, 784) features."""
    return x.reshape(len(x), -1)


def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(
        # Will probably have to increaser max_iter later just working based on the dummy data
        max_iter=200, 
        solver="lbfgs"
    )

    model.fit(flatten(x_train), y_train)

    return model


def evaluate_model(model, x, y):
    predictions = model.predict(flatten(x))

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions
        ).tolist()
    }


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()

    # Training
    start = time.time()
    model = train_logistic_regression(x_train, y_train)
    train_time = time.time() - start

    # Evaluating: Will compare val_result with test_result
    val_results = evaluate_model(model, x_val, y_val)
    test_results = evaluate_model(model, x_test, y_test)

    # Save trained model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Save confusion matrix
    CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        CONFUSION_PATH,
        test_results["confusion_matrix"],
        delimiter=",",
        fmt="%d"
    )

    # Save evaluation results
    results = {
        "model": "logistic_regression",
        "validation_accuracy": val_results["accuracy"],
        "test_accuracy": test_results["accuracy"],
        "train_seconds": round(train_time, 2),
        "classification_report": test_results["classification_report"],
        "model_path": str(MODEL_PATH)
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
