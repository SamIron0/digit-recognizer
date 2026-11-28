"""Work package 5 — Training script.

Trains build_model() on MNIST from load_data(), stops early when validation
loss stops improving, and saves the model, training curves, and metrics.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from src.data import load_data
from src.model import build_model

MODEL_PATH = Path("models") / "digit_cnn.keras"
METRICS_PATH = Path("models") / "metrics.json"
CURVES_PATH = Path("docs") / "training_curves.png"


def train(epochs: int = 15):
    """Fit the CNN on MNIST and save the model, curves, and metrics."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    model = build_model()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=[early_stopping],
    )

    _, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    _save_curves(history.history)

    metrics = {
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"saved {MODEL_PATH}")
    print(f"saved {CURVES_PATH}")
    print(f"saved {METRICS_PATH}")
    print(f"val accuracy: {val_accuracy:.4f}  test accuracy: {test_accuracy:.4f}")
    return model


def _save_curves(history: dict):
    """Write accuracy and loss vs. epoch to docs/training_curves.png."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["accuracy"], label="train")
    axes[0].plot(epochs, history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CURVES_PATH)
    plt.close(fig)


if __name__ == "__main__":
    train()
