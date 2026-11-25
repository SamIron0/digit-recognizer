"""Work package 5 — Training script.

Hour one trains build_model() on fake_data for one epoch and saves
models/digit_cnn.keras. Labels are random, so accuracy stays near 10%.
The point of this run is that the script finishes and writes the file.
"""

from pathlib import Path

from src.data import fake_data
from src.model import build_model

MODEL_PATH = Path("models") / "digit_cnn.keras"


def train(n: int = 100, epochs: int = 1):
    """Fit the CNN on stand-in data and save the model file."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = fake_data(n)
    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved {MODEL_PATH}")
    print(f"test loss: {loss:.4f}  test accuracy: {accuracy:.4f}")
    return model


if __name__ == "__main__":
    train()
