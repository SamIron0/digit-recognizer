"""Work package 4 — CNN model definition.

build_model() returns the compiled CNN from the PRD:
input (28, 28, 1) float32 in 0-1, output 10 softmax probabilities.
"""

import keras
from keras import layers


def build_model() -> keras.Model:
    """Return the compiled CNN: 2 conv, 2 pooling, dropout, softmax."""
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    import numpy as np

    model = build_model()
    model.save("models/digit_cnn.keras")
    probs = model.predict(np.zeros((1, 28, 28, 1)))
    print("output shape:", probs.shape)   # expect (1, 10)
    print("sum:", probs.sum())            # expect ~1.0
