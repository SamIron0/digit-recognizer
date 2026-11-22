"""
Work package 2 — Data pipeline
Owner: Okoro Ebubechukwu Prince 24/0253.

Deliverable: load MNIST, normalize, reshape, and create a 90/10 train/validation split.
Depends on: package 1.

"""
from keras.datasets import mnist #import the mnist dataset
import numpy as np

def load_mnist():
    """Load MNIST and return the raw training and test sets"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def normalize(image):
    """Scale pixel values from 0-255 to 0-1."""
    return image / 255.0

def fake_data(n = 100):
    """Dummy data for testing while the main dataset is being prepered"""
    x_train = np.random.rand(n, 28, 28, 1).astype("float32")
    x_val = np.random.rand(n // 10, 28, 28, 1).astype("float32")
    x_test = np.random.rand(n // 10, 28, 28, 1).astype("float32")

    y_train = np.random.randint(0, 10, n)
    y_val = np.random.randint(0, 10, n // 10)
    y_test = np.random.randint(0, 10, n // 10)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = fake_data()
    print(x_train.shape, x_val.shape, x_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)