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

if __name__ == "__main__":
    (x_train, _), (_,_) = load_mnist()
    normal = normalize(x_train)
    print(normal.min(), normal.max())
