"""
Work package 2 — Data pipeline
Owner: Okoro Ebubechukwu Prince 24/0253.

Deliverable: load MNIST, normalize, reshape, and create a 90/10 train/validation split.
Depends on: package 1.

"""
from keras.datasets import mnist #import the mnist dataset
import numpy as np

def load_mnist():
    """"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def normalize(image):
    

if __name__ == "__main__":
    (x_train, y_train), _ = load_mnist()
    print(y_train.shape)