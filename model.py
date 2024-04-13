import numpy as np
import random

class Model():
    def __init__(self, inputSize, outputSize):
        fraction = 2/3

        self.W1 = np.random.rand(10, inputSize) - 0.5
        self.B1 = np.random.rand(10, 1) - 0.5

        self.W2 = np.random.rand(int(inputSize * fraction + outputSize), 10) - 0.5
        self.B2 = np.random.rand(int(inputSize * fraction + outputSize), 1) - 0.5

        self.W3 = np.random.rand(10, int(inputSize * fraction + outputSize)) - 0.5
        self.B3 = np.random.rand(10, 1) - 0.5

        print("W1", self.W1)
        print("B1", self.B1)

        print("W2", self.W2)
        print("B2", self.B2)

        print("W3", self.W3)
        print("B3", self.B3)

    def fit():
        print("fit")
    def evaluate():
        print("evaluate")

