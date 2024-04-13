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
    def updateParams(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
        W1 = W1 - alpha * dW1
        B1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        B2 = b2 - alpha * db2
        W3 = W3 - alpha * dW3
        B3 = b3 - alpha * db3
    
    return W1, B1, W2, B2, W3, B3
    def evaluate():
        print("evaluate")
