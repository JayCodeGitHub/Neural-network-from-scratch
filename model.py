import numpy as np
import random
from forwardpropagation import forwardpropagation
from backpropagation import backpropagation

class Model():
    def __init__(self, inputSize, outputSize):
        fraction = 2/3

        self.W1 = np.random.rand(10, inputSize) - 0.5
        self.B1 = np.random.rand(10, 1) - 0.5

        self.W2 = np.random.rand(int(inputSize * fraction + outputSize), 10) - 0.5
        self.B2 = np.random.rand(int(inputSize * fraction + outputSize), 1) - 0.5

        self.W3 = np.random.rand(10, int(inputSize * fraction + outputSize)) - 0.5
        self.B3 = np.random.rand(10, 1) - 0.5

    def updateParams(self, W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
        W1 = W1 - alpha * dW1
        B1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        B2 = b2 - alpha * db2
        W3 = W3 - alpha * dW3
        B3 = b3 - alpha * db3
    
        return W1, B1, W2, B2, W3, B3
    
    def getPredictions(self, A):
        return np.argmax(A,0)

    def getAccuracy(self, predictions,Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradientDescent(self, X,Y,iterations,alpha):
        W1 = self.W1
        B1 = self.B1
        W2 = self.W2
        B2 = self.B2
        W3 = self.W3
        B3 = self.B3
        
        for i in range(iterations):
            Z1,A1,Z2,A2,Z3,A3 = forwardpropagation(W1, B1, W2, B2, W3, B3, X)
            dW1, db1, dW2, db2, dW3, db3 = backpropagation(Z1,A1,Z2,A2,W2,Z3,A3,W3,X,Y)
            W1, B1, W2, B2, W3, B3 = self.updateParams(W1, B1, W2, B2, W3, B3, dW1, db1, dW2, db2, dW3, db3, alpha)
            if i % 50 == 0:
                print('Iteration: ',i)
                print('Accuracy: ',self.getAccuracy(self.getPredictions(A3),Y))
        return W1,B1,W2,B2,W3,B3

    def fit(self, x_train, y_train):
        self.W1,self.B1,self.W2,self.B2, self.W3,self.B3 = self.gradientDescent(x_train,y_train,500,0.10)
