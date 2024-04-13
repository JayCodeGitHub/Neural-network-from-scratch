import numpy as np
from rel import ReLU
from softmax import Softmax

def forwardpropagation(W1,B1,W2,B2,W3,B3,X):
    Z1 = W1.dot(X.T) + B1
    A1 = ReLU(Z1)
    
    Z2 = W2.dot(A1) + B2
    A2 = ReLU(Z2)
    
    Z3 = W3.dot(A2) + B3
    A3 = Softmax(Z3)
    
    return Z1,A1,Z2,A2,Z3,A3
