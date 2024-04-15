import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import x_train, x_test, y_train
from model import Model

def network():
    inputSize = len(x_train[0])
    outputSize = 10
    myModel = Model(inputSize, outputSize)
    myModel.fit(x_train, y_train)
    myModel.testPrediction(8, x_train, y_train)
    myModel.testPrediction(3, x_train, y_train)
    myModel.testPrediction(93, x_train, y_train)
    myModel.testPrediction(43, x_train, y_train)
    myModel.testPrediction(2, x_train, y_train)
    myModel.testPrediction(1, x_train, y_train)
    myModel.testPrediction(5, x_train, y_train)
    myModel.testPrediction(93, x_train, y_train)
    myModel.testPrediction(34, x_train, y_train)
    myModel.testPrediction(467, x_train, y_train)
    myModel.testPrediction(345, x_train, y_train)
    myModel.testPrediction(65, x_train, y_train)


