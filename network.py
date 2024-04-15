import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import x_train, x_test, y_train
from model import Model

def network():
    inputSize = len(x_train[0])
    outputSize = 10
    myModel = Model(x_train, outputSize)
    myModel.fit(x_train, y_train)
    myModel.testPrediction(8, y_train)
    myModel.testPrediction(3, y_train)
    myModel.testPrediction(93, y_train)
    myModel.testPrediction(43, y_train)
    myModel.testPrediction(2, y_train)
    myModel.testPrediction(1, y_train)
    myModel.testPrediction(5, y_train)
    myModel.testPrediction(93, y_train)
    myModel.testPrediction(34, y_train)
    myModel.testPrediction(467, y_train)
    myModel.testPrediction(345, y_train)
    myModel.testPrediction(65, y_train)


