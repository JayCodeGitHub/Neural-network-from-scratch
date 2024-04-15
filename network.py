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
    myModel.testPrediction(8)
    myModel.testPrediction(3)
    myModel.testPrediction(93)
    myModel.testPrediction(43)
    myModel.testPrediction(2)
    myModel.testPrediction(1)
    myModel.testPrediction(5)
    myModel.testPrediction(96)
    myModel.testPrediction(34)
    myModel.testPrediction(467)
    myModel.testPrediction(345)
    myModel.testPrediction(65)


