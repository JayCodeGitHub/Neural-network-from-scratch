import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import x_train, x_test, y_train
from model import Model


def network():
    inputSize = len(x_train[0]) * len(x_train[0][0])
    outputSize = 10
    myModel = Model(inputSize, outputSize)


