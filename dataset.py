import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)
