import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

x = tf.constant([[1., 2., 3.],[4., 5., 6.]])

print(x)
