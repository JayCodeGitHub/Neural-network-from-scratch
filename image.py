import numpy as np
import matplotlib.pyplot as plt

def showImage(image):
    image = image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image, interpolation="nearest")
    plt.show()
    
    
