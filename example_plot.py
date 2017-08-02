# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.misc import toimage
import numpy as np
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# create a grid of 3x3 images
plt.figure(1)
for i in range(0, 9):
	plt.subplot(330 + 1 + i)
	plt.imshow(toimage(X_train[i]))
# show the plot
plt.show()

color_train = np.zeros((3,32,32,3))
plt.figure(2)
for i in range(0, 3):
        color_train[i,:,:,i] = X_train[5,:,:,i]
        plt.subplot(330 + 1 + i)
        plt.imshow(toimage(color_train[i]))
plt.show()
print color_train[2,:,:,2]
