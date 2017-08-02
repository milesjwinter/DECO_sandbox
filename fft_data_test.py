import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.ndimage.interpolation import rotate
import theano
import theano.tensor as T
from theano.tensor import fft
 
#fix random seed for reproducibility
#seed = 7
#np.random.seed(seed)

index = 12
num_classes=3
#load images and labels
f = h5py.File('Big_gray_DECO_Image_Database_v4.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']

images = np.array(images,dtype='float32')/255.
#images = np.sqrt(images)
#for i in range(len(images)):
#    norm = np.amax(images[i,:,:,0])
#    print norm
#    images[i] = images[i]/norm

train_images = images[index:]
train_labels = labels[index:]
test_images = images[:index]
test_labels = labels[:index]
f.close()

print train_images[-1].shape
x = T.tensor4('x', dtype='float32')
rfft = fft.rfft(x,norm='ortho')
f_rfft = theano.function([x], rfft)
out = f_rfft(test_images)
c_out = np.asarray(out[:,:,:,0,0] + 1j*out[:,:,:,0,1])
#out = out.reshape((out.shape[0],out.shape[1],out.shape[2],out.shape[4]))
y = T.tensor4('y', dtype='float32')
irfft = fft.irfft(y,norm='ortho')
f_irfft = theano.function([y], irfft)
iout = f_irfft(c_out)

print out.shape
print iout.shape

fft_image = np.fft.rfft2(test_images,axes=(1,2))
print fft_image.shape
print out[0,0,0,0]
print out[0,0,0,1]
print fft_image[0,0,0,0]
#print fft_image
ifft_image = np.fft.irfft2(fft_image,axes=(1,2))
#print ifft_image.shape
plt.figure()
#plt.imshow(np.absolute(fft_image))
plt.imshow(ifft_image[0,:,:,0])
plt.show()
