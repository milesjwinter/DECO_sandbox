from __future__ import print_function
import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.models import load_model
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils.vis_utils import plot_model
from keras import metrics
from scipy.ndimage.interpolation import rotate
from keras import backend as K
import theano.tensor as T
from theano.tensor import fft
import numpy as np

def discrete_rotation(img):
    angle = np.random.randint(4)*90
    return rotate(img,angle)

def fft(img):
    #img = np.array(img)
    #return K.function(img,np.fft.rfft2(img,axes=(1,2)))
    img = np.fft.rfft2(img,axes=(0,1))
    print(img.shape)
    return img

def ifft(img):
    return K.function(img,np.fft.irfft2(img,axes=(1,2)))

#def sorted_img(x):
#    return K.Sort(x)
x = T.tensor4(name='x',dtype='float32')
#fft_img = np.fft.rfft2(x,axes=(0,1))
fft_func = K.function(inputs=[x],outputs=np.fft.rfft2(x,axes=(1,2)))
#ifft_func = K.function(ifft)   

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#model/training parameters
batch_size = 22
num_classes = 3
epochs = 2
index = 22
# input image parameters
img_rows, img_cols = 64, 64
img_channels = 3

#load images and labels
f = h5py.File('Big_gray_DECO_Image_Database_v6.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']
train_images = images[index:]
train_labels = labels[index:]
test_images = images[:index]
test_labels = labels[:index]
f.close()

class_weight = {0:1.0,1:3.2,2:2.5}
#normalize
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

#make labels into a matrix format
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)
'''
#define model structure
model = Sequential()
model.add(Conv2D(64, (8, 8),activation='relu',padding='same',input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (4, 4), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(4)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu',kernel_constraint=maxnorm(4)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Nadam(),
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''

#really simple model
model = Sequential()
model.add(Conv2D(64, (8, 8),activation='relu', padding='same', input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(Lambda(lambda x: K.function(x,fft(x)),output_shape = (64,33,1)))
model.add(Lambda(fft_img,output_shape = (64,33,1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(Lambda(ifft,output_shape=(64,64,1)))
#model.add(Lambda(lambda x: ifft(x)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(4)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

'''
#define model structure (First Model)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=train_images.shape[1:]))
#model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''

print(model.summary())
#plot_model(model, to_file='model.png',show_shapes=True)

#preprocess images
datagen = ImageDataGenerator(horizontal_flip=True, 
                             vertical_flip=True,
                             #width_shift_range=0.1, 
                             #height_shift_range=0.1,
                             #rotation_range=10.,
                             #zoom_range=.1,
                             #fill_mode="constant", 
                             #cval=0,
                             preprocessing_function=discrete_rotation) 

#fit the model
datagen.fit(train_images)
model.fit_generator(datagen.flow(train_images, train_labels,
                    batch_size=batch_size),
                    steps_per_epoch=train_images.shape[0] // batch_size,
                    epochs=epochs,
                    class_weight=class_weight,
                    validation_data=(test_images, test_labels))
#save model weights and structure
#model.save('Big_gray_trained_model.h5')

#evaluate  model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

