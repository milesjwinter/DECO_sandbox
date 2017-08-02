from __future__ import print_function
import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, MaxoutDense, AveragePooling2D, Lambda, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils.vis_utils import plot_model
from keras import metrics
from scipy.ndimage.interpolation import rotate
#import matplotlib.pyplot as plt
import numpy as np
import time
from keras import backend as K

start_time = time.time()

def smooth_labels(y, smooth_factor):
    '''
    Convert a matrix of one-hot row-vector labels into smoothed versions.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def augment_image(img):
    #random square-root to square
    #exp_val = 2.0**np.random.uniform(-1.,1.)
    #img = img**exp_val
    #random brightness correction
    img = img*np.random.uniform(0.9,1.0)
    #random rotation
    angle = np.random.randint(4)*90
    return rotate(img,angle)

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def crop_center(img):
    cropx = 64
    cropy = 64
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]

#fix random seed for reproducibility
#seed = 357
#np.random.seed(seed)

#model/training parameters
batch_size = 25 #17
num_classes = 2 #3
epochs = 500
index = 252 #34
# input image parameters
img_rows, img_cols = 64, 64
img_channels = 3 #3

#load images and labels
#f = h5py.File('Big_gray_DECO_Image_Database_v7.h5','r')
#f = h5py.File('Binary_DECO_Image_Database_100.h5','r')
f = h5py.File('Binary_RGB_DECO_Image_Database_100.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']

#make labels into a matrix format
#labels = keras.utils.to_categorical(labels, num_classes)
#labels = smooth_labels(keras.utils.to_categorical(labels, num_classes),0.002)

#normalize images
images = np.array(images,dtype='float32')/255.
#for i in range(len(images)):
#    norm = np.amax(images[i,:,:,0])
#    images[i,:,:,0] = images[i,:,:,0]/norm

train_images = images[index:]
train_labels = smooth_labels(keras.utils.to_categorical(labels[index:], num_classes),0.002)
test_images = images[:index]
test_labels = keras.utils.to_categorical(labels[:index], num_classes)

'''
i = 0
index0=44*i
index1=44*(i+1)
test_indices = np.arange(index0,index1,1)
train_indices = np.arange(0,index0,1)
train_indices = np.append(train_indices,np.arange(index1,440,1))
train_images = images[train_indices]
test_images = images[test_indices]
train_labels = labels[train_indices]
test_labels = labels[test_indices]
'''
#for i in range(len(train_images)):
#    norm = np.amax(train_images[i,:,:,0])
#    train_images[i,:,:,0] = train_images[i,:,:,0]/norm

f.close()

#class_weight = {0:1.0,1:3.2,2:2.5}
#class_weight = {0:1.0,1:3.66,2:2.85}  # for v6
#class_weight = {0:1.0,1:8.2,2:2.2}  #for v7
class_weight = {0:1.0,1:2.6} #for binary
#normalize

#make labels into a matrix format
#train_labels = keras.utils.to_categorical(train_labels, num_classes)
#test_labels = keras.utils.to_categorical(test_labels, num_classes)


'''
#define model structure (Old Best, train for 450-500?)
model = Sequential()
model.add(Conv2D(16, (16, 16),activation='relu',padding='same',input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (8, 8), activation='relu',padding='same'))
model.add(Conv2D(32, (5, 5), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (4, 4), activation='relu',padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(4)))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(4)))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''

'''
#define model structure (trained to 1000, continuing training)
model = Sequential()
#model.add(Lambda(crop_center,output_shape=(64,64,1),input_shape=(100,100,1)))
model.add(Cropping2D(cropping=18,input_shape=(100, 100, img_channels)))
model.add(Conv2D(32, (3, 3),activation='relu', padding='same')) #, input_shape=(64,64,1)))
#model.add(Conv2D(32, (3, 3),activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu',strides=(2, 2), padding='same'))
model.add(BatchNormalization(axis=-1))
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2),padding='same'))
model.add(BatchNormalization(axis=-1))
#model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(BatchNormalization(axis=-1))
#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu',strides=(2, 2), padding='same'))
model.add(BatchNormalization(axis=-1))
#model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
#model.add(Conv2D(256, (6, 6), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
#model.add(Dense(2048, activation='relu'))
#model.add(Dropout(0.4))
#model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Nadam(),
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''




#define model structure (using leakyrelu)
model = Sequential()
model.add(Cropping2D(cropping=18,input_shape=(100, 100, img_channels)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dropout(0.1))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dropout(0.1))
model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dropout(0.3))
model.add(Conv2D(128, (1, 1), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

'''
#define model structure
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(64,64,1)))
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
'''
#define model structure (adding batch normalization)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='valid', input_shape=(64,64,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1024))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''

#model = load_model('Binary_trained_model.h5')
print(model.summary())
#plot_model(model, to_file='best_model_structure.png',show_shapes=True)

# checkpoint
#filepath="saved_model_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = "saved_model_checkpoints/best_checkpointed_model.h5"
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

#preprocess images
'''
datagen = ImageDataGenerator(horizontal_flip=True, 
                             vertical_flip=True,
                             samplewise_center=True,
                             samplewise_std_normalization=True,
                             width_shift_range=0.08, 
                             height_shift_range=0.08,
                             rotation_range=5.,
                             zoom_range=[0.9,1.1],
                             fill_mode="constant", 
                             cval=0,
                             preprocessing_function=augment_image) 
'''
datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             #samplewise_center=True,
                             #samplewise_std_normalization=True,
                             width_shift_range=0.08,
                             height_shift_range=0.08,
                             rotation_range=360.,
                             zoom_range=[0.9,1.1],
                             fill_mode="constant",
                             cval=0)

#fit the model
datagen.fit(train_images)
history = model.fit_generator(datagen.flow(train_images, train_labels,
                    batch_size=batch_size),
                    steps_per_epoch=train_images.shape[0] // batch_size,
                    epochs=epochs,
                    class_weight=class_weight,
                    validation_data=(test_images, test_labels),
                    callbacks=[checkpointer])
                    #initial_epoch=1000)

#save model weights and structure
model.save('Early_Stop_gray_trained_model.h5')

#evaluate  model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("job time: ", time.time() - start_time)

# list all data in history
print(history.history.keys())
history_vals = np.array([history.history['acc'],history.history['val_acc'],history.history['loss'],history.history['val_loss']])
np.savetxt('early_stop_history_vals.txt',np.transpose(history_vals))


