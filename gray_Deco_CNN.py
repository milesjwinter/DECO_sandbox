from __future__ import print_function
import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate
from keras.layers import Conv2D, MaxPooling2D, MaxoutDense
from keras.models import load_model
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils.vis_utils import plot_model
from keras import metrics
from scipy.ndimage.interpolation import rotate
#import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

def discrete_rotation(img):
    #random square-root to square
    exp_val = 2.0**np.random.uniform(-1,1)
    img = img**exp_val
    #random brightness correction
    #img = img*np.random.uniform(0.5,1.0)
    #random rotation
    angle = np.random.randint(4)*90
    return rotate(img,angle)

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#model/training parameters
batch_size = 22 #17
num_classes = 3
epochs = 350
index = 44 #34
# input image parameters
img_rows, img_cols = 64, 64
img_channels = 3

#load images and labels
f = h5py.File('Big_gray_DECO_Image_Database_v6.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']

#make labels into a matrix format
labels = keras.utils.to_categorical(labels, num_classes)

#normalize images
images = np.array(images,dtype='float32')/255.
for i in range(len(images)):
    norm = np.amax(images[i,:,:,0])
    images[i,:,:,0] = images[i,:,:,0]/norm

#train_images = images[index:]
#for i in range(len(train_images)):
#    norm = np.amax(train_images[i,:,:,0])
#    train_images[i,:,:,0] = train_images[i,:,:,0]/norm
#train_labels = labels[index:]
#test_images = images[:index]
#test_labels = labels[:index]

i = 4
index0=44*i
index1=44*(i+1)
test_indices = np.arange(index0,index1,1)
train_indices = np.arange(0,index0,1)
train_indices = np.append(train_indices,np.arange(index1,440,1))
train_images = images[train_indices]
test_images = images[test_indices]
train_labels = labels[train_indices]
test_labels = labels[test_indices]

#for i in range(len(train_images)):
#    norm = np.amax(train_images[i,:,:,0])
#    train_images[i,:,:,0] = train_images[i,:,:,0]/norm

f.close()

#class_weight = {0:1.0,1:3.2,2:2.5}
class_weight = {0:1.0,1:3.66,2:2.85}
#normalize
#train_images = train_images.astype('float32')
#test_images = test_images.astype('float32')
#train_images /= 255
#test_images /= 255
#train_images = (train_images/255.)**2
#test_images = (test_images/255.)**2
#train_images = np.sqrt(train_images)
#test_images = np.sqrt(test_images)

#make labels into a matrix format
#train_labels = keras.utils.to_categorical(train_labels, num_classes)
#test_labels = keras.utils.to_categorical(test_labels, num_classes)

#Define SGD optimizer schedule for learning rate
#learning_rate = 0.1
#decay_rate = learning_rate / epochs
#momentum = 0.9
#sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)

'''
#define model structure (New Best, train for 450-500?)
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
#define model structure (Best Model)
model = Sequential()
model.add(Conv2D(16, (16, 16),activation='relu',padding='same',input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (8, 8), activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (4, 4), activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
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
#define model structure
model = Sequential()
model.add(Conv2D(128, (6, 6),activation='relu',input_shape=(64,64,1)))
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (4, 4), activation='relu'))
model.add(Conv2D(256, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Nadam(),
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
#define model structure with maxout dense
model = Sequential()
model.add(Conv2D(64, (12, 12),activation='relu', padding='same', input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (6, 6), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
#model.add(Conv2D(256, (6, 6), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu',kernel_constraint=maxnorm(3.)))
#model.add(MaxoutDense(256,nb_feature=2))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu',kernel_constraint=maxnorm(3.)))
#model.add(MaxoutDense(128,nb_feature=2))
model.add(Dropout(0.25))
#model.add(Dense(64, activation='relu',kernel_constraint=maxnorm(3.)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Nadam(),
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
#define model structure
model = Sequential()
model.add(Conv2D(64, (12, 12),activation='relu', input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (8, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(3.)))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(3.)))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Nadam(),
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
'''
#online model example
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
#lrate = 0.01
#decay = lrate/epochs
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
'''
'''
#really simple model
model = Sequential()
model.add(Conv2D(16, (5, 5),activation='relu', padding='same', input_shape=(64,64,1)))
model.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
model.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(4)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
'''
#define model structure(previous best)
model = Sequential()
model.add(Conv2D(64, kernel_size=(6, 6),activation='relu',input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
'''
#define model structure
model = Sequential()
model.add(Conv2D(64, (10, 10),activation='relu',input_shape=(64,64,1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
'''
#define model structure (BEST SO FAR)
model = Sequential()
model.add(Conv2D(32, kernel_size=(16, 16),activation='relu',input_shape=train_images.shape[1:]))
#model.add(Dropout(0.25))
model.add(Conv2D(128, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
#model.add(Dropout(0.25))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
'''
#define model structure
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
'''
#define model structure (test gal zoo)
model = Sequential()
model.add(Conv2D(64, kernel_size=(6, 6),activation='relu',input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
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
'''
model = Sequential()
model.add(Conv2D(64, kernel_size=(16, 16),activation='relu',input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
'''
#define model structure 
model = Sequential()
model.add(Conv2D(32, kernel_size=(10, 10),activation='relu', padding='same',input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(4, 4)))
#model.add(Dropout(0.2))
model.add(Conv2D(64, (6, 6), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
              optimizer=keras.optimizers.Adam(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
'''
#define model structure
model = Sequential()
model.add(Conv2D(64, (12, 12),activation='relu',padding='same',input_shape=train_images.shape[1:]))
#model.add(Conv2D(32, (5, 5), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
'''
model = Sequential()
model.add(Conv2D(32, (4, 4), activation='relu', input_shape=train_images.shape[1:]))
model.add(Conv2D(32, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
print(model.summary())
#plot_model(model, to_file='best_model_structure.png',show_shapes=True)

#preprocess images
datagen = ImageDataGenerator(horizontal_flip=True, 
                             vertical_flip=True,
                             #width_shift_range=0.1, 
                             #height_shift_range=0.1,
                             rotation_range=10.,
                             zoom_range=0.1,
                             fill_mode="constant", 
                             cval=0,
                             preprocessing_function=discrete_rotation) 

#fit the model
datagen.fit(train_images)
history = model.fit_generator(datagen.flow(train_images, train_labels,
                    batch_size=batch_size),
                    steps_per_epoch=train_images.shape[0] // batch_size,
                    epochs=epochs,
                    class_weight=class_weight,
                    validation_data=(test_images, test_labels))

#save model weights and structure
model.save('Big_gray_trained_model.h5')

#evaluate  model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("job time: ", time.time() - start_time)

# list all data in history
print(history.history.keys())
history_vals = np.array([history.history['acc'],history.history['val_acc'],history.history['loss'],history.history['val_loss']])
np.savetxt('history_vals.txt',np.transpose(history_vals))


'''
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
