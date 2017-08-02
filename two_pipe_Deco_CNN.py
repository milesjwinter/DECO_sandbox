from __future__ import print_function
import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils.vis_utils import plot_model
from keras import metrics
import numpy as np

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#model/training parameters
batch_size = 17
num_classes = 3
epochs = 2
index = 68

# input image parameters
img_rows, img_cols = 64, 64
img_channels = 3

#load images and labels
f = h5py.File('Big_gray_DECO_Image_Database_v4.h5','r')
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
model.add(Conv2D(16, (16, 16),activation='relu',padding='same',input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (8, 8), activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (4, 4), activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu',kernel_constraint=maxnorm(4)))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(4)))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Nadam(),
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])
'''
#define left branch of model
left_model = Sequential()
left_model.add(Conv2D(64, (3, 3),activation='relu',input_shape=(64,64,1)))
left_model.add(MaxPooling2D(pool_size=(2, 2)))
left_model.add(Dropout(0.2))
left_model.add(Flatten())

#define right branch of model
right_model = Sequential()
right_model.add(Conv2D(64, (3, 3),activation='relu',input_shape=(64,64,1)))
right_model.add(MaxPooling2D(pool_size=(2, 2)))
right_model.add(Dropout(0.2))
right_model.add(Flatten())

#merge left and right branches
#merged = concatenate([left_model, right_model])
merged = Merge([left_model, right_model], mode='concat')
#merged = keras.layers.Concatenate([left_model, right_model])
#print(merged.input_shape)

#define composite model
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(128, activation='relu'))
final_model.add(Dropout(0.4))
final_model.add(Dense(num_classes, activation='softmax'))
final_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(final_model.inputs)
print(final_model.summary())
#plot_model(model, to_file='model.png',show_shapes=True)

#preprocess images
left_datagen = ImageDataGenerator(#horizontal_flip=True, 
                             #vertical_flip=True,
                             #width_shift_range=0.2, 
                             #height_shift_range=0.2,
                             rotation_range=360,
                             #zoom_range=.2,
                             fill_mode="constant", 
                             cval=0) 
left_images = left_datagen.fit(train_images,seed=123)

right_datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             #width_shift_range=0.2,
                             #height_shift_range=0.2,
                             #rotation_range=360,
                             zoom_range=.2)
                             #fill_mode="constant",
                             #cval=0)
right_images = right_datagen.fit(train_images,seed=123)

#fit the model
'''
final_model.fit_generator([left_datagen.flow(train_images, train_labels, batch_size=batch_size), 
                    right_datagen.flow(train_images, train_labels,batch_size=batch_size)],
                    steps_per_epoch=train_images.shape[0] // batch_size,
                    epochs=epochs,
                    class_weight=class_weight)
                    #validation_data=(test_images, test_labels))
'''
final_model.fit_generator([left_images, right_images], train_labels,steps_per_epoch=16,epochs=epochs) 
#save model weights and structure
model.save('two_pipe_gray_trained_model.h5')

#evaluate  model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

