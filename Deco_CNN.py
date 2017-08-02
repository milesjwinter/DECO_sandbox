from __future__ import print_function
import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.optimizers import SGD

batch_size = 25
num_classes = 3
epochs = 200
index = 62

# input image parameters
img_rows, img_cols = 64, 64
img_channels = 3

#load images and labels
f = h5py.File('Big_DECO_Image_Database.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']
train_images = images[:-index]
train_labels = labels[:-index]
test_images = images[-index:]
test_labels = labels[-index:]
f.close()
del images
del labels

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
model.add(Conv2D(32, (6, 6),activation='relu',input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
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

#define model structure
model = Sequential()
model.add(Conv2D(64, (16, 16),activation='relu',input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(4, 4)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=sgd,
              metrics=['accuracy'])

'''
#define model structure
model = Sequential()
model.add(Conv2D(64, kernel_size=(4, 4),activation='relu',input_shape=train_images.shape[1:]))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
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
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_images.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
'''
print(model.summary())

#preprocess images
datagen = ImageDataGenerator(rotation_range=90, 
                             horizontal_flip=True, 
                             vertical_flip=True, 
                             fill_mode="constant", 
                             cval=0) 

#fit the model
datagen.fit(train_images)
model.fit_generator(datagen.flow(train_images, train_labels,
                    batch_size=batch_size),
                    steps_per_epoch=train_images.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(test_images, test_labels))

#save model weights and structure
model.save('Big_trained_model.h5')

#evaluate  model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
