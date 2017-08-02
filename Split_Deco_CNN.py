from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras import metrics
from scipy.ndimage.interpolation import rotate
import numpy as np
import h5py
import time

def discrete_rotation(img):
    #random square-root to square
    #exp_val = 2.0**np.random.uniform(-5.,5.)
    #img = img**exp_val
    #random brightness correction
    #img = img*np.random.uniform(0.5,1.0)
    #random rotation
    angle = np.random.randint(4)*90
    return rotate(img,angle)

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#model/training parameters
batch_size = 22
num_classes = 3
epochs = 300

# input image parameters
img_rows, img_cols = 64, 64
img_channels = 3

#load images and labels
f = h5py.File('Big_gray_DECO_Image_Database_v6.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']

#class_weight = {0:1.0,1:3.2,2:2.5}
class_weight = {0:1.0,1:3.66,2:2.85}

#normalize
images = np.array(images,dtype='float32')/255.

#make labels into a matrix format
labels = keras.utils.to_categorical(labels, num_classes)

start_time = time.time()

# k-fold cross validation
cvscores = []
for i in range(10):
    index0=44*i
    index1=44*(i+1)
    test_indices = np.arange(index0,index1,1)
    train_indices = np.arange(0,index0,1)
    train_indices = np.append(train_indices,np.arange(index1,440,1))
    train_images = images[train_indices]
    test_images = images[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    #normalize max val of training set to one
    #for q in range(len(train_images)):
    #    norm = np.amax(train_images[q,:,:,0])
    #    train_images[q,:,:,0] /= norm
    '''   
    #define model structure
    model = Sequential()
    model.add(Conv2D(16, (16, 16),activation='relu',padding='same',input_shape=(64,64,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (8, 8), activation='relu',padding='same'))
    model.add(Conv2D(32, (5, 5), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (4, 4), activation='relu',padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
		  optimizer=keras.optimizers.Adadelta(),
		  metrics=['accuracy'])
    '''

    #define model structure
    model = Sequential()
    model.add(Conv2D(16, (16, 16),activation='relu',padding='same',input_shape=(64,64,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (8, 8), activation='relu',padding='same'))
    model.add(Conv2D(32, (5, 5), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(32, (4, 4), activation='relu',padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
		  optimizer=keras.optimizers.Adadelta(),
		  metrics=['accuracy'])

    print(model.summary())

    #preprocess images
    datagen = ImageDataGenerator(horizontal_flip=True, 
				 vertical_flip=True,
				 rotation_range=10.,
				 zoom_range=.1,
				 fill_mode="constant", 
				 cval=0,
				 preprocessing_function=discrete_rotation) 

    #fit the model
    datagen.fit(train_images)
    history = model.fit_generator(datagen.flow(train_images, train_labels,
			batch_size=batch_size),
			steps_per_epoch=train_images.shape[0] // batch_size,
			epochs=epochs,
                        validation_data=(test_images, test_labels),
			class_weight=class_weight)
    #save model and history
    model.save('Big_gray_trained_model_%s.h5' %i)
    history_vals = np.array([history.history['acc'],
                             history.history['val_acc'],
                             history.history['loss'],
                             history.history['val_loss']])
    np.savetxt('history_vals_%s.txt' %i,np.transpose(history_vals))

    #evaluate  model
    scores = model.evaluate(test_images, test_labels, verbose=0)
    print("job time: ", time.time() - start_time)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
