import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import h5py
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

def get_type(x): #Returns string version of type
    return {
        0 : 'Worm',
        1 : 'Spot',
        2 : 'Track',
    }[x]

#load images and labels
num_classes = 3
index = 34
f = h5py.File('Big_gray_DECO_Image_Database_v7.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']

#make labels into a matrix format
labels = keras.utils.to_categorical(labels, num_classes)

#normalize
images = np.array(images,dtype='float32')
images = images/255.


index0=34
index1=68
test_indices = np.arange(index0,index1,1)
train_indices = np.arange(0,index0,1)
train_indices = np.append(test_indices,np.arange(index1,340,1))
#train_images = images[train_indices]
test_images = images[test_indices]
#train_labels = labels[train_indices]
test_labels = labels[test_indices]

#make labels into a matrix format
#train_labels = keras.utils.to_categorical(train_labels, num_classes)
#labels = keras.utils.to_categorical(labels, num_classes)

#Load trained model
#model = load_model('Best_Models/Big_gray_trained_model.h5')
model = load_model('saved_model_checkpoints/very_best_checkpointed_model.h5')
predictions = model.predict(test_images, batch_size=34, verbose=0)
#predictions = model.predict(images, batch_size=34, verbose=0)

#score = model.evaluate(images, labels, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

'''
# create a grid of 4x4 images
plt.figure(1,figsize=(10,10))
for i in range(9):
	plt.subplot(3,3,i+1-5)
	plt.imshow(test_images[i,:,:,0],cmap=mpl.cm.hot,interpolation="nearest", aspect="auto")
        label = '%s | w %.1f, s %.1f, t %.1f' %(get_type(np.argmax(test_labels[i])),predictions[i,0]*100.,predictions[i,1]*100.,predictions[i,2]*100.)
        plt.title(label,fontsize=12)
        plt.axis('off')
# show the plot
plt.tight_layout()
plt.savefig('plots/grid_predictions.png',pad_inches=0)
plt.show()
'''
print predictions[0]
predictions = predictions*100.
i=3
for i in range(len(predictions)):
    tick_labels = ('Worm','Spot','Track')
    values = np.array([0,1,2])
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax0.imshow(test_images[i,:,:,0],cmap=mpl.cm.hot,interpolation="nearest", aspect="auto")
    ax0.axis('off')
    ax2 = plt.subplot(gs[1])
    #ax2.plot(times,corr_samples[i,:],orientation='horizontal')
    ax2.barh(values, predictions[i],align='center')
    #ax2.step(values, predictions[i],where='mid',orientation='horizontal')
    ax2.set_xlabel('%',fontsize=20)
    #ax2.set_yticks(values)
    #ax2.set_yticklabels(tick_labels)
    ax2.set_xticks([0.,50.,100.])
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.invert_yaxis()
    ax2.text(20,-0.05,'Worm')
    ax2.text(22,0.15,'%.1f %s' %(predictions[i,0],'%'))
    ax2.text(20,0.95,'Spot')
    ax2.text(22,1.15,'%.1f %s' %(predictions[i,1],'%'))
    ax2.text(20,1.95,'Track')
    ax2.text(22,2.15,'%.1f %s' %(predictions[i,2],'%'))
    ax2.minorticks_off()
    plt.tight_layout()
    plt.show()
