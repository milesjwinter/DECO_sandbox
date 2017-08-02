from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from scipy.ndimage.interpolation import rotate
import numpy as np
import h5py
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
def discrete_rotation(img):
    angle = np.random.randint(4)*90
    return rotate(img,angle)

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#model/training parameters
batch_size = 17
num_classes = 3
epochs = 250

# input image parameters
img_rows, img_cols = 64, 64
img_channels = 3

#load images and labels
f = h5py.File('Big_gray_DECO_Image_Database_100.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']

class_weight = {0:1.0,1:3.2,2:2.5}
#normalize
images = np.array(images,dtype='float32')
images = images/255.

#make labels into a matrix format
labels = keras.utils.to_categorical(labels, num_classes)

# k-fold cross validation
cvscores = []
cvloss = []
#conf_mat = np.zeros((3,3,1))
conf_mat = np.zeros((3,3))
history = np.zeros((300,4,1))
for i in range(1):
    #split data into testing and training sets

    index0=109*i
    index1=109*(i+1)
    test_indices = np.arange(index0,index1,1)
    train_indices = np.arange(0,index0,1)
    train_indices = np.append(train_indices,np.arange(index1,440,1))
    train_images = images[train_indices]
    test_images = images[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    #load history from file
    #history[:,:,i] = np.loadtxt('Best_Models/history_vals_%s.txt' %i)  
    test_comp = np.zeros(3)
    for q in range(3):
        test_comp[q] = np.sum(np.argmax(test_labels,axis=1)==q)
    print(test_comp)
    test_comp = test_comp/88.
    print(test_comp)
    #Load model weights and structure
    model = load_model('best_gpu_models/Early_Stop_gray_trained_model.h5')
    #model = load_model('best_gpu_models/best_checkpointed_model.h5')
    ''' 
    datagen = ImageDataGenerator(horizontal_flip=True
				 vertical_flip=True,
				 rotation_range=10.,
				 zoom_range=.1,
				 fill_mode="constant",
				 cval=0.,
				 preprocessing_function=discrete_rotation)
    
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

    #datagen.fit(test_images)
    scores = model.evaluate_generator(datagen.flow(test_images, test_labels,batch_size=109),
                        steps=8)
    predictions = model.predict_generator(datagen.flow(test_images, test_labels, batch_size=109, shuffle=False),
                        steps=8)
        
    #layer_name = 'max_pooling2d_3'
    layer_name = ['conv2d_1','max_pooling2d_1','conv2d_2','conv2d_3','max_pooling2d_2' ,'conv2d_4','conv2d_5','max_pooling2d_3']
    layer_size = np.array([16,16,16,32,32,32,128,128])    
    
    for k in range(len(layer_name)):
        print('Getting output from layer: ',layer_name[k])
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name[k]).output)
        intermediate_output = intermediate_layer_model.predict(test_images)
        
        for q in range(layer_size[k]):
            int_img = intermediate_output[5,:,:,q]
            #test_img = test_images[5,:,:,0]
        
            fig = plt.figure(figsize=(3,3))
            axg = fig.add_subplot(111)
            plt.imshow(int_img,cmap=mpl.cm.hot,interpolation="nearest", aspect="auto")
            plt.axis('off')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            fig.savefig('cnn_layer_images/worm2/%s_image_%s.png' %(layer_name[k],q),pad_inches=0)
            #plt.title('i=%s'%q)
            #plt.show()
        
    
    '''
    predictions = model.predict(test_images, batch_size=109, verbose=0)
    print(predictions)
    '''
    plt.figure(figsize=(4,4))
    plt.imshow(test_images[5,:,:,0],cmap=mpl.cm.hot,interpolation="nearest", aspect="auto")
    label = 'Worm %.1f, Spot %.1f, Track %.1f' %(predictions[5,0]*100.,predictions[5,1]*100.,predictions[5,2]*100.)
    plt.title(label,fontsize=10)
    plt.axis('off')   
    plt.show() 
    '''
    #evaluate  model
    scores = model.evaluate(test_images, test_labels, verbose=0)
    
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print(test_comp)
    cvscores.append(scores[1] * 100)
    cvloss.append(scores[0] * 100)

    new_labels = np.array([test_labels]*4)
    new_labels = new_labels.reshape(len(test_labels)*4,3)  
    #CM = confusion_matrix(np.argmax(new_labels,axis=1),np.argmax(predictions,axis=1))  
    CM = confusion_matrix(np.argmax(test_labels,axis=1),np.argmax(predictions,axis=1))
    CM = np.array(CM,dtype='float32')
    print(CM)
    #for q in range(3):
    #    CM[q,:] = CM[q,:]/test_comp[q]*100.
    #conf_mat[:,:] = CM
    #print(CM) 
    

#print("acc %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#print("loss %.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))
#avg_conf_mat = np.mean(conf_mat,axis=2)
np.savetxt('confusion_matrix_gpu.txt',CM)
print(conf_mat)

'''
avg_hist = np.mean(history, axis=2)
hist_acc = avg_hist[:,0]
hist_val_acc = avg_hist[:,1]
hist_loss = avg_hist[:,2]
hist_val_loss = avg_hist[:,3]
# summarize history for accuracy
fig, ax = plt.subplots(figsize=(10,5))
plt.subplot(121)
plt.plot(hist_acc*100.)
plt.plot(hist_val_acc*100.)
plt.title('Model Accuracy',fontsize=18)
plt.ylabel('Accuracy (%)',fontsize=15)
plt.xlabel('Epoch',fontsize=15)
plt.legend(['Training', 'Testing'], loc='upper left')
plt.subplot(122)
# summarize history for loss
plt.plot(hist_loss)
plt.plot(hist_val_loss)
plt.title('Model Loss',fontsize=18)
plt.ylabel('Loss',fontsize=15)
plt.xlabel('Epoch',fontsize=15)
plt.legend(['Training', 'Testing'], loc='upper right')
plt.tight_layout()
plt.show()
'''
