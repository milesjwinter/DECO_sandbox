import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.1f'%(100.*cm[i, j]/1003.),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Miles Label')
    plt.ylabel('Raaha label')

def get_numtype(x): #Returns number value assigned to each type
    return {'worm':0,'spot':1,'track':2,'ambig':3,'noise':4,'edge':5,'empty':6}[x]
vec_get_numtype = np.vectorize(get_numtype)



#load data
miles_list = np.loadtxt('miles_track_labels.txt',dtype='S')
raaha_list = np.loadtxt('raaha_track_labels.txt',dtype='S')

#get image names
miles_paths = np.array([r.split('/')[-1].split('.')[:-1] for r in miles_list[:,-1]],dtype=int)
raaha_paths = np.array([r.split('/')[-1].split('.')[:-1] for r in raaha_list[:,-1]],dtype=int)

#get image labels
miles_labels = vec_get_numtype(miles_list[:,2])
raaha_labels = vec_get_numtype(raaha_list[:,2])

#sort numerically by image name
miles_indices = np.argsort(miles_paths,axis=0)
miles_sorted_paths = miles_paths[miles_indices]
miles_sorted_labels = miles_labels[miles_indices]

raaha_indices = np.argsort(raaha_paths,axis=0)
raaha_sorted_paths = raaha_paths[raaha_indices]
raaha_sorted_labels = raaha_labels[raaha_indices]

#compute confusion matrix
CM = confusion_matrix(raaha_sorted_labels,miles_sorted_labels)
CM = np.array(CM,dtype='float32')
#print(CM)

#CM = np.loadtxt('miles_raaha_cm.txt')

#plot Confusion Matrix
class_names = ['Worm','Spot','Track','Ambig','Noise','Edge','Empty']
plt.figure(figsize=(6,6))
plot_confusion_matrix(CM, classes=class_names,normalize=False,title='Confusion Matrix')
plt.tight_layout()
plt.show()


