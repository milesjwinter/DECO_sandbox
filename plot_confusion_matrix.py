import numpy as np
import matplotlib.pyplot as plt
import itertools


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
    clb = plt.colorbar()
    clb.set_label('Number of Images',labelpad=20, rotation=270)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        #cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        cm = cm.astype('float') / np.sum(cm,axis=0)#cm.sum(axis=1)[np.newaxis,:]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.2f'%(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh and cm[i, j] < 1 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#load data
CM = np.loadtxt('confusion_matrix_gpu.txt')
class_names = ['Worm','Spot','Track']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(CM, classes=class_names,normalize=True,
                      title='Confusion Matrix')
plt.show()
