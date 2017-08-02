import matplotlib.pyplot as plt
import numpy as np
import h5py

f = h5py.File('DCGAN_DECO_Image_Database_64.h5','r')
#f = h5py.File('Binary_DECO_Image_Database.h5','r')
labels = f['train/train_labels']
print labels.shape
labels = np.array(labels)
index = 253
worm = np.sum(labels[index:]==0,dtype=float)
spot = np.sum(labels[index:]==1,dtype=float)
track = np.sum(labels[index:]==2,dtype=float)
print 'worm ',worm, '  weight= ',worm/worm
print 'spot ',spot,'  weight= ',worm/spot
print 'track ',track,'  weight= ',worm/track

plt.figure()
plt.hist(labels,bins=3)
plt.show()

for i in range(10):
    #split data into testing and training sets
    index0=34*i
    index1=34*(i+1)
    test_indices = np.arange(index0,index1,1)
    train_indices = np.arange(0,index0,1)
    train_indices = np.append(test_indices,np.arange(index1,340,1))
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    test_comp = np.zeros(3)
    for q in range(3):
        test_comp[q] = np.sum(test_labels==q)
    print test_comp
