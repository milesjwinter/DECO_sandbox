import os
import numpy as np

data = np.loadtxt('bad_iOS_labels.txt',dtype='S')
files = data[:,-1]

for image_path in files:
    print 'removing image: ',image_path
    os.system('rm %s' %(image_path))
