import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import h5py

def get_blobtype(x): #Returns number value assigned to each type
    return {1:'spot',0:'worm',3:'ambig',2:'track',5:'noise'}[x]

def get_numtype(x): #Returns number value assigned to each type
    return {'spot':1,'worm':0,'ambig':3,'track':2,'noise':5}[x]
vec_get_numtype = np.vectorize(get_numtype)

def get_whitespace(image): #checks if image is on the edge of the sensor
    stacked_image = np.sum(image,axis=2)
    return np.sum(stacked_image==255*3)

#If True, visually inspect images and labels before adding to database
visualize = False

#Load training image file paths
path = 'event_database'
files = glob.glob( os.path.join(path, '*.jpeg'))

#load initial blob type classifications
data = np.loadtxt('class_list.txt',dtype='S')
file_number = data[:,0]
sorted_type = data[:,1]
sorted_num = vec_get_numtype(sorted_type)

#create database for holding images and labels
f = h5py.File("DECO_Image_Database.h5","w")
training_images = f.create_dataset("train/train_images", (1,64,64,3), maxshape=(None,64,64,3), dtype="uint8",chunks=True)
training_labels = f.create_dataset("train/train_labels", (1,), maxshape=(None,), dtype="uint8", chunks=True)
training_files = f.create_dataset("train/train_files", (1,), maxshape=(None,), dtype="S%s" % len(file_number[0]), chunks=True)

#Loop through all images for classifying purposes
choices = {"y","n","exit"}
possilities = {"0","1","2","3"}
random_index = np.random.permutation(len(files)).astype(int) 
for i in random_index:
    image = mpimg.imread(files[i])
    if get_whitespace(image)==0:	    
	index = np.where(file_number==files[i][-28:])
        dims = index[0].shape
        if dims[0] == 0:
            print "Skipping Image: No label found"
            print files[i]
        else:
	    if visualize==True:
		plt.figure(figsize=(6,6))
		plt.imshow(image)
		plt.axis('off')
		plt.ion()
		plt.title('%s' % sorted_type[index])
		plt.show()
		print '-----------------------------'
		print files[i]
		print 'Is the classification correct? (exit to quit)'
		blob_type = ""
		class_type = ""
		while True:
		    class_type = raw_input("y/n : ")
		    if class_type in choices:
			break
		    else:
			print "Not an acceptable input, try again"
                if class_type == 'exit':
                    labels = np.array([file_number,sorted_type])
                    np.savetxt('class_list.txt',np.transpose(labels),fmt='%s')
                    raise SystemExit 
		elif class_type == 'n':
		    print "Worm=0, Spot=1, Track=2, Ambiguous=3"
		    while True:
			blob_type = raw_input("Enter Correct Clasification: ")
			if blob_type in possilities:
			    break
			else:
			    print "Not an acceptable input, try again"
                    sorted_type[index] = get_blobtype(int(blob_type))
		else:
		    blob_type = sorted_num[index]
		plt.close()
		if int(blob_type)<3:
		    training_images[-1] = image
		    training_labels[-1] = int(blob_type)
		    training_files[-1] = file_number[index]
		    #resize database unless in final loop
		    if i != random_index[-1]:
			training_images.resize(training_images.shape[0]+1,axis=0)
			training_labels.resize(training_labels.shape[0]+1,axis=0)
			training_files.resize(training_files.shape[0]+1,axis=0)
            else:
                blob_type = sorted_num[index]
                if int(blob_type)<3:
                    training_images[-1] = image
                    training_labels[-1] = int(blob_type)
                    training_files[-1] = file_number[index]
                    #resize database unless in final loop
                    if i != random_index[-1]:
                        training_images.resize(training_images.shape[0]+1,axis=0)
                        training_labels.resize(training_labels.shape[0]+1,axis=0)
                        training_files.resize(training_files.shape[0]+1,axis=0)                
    else:
        print "Skipping Image: on the edge of the camera sesnor"
        print files[i]
f.close()
labels = np.array([file_number,sorted_type])
np.savetxt('class_list.txt',np.transpose(labels),fmt='%s')
