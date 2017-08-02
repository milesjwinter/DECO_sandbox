import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from PIL import Image
import h5py

def get_blobtype(x): #Returns number value assigned to each type
    return {1:'spot',0:'worm',3:'ambig',2:'track',5:'noise',6:'edge',7:'empty'}[x]

def get_numtype(x): #Returns number value assigned to each type
    return {'spot':1,'worm':0,'ambig':3,'track':2,'noise':5,'edge':6,'empty':7}[x]
vec_get_numtype = np.vectorize(get_numtype)

def get_whitespace(image): #checks if image is on the edge of the sensor
    stacked_image = np.sum(image,axis=2)
    return np.sum(stacked_image==255*3)

def getZoomedBoundingBox(xavg, yavg, size=32):
    return (xavg-size, xavg+size, yavg-size, yavg+size)

#load initial blob type classifications
data = np.loadtxt('training_blobs.txt',dtype='S')
xavg = data[:,0].astype(int)
yavg = data[:,1].astype(int)
blob_type = data[:,2]
file_path = data[:,3]
blob_num = vec_get_numtype(blob_type)

#create database for holding images and labels
f = h5py.File("Big_DECO_Image_Database.h5","w")
training_images = f.create_dataset("train/train_images", (1,64,64,3), maxshape=(None,64,64,3), dtype="uint8",chunks=True)
training_labels = f.create_dataset("train/train_labels", (1,), maxshape=(None,), dtype="uint8", chunks=True)
#training_files = f.create_dataset("train/train_files", (1,), maxshape=(None,), dtype="S%s" % len(file_number[0]), chunks=True)

#Loop through all images for classifying purposes
random_index = np.random.permutation(len(xavg)).astype(int) 
for i in random_index:	  
    if int(blob_num[i])<3:
        img = Image.open(file_path[i])
        X0, X1, Y0, Y1 = getZoomedBoundingBox(xavg[i],yavg[i])
        y0 = img.size[1]-Y1
        y1 = img.size[1]-Y0
        x0 = X0
        x1 = X1
        try: 
            cropped_img = img.crop((x0,y0,x1,y1))
            #plt.figure(figsize=(6,6))
            #plt.imshow(cropped_img)
            #plt.show()
	    training_images[-1] = np.array(cropped_img,dtype=float)
	    training_labels[-1] = int(blob_num[i])
	    #training_files[-1] = file_number[index]
            print "Adding Image to Database: "
            print file_path[i]
	    #resize database unless in final loop
	    if i != random_index[-1]:
		training_images.resize(training_images.shape[0]+1,axis=0)
		training_labels.resize(training_labels.shape[0]+1,axis=0)
		#training_files.resize(training_files.shape[0]+1,axis=0)    
        except TypeError:
            print "Skipping Image: on the edge of the camera sensor"
            print file_path[i]
f.close()
