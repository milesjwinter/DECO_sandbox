import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from PIL import Image

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
data = np.loadtxt('blob_labels.txt',dtype='S')
xavg = data[:,0].astype(int)
yavg = data[:,1].astype(int)
blob_type = data[:,2]
file_path = data[:,3]
blob_num = vec_get_numtype(blob_type)

for i in range(len(file_path[:3])):
    if int(blob_num[i])<3:
        img = Image.open(file_path[i])
        L_img = Image.open(file_path[i]).convert('L')
        X0, X1, Y0, Y1 = getZoomedBoundingBox(xavg[i],yavg[i])
        print getZoomedBoundingBox(xavg[i],yavg[i])
        y0 = img.size[1]-Y1
        y1 = img.size[1]-Y0
        print X0,' ',X1,' ',y0,' ',y1
        try:
            cropped_img = img.crop((X0,y0,X1,y1))
            new_img = np.array(cropped_img,dtype=float)
            gray_img = np.mean(new_img,axis=2)

            L_crop_img = L_img.crop((X0,y0,X1,y1))
            print new_img.shape
            plt.figure(figsize=(8,4))
            gs = gridspec.GridSpec(1,2)
            ax0 = plt.subplot(gs[0])
            ax0.imshow(gray_img,cmap=mpimg.cm.hot, interpolation="nearest", aspect="auto")
            ax0.axis('off')
            ax1 = plt.subplot(gs[1])
            ax1.imshow(L_crop_img,cmap=mpimg.cm.hot, interpolation="nearest", aspect="auto")
            ax1.axis('off')   
            plt.show()
            print "Adding Image to Database: "
            print file_path[i]
        except TypeError:
            print "Skipping Image: on the edge of the camera sesnor"
            print file_path[i]
