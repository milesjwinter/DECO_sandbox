from PIL import Image, ImageChops
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import toimage
from scipy.ndimage.interpolation import rotate, shift
def trim(im, border):
    bg = Image.new(im.mode, im.size, border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def check_whitespace(image):
    stacked_image = np.sum(image,axis=2)
    return np.sum(stacked_image==255*3)

files = ['event_database/604_101787241_cluster01.jpeg','event_database/605_101787000_cluster01.jpeg']
for infile in files:
    image = mpimg.imread(infile)
    print check_whitespace(image)
    
    plt.figure()
    plt.imshow(rotate(image,45))
    plt.show()
