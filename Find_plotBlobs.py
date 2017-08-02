#!/usr/bin/env python

try:
    import argparse, os, sys, glob, warnings
    from PIL import Image

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from skimage import measure, io
    from skimage.color import rgb2gray

except ImportError,e:
    print e
    raise SystemExit


class Blob(object):
    """Class that defines a 'blob' in an image: the contour of a set of pixels
       with values above a given threshold."""

    def __init__(self, x, y):
        """Define a counter by its contour lines (an list of points in the xy
           plane), the contour centroid, and its enclosed area."""
        self.x = np.array(x)
        self.y = np.array(y)
        self.xc = np.mean(x)
        self.yc = np.mean(y)
        self._length = x.shape[0]

        # Find the area inside the contour
        area = 0
        for i in range(self._length):
            area += 0.5*(y[i]+y[i-1])*(x[i]-x[i-1])
        self.area = area

    def length(self):
        """ Find the approx length of the blob from the max points of the
            contour. """
        xMin = self.x.min()
        xMax = self.x.max()
        yMin = self.y.min()
        yMax = self.y.max()
        len_ = np.sqrt( (xMin - xMax)**2 + (yMin - yMax)**2 )
        return len_

    def distance(self, blob):
        """Calculate the distance between the centroid of this blob contour and
           another one in the xy plane."""
        return np.sqrt((self.xc - blob.xc)**2 + (self.yc-blob.yc)**2)


class BlobGroup(object):
    """A list of blobs that is grouped or associated in some way, i.e., if
       their contour centroids are relatively close together."""

    def __init__(self):
        """Initialize a list of stored blobs and the bounding rectangle which
        defines the group."""
        self.blobs = []
        self.xmin =  1e10
        self.xmax = -1e10
        self.ymin =  1e10
        self.ymax = -1e10

    def addBlob(self, blob):
        """Add a blob to the group and enlarge the bounding rectangle of the
           group."""
        self.blobs.append(blob)
        self.xmin = min(self.xmin, blob.x.min())
        self.xmax = max(self.xmax, blob.x.max())
        self.ymin = min(self.ymin, blob.y.min())
        self.ymax = max(self.ymax, blob.y.max())
        self.cov  = None

    def getBoundingBoxCenter(self):
        """Get the center pixel (x,y) of the group"""
        xmin, xmax, ymin, ymax = (self.xmin, self.xmax, self.ymin, self.ymax)
        xavg = int((xmin+xmax)/2)
        yavg = int((ymin+ymax)/2)
        return (xavg, yavg)
 
    def getZoomedBoundingBox(self, size=32):
        xavg, yavg = bg.getBoundingBoxCenter()
        return (xavg-size, xavg+size, yavg-size, yavg+size)

def findBlobs(image, threshold, minArea=2., maxArea=1000.):
    """Pass through an image and find a set of blobs/contours above a set
       threshold value.  The minArea parameter is used to exclude blobs with an
       area below this value."""
    blobs = []
    ny, nx = image.shape

    # Find contours using the Marching Squares algorithm in the scikit package.
    contours = measure.find_contours(image, threshold)
    for contour in contours:
        x = contour[:,1]
        y = ny - contour[:,0]
        blob = Blob(x, y)
        if blob.area >= minArea and blob.area <= maxArea:
            blobs.append(blob)
    return blobs

def groupBlobs(blobs, maxDist):
    """Given a list of blobs, group them by distance between the centroids of
       any two blobs.  If the centroids are more distant than maxDist, create a
       new blob group."""
    n = len(blobs)
    groups = []
    if n >= 1:
        # Single-pass clustering algorithm: make the first blob the nucleus of
        # a blob group.  Then loop through each blob and add either add it to
        # this group (depending on the distance measure) or make it the
        # nucleus of a new blob group
        bg = BlobGroup()
        bg.addBlob(blobs[0])
        groups.append(bg)

        for i in range(1, n):
            bi = blobs[i]
            isGrouped = False
            for group in groups:
                # Calculate distance measure for a blob and a blob group:
                # blob just has to be < maxDist from any other blob in the group
                for bj in group.blobs:
                    if bi.distance(bj) < maxDist:
                        group.addBlob(bi)
                        isGrouped = True
                        break
            if not isGrouped:
                bg = BlobGroup()
                bg.addBlob(bi)
                groups.append(bg)

    return groups

if __name__ == '__main__':
    
    # Get an image file name from the command line
    parser = argparse.ArgumentParser(description="Histogram luminance of a JPG image")
    #parser.add_argument("image", help="JPG file name")
    parser.add_argument("image_dir", help="Image Directory")
    parser.add_argument("-x", "--threshold", dest="thresh", type=float,
                   default=None,
                   help="Threshold the pixel map")
    parser.add_argument("-o", "--output", dest="output",
                   default="blob_locations",
                   help="Name of output text file for label storage")

    blobGroup = parser.add_argument_group("Blob Identification")
    blobGroup.add_argument("-c", "--contours", dest="contours", type=float,
                           default=30,
                           help="Identify blob contours above some threshold")
    blobGroup.add_argument("-d", "--distance", dest="distance", type=float,
                           default=64.,
                           help="Group blobs within some distance of each other")
    blobGroup.add_argument("-a", "--min-area", dest="area", type=float,
                           default=10.,
                           help="Remove blobs below some minimum area")
    blobGroup.add_argument("-m", "--max-area", dest="max_area", type=float,
                           default=1000.,
                           help="Remove blobs above some maximum area")

    args = parser.parse_args()
    

    #open labeling file
    label_file = args.output+'.txt'
    existing_labels = ''
    f = open(label_file,'a')
    print 'Opening labeling file: '+label_file
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prev_labels = np.loadtxt(label_file,dtype='S')
        if prev_labels.shape == (0,):
            existing_labels = []
        elif prev_labels.shape ==(3,):
            existing_labels = prev_labels[2]
        else:
            existing_labels = prev_labels[:,2]
        del prev_labels

    #Loop through all images in the input directory
    indir = args.image_dir
    for filename in glob.iglob('%s/*.jpg' % indir):
        #Check if this file has been processed previously
        if filename not in existing_labels:
            # Load the image and convert pixel values to grayscale intensities
            print 'processing file: %s' % filename
            img = Image.open(filename).convert("L")
 
            # Stuff image values into a 2D table called "image"
            x0, y0, x1, y1 = (0, 0, img.size[0], img.size[1])
            image = np.array(img, dtype=float)
    
            # Calculate contours using the scikit-image marching squares algorithm,
            # store as Blobs, and group the Blobs into associated clusters
            blobs = findBlobs(image, threshold=args.contours, minArea=args.area, maxArea=args.max_area)
            groups = groupBlobs(blobs, maxDist=args.distance)

            # Apply a threshold to the image pixel values
            if args.thresh is not None:
                image[image < args.thresh] = 0.

            # Zoom in on grouped blobs in separate figures
            title = os.path.basename(filename)
            possilities = {"spot","worm","track","noise","ambig","edge","exit"}
	    for i, bg in enumerate(groups):
                X0, X1, Y0, Y1 = bg.getZoomedBoundingBox()
                Xavg, Yavg = bg.getBoundingBoxCenter()

                f.write("{}\t{}\t{}".format(Xavg, Yavg, filename))
                f.write("\n") 
        else:
            print 'Image %s was processed previously' % filename.split('/')[-1]
    f.close() 
