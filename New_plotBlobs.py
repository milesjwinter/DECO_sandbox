#!/usr/bin/env python

################################################################################
# Take a JPG image taken by an Android phone and convert it to a 2D numpy array
# (a bitmap) with RGB values converted to grayscale luminance.
################################################################################
#UPDATE: Now prints at least one line per eventID, events not meeting requirements output "-999" for all variables

try:
    import argparse
    from PIL import Image
    import os
    import sys
    import json

    import numpy as np
    import matplotlib as mpl
    mpl.use('Agg') # for batch mode plotting
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

    def getBoundingBox(self):
        """Get the bounding rectangle of the group."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    def getSquareBoundingBox(self):
        """Get the bounding rectangle, redefined to give it a square aspect
           ratio."""
        xmin, xmax, ymin, ymax = (self.xmin, self.xmax, self.ymin, self.ymax)
        xL = np.abs(xmax - xmin)
        yL = np.abs(ymax - ymin)
        if xL > yL:
            ymin -= 0.5*(xL-yL)
            ymax += 0.5*(xL-yL)
        else:
            xmin -= 0.5*(yL-xL)
            xmax += 0.5*(yL-xL)
        xavg = int((xmin+xmax)/2.)
        yavg = int((ymin+ymax)/2.)
        #return (xmin, xmax, ymin, ymax)
        return (xavg-32,xavg+32,yavg-32,yavg+32)

    def getSubImage(self, image):
        """Given an image, extract the section of the image corresponding to
           the bounding box of the blob group."""
        ny,nx = image.shape
        x0,x1,y0,y1 = bg.getBoundingBox()

        # Account for all the weird row/column magic in the image table...
        i0,i1 = [ny - int(t) for t in (y1,y0)]
        j0,j1 = [int(t) for t in (x0,x1)]

        # Add a pixel buffer around the bounds, and check the ranges
        buf = 1
        i0 = 0 if i0-buf < 0 else i0-buf
        i1 = ny-1 if i1 > ny-1 else i1+buf
        j0 = 0 if j0-buf < 0 else j0-buf
        j1 = nx-1 if j1 > nx-1 else j1+buf

        return image[i0:i1, j0:j1]

    def getRawMoment(self, image, p, q):
        """Calculate the image moment given by
           M_{ij}=\sum_x\sum_y x^p y^q I(x,y)
           where I(x,y) is the image intensity at location x,y."""
        nx,ny = image.shape
        Mpq = 0.
        if p == 0 and q == 0:
            Mpq = np.sum(image)
        else:
            for i in range(0,nx):
                #x = 0.5 + i
                x = i
                for j in range(0,ny):
                   # y = 0.5 + j
                    y = j
                    Mpq += x**p * y**q * image[i,j]
        return Mpq

    def getHuMoments(self, image):
        """Calcuate the Hu moments, which are translation, scale, and
        rotation invariant using skimage.measure.

        Additionally calculate an 8th moment to complete the set as
        described here:
        https://en.wikipedia.org/wiki/Image_moment#Raw_moments
        """
        
        subImage = self.getSubImage(image).transpose()
        moments = measure.moments(subImage)
        centroid_row = moments[0, 1] / moments[0, 0]
        centroid_column = moments[1, 0] / moments[0, 0]
        moments_central = measure.moments_central(image, 
                          centroid_row, centroid_column)
        moments_normalized = measure.moments_normalized(moments_central)
        moments_hu = measure.moments_hu(moments_normalized)

        # Calculate the 8th Hu moment
        mn = moments_normalized
        hu_8 = mn[1][1]*((mn[3][0]+mn[1][2])**2-(mn[0][3]+mn[2][1])**2)\
               - (mn[2][0]-mn[0][2])*(mn[3][0]+mn[1][2])*(mn[0][3]+mn[2][1])
        moments_hu = np.append(moments_hu,hu_8)
        return moments_hu
         
    def getMaxIntensity(self, image):
        """Find the maximum intensity within the blob
           where I(x,y) is the image intensity at location x,y."""
        nx,ny = image.shape
        maxI = 0.
        for i in range(0,nx):
            for j in range(0,ny):
                if image[i,j] > maxI: maxI = image[i,j]
        return maxI

    # Not sure if this works
    def getArea(self, image):
        """Calculate image area from image moment 00."""
        M00 = -999
        subImage = self.getSubImage(image).transpose()
        M00 = self.getRawMoment(subImage, 0, 0)
        return M00

    def getCovariance(self, image):
        """Get the raw moments of the image region inside the bounding box
           defined by this blob group and calculate the image covariance
           matrix."""
        if self.cov == None:
            subImage = self.getSubImage(image).transpose()
            M00 = self.getRawMoment(subImage, 0, 0)
            M10 = self.getRawMoment(subImage, 1, 0)
            M01 = self.getRawMoment(subImage, 0, 1)
            M11 = self.getRawMoment(subImage, 1, 1)
            M20 = self.getRawMoment(subImage, 2, 0)
            M02 = self.getRawMoment(subImage, 0, 2)
            xbar = M10/M00
            ybar = M01/M00
            self.cov = np.vstack([[M20/M00 - xbar*xbar, M11/M00 - xbar*ybar],
                                  [M11/M00 - xbar*ybar, M02/M00 - ybar*ybar]])
        return self.cov

    def getPrincipalMoments(self, image):
        """Return the maximum and minimum eigenvalues of the covariance matrix,
           as well as the angle theta between the maximum eigenvector and the
           x-axis."""
        cov = self.getCovariance(image)
        u20 = cov[0,0]
        u11 = cov[0,1]
        u02 = cov[1,1]

        theta = 0.5 * np.arctan2(2*u11, u20-u02)
        l1 = 0.5*(u20+u02) + 0.5*np.sqrt(4*u11**2 + (u20-u02)**2)
        l2 = 0.5*(u20+u02) - 0.5*np.sqrt(4*u11**2 + (u20-u02)**2)
        return l1, l2, theta

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
    parser.add_argument("image", help="JPG file name")
    parser.add_argument("-H", "--histogram", dest="hist", action="store_true",
                   default=False,
                   help="Produce a histogram of the pixel luminances")
    parser.add_argument("-l", "--log", dest="log", action="store_true",
                   default=False,
                   help="If specified, plot logarithm of pixel luminances")
    parser.add_argument("-x", "--threshold", dest="thresh", type=float,
                   default=None,
                   help="Threshold the pixel map")
    parser.add_argument("-o", "--output", dest="output",
                   default=None,
                   help="Name of image output file (for batch processing)")

    blobGroup = parser.add_argument_group("Blob Identification")
    blobGroup.add_argument("-c", "--contours", dest="contours", type=float,
                           default=40,
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
    
    # Load the image and convert pixel values to grayscale intensities
    filename = args.image
    print 'processing file: %s' % filename
    # img_new = io.imread(filename)
    # img_new_grayscale = rgb2gray(img_new)
    # print('img_new_grayscale = {}'.format(img_new_grayscale[0, :]))
    img = Image.open(filename).convert("L")
    rgb_img = Image.open(filename)
    image = []
    pix = img.load()

    # Stuff image values into a 2D table called "image"
    nx, ny = img.size[0], img.size[1]
    x0, y0, x1, y1 = (0, 0, nx, ny)
    for y in xrange(ny):
        image.append([pix[x, y] for x in xrange(nx)])

    image = np.array(image, dtype=float)

    # eventsWithBlobs = []

    # Calculate contours using the scikit-image marching squares algorithm,
    # store as Blobs, and group the Blobs into associated clusters
    blobs = findBlobs(image, threshold=args.contours, minArea=args.area, maxArea=args.max_area)
    groups = groupBlobs(blobs, maxDist=args.distance)

    # Apply a threshold to the image pixel values
    if args.thresh is not None:
        image[image < args.thresh] = 0.
    
    '''
    # Draw the log(intensity) of the pixel values
    if args.log:
        image = np.log10(1. + image)

    fig, ax = plt.subplots(figsize=(8,7))
    #im = ax.imshow(image, cmap='viridis', interpolation="nearest", aspect="auto",extent=[x0, x1, y0, y1])
    im = ax.imshow(image, cmap=mpl.cm.hot, interpolation="nearest", aspect="auto",
                   extent=[x0, x1, y0, y1])
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    title = os.path.basename(filename)
    ax.set_title(title)
    ax.grid(color='white')
    cb = fig.colorbar(im, orientation="horizontal",
                           shrink=0.8,
                           fraction=0.05,
                           pad=0.1)
    label = "log(luminance)" if args.log else "pixel luminance"
    cb.set_label(label)

    # Plot any identified blobs
    # Show highlighted blobs on the main plot
    for blob in blobs:
        ax.plot(blob.x, blob.y, linewidth=2, color="#00dd00")
    if args.output:
        fig.savefig(args.output)
    '''

    # Zoom in on grouped blobs in separate figures
    title = os.path.basename(filename)
    numGroups = len(groups)
    for i, bg in enumerate(groups):
        X0, X1, Y0, Y1 = bg.getSquareBoundingBox()
        l1, l2, theta = bg.getPrincipalMoments(image)
        eccentricity = np.sqrt( l1**2 - l2**2 ) / l1
        maxI = bg.getMaxIntensity(image)
        hu_moments = bg.getHuMoments(image)
        #fig = plt.figure(figsize=(6.4,6.4))
        #axg = fig.add_subplot(111)
        #im = axg.imshow(image, cmap='viridis',interpolation="nearest", aspect="auto",extent=[x0, x1, y0, y1])
        #im = axg.imshow(image, cmap=mpl.cm.hot,interpolation="nearest", aspect="auto",extent=[x0, x1, y0, y1])
        
        fig = plt.figure(figsize=(6.4,6.4))
        axg = fig.add_subplot(111)
        #im = axg.imshow(rgb_img,interpolation="nearest", aspect="equal",extent=[x0, x1, y0, y1])
        im = axg.imshow(image, cmap=mpl.cm.hot,interpolation="nearest", aspect="auto",extent=[x0, x1, y0, y1])
        axg.set_xlim([X0-5, X1+5])
        axg.set_ylim([Y0-5, Y1+5])
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        if args.output:
            name, ext = os.path.splitext(args.output)
            image_path = 'event_database/{}_cluster{:02d}.jpeg'.format(name, i+1)
            fig.savefig(image_path,pad_inches=0,dpi=10)
        else:
            image_path = 'None'
        
        '''
        fig = plt.figure(figsize=(6,6))
        axg = fig.add_subplot(111)
       
        im = axg.imshow(image, cmap=mpl.cm.hot,
                        interpolation="nearest", aspect="auto",
                        extent=[x0, x1, y0, y1])
        
        im = axg.imshow(rgb_img,
                        interpolation="nearest", aspect="auto",
                        extent=[x0, x1, y0, y1])
    	# Get number for blobs in this grouping
        numBlobs = len(bg.blobs)
        for blob_idx, blob in enumerate(bg.blobs, start=1):
            axg.plot(blob.x, blob.y, linewidth=2, color="#00dd00")
	    area = blob.area
	    blog_length = blob.length()
        aOverMaj = area / l2
        xCent = blob.xc
        yCent = blob.yc
        if args.output:
            name, ext = os.path.splitext(args.output)
            image_path = 'sortedEvents/{}_cluster{:02d}.jpeg'.format(name, i+1)
        else:
            image_path = 'None'

        axg.set_xlim([X0-5, X1+5])
        axg.set_ylim([Y0-5, Y1+5])
        #plt.axis('off')
        #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        axg.set_xlabel('X (pixels)')
        axg.set_ylabel('Y (pixels)')
        axg.set_title("%s: Cluster %d" % (title, i+1))
        axg.grid(color='white')

        #if args.output: fig.savefig(image_path,pad_inches=0,dpi=10)
        if args.output: fig.savefig(image_path)
        '''
        # print "evt: %s totBG: %i totBlobs: %i bg#: %i 'blob'#: %i majA: %g minA: %g theta: %g ecc: %f area: %f aOverMaj: %f blog_length: %f xCent: %i yCent: %i maxI: %f file: %s image: %s" % \
        #   (str( os.path.basename(filename) )[:-4], numGroups, numBlobs, i+1, blob_idx, l1, l2, theta*180./np.pi, \
        #    eccentricity, area, aOverMaj, blog_length, xCent, yCent, maxI, filename, image_path )
        '''
        data_dict = {'evt': str( os.path.basename(filename) )[:-4],
                     'totBG': numGroups,
                     'totBlobs': numBlobs,
                     'bg#': i+1,
                     'blob#': blob_idx,
                     'majA': l1,
                     'minA': l2,
                     'theta': theta*180./np.pi,
                     'ecc': eccentricity,
                     'area': area,
                     'aOverMaj': aOverMaj,
                     'blog_length': blog_length,
                     'xCent': xCent,
                     'yCent': yCent,
                     'maxI': maxI,
                     'file': filename,
                     'image': image_path
                     }
        # Write data_dict to disk
        with open('data-dict.json', 'w') as outfile:
            json.dump(data_dict, outfile)
        '''

    # Plot the luminance histogram if desired
    if args.hist:
        fig2 = plt.figure(figsize=(7,4.5))
        ax = fig2.add_subplot(111)
        if args.log:
            image = 10**image
            label = 'Luminance'
        ax.hist(image.flatten(), bins=100, log=args.log)
        ax.set_xlabel(label)
        ax.set_ylabel('Counts')
        # ax.set_title(filename)
        fig2.tight_layout()
        if args.output:
            fig2.savefig(args.output + "_hist")

    else:
        plt.show()
    

