#!/usr/bin/env python
# Justin Vandenbroucke
# Created Jun 4 2015
# Zoom in on the brightest pixel in a DECO image.
# Parts are taken from Segev's code plotBlobs.py.
# Usage:
# Option 1: ./zoom.py, to loop over images in directories.
# Option 2: ./zoom.py <date string> <event id> to process a particular event.
# Example: ./zoom.py 2015.01.8 196782816
# Zoomed out is ~2000 x ~2000 pixels, depending on camera resolution.
# Zoomed in is 40 x 40 (one part in ~2500 by area of the zoomed out).
# Zoomed medium is 300 x 300 (factor 7 in either direction)

# Import modules
# Use Agg backend to generate plots without window appearing
# (http://matplotlib.org/faq/howto_faq.html)
import sys, os, glob, time, matplotlib
matplotlib.use('Agg')
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy, shutil

# Save three zoom levels for each image: full, zoom (50 pixels square), and medium (300 pixels square)
# Set parameters
halfWidthZoom = 32		# the full zoomed in image is a square with side length twice this number (in pixels)
halfWidthMedium = 150
publicDirectory = '/net/deco/deco_plots/zoom/'

# Function to process a single event
def zoomIn(infilename,outfilebase):
	# Load the image and convert pixel values to grayscale intensities
	#img = Image.open(infilename).convert("L")
	try:
		img = Image.open(infilename).convert("L")
                rgb_img = Image.open(infilename)
	except IOError,e:
		print e
		print "ERROR: corrupt event!  ", infilename
		if os.path.exists(lockfilename):
			os.remove(lockfilename)
		return IOError

	# Explanation of "L":
	# When converting from a colour image to black and white, the library uses the ITU-R 601-2 luma transform:
	# L = R * 299/1000 + G * 587/1000 + B * 114/1000
	# See http://effbot.org/imagingbook/image.htm
	# Also http://en.wikipedia.org/wiki/Rec._601

	image = []
	pix = img.load()

	# Stuff image values into a 2D table called "image"
	nx = img.size[0]
	ny = img.size[1]
	x0, y0, x1, y1 = (0, 0, nx, ny)
	for y in xrange(ny):
		image.append([pix[x, y] for x in xrange(nx)])
	image = np.array(image, dtype=float)
	maxLuminance = np.amax(image)
	maxIndex = np.argmax(image)
	print "Event max luminance: %.1f" % (maxLuminance)
	(maxY,maxX) = np.unravel_index(image.argmax(), image.shape)

	# Plot full image
	X0 = maxX-halfWidthZoom
	Y0 = (ny-maxY)-halfWidthZoom
	X1 = maxX+halfWidthZoom
	Y1 = (ny-maxY)+halfWidthZoom
	mpl.rc("font", family="serif", size=14)
	title = "DECO Event"
	fig1 = plt.figure(1, figsize=(6.4,6.4))
	ax = fig1.add_subplot(111)
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	#im = ax.imshow(image, cmap=mpl.cm.hot, interpolation="nearest", aspect="auto",extent=[x0, x1, y0, y1])
        #im = ax.imshow(rgb_img,extent=[x0, x1, y0, y1])
	#ax.set_xlabel("X (pixels)")
	#ax.set_ylabel("Y (pixels)")
	#ax.set_title(title)
	#cb = fig1.colorbar(im, orientation="horizontal",shrink=0.8,fraction=0.05,pad=0.1)
	label = "Luminance"
	#cb.set_label(label)
	#fig1.savefig(outfilebase + '_full.png',format='png')
	
	# Zoom in
	ax.set_xlim([X0, X1])
	ax.set_ylim([Y0, Y1])
        pix  = np.array(rgb_img)
        print pix.shape
        print X0,' ',X1,' ',Y0,' ',Y1
        #cropped_img = pix[X0:X1,Y0:Y1,:]
        X0 = maxX-halfWidthZoom
        #Y0 = (ny-maxY)-halfWidthZoom
        Y0 = (maxY)-halfWidthZoom
        X1 = maxX+halfWidthZoom
        #Y1 = (ny-maxY)+halfWidthZoom
        Y1 = (maxY)+halfWidthZoom
        print X0,' ',X1,' ',Y0,' ',Y1
        rgb_img = Image.open(infilename)
        cropped_img = rgb_img.crop((X0,Y0,X1,Y1))
        cropped_img.save("img2.jpg")
        im = ax.imshow(cropped_img)
	zoomfilename  = outfilebase + '_zoom.png'
	fig1.savefig(zoomfilename,format='png',pad_inches=0,dpi=10)
	#print "dateString: ", dateString
	#publicSubdir = publicDirectory + '/' + dateString + '/'
	#print "publicSubdir:" , publicSubdir
	#if not os.path.exists(publicSubdir):
	#	os.mkdir(publicSubdir)
	#shutil.copy(zoomfilename,publicSubdir)
 
	# Zoom in medium
	X0 = maxX-halfWidthMedium
	Y0 = (ny-maxY)-halfWidthMedium
	X1 = maxX+halfWidthMedium
	Y1 = (ny-maxY)+halfWidthMedium
	ax.set_xlim([X0, X1])
	ax.set_ylim([Y0, Y1])
	#fig1.savefig(outfilebase + '_medium.png',format='png')
	print outfilebase
	plt.clf()
        '''
	# Save output data
	basename = os.path.basename(infilename)
	parts = os.path.split(infilename)
	basename = parts[-1]
	indirname = parts[-2]
	parts = os.path.split(indirname)
	subdirname = parts[-1]
	outdirname = '/data/user/justin/deco_analysis/zoom/' + subdirname
	if not os.path.exists(outdirname):
		os.mkdir(outdirname)
	outfilename = outdirname + '/' + basename
	parts = os.path.splitext(outfilename)
	outfilename = parts[0] + '.npz'
	numpy.savez(outfilename,maxLuminance,maxX,maxY)
	print outfilename

	# Stop timer
	if os.path.exists(lockfilename):
		os.remove(lockfilename)
	stopTime = time.time()
	print "Time to process event: %.2f sec." % (stopTime-startTime)
        '''
	return False

if __name__ == "__main__":
        infilename = './new_CNN_images/100076502.jpg'
        outfilebase = 'sortedEvents/track_zoom/100076502'
        zoomIn(infilename,outfilebase)
        rgb_img = Image.open(infilename)
        x0, x1, y0, y1 = (2249, 2341, 1895, 1987)
        cropped_img = rgb_img.crop((y0,x0,y1,x1))
        cropped_img.save("sortedEvents/track_zoom/test_100076502.jpg")
        '''
	# This is the path the automated processing follows
	if len(sys.argv) > 1:
		#dateString = sys.argv[1]
		#eventID = int(sys.argv[2])
		infilename = sys.argv[1]
		outfilebase = sys.argv[2]
		parts = infilename.split("/")
		#print "parts: " , parts
		dateString = parts[5]
		print "dateString: ", dateString
		if len(sys.argv)>3:	# optional 4th argument is event id (do not provide for DECO v2 and DECO B)
			eventID = int(sys.argv[3])
		#dateString = ''
		zoomIn(infilename,outfilebase,eventID,dateString)
		sys.exit(1)

	# Standard syntax 
	list = glob.glob('/data/user/justin/deco_data/????.??.??')
	for directory in list:
		parts = directory.split("/")
		dateString = parts[-1]
		print "dateString: ", dateString
		processOneDay(dateString)
        '''
