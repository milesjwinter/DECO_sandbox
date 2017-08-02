import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.ndimage.interpolation import rotate
import scipy.ndimage as ndi
from PIL import Image

def smooth_labels(y, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.
    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)
    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))
    return y

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='constant', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='constant', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='constant', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def crop_center(img,cropx,cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]


def transform_image(img):
    img = random_rotation(img, 360.)
    #img = random_zoom(img, [1.4,1.5])
    img = random_shift(img, 0.1, 0.1)
    return img
    #print img.shape
    #return crop_center(img,64,64)    

def discrete_rotation(img):
    #random square-root to square
    #exp_val = 2.0**np.random.uniform(-1.,1.)
    #img = img**exp_val
    #random brightness correction
    #img = img*np.random.uniform(0.5,1.0)
    #random rotation
    angle = np.random.randint(4)*90
    return rotate(img,angle)

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

index = 2
num_classes=2 #3
dims = 64

#load images and labels
f = h5py.File('Binary_RGB_DECO_Image_Database_100.h5','r')
images = f['train/train_images']
labels = f['train/train_labels']

angle = np.random.randint(4)*90
rot_img = rotate(images[0],angle)
new_rot_img = crop_center(rot_img,64,64)
print new_rot_img.dtype
plt.figure()
plt.imshow(new_rot_img)
plt.show()

#images = np.array(images,dtype='float32')#/255.
#images = np.sqrt(images)
#for i in range(len(images)):
#    norm = np.amax(images[i,:,:,0])
#    images[i] = images[i]/norm

train_images = images[index:]
train_labels = labels[index:]
test_images = images[:index]
test_labels = labels[:index]
f.close()

print test_images[0,:,:,0]
print test_images[0,:,:,1]
print test_images[0,:,:,2]
#make labels into a matrix format
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)
#smoothed_labels = smooth_labels(train_labels,0.1)

#normalize
#train_images = train_images.astype('float32')
#test_images = test_images.astype('float32')
#train_images /= 255
#test_images /= 255

#train_images = (train_images/255.)**2
#test_images = (test_images/255.)**2
#train_images = np.sqrt(train_images/255.)
#test_images = np.sqrt(test_images/255.)
#preprocess images

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             #width_shift_range=0.08,
                             #height_shift_range=0.08,
                             #samplewise_center=True,
                             #samplewise_std_normalization=True,
                             #rotation_range=360.,
                             #zoom_range=[0.8,1.2],
                             fill_mode="constant",
                             #shear_range=0.5,
                             cval=.5)
                             #preprocessing_function=transform_image)
datagen.fit(train_images,seed=123)
#datagen.fit(train_images)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
                new_img = X_batch[i].astype('uint8')
                print new_img[:,:,0]
                #new_img = crop_center(X_batch[i],64,64)
                #new_img = np.array(new_img,dtype=uint)
                print new_img.shape
                #new_img[0,0]=1.
		plt.subplot(330 + 1 + i)
		#plt.imshow(X_batch[i].reshape(100, 100),cmap=mpl.cm.hot)
                #plt.imshow(new_img[:,:,0],cmap=mpl.cm.hot)
                plt.imshow(new_img)
                plt.axis('off')
	# show the plot
        #plt.tight_layout()
	plt.show()
	break

