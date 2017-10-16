#!/usr/bin/env python
import sys
import numpy as np
from   PIL import Image
from   keras.preprocessing.image import ImageDataGenerator

# allowed divisors
divisors = [2, 3, 4, 5, 8, 15, 16, 20, 30, 60, 75, 100, 200]


def read(fname):
    """ read the specified file and return a numpy 2d array of data points as u16
        fname : name of file to read
        the last row and column overlap are trimmed off
        returns 2D array of 16 bit integers
    """
    try:
        # read the file as a 1d array
        m = np.fromfile(fname, dtype=">H")
    except Exception as ex:
        print(ex)
        print("invalid input file")
        sys.exit(1)

    # fill voids which are == 32768
    # fill with min value
    # should probably do an interpolation of surrounding non-void points
    imin = np.min(m)
    m[m >= 32768] = imin

    # get size of square dimensions
    s = int(np.sqrt(m.shape[0]))

    # reshape to a square 2d array
    m = np.reshape(m, (s, s))

    # trim off overlap of last row/col
    s -= 1
    m = m[0:s:1, 0:s:1]

    x = m[m == imin]
    return m


def toimage(m, fname=None, size=(512, 512)):
    """ create a image of the elevation matrix
        m     : 2d matrix of height postings
        fname : optional, if given, the resulting image is written to the file
                the type suffix determines the save image type
        size  : optional, tuple specifying resulting image size
        returns the image object
    """
    # normalize to 0..255
    m = normalize(m, 256)

    # create an image
    img = Image.fromarray(m)

    # convert to RGB format so that either .png or .jpg works
    img = img.convert("RGB")

    # resize as specified
    img = img.resize(size)

    # if filename is specified, save to file
    if fname:
        img.save(fname)
    return img


def normalize(m, scale=1.0):
    """ rescale the 2d array to float range (0.0..scale)
        - m     : 2d matrix of height postings
        - scale : optional, max value after normalization
        returns the normalized array as floating point
    """
    # compute range
    imin = np.min(m)
    imax = np.max(m)
    range = imax - imin

    # prevent divide by 0
    if range <= 0:
        range = 1

    # compute scale
    scale = scale / float(range)

    # scale all values
    m = (m - imin) * scale
    return m


def subdivide(m, divisor):
    """ subdivide the 2d array in nxn 2d arrays
        m     : 2d matrix of height postings
        d     : number of subdivisions in one axis
        allowable subdivisions are [2, 3, 4, 8, 15, 16, 20, 30, 60, 75, 100]
        return an array of all subdivided arrays or None if divisor specified is illegal
    """
    global divisors
    # get array size
    s = m.shape[0]

    # check for legal divisor
    # divisors = [2, 3, 4, 8, 15, 16, 20, 30, 60, 75, 100]
    cond = (divisor in divisors)
    if cond:
        # proceed with subdivision

        # number of results
        count = divisor * divisor

        # size of result
        size = int(s / divisor)

        # output container
        a = []

        # generate the subdivisions
        for row in range(0, s, size):
            for col in range(0, s, size):
                sub = m[row:row + size:1, col:col + size:1]
                a.append(sub)

        ret = np.array(a)
    else:
        # return None
        ret = None
    return ret


def mark(m, divisor):
    """ mark the subdivisions in the 2d array for visualizationn
        m     : 2d matrix of height postings
        d     : number of subdivisions in one axis
        allowable subdivisions are [2, 3, 4, 8, 15, 16, 20, 30, 60, 75, 100]
        return an array of all subdivided arrays or None if divisor specified is illegal
    """
    global divisors

    # get array size
    s = m.shape[0]

    # normalize array to 256
    m = normalize(m, 255)

    # check for legal divisor
    # divisors = [2, 3, 4, 8, 15, 16, 20, 30, 60, 75, 100, 200]
    cond = (divisor in divisors)
    if cond:
        # proceed with subdivision

        # number of results
        count = divisor * divisor

        # size of result
        size = int(s / divisor)

        # generate the row subdivisions
        for row in range(0, s, size):
            m[row + 0] = 255
            m[row + 1] = 255
        m[s - 1] = 255

        # generate the col subdivisions
        for col in range(0, s, size):
            for row in range(0, s):
                m[0:s:1, col + 0] = 255
                m[0:s:1, col + 1] = 255

        m[s - 1] = 255
        ret = m
    else:
        # return None
        ret = None
    return ret


def stack1(m):
    """stack a single array of 1 channel, 2d matrix into a monochrome image tensor for ImageDataGenerator
       changes [[1,2],[3,4]] to [ [[1],[2]] , [[3],[4]] ]
       shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
       returns the new ndarray
    """
    a = m.reshape(m.shape + (1,))  # convert to shape (rows,cols,1)
    b = a.reshape((1,) + a.shape)  # convert to shape (1,rows,cols,1)
    return b


def stack1m(m):
    """stack an array of  1 channel, 2d matrix into an array of monochrome image tensors for ImageDataGenerator
       typically input is from an srtm.subdivide operation
       changes [[1,2],[3,4]] to [ [[1],[2]] , [[3],[4]] ]
       shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
       returns the new ndarray
    """
    a = m.reshape(m.shape + (1,))  # convert to shape (rows,cols,1)
    b = a.reshape((1,) + a.shape)  # convert to shape (1,rows,cols,1)
    return b


def stack3(m):
    """stack a 1 channel, 2d matrix into an RGB image tensor for ImageDataGenerator
       changes [[1,2],[3,4]] to [[[1,1,1][2,2,2],[[3,3,3],[4,4,4]]]
       shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
       returns the new ndarray
    """
    a = np.dstack((m, m, m))  # convert to shape (rows,cols,3)
    b = a.reshape((1,) + a.shape)  # convert to shape (1,rows,cols,3)
    return b


def stack3m(m):
    """stack an array of 1 channel, 2d matrix into an array of RGB image tensors for ImageDataGenerator
       typically input is from an srtm.subdivide operation
       returns the new ndarray
    """
    b = []
    for a in m:
        c = np.dstack((a, a, a))
        b.append(c)
    r = np.array(b)
    return r


def get_divisors():
    """get the list of allowed divisors"""
    global divisors
    return divisors


def get_subdivisions():
    """ get a list of divisors and subdivision counts"""
    global divisors
    a = []
    for d in divisors:
        a.append((d, d * d))
    return a


def array_to_tensor(m):
    """reshape (n,n) to keras image tensor (batch,rows,cols,depth)"""
    x = m.reshape((1,) + m.shape + (1,))
    return x


def tensor_to_array(t):
    """write single tensor as an image"""
    m = t[0]
    n = m.reshape(m.shape[0], m.shape[1])
    return n


def datagen(a, count):
    """generate 'count' modified images from single image array
       images are returned as a tuple of
       (keras image tensor,label)
    """
    x = array_to_tensor(a)

    # create the datagenerator
    datagen = ImageDataGenerator(
        rotation_range=30,
        height_shift_range=0.1,
        width_shift_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    # generate 'count' modified images
    i = 0
    y = [1]
    xr = []

    # keep original
    xr.append(x)
    # create modified copies
    for xb, yb in datagen.flow(x, y, batch_size=1):
        # keep the x value
        xr.append(xb)
        # number of iterations
        i += 1
        if i >= count:
            break  # otherwise the generator would loop indefinitely

    # return array of x's. labels will applied separately
    return np.array(xr)


def generate(m, count):
    """given an array of N 'images', generate count modified images for each of N
       return a an array of N sets of images
    """

    label = 0
    x = []
    y = []
    for a in m:
        xb = datagen(a, count)
        # create array of samples and matching labels
        for a in xb:
            x.append(a)
            y.append(label)
        label += 1
    return np.array(x), np.array(y)


def train_test_split(X, y, stride, split=(.70, .15, .15)):
    """ divide data into train_test_validate
       returns train_x,valid_x,test_x,train_y,valid_y,test_y
    """

    # create train/validate/test groups
    # data is ordered by label [0,0,0..,1,1,1..]
    # divide each set into the 3 groups
    count = y.shape[0]

    train_count = int(stride * split[0])
    train_offset = 0
    valid_count = int(stride * split[1])
    valid_offset = train_offset + train_count
    test_count = int(stride * split[2])
    test_offset = valid_offset + valid_count

    train_x = []
    valid_x = []
    test_x = []
    train_y = []
    valid_y = []
    test_y = []

    for i in range(0, count, stride):
        for j in range(train_offset, train_count):
            k = i + j
            train_x.append(X[k])
            train_y.append(y[k])
        for j in range(valid_offset, valid_offset + valid_count):
            k = i + j
            valid_x.append(X[k])
            valid_y.append(X[k])
        for j in range(test_offset, test_offset + test_count):
            k = i + j
            test_x.append(X[k])
            test_y.append(y[k])
        for j in range(i, i + stride):
            shp = X[j].shape
            arr = tensor_to_array(X[j])
            toimage(arr, "preview/img{0}_{1}.jpg".format(y[j], j), (128, 128))

        print()

    print(train_count, valid_count, test_count)

    return np.array(train_x), \
           np.array(valid_x), \
           np.array(test_x),  \
           np.array(train_y), \
           np.array(valid_y), \
           np.array(test_y)
