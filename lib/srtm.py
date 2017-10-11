#!/usr/bin/env python
import numpy as np
import scipy.ndimage.interpolation as terp
from PIL import Image


def read(fname):
    """ read the specified file and return a numpy 2d array of data points as u16
        fname : name of file to read
        the last row and column overlap are trimmed off
        returns 2D array of 16 bit integers
    """
    # read the file as a 1d array
    m = np.fromfile(fname, dtype=">H")

    # fill voids which are == 32768
    # fill with min value
    imin = np.min(m)
    m[m >= 32768] = imin

    # get size of square dimensions
    s = int(np.sqrt(m.shape[0]))

    # reshape to a square 2d array
    m = np.reshape(m, (s, s))

    # trim off overlap of last row/col
    s -= 1
    m = m[0:s:1, 0:s:1]

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
    """ rescale the 2d array to float range (0.0..scale]
        - m     : 2d matrix of height postings
        - scale : optional, max value after normalization
        returns the normalized array as floating point
    """
    imin = np.min(m)
    imax = np.max(m)
    range = imax - imin
    if range <= 0:
        range = 1
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
    # get array size
    s = m.shape[0]

    # check for legal divisor
    divisors = [2, 3, 4, 8, 15, 16, 20, 30, 60, 75, 100]
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
    # get array size
    s = m.shape[0]

    # normalize array to 256
    m = normalize(m, 256)

    # check for legal divisor
    divisors = [2, 3, 4, 8, 15, 16, 20, 30, 60, 75, 100, 200]
    cond = (divisor in divisors)
    if cond:
        # proceed with subdivision

        # number of results
        count = divisor * divisor

        # size of result
        size = int(s / divisor)

        # generate the row subdivisions
        for row in range(0, s, size):
            print(m[row].shape)
            m[row] = 255
        m[3599] = 255

        # generate the col subdivisions
        for col in range(0, s, size):
            for row in range(0, s):
                m[0:s:1, col] = 255

        m[3599] = 255
        ret = m
    else:
        # return None
        ret = None
    return ret


def rotate(m, angle=90):
    """rotate a matrix by angle
       returns the rotate image with no fill
    """
    r = terp.rotate(m, angle)
    return r


def stack1(m):
    """stack a single array of 1 channel, 2d matrix into a monochrome image tensor for ImageDataGenerator
       changes [[1,2],[3,4]] to [ [[1],[2]] , [[3],[4]] ]
       shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
       returns the new ndarray
    """
    a = m.reshape(m.shape + (1,))   # convert to shape (rows,cols,1)
    b = a.reshape((1, ) + a.shape)  # convert to shape (1,rows,cols,1)
    return b


def stack1m(m):
    """stack an array of  1 channel, 2d matrix into an array of monochrome image tensors for ImageDataGenerator
       typically input is from an srtm.subdivide operation
       changes [[1,2],[3,4]] to [ [[1],[2]] , [[3],[4]] ]
       shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
       returns the new ndarray
    """
    a = m.reshape(m.shape + (1,))   # convert to shape (rows,cols,1)
    b = a.reshape((1, ) + a.shape)  # convert to shape (1,rows,cols,1)
    return b

def stack3(m):
    """stack a 1 channel, 2d matrix into an RGB image tensor for ImageDataGenerator
       changes [[1,2],[3,4]] to [[[1,1,1][2,2,2],[[3,3,3],[4,4,4]]]
       shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
       returns the new ndarray
    """
    a = np.dstack((m, m, m))       # convert to shape (rows,cols,3)
    b = a.reshape((1, ) + a.shape)  # convert to shape (1,rows,cols,3)
    return b


def stack3m(m):
    """stack an array of 1 channel, 2d matrix into an array of RGB image tensors for ImageDataGenerator
       typically input is from an srtm.subdivide operation
       returns the new ndarray
    """
    b = []
    for a in m:
        c = np.dstack((a, a, a))
        print(c.shape)
        b.append(c)
    r = np.array(b)
    return r

def test_images():
    m = read("../data/level1/N37W098.hgt")
    print(m.shape)
    toimage(m, "../doc/L1_n37w098.jpg")

    m = read("../data/level1/N39W120.hgt")
    print(m.shape)
    toimage(m, "../doc/L1_n39w120.jpg")

    m = read("../data/level2/N37W098.hgt")
    print(m.shape)
    toimage(m, "../doc/L2_n37w098.jpg")

    m = read("../data/level2/N39W120.hgt")
    print(m.shape)
    toimage(m, "../doc/L2_n39w120.jpg")


def test_subdiv():
    m = read("../data/level2/N39W120.hgt")
    d = subdivide(m, 8)
    i = 0
    print(len(d))
    for a in d:
        name = "a" + str(i) + ".jpg"
        print(a.shape)
        toimage(a, name, size=a.shape)
        i += 1


def test_mark():
    m = read("../data/level2/N39W120.hgt")
    m = mark(m, 16)
    toimage(m, "img1.jpg", size=m.shape)
    m = read("../data/level2/N39W120.hgt")
    toimage(m, "img2.jpg")


def test_rotate():
    m = read("../data/level2/N39W120.hgt")
    print(m.shape)
    toimage(m, "r1.jpg")
    r = terp.rotate(m, 45.0)
    print(r.shape)
    toimage(r, "r2.jpg")
    d = subdivide(m, 8)
    toimage(d[0],"d0.jpg")
    d1 = terp.rotate(d[0],90.0)
    toimage(d1,"d1.jpg")


# local test code
if __name__ == "__main__":
    test_subdiv()
    # test_mark()
    # test_rotate()
