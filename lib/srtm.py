#!/usr/bin/env python
import numpy as np
from PIL import Image


def srtm_read(fname):
    """read the specified file and return a numpy 2d array of data points as u16
       the last row and column overlap are trimmed off
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


def srtm_toimage(m, fname=None, size=(512, 512)):
    """create a image image of the elevation matrix
       the filename.<type> determine image type (.png or .jpg)
       .jpg will be smaller file size
    """
    # normalize to 0..255
    m = srtm_normalize(m, 256)
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


def srtm_normalize(m, scale):
    """rescale the 2d array to float range (0.0..scale]"""
    imin = np.min(m)
    imax = np.max(m)
    range = imax - imin
    scale = scale / float(range)
    # scale all values
    m = (m - imin) * scale
    return m


# local test code
if __name__ == "__main__":
    m = srtm_read("../data/level1/N37W098.hgt")
    print(m.shape)
    srtm_toimage(m, "L1n37W098.jpg")

    m = srtm_read("../data/level1/N39W120.hgt")
    print(m.shape)
    srtm_toimage(m, "L1n39w120.jpg")

    m = srtm_read("../data/level2/N37W098.hgt")
    print(m.shape)
    srtm_toimage(m, "L2n37w098.jpg")

    m = srtm_read("../data/level2/N39W120.hgt")
    srtm_toimage(m, "L2n39w120.jpg")
