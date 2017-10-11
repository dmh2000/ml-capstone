Module srtm
-----------

Functions
---------
<pre>

mark(m, divisor)
    mark the subdivisions in the 2d array for visualizationn
    m     : 2d matrix of height postings
    d     : number of subdivisions in one axis
    allowable subdivisions are [2, 3, 4, 8, 15, 16, 20, 30, 60, 75, 100]
    return an array of all subdivided arrays or None if divisor specified is illegal

-------
normalize(m, scale=1.0)
    rescale the 2d array to float range (0.0..scale]
    - m     : 2d matrix of height postings
    - scale : optional, max value after normalization
    returns the normalized array as floating point

-------
read(fname)
    read the specified file and return a numpy 2d array of data points as u16
    fname : name of file to read
    the last row and column overlap are trimmed off
    returns 2D array of 16 bit integers

-------
rotate(m, angle=90)
    rotate a matrix by angle
    returns the rotate image with no fill

-------
stack1(m)
    stack a single array of 1 channel, 2d matrix into a monochrome image tensor for ImageDataGenerator
    changes [[1,2],[3,4]] to [ [[1],[2]] , [[3],[4]] ]
    shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
    returns the new ndarray

-------
stack1m(m)
    stack an array of  1 channel, 2d matrix into an array of monochrome image tensors for ImageDataGenerator
    typically input is from an srtm.subdivide operation
    changes [[1,2],[3,4]] to [ [[1],[2]] , [[3],[4]] ]
    shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
    returns the new ndarray

-------
stack3(m)
    stack a 1 channel, 2d matrix into an RGB image tensor for ImageDataGenerator
    changes [[1,2],[3,4]] to [[[1,1,1][2,2,2],[[3,3,3],[4,4,4]]]
    shape (rows,cols) -> (rows,cols,3) -> (1,rows,cols,3)
    returns the new ndarray

-------
stack3m(m)
    stack an array of 1 channel, 2d matrix into an array of RGB image tensors for ImageDataGenerator
    typically input is from an srtm.subdivide operation
    returns the new ndarray

-------
subdivide(m, divisor)
    subdivide the 2d array in nxn 2d arrays
    m     : 2d matrix of height postings
    d     : number of subdivisions in one axis
    allowable subdivisions are [2, 3, 4, 8, 15, 16, 20, 30, 60, 75, 100]
    return an array of all subdivided arrays or None if divisor specified is illegal

-------
toimage(m, fname=None, size=(512, 512))
    create a image of the elevation matrix
    m     : 2d matrix of height postings
    fname : optional, if given, the resulting image is written to the file
            the type suffix determines the save image type
    size  : optional, tuple specifying resulting image size
    returns the image object
</pre>