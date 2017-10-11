Module srtm
-----------

Functions
---------
srtm_normalize(m, scale=1.0)
    rescale the 2d array to float range (0.0..scale]

            - m     : 2d matrix of height postings
            - scale : optional, max value after normalization
            
    returns the normalized array as floating point

srtm_read(fname)
    read the specified file and return a numpy 2d array of data points as u16

            - fname : name of file to read
            the last row and column overlap are trimmed off
            
    returns 2D array of 16 bit integers

srtm_toimage(m, fname=None, size=(512, 512))
    create a image image of the elevation matrix

            - m     : 2d matrix of height postings
            - fname : optional, if given, the resulting image is written to the file
                      the type suffix determines the save image type
            - size  : optional, tuple specifying resulting image size
            
    returns the image object
