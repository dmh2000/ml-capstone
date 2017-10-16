import sys
import numpy as np
import keras
import lib.srtm as srtm
import lib.benchmark as benchmark
import lib.solution  as solution

if __name__ == "__main__":
    # check input arguments
    if len(sys.argv) < 4:
        print("capstone <.hgt filename> <divisor> <model>")
        print("<filname> : .hgt file from SRTM dataset")
        print("<divisor> : number of rows/cols to subdivide the input image")
        print("<model>   : string 'benchmark' or 'solution'")
        sys.exit(1)

    # get the input arguments
    fname = None
    divisions = None
    try:
        fname = sys.argv[1]
        divisor = int(sys.argv[2])
        selected_model = sys.argv[3]
        print(fname, divisor, selected_model)
    except Exception as ex:
        print(type(ex))
        print(ex)
        sys.exit(1)

    # get data divided into nxn 'images'
    m = srtm.read_and_subdivide(fname, divisor)
    print("subdivided shape : {0}".format(m.shape))

    # use datagen to augment the features and labels for each image
    # each one is processed individually

    # divide the set of images into train/validate/test sets
    # be sure that an equal proportion of each original image
    # is added to each set
    train = validate = test = None

    # execute the selected model
    if selected_model == 'benchmark':
        benchmark.run(train, validate, test)
    elif selected_model == 'solution':
        solution.run(train, validate, test)
    else:
        print("model parameter is not valid : must be benchmark or solution")

    pass
