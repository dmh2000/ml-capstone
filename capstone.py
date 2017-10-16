#!/usr/bin/env python
import sys
import numpy as np
import keras
import lib.srtm as srtm
import lib.benchmark as benchmark
import lib.solution  as solution

if __name__ == "__main__":
    # check input arguments
    if len(sys.argv) < 4:
        print("capstone <.hgt filename> <divisor> <gen_count> <model> ")
        print("<filename>   : .hgt file from SRTM dataset")
        print("<divisor>    : number of rows/cols to subdivide the input image")
        print("    allowed values: [2, 3, 4, 5, 8, 15, 16, 20, 30, 60, 75, 100, 200]")
        print("<gen_count>  : number of augmented images to generate")
        print("<model>      : string 'benchmark' or 'solution'")
        sys.exit(1)

    # get the input arguments
    try:
        fname = sys.argv[1]           # input filename
        divisor = int(sys.argv[2])    # number of rows/cols to subdivide input image
        gen_count = int(sys.argv[3])  # number of datagen produced images
        selected_model = sys.argv[4]  # which model to run
        print("filename  : {0}".format(fname))
        print("divisor   : {0}".format(divisor))
        print("gen_count : {0}".format(gen_count))
        print("model     : {0}".format(selected_model))

    except Exception as ex:
        print(type(ex))
        print(ex)
        sys.exit(1)

    # read the input file
    m = srtm.read(fname)
    print("input shape      : {0}",format(m.shape))

    # subdivide into NxN images
    s = srtm.subdivide(m, divisor)
    print("subdivided shape : {0}".format(s.shape))

    # normalize to increase contrast per image
    n = [srtm.normalize(x,255) for x in s]
    n = np.array(n)
    print("normalized shape : {0}".format(n.shape))

    # use datagen to augment the features and labels for each image
    # each one is processed individually
    # labeling is in order of the input array
    X, y = srtm.generate(n, gen_count)
    print("X shape          : {0}".format(X.shape))
    print("y shape          : {0}".format(y.shape))

    # execute the selected model
    if selected_model == 'benchmark':
        benchmark.run(X, y, gen_count + 1)
    elif selected_model == 'solution':
        solution.run(X, y, gen_count + 1)
    else:
        print("model parameter is not valid : must be benchmark or solution")

    pass
