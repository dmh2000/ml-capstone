#!/usr/bin/env python
import os
import sys
import json
from datetime import datetime
from time import time
import lib.srtm as srtm
import lib
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, History


def run(X, y, cfg):  # labels, groups, epochs, timestamp, cfg=None):
    """execute the selected model"""

    # extract parameters
    labels = cfg['divisor'] * cfg['divisor']
    epochs = cfg['epochs']
    timestamp = cfg['timestamp']
    groups = cfg['augments'] + 1

    train_X, valid_X, test_X, train_y, valid_y, test_y = lib.srtm.train_test_split(X, y, labels, groups)

    print("train data      X: " + str(train_X.shape) + " y: " + str(train_y.shape))
    print("validation data X: " + str(valid_X.shape) + " y: " + str(valid_y.shape))
    print("test data       X: " + str(test_X.shape) + " y: " + str(test_y.shape))

    # get the model object
    model_type = lib.models.get_model(cfg['model'])

    # instantiate it
    model = model_type(labels, train_X.shape[1:])

    # model = Sequential()
    # model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=train_X.shape[1:]))
    # model.add(GlobalAveragePooling2D())
    # model.add(Dense(labels, activation='softmax'))

    # print summary
    model.summary()

    # compile the model
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # create fit callbacks
    weights_file = 'saved_models/weights.benchmark.hdf5'
    checkpointer = ModelCheckpoint(filepath=weights_file,
                                   verbose=0,
                                   save_best_only=True)

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)

    historycb = History()

    progress = lib.metrics.Progress()

    # get start time
    t0 = time()

    # train the model
    history = model.fit(train_X, train_y,
                        validation_data=(valid_X, valid_y),
                        epochs=epochs,
                        batch_size=20,
                        callbacks=[checkpointer, tensorboard, historycb, progress],
                        verbose=0
                        )

    # get end time
    t1 = time()

    # load best weights
    model.load_weights(weights_file)

    # evaluate the model
    score= model.evaluate(test_X,test_y,verbose=0)
    print()
    print("loss : {0}".format(score[0]))
    print("acc  : {0}".format(score[1]))

    # get the prediction for each test data image
    # this one only gets the top prediction index
    predictions = model.predict_classes(test_X)

    # report test metrics
    lib.metrics.print_metrics(predictions, test_y, history.history, epochs, t0, t1)

    # plot history
    lib.metrics.plot_history("benchmark", timestamp, history.history)


if __name__ == "__main__":
    # check input arguments

    if len(sys.argv) < 2:
        print("capstone <config.cfg> <timestamp>")
        print("<filename>   : .hgt file from SRTM dataset")
        print("<timestamp>  : timestamp of test run")
        sys.exit(1)

    # get file timestamp for output
    if len(sys.argv) >= 3:
        # get timestamp
        timestamp = sys.argv[2]
    else:
        # create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print("timestamp : {0}".format(timestamp))

    # get the input arguments from the config file
    try:
        fname = sys.argv[1]  # input filename
        f = open(fname)
        jcfg = json.load(f)
        datafile = jcfg["datafile"]  # datafile to read
        divisor = int(jcfg["divisor"])  # number of rows/cols to subdivide input image
        augments = int(jcfg["augments"])  # number of datagen produced images
        epochs = int(jcfg["epochs"])  # number of training epochs
        selected_model = jcfg["model"]  # which model to run
        print("config    : {0}".format(fname))
        print("datafile  : {0}".format(datafile))
        print("divisor   : {0}".format(divisor))
        print("augments  : {0}".format(augments))
        print("epochs    : {0}".format(epochs))
        print("model     : {0}".format(selected_model))

        # convert strings to numbers in config object
        jcfg["divisor"] = divisor
        jcfg["augments"] = augments
        jcfg["epochs"] = epochs

    except Exception as ex:
        print("Error in JSON file : " + str(type(ex)))
        print(ex)
        sys.exit(1)

    # add timestamp to config object
    jcfg["timestamp"] = timestamp

    # read the input file
    m = srtm.read(datafile)
    print("input shape      : {0}".format(m.shape))

    # subdivide into NxN images
    s = srtm.subdivide(m, divisor)
    print("subdivided shape : {0}".format(s.shape))

    # normalize to increase contrast per image
    n = [srtm.normalize(x, 255) for x in s]
    n = np.array(n)
    print("normalized shape : {0}".format(n.shape))

    # use datagen to augment the features and labels for each image
    # each one is processed individually
    # labeling is in order of the input array
    X, y = srtm.generate(n, augments)
    print("X shape          : {0}".format(X.shape))
    print("y shape          : {0}".format(y.shape))

    # number of labels
    labels = divisor * divisor

    # release memory
    del (m)
    del (s)
    del (n)

    # initialize models
    lib.models.init()

    # execute the selected model
    run(X, y, jcfg)

    print("done")
