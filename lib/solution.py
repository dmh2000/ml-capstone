from time import time
import lib.srtm as srtm
import lib
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, History
import keras.metrics as metrics


def et(t):
    t = int(t)
    m = int(t / 60)
    s = int(t % 60)
    return str(m) + ":" + str(s)


def run(X, y, labels, groups):
    """solution model
    """
    print("benchmark")

    train_X, valid_X, test_X, train_y, valid_y, test_y = srtm.train_test_split(X, y, labels, groups)

    print(train_X.shape, train_y.shape)
    print(valid_X.shape, valid_y.shape)
    print(test_X.shape, test_y.shape)

    print(train_X.shape[1:])

    # create model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=train_X.shape[1:]))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(labels, activation='softmax'))

    # print summary
    model.summary()

    # compile the model
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    print("compiled")

    epochs = 20
    print("epochs : {0}".format(epochs))

    # create fit callbacks
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.benchmark.hdf5',
                                   verbose=0,
                                   save_best_only=True)

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=True,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)

    historycb = History()

    # get start time
    t0 = time()

    # train the model
    history = model.fit(train_X, train_y,
                        validation_data=(valid_X, valid_y),
                        epochs=epochs,
                        batch_size=20,
                        callbacks=[checkpointer,tensorboard],
                        verbose=0
                        )
    # get end time
    t1 = time()

    # load best weights
    model.load_weights('saved_models/weights.benchmark.hdf5')

    # get the prediction for each test data image
    # this one only gets the top prediction index
    predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_X]

    lib.metrics.print_metrics(predictions, test_y, history.history, epochs, t0, t1)
