from time import time
import lib
import numpy as np
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, History

def run(X, y, labels, groups):
    """solution model
    """
    print("benchmark")

    train_X, valid_X, test_X, train_y, valid_y, test_y = lib.srtm.train_test_split(X, y, labels, groups)

    print(train_X.shape, train_y.shape)
    print(valid_X.shape, valid_y.shape)
    print(test_X.shape, test_y.shape)

    print(train_X.shape[1:])

    # create model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=train_X.shape[1:]))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(labels, activation='softmax'))

    # print summary
    model.summary()

    # compile the model
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    print("compiled")

    epochs = 5
    print("epochs : {0}".format(epochs))

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.benchmark.hdf5',
                                   verbose=1, save_best_only=True)

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

    t0 = time()
    t0 = time()

    # train the model
    history = model.fit(train_X, train_y,
              validation_data=(valid_X, valid_y),
              epochs=epochs,
              batch_size=20,
              callbacks=[checkpointer, tensorboard,historycb],
              verbose=1
              )

    t1 = time()

    # load best weights
    model.load_weights('saved_models/weights.benchmark.hdf5')

    # get the prediction for each test data image
    # this one only gets the top prediction index
    predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_X]

    # report test metrics
    lib.metrics.print_metrics(predictions, test_y, history.history, epochs, t0, t1)
