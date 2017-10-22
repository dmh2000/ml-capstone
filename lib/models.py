from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import Dense, MaxPooling2D, Dropout
from keras.models import Sequential

models = dict()


def model0(labels,input_shape):
    """ Benchmark """
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(labels, activation='softmax'))
    return model


def model1(labels,input_shape):
    """ 2 layer """
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(labels, activation='softmax'))
    return model


def init():
    global models
    print("model init")

    models['benchmark'] = model0
    models['model1'] = model1


def get_model(model):
    return models[model]


