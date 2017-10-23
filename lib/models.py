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
    """ 2 layer, 32 filters, kernel = 3 """
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


def model2(labels,input_shape):
    """ 3 layer , 16 filters, kernel = 3 """
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(labels, activation='softmax'))
    return model


def model3(labels,input_shape):
    """ 3 layer , 32 filters, kernel = 5 """
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(labels, activation='softmax'))
    return model


def model4(labels,input_shape):
    """ 4 layer , 16 filters, kernel = 5 """
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
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
    models['model2'] = model2
    models['model3'] = model3
    models['model4'] = model4


def get_model(model):
    return models[model]


