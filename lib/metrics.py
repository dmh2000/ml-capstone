#!/usr/bin/env python
import numpy as np
import keras
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def et(t0, t1):
    """compute minutes:seconds and return as a string"""
    t = t1 - t0
    t = int(t)
    m = int(t / 60)
    s = int(t % 60)
    return "{0:02d}".format(m) + ":" + "{0:02d}".format(s)


def print_metrics(predictions, test_y, history, epochs, t0, t1):
    """print metrics from benchmark or solution model"""
    h = history
    last = epochs - 1
    print("final accuracy : {0:.4f}".format(h['acc'][last]))
    print("best  accuracy : {0:.4f}".format(h['acc'][np.argmax(h['acc'])]))
    print("final loss     : {0:.4f}".format(h['loss'][last]))
    print("best loss      : {0:.4f}".format(h['loss'][np.argmin(h['loss'])]))
    print("final val acc  : {0:.4f}".format(h['val_acc'][last]))
    print("best  val acc  : {0:.4f}".format(h['val_acc'][np.argmax(h['val_acc'])]))
    print("final val loss : {0:.4f}".format(h['val_loss'][last]))
    print("best  val loss : {0:.4f}".format(h['val_loss'][np.argmin(h['val_loss'])]))

    # report test accuracy
    actual = np.argmax(test_y, axis=1)
    test_accuracy = 100 * np.sum(np.array(predictions) == actual) / len(predictions)
    print('Avg Accuracy   : %.4f%%' % test_accuracy)

    # print epochs
    print("epochs         : {0}".format(epochs))
    # print training time
    print("training time  : " + et(t0, t1))


def plot(ax, data, label):
    ax.set_title(label)
    ax.plot(data)
    ax.grid()


def plot_history(title, timestamp, history):
    """plot the history graphs"""
    fig, ax = plt.subplots(2, 2)

    plot(ax[0, 0], history['acc'], 'acc')
    plot(ax[0, 1], history['loss'], 'loss')
    plot(ax[1, 0], history['val_acc'], 'val_acc')
    plot(ax[1, 1], history['val_loss'], 'val_loss')
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    fig.suptitle(title + " " + timestamp)
    plt.show()
    fig.savefig("results/" + timestamp + "/" + timestamp + ".svg")


class Progress(keras.callbacks.Callback):
    """custom progress callback to minimize verbosity but still see progress"""

    def __init__(self):
        keras.callbacks.Callback.__init__(self)
        self.count = 0

    def on_train_begin(self, logs=None):
        super()
        print("<<START>>")

    def on_epoch_begin(self, epoch, logs=None):
        super()
        print(".", end='', flush=True)

    def on_epoch_end(self, epoch, logs=None):
        super()
        self.count += 1
        if self.count >= 50:
            print()
            self.count = 0

    def on_train_end(self, logs=None):
        super()
        print("\n<<DONE>>")
