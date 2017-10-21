#!/usr/bin/env python
import numpy as np


def et(t0,t1):
    """compute minutes:seconds and return as a string"""
    t = t1 - t0
    t = int(t)
    m = int(t / 60)
    s = int(t % 60)
    return str(m) + ":" + str(s)


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
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_y, axis=1))/len(predictions)
    print('Avg Accuracy   : %.4f%%' % test_accuracy)

    # print training time
    print("training time  : " + et(t0, t1))
2