"""
MNIST Handwritten Digit Data Loader

Author: Ryan Gorey

Written in consultation with (but not copied from) Michael Nielsen's resource
at neuralnetworksanddeeplearning.com

Program to load and organize MNIST handwritten digit data.
"""

import numpy as np
import gzip
import cPickle

def load_data(file_name):
    """

    :return:
    """

    f = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = cPickle.load(f)
    f.close()
    return (train_data, val_data, test_data)

