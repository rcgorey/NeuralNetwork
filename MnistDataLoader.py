"""
MNIST Handwritten Digit Data Loader

Author: Ryan Gorey

Credit for chunks of the code and the organization to Michael J. Nielsen. Ryan
Gorey did not directly copy the entirety of the code, but used Nielsen's code
as a reference, used the organization, and several chunks of code.

Program to load and organize MNIST handwritten digit data.
"""

import numpy as np
import gzip
import pickle

def load_data():
    """

    :return:
    """

    f = gzip.open("neural-networks-and-deep-learning\data\mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (train_data, val_data, test_data)

def get_formatted_data():
    """
    Format the raw data into a tuple with three components:

    0. A list containing 50k two-element tuples. The first element is a 784x1
    np array containing the greyscale float values of each pixel in an image of
    a handwritten digit (in a consistent ordering). The second element is a
    10x1 np array composed of zeroes, with the correct label for the image
    represented by float 1.0 at the index representing that digit. This data
    is used for training.

    1. A list containing 10k two-element tuples. The first element is a 784x1
    np array containing the greyscale float values of each pixel in an image of
    a handwritten digit (in a consistent ordering). The second element is an
    integer label for the digit the image represents. This data is used for
    validation.

    2. A list containing 10k two-element tuples. The first element is a 784x1
    np array containing the greyscale float values of each pixel in an image of
    a handwritten digit (in a consistent ordering). The second element is an
    integer label for the digit the image represents. This data is used for
    testing.

    :return: the tuple described above.
    """
    train_data, val_data, test_data = load_data()


    train_input = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_label = [np.reshape(vectorize_label(y), (10, 1)) for y in train_data[1]]
    formatted_train_data = list(zip(train_input, train_label))

    val_input = [np.reshape(x, (784,1)) for x in val_data[0]]
    formatted_val_data = list(zip(val_input, val_data[1]))

    test_input = [np.reshape(x, (784,1)) for x in val_data[0]]
    formatted_test_data = list(zip(test_input, test_data[1]))

    return (formatted_train_data, formatted_val_data, formatted_test_data)

def vectorize_label(label):
    """
    Turn an integer label representing a digit into a 10x1 np array vector with
    a 1.0 at the correct training

    :param label: int representing the correct label for a training input.
    :return: 10x1 np array of mostly zeroes, with 1.0 at the index of the label.
    """
    vectorized_label = np.zeros((10,1))
    vectorized_label[label] = 1.0
    return vectorized_label
