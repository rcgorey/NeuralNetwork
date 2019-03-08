"""
Neural Network

Author: Ryan Gorey
Created: 2/8/2019

Defines structure and behaviors for a Neural Network object.

Intended to be applied to the MNIST handwritten digit database.

Created independently, using http://neuralnetworksanddeeplearning.com/
as a resource.
"""
import numpy as np
from scipy.special import expit

import MnistDataLoader as data_loader

class NeuralNetwork:
    """
    Implementation of neural network object for classifying handwritten digits
    from the MNIST database.

    XXX - fill in with PEP 8 recommendations for proper docstring
    documentation.
    """

    def __init__(self, dims, learning_rate=1.0):
        """
        Initializes an instance of the NeuralNetwork class.

        :param dims: list containing the number of nodes in each layer of the
            desired network (one int for each layer)
        :param learning_rate: rate the network learns from training data with
            default value = 1.0
        """

        '''
        self.num_input = num_input
        self.num_output = num_output
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_dims = hidden_layer_dims
        self.learning_rate = learning_rate

        self.weights = np.random.rand((3, 2))
        self.biases = np.random.rand(())
        '''
        self.dims = dims
        self.learning_rate = learning_rate
        self.weights = self.init_weight_matrices()
        self.biases = self.init_bias_vectors()

    def init_weight_matrices(self):
        """
        Initialize the structure of the weight matrices with random uniformally
        distributed float values between 0 and 1.

        :return: list of initialized weight numpy matrices.
        """
        weights = []
        for i in range(len(self.dims) - 1):
            cur_weight_matrix = np.random.rand(self.dims[i + 1], self.dims[i])
            weights.append(cur_weight_matrix)
        return weights

    def init_bias_vectors(self):
        """
        Initialize the structure of the bias vectors with random uniformally
        distributed float values between 0 and 1.

        :return:  list of initialized bias numpy column vectors.
        """
        biases = []
        for i in range(len(self.dims) - 1):
            cur_bias_vector = np.random.rand(self.dims[i + 1], 1)
            biases.append(cur_bias_vector)
        return biases

    def set_learned_values(self, filename):
        """
        Reads textfile to populate a NeuralNetwork instance with specific
        weights and biases.
        :param filename:
        :return:
        """
        pass

    def write_learned_values(self):
        """
        Writes a textfile with NeuralNetwork's weights and biases.
        """
        pass

    def train(self, training_data, num_epochs = 10, batch_size = 10,
              test_data = None):
        """
        Trains the network using Stochastic Gradient Descent (SGD).

        :param training_data: zipped 2-element tuple containing the pixel
        values and labels.
        :param num_epochs: int number of epochs to train over, default 10.
        :param batch_size: int designating mini-batch size for SGD, default 10.
        :param learning_rate: float between 0 and 1 adjusting the rate the
        network learns from each mini batch.
        """
        for epoch in range(num_epochs):
            np.random.shuffle(training_data)
            batches = [training_data[k: k + batch_size]
                       for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.process_mini_batch(batch)

            print("Finished epoch " + str(epoch + 1) + " of " + str(num_epochs))
            if test_data:
                self.evaluate(test_data)

    def process_mini_batch(self, batch):
        """
        Applies stochastic gradient descent for one batch of training data

        :param batch: a list containing zipped tuples of training data and
        labels.
        """
        w_grads = [np.zeros(w_mat.shape) for w_mat in self.weights]
        b_grads = [np.zeros(b_vec.shape) for b_vec in self.biases]

        for data, label in batch:
            w_deltas, b_deltas = self.backprop(data, label)
            w_grads = [w + w_delta
                            for w, w_delta in zip(w_grads, w_deltas)]
            b_grads = [b + b_delta
                            for b, b_delta in zip(b_grads, b_deltas)]

        w_grads = [w * (self.learning_rate/len(batch))
                        for w in w_grads]
        b_grads = [b * (self.learning_rate/len(batch))
                        for b in b_grads]

        print(w_grads)

        self.weights = [w + w_grad
                        for w, w_grad in zip(self.weights, w_grads)]
        self.biases = [b + b_grad
                        for b, b_grad in zip(self.biases, b_grads)]


    def backprop(self, data, label):

        ### 1. Feed Forward, capturing data as needed.
        a_vecs = [data]
        z_vecs = []

        for w, b in zip(self.weights, self.biases):
            z_vecs.append(np.dot(w, a_vecs[-1]) + b)
            a_vecs.append(z_vecs[-1])

        w_grads = [np.zeros(w_mat.shape) for w_mat in self.weights]
        b_grads = [np.zeros(b_vec.shape) for b_vec in self.biases]

        ### 2. Find last layer's respective gradient elements
        error = self.calc_output_error(z_vecs[-1], a_vecs[-1], label)
        b_grads[-1] = error
        w_grads[-1] = np.dot(error, a_vecs[-2].T)

        ### 3. Backpropagate
        for layer_ind in range(2, len(self.dims)):
            s_prime = sig_prime(z_vecs[-layer_ind])
            b_grads[-layer_ind] = np.dot(self.weights[-layer_ind + 1].T,
                                         b_grads[-layer_ind + 1]) \
                                  * s_prime
            w_grads[-layer_ind] = np.dot(b_grads[-layer_ind], a_vecs[-layer_ind-1].T)

        return w_grads, b_grads

    def calc_output_error(self, z, a_output, label):
        output_err = ((a_output - label) * (sig(z) * (1 - sig(z))))
        return output_err

    def feed_forward(self, obs):
        """
        Feed input observation through neural network instance. Return the
        output activations.

        :param obs: input_activations as numpy array
        :return: tuple containing the vector of weighted inputs to the output
        layer, and the output activations as numpy array
        """
        a = obs
        for w, b in zip(self.weights, self.biases):
            a = sig(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        """
        Evaluate the accuracy of the neural network over a set of training data.
        Print the results.

        :param test_data: a list of tuples containing the raw pixel data as an
        np array, and the integer label.
        """
        num_obs = len(test_data)
        num_correct = 0
        for obs, label in test_data:
            if label == np.argmax(self.feed_forward(obs)):
                num_correct += 1
        print("Accuracy = " + str(num_correct) + "/ " + str(num_obs) + " = " +
              str(num_correct/num_obs))


    def print_weight_matrix(self):
        """
        Print the weight matrices and their values.
        """

        for i in range(len(self.weights)):
            print("Layer " + str(i + 1) + ": \n")
            print(self.weights[i])
            print()

    def print_bias_vectors(self):
        """
        Print the bias vectors and their values.
        """
        for i in range(len(self.biases)):
            print("Layer " + str(i + 1) + ": \n")
            print(self.biases[i])
            print()


def sig(z):
    """
    Return the sigmoid function applied to some vector of weighted inputs z.

    :param z: np.array of weighted inputs.
    :return: np.array of activations
    """
    return 1.0/(1.0 + np.exp(-z))

def sig_prime(z):
    return sig(z) * (1 - sig(z))

myNetwork = NeuralNetwork([784, 3, 3, 10])

#load data
training_data, val_data, test_data = data_loader.get_formatted_data()
myNetwork.train(training_data, test_data=test_data)
