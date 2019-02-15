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


class NeuralNetwork():
    """
    """

    def __init__(self, dims, learning_rate=0.05):
        """
        Initializes an instance of the NeuralNetwork class.

        :param dims: list containing the number of nodes in each layer of the
            desired network (one int for each layer)
        :param learning_rate: rate the network learns from training data with
            default value = 0.05
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
        self.weights = []
        self.initWeightMatrices()

    def initWeightMatrices(self):
        weights = []
        for i in range(len(self.dims)-1):
            cur_weight_matrix = np.random.rand(self.dims[i+1], self.dims[i])
            weights.append(cur_weight_matrix)
        self.weights = weights

    def feed_forward(self):
        return null

    def train(self, training_data):
        return self

    def backpropagation(self):
        return self

    def print_weight_matrix(self):

        for i in range(len(self.weights)):
            print("Layer " + str(i + 1) + ": \n")
            print(self.weights[i])
            print()


myNetwork = NeuralNetwork([])
myNetwork.print_weight_matrix()
