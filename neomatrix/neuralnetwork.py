import dataset, optimization
from neomatrix import *

class NeuralNetwork:
    def __init__(self, inputs, output, layers, learning_rate):
        self.inputs = inputs
        self.output = output
        self.layers = layers
        self.learning_rate = learning_rate