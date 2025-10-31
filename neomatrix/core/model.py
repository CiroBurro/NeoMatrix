"""
Module for ML models. It provides classes for the most common ML algorithms:
- Neural Network
- Linear Regression
- Logistic Regression
- Softmax Regression
"""
import json

from neomatrix.core import Layer, Cost, Tensor, get_cost, Activation
import neomatrix.utils as utils
import neomatrix.core.optimizer as opt
import os

__all__ = [
    'NeuralNetwork',
    'LinearRegression',
    'LogisticRegression',
    'SoftmaxRegression',
]

class NeuralNetwork:
    """
    A class representing a generic neural network model.

    Attributes:
        layers (list[core.Layer]): List of neural network layers.
        cost_function (core.Cost): Cost function used for training.
        learning_rate (float): Learning rate for updating parameters.
    """
    def __init__(self, layers: list[Layer], cost_function: Cost, learning_rate: float):
        self.layers = layers
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        """
        Initializes a new NeuralNetwork instance.

        :param: layers (list[core.Layer]): List of neural network layers.
        :param: cost_function (core.Cost): Cost function used for training.
        :param: learning_rate (float): Learning rate for updating parameters.
        """

    def predict(self, ntwk_inputs: Tensor, batch_processing: bool=True, parallel: bool=False) -> Tensor:
        """
        Performs forward propagation through the network.

        :param: ntwk_inputs (core.Tensor): Input tensor for the network.
        :param: batch_processing (bool, optional): If True, processes the data in batch mode. Default is True.
        :param: parallel (bool, optional): If True, enables parallel processing. Default is False.

        :return core.Tensor: Output tensor from the network.
        """
        inputs = ntwk_inputs
        if batch_processing:
            for layer in self.layers:
                output = layer.forward_batch(inputs, parallel=parallel)
                inputs = output
            return inputs
        else:
            if inputs.dimension == 2:
                exit(1)
            for layer in self.layers:
                output = layer.forward(inputs, parallel=parallel)
                inputs = output
            return inputs

    def backward(self, ntwk_inputs: Tensor, t: Tensor, z: Tensor, optimizer: opt.Optimizer):
        """
        Performs backpropagation to compute gradients and update parameters.

        :param: ntwk_inputs (core.Tensor): Input tensor for the network.
        :param: t (core.Tensor): Target tensor.
        :param: z (core.Tensor): Output tensor from the network.
        :param: optimizer (opt.Optimizer): Optimizer for parameter updates.
        """
        all_outputs = []
        deltas = self.layers[-1].get_output_deltas(self.cost_function, t, z)
        for (i, layer) in enumerate(reversed(self.layers)):
            out_layer = True
            next_weights = None
            if i != 0:
                out_layer = False
                next_weights = self.layers[-i].weights
            
            if i + 1 < len(self.layers):
                prev_layer = self.layers[- (i + 2)]
                all_outputs = prev_layer.output
            elif i + 1 == len(self.layers):
                all_outputs = ntwk_inputs
                 
            (w_grads, b_grads, new_deltas) = layer.backward(out_layer, deltas, next_weights, all_outputs)
            optimizer.params_update(layer=layer, w_grads=w_grads, b_grads=b_grads, learning_rate=self.learning_rate)
            deltas = new_deltas
    
    def fit(self, training_set: Tensor, training_targets: Tensor, val_set: Tensor, val_targets: Tensor, optimizer: opt.Optimizer, epochs: int, parallel: bool = False):
        """
        Trains the network using the provided data.

        :param: training_set (core.Tensor): Training data tensor.
        :param: training_targets (core.Tensor): Training targets tensor.
        :param: val_set (core.Tensor): Validation data tensor.
        :param: val_targets (core.Tensor): Validation targets tensor.
        :param: optimizer (opt.Optimizer): Optimizer used for training.
        :param: epochs (int): Number of training epochs.
        :param: parallel (bool, optional): If True, enables parallel processing. Default is False.
        """
        batch_processing: bool = True
        training_batch_size: int = 1
        validation_batch_size: int = 1

        match optimizer:
            case opt.BatchGD():
                training_batch_size = training_set.length()
                validation_batch_size = val_set.length()
            case opt.SGD():
                batch_processing = False
            case opt.MiniBatchGD():
                training_batch_size= optimizer.training_batch_size
                validation_batch_size = optimizer.validation_batch_size
            case _:
                print("Invalid optimizer")
                exit(1)
        
        for (i, epoch) in enumerate(range(epochs)):
            os.system('cls' if os.name == 'nt' else 'clear')
            total_loss = 0
            train_batches = utils.get_batches(training_set, training_batch_size)
            train_target_batches = utils.get_batches(training_targets, training_batch_size)
            val_batches = utils.get_batches(val_set, validation_batch_size)
            val_targets_batches = utils.get_batches(val_targets, validation_batch_size)

            for (j, batch) in enumerate(train_batches):
                outputs = self.predict(ntwk_inputs=batch, batch_processing=batch_processing, parallel=parallel)
                self.backward(ntwk_inputs=batch, t=train_target_batches[j], z=outputs, optimizer=optimizer)
                loss = get_cost(self.cost_function, train_target_batches[j], z=outputs, parallel=parallel, batch_processing=batch_processing)
                total_loss += loss
                print("-------------------------")
                print(f"Epoch: {i + 1}, Training batch: {j}, Loss: {loss}, Total loss: {total_loss}")

            total_loss = 0
            for (k, batch) in enumerate(val_batches):
                outputs = self.predict(ntwk_inputs=batch, batch_processing=batch_processing, parallel=parallel)
                val_loss = get_cost(self.cost_function, t=val_targets_batches[k], z=outputs, parallel=parallel, batch_processing=batch_processing)
                total_loss += val_loss
                print("-------------------------")
                print(f"Epoch: {i + 1}, Validation batch: {k}, Loss: {val_loss}, Total loss: {total_loss}")

    def to_dict(self):
        """
        Writes a NeuralNetwork class as a python dict.
        :return: A python dictionary
        """
        d = {
            "layers" : [layer.to_dict() for layer in self.layers],
            "cost" : self.cost_function.to_dict(),
            "learning_rate" : self.learning_rate
        }

        print(d)
        return d

    @staticmethod
    def from_json(d):
        learning_rate = d["learning_rate"]
        cost_dict = d["cost"]
        layers = d["layers"]

        return NeuralNetwork([Layer.from_dict(layer_dict) for layer_dict in layers ], Cost.from_dict(cost_dict), learning_rate=float(learning_rate))

    def save(self, path):
        """
        Serialize a NeuralNetwork class into a json file
        :param path: path of the file where serialized model will be stored
        """
        with open(path, mode='w') as f:
            f.write(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            model = cls.from_json(json.load(f))

        return model

class LinearRegression(NeuralNetwork):
    """
    A class representing a linear regression model.
    """
    def __init__(self, input_nodes: int, output_nodes: int, learning_rate: float):
        """
        Initializes a Linear Regression model.

        Args:
            input_nodes (int): Number of input nodes.
            output_nodes (int): Number of output nodes.
            learning_rate (float): Learning rate.
        """
        super().__init__([
            Layer(output_nodes, input_nodes, Activation.Linear)
        ], Cost.MeanSquaredError(), learning_rate)

class LogisticRegression(NeuralNetwork):
    """
    A class representing a logistic regression model.
    """
    def __init__(self, input_nodes: int, learning_rate: float):
        """
        Initializes a Logistic Regression model.

        Args:
            input_nodes (int): Number of input nodes.
            learning_rate (float): Learning rate.
        """
        super().__init__([
            Layer(1, input_nodes, Activation.Sigmoid)
        ], Cost.BinaryCrossEntropy(), learning_rate)

class SoftmaxRegression(NeuralNetwork):
    """
    A class representing a softmax regression model.
    """
    def __init__(self, input_nodes: int, output_nodes: int, learning_rate: float):
        """
        Initializes a Softmax Regression model.

        Args:
            input_nodes (int): Number of input nodes.
            output_nodes (int): Number of output nodes.
            learning_rate (float): Learning rate.
        """
        super().__init__([
            Layer(output_nodes, input_nodes, Activation.Softmax)
        ], Cost.CategoricalCrossEntropy(), learning_rate)
