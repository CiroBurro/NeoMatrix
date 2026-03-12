"""
Module for ML models. It provides classes for the most common ML algorithms:
- Neural Network
- Linear Regression
- Logistic Regression
- Softmax Regression
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neomatrix._backend import Tensor
    from neomatrix import layers, losses, utils, optimizers

import os

__all__ = [
    'Model',
    'LinearRegression',
    'LogisticRegression',
    'SoftmaxRegression',
]

class Model:

    def __init__(self, layers: list[layers.Layer]):
        self.layers = layers
        self._use_optimize = None

    def _detect_optimization(self):
        if self._use_optimize is not None:
            return

        last_layer = self.layers[-1]
        if isinstance(last_layer, layers.Softmax) and isinstance(self.loss_function, losses.CategoricalCrossEntropy):
            self._use_optimize = True
        elif isinstance(last_layer, layers.Sigmoid) and isinstance(self.loss_function, losses.BinaryCrossEntropy):
            self._use_optimize = True
        else:
            self._use_optimize = False

    @property
    def use_optimized_gradient(self) -> bool:
        if self._use_optimized is None:
            self._detect_optimization()
        return self._use_optimized or False
    
    def compile(self, loss_function: losses.LossFunction, optimizer: optimizers.Optimizer, metrics = None):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics or []
        self._use_optimize = self._detect_optimization()

        params = []

        for l in self.layers:
            if hasattr(layer, 'get_parameters'):
                p = layer.get_parameters()
                if p:
                    params.append(p)



    def predict(self, x: Tensor, training: bool = False) -> Tensor:
        for l in self.layers:
            x = l.forward(x, training)

        return x

    def backward(self, y_true: Tensor, y_pred: Tensor):
        if self.use_optimized_gradient:
            output_grads = self.loss_function.backward_optimized(y_true=y_true, y_pred=y_pred)
            self.layers.remove(self.layers[-1])
        else:
            output_grads = self.loss_function.backward(y_true=y_true, y_pred=y_pred)

        for l in self.layers.reverse():
            output_grads = l.backward(output_grads)

        return
    
    def fit(self, training_x: Tensor, training_y: Tensor, val_x: Tensor, val_y: Tensor, epochs: int, batch_size: int):
        
        if self.loss is None or self.optimizer is None:
            raise ValueError("Call model.compile(loss_function=..., optimizer=...) first")
    
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches_x = utils.get_batches(training_x, batch_size)
            batches_y = utils.get_batches(training_y, batch_size)
            
            for i, (batch_x, batch_y) in enumerate((batches_x, batches_y)):

                self.optimizer.zero_grad()
                pred = self.forward(batch_x, parallel=True)
                loss_val = self.loss.call(pred, batch_y)
                grad = self.loss.backward(pred, batch_y)
                self.backward(grad, parallel=True)

                self.optimizer.step()
                
                epoch_loss += loss_val
            

        return



class LinearRegression(Model):
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

class LogisticRegression(Model):
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

class SoftmaxRegression(Model):
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
