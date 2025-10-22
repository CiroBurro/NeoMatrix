"""
Module for optimizer algorithms used in neural network training.
Provides implementations for Batch Gradient Descent (BatchGD),
Stochastic Gradient Descent (SGD), and Mini Batch Gradient Descent (MiniBatchGD).
"""

from dataclasses import dataclass
from neomatrix.core import Layer, Tensor

class Optimizer:
    """Base class for optimizer algorithms."""

    @staticmethod
    def params_update(layer: Layer, w_grads: Tensor, b_grads: Tensor, learning_rate: float):
        """
        Update the parameters of a layer using gradient descent.

        Args:
            layer (Layer): The neural network layer to update.
            w_grads (Tensor): Gradients for the weights.
            b_grads (Tensor): Gradients for the biases.
            learning_rate (float): Learning rate for the update.
        """
        layer.weights = layer.weights.tensor_subtraction(w_grads.scalar_multiplication(learning_rate))
        layer.biases = layer.biases.tensor_subtraction(b_grads.scalar_multiplication(learning_rate))

class BatchGD(Optimizer):
    """Implementation of the Batch Gradient Descent optimization algorithm."""
    pass

class SGD(Optimizer):
    """Implementation of the Stochastic Gradient Descent optimization algorithm."""
    pass

@dataclass
class MiniBatchGD(Optimizer):
    """
    Implementation of the Mini Batch Gradient Descent optimization algorithm.

    Attributes:
        training_batch_size (int): Size of the batches used during training.
        validation_batch_size (int): Size of the batches used during validation.
    """
    training_batch_size: int
    validation_batch_size: int