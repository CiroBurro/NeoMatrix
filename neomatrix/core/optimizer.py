from dataclasses import dataclass
from neomatrix.core import Layer, Tensor

class Optimizer:
    @staticmethod
    def params_update(layer: Layer, w_grads: Tensor, b_grads: Tensor, learning_rate: float):
        layer.weights = layer.weights.tensor_subtraction(w_grads.scalar_multiplication(learning_rate))
        layer.biases = layer.biases.tensor_subtraction(b_grads.scalar_multiplication(learning_rate))

class BatchGD(Optimizer):
    pass

class SGD(Optimizer):
    pass

@dataclass
class MiniBatchGD(Optimizer):
    training_batch_size: int
    validation_batch_size: int