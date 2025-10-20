from dataclasses import dataclass
from neomatrix.core import Layer, Tensor

'''
General Class fo optimizer algorithms
'''
class Optimizer:
    @staticmethod
    def params_update(layer: Layer, w_grads: Tensor, b_grads: Tensor, learning_rate: float):
        layer.weights = layer.weights.tensor_subtraction(w_grads.scalar_multiplication(learning_rate))
        layer.biases = layer.biases.tensor_subtraction(b_grads.scalar_multiplication(learning_rate))

'''
Batch gradient descent class
'''
class BatchGD(Optimizer):
    pass


'''
Stochastic gradient descent class
'''
class SGD(Optimizer):
    pass

'''
Mini batch gradient descent
'''
@dataclass
class MiniBatchGD(Optimizer):
    # Needs a batch size for both training and validation sets
    training_batch_size: int
    validation_batch_size: int