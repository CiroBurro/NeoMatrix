from dataclasses import dataclass

class Optimizer:
    pass

class BatchGD(Optimizer):
    pass

class SGD(Optimizer):
    pass

@dataclass
class MiniBatchGD(Optimizer):
    training_batch_size: int
    validation_batch_size: int