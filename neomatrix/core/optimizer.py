from dataclasses import dataclass

class Optimizer:
    pass

class BatchGD(Optimizer):
    pass

class SGD(Optimizer):
    pass

@dataclass
class MiniBatchGD(Optimizer):
    batch_size: int