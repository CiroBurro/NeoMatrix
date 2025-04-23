from rustybrain import Tensor
import numpy as np

def get_batches(tensor: Tensor, batch_size: int):
    array = tensor.data

    total_samples = tensor.shape[0]
    num_batches = total_samples // batch_size
    
    subarrays = np.array_split(array, num_batches, axis=0)

    tensors = []
    for arr in subarrays:
        tensor = Tensor.from_numpy(arr)
        tensors.append(tensor)
    
    return tensors

