import neomatrix.core as core
import numpy as np

'''
Split a tensor in smaller ones (batches) of an arbitrary size

Parameters:
- tensor: tensor to be splitted
- batch_size: number of samples of a single batch
'''

def get_batches(tensor: core.Tensor, batch_size: int):

    array = tensor.data

    total_samples = tensor.shape[0]
    num_batches = total_samples // batch_size
    
    try:
        subarrays = np.array_split(array, num_batches, axis=0)
    except ValueError:
        subarrays = [array]

    tensors = []
    for arr in subarrays:
        tensor = core.Tensor.from_numpy(arr=arr)
        if batch_size == 1:
            tensor.flatten()
        tensors.append(tensor)
    
    return tensors



