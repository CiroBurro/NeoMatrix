"""
Module for dataset utilities and batching operations.

Provides functions to split a tensor into smaller batch tensors.
"""

from neomatrix.core import Tensor
import numpy as np

def get_batches(tensor: Tensor, batch_size: int) -> list[Tensor]:
    """
    Split a tensor into smaller batches of the specified size.

    :param: tensor (core.Tensor): The tensor to be split.
    :param: batch_size (int): Number of samples per batch.

    :return list[core.Tensor]: A list of tensor batches.
    """

    if batch_size == tensor.length():
        return [tensor]

    array = tensor.data

    total_samples = tensor.shape[0]
    num_batches = total_samples // batch_size
    
    try:
        subarrays = np.array_split(array, num_batches, axis=0)
    except ValueError:
        subarrays = [array]

    tensors = []
    for arr in subarrays:
        t = Tensor.from_numpy(array= arr)
        if t.dimension == 1:
            length = t.length()
            t.reshape([length, 1])
        tensors.append(t)
    
    return tensors
