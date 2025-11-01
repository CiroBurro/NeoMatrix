"""
Module for dataset utilities and batching operations.

Provides functions to split a tensor into smaller batch tensors.
"""

import neomatrix.core as core
import numpy as np

def get_batches(tensor: core.Tensor, batch_size: int) -> list[core.Tensor]:
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
        tensor = core.Tensor.from_numpy(array= arr)
        if batch_size == 1:
            tensor.flatten()
        tensors.append(tensor)
    
    return tensors
