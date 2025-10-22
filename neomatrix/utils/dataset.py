"""
Module for dataset utilities and batching operations.

Provides functions to split a tensor into smaller batch tensors.
"""

import neomatrix.core as core
import numpy as np

def get_batches(tensor: core.Tensor, batch_size: int) -> list[core.Tensor]:
    """
    Split a tensor into smaller batches of the specified size.

    Args:
        tensor (core.Tensor): The tensor to be split.
        batch_size (int): Number of samples per batch.

    Returns:
        list[core.Tensor]: A list of tensor batches.
    """
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
