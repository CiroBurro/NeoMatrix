import numpy as np

from neomatrix._backend import Tensor

__all__ = ["get_batches"]


def get_batches(tensor: Tensor, batch_size: int) -> list[Tensor]:
    """Split a tensor into smaller batches of the specified size.

    Args:
        tensor: The tensor to split
        batch_size: Number of samples per batch

    Returns:
        List of tensor batches
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
        t = Tensor.from_numpy(array=arr)
        if t.dimension == 1:
            length = t.length()
            t.reshape([length, 1])
        tensors.append(t)

    return tensors
