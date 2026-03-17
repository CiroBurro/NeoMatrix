from neomatrix._backend import Tensor

__all__ = ["get_batches"]


def get_batches(tensor: Tensor, batch_size: int) -> list[Tensor]:
    if batch_size == len(tensor):
        return [tensor]

    array = tensor.data
    total_samples = tensor.shape[0]

    batches = []
    for i in range(0, total_samples, batch_size):
        batch_arr = array[i : i + batch_size]
        t = Tensor.from_numpy(batch_arr)
        if t.ndim == 1:
            length = len(t)
            t.reshape([length, 1])
        batches.append(t)

    return batches
