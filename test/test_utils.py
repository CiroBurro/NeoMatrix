import numpy as np

from neomatrix import utils
from neomatrix._backend import Tensor


class TestGetBatches:
    def test_exact_batch_size(self):
        t = Tensor.from_numpy(np.arange(32.0, dtype=np.float32).reshape(32, 1))
        batches = utils.get_batches(t, batch_size=32)
        assert len(batches) == 1
        assert batches[0].shape == [32, 1]

    def test_single_sample_batches(self):
        t = Tensor.from_numpy(np.arange(10.0, dtype=np.float32).reshape(10, 1))
        batches = utils.get_batches(t, batch_size=1)
        assert len(batches) == 10
        for batch in batches:
            assert batch.shape == [1, 1]

    def test_multiple_batches(self):
        t = Tensor.from_numpy(np.arange(32.0, dtype=np.float32).reshape(32, 1))
        batches = utils.get_batches(t, batch_size=8)
        assert len(batches) == 4
        for batch in batches:
            assert batch.shape == [8, 1]

    def test_unequal_batches(self):
        t = Tensor.from_numpy(np.arange(30.0, dtype=np.float32).reshape(30, 1))
        batches = utils.get_batches(t, batch_size=8)
        assert len(batches) == 4
        assert batches[0].shape == [8, 1]
        assert batches[1].shape == [8, 1]
        assert batches[2].shape == [8, 1]
        assert batches[3].shape == [6, 1]

    def test_2d_tensor_feature_dim_preserved(self):
        t = Tensor.from_numpy(np.random.randn(20, 5).astype(np.float32))
        batches = utils.get_batches(t, batch_size=4)
        assert len(batches) == 5
        for batch in batches:
            assert batch.shape[1] == 5

    def test_batch_larger_than_data(self):
        t = Tensor.from_numpy(np.arange(10.0, dtype=np.float32).reshape(10, 1))
        batches = utils.get_batches(t, batch_size=100)
        assert len(batches) == 1
        assert batches[0].shape == [10, 1]
