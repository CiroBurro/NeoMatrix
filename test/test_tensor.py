import numpy as np

from neomatrix._backend import Tensor


class TestTensorCreationFromNumpy:
    def test_1d(self):
        t = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert t.shape == [3]
        assert np.allclose(t.to_numpy(), [1.0, 2.0, 3.0])

    def test_2d(self):
        arr = np.arange(6.0, dtype=np.float32).reshape(2, 3)
        t = Tensor.from_numpy(arr)
        assert t.shape == [2, 3]
        assert np.allclose(t.to_numpy(), arr)

    def test_preserves_dtype(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        t = Tensor.from_numpy(arr)
        assert t.to_numpy().dtype == np.float32


class TestTensorCreationDirect:
    def test_1d(self):
        t = Tensor([3], [1.0, 2.0, 3.0])
        assert t.shape == [3]
        assert np.allclose(t.to_numpy(), [1.0, 2.0, 3.0])

    def test_2d(self):
        t = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert t.shape == [2, 3]
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.allclose(t.to_numpy(), expected)

    def test_single_element(self):
        t = Tensor([1], [42.0])
        assert t.shape == [1]
        assert np.allclose(t.to_numpy(), [42.0])


class TestTensorCreationZeros:
    def test_1d(self):
        t = Tensor.zeros([5])
        assert t.shape == [5]
        assert np.allclose(t.to_numpy(), 0.0)

    def test_2d(self):
        t = Tensor.zeros([3, 4])
        assert t.shape == [3, 4]
        assert np.allclose(t.to_numpy(), 0.0)


class TestTensorCreationRandom:
    def test_1d(self):
        t = Tensor.random([5])
        assert t.shape == [5]
        assert len(t) == 5

    def test_2d(self):
        t = Tensor.random([3, 4])
        assert t.shape == [3, 4]

    def test_custom_range(self):
        t = Tensor.random([100], start=0.0, end=1.0)
        data = t.to_numpy()
        assert np.all(data >= 0.0) and np.all(data <= 1.0)

    def test_default_range(self):
        t = Tensor.random([1000])
        data = t.to_numpy()
        assert np.all(data >= -1.0) and np.all(data <= 1.0)


class TestTensorProperties:
    def test_shape_returns_list(self, tensor_2d):
        assert isinstance(tensor_2d.shape, list)
        assert tensor_2d.shape == [3, 4]

    def test_ndim(self, tensor_2d):
        assert tensor_2d.ndim == 2

    def test_ndim_1d(self, tensor_1d):
        assert tensor_1d.ndim == 1

    def test_len(self, tensor_1d):
        assert len(tensor_1d) == 5

    def test_len_2d(self, tensor_2d):
        assert len(tensor_2d) == 12

    def test_data_getter(self, tensor_1d):
        data = tensor_1d.data
        assert isinstance(data, np.ndarray)
        assert np.allclose(data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_data_setter(self):
        t = Tensor.zeros([3])
        new_data = np.array([7.0, 8.0, 9.0], dtype=np.float32)
        t.data = new_data
        assert np.allclose(t.to_numpy(), [7.0, 8.0, 9.0])


class TestTensorArithmeticTensorTensor:
    def test_add(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
        b = Tensor.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        c = a + b
        assert np.allclose(c.to_numpy(), [4.0, 6.0])

    def test_sub(self):
        a = Tensor.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        b = Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
        c = a - b
        assert np.allclose(c.to_numpy(), [2.0, 2.0])

    def test_mul(self):
        a = Tensor.from_numpy(np.array([2.0, 3.0], dtype=np.float32))
        b = Tensor.from_numpy(np.array([4.0, 5.0], dtype=np.float32))
        c = a * b
        assert np.allclose(c.to_numpy(), [8.0, 15.0])

    def test_div(self):
        a = Tensor.from_numpy(np.array([6.0, 8.0], dtype=np.float32))
        b = Tensor.from_numpy(np.array([2.0, 4.0], dtype=np.float32))
        c = a / b
        assert np.allclose(c.to_numpy(), [3.0, 2.0])


class TestTensorArithmeticScalar:
    def test_add_scalar_right(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = a + 1.0
        assert np.allclose(b.to_numpy(), [2.0, 3.0, 4.0])

    def test_add_scalar_left(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = 1.0 + a
        assert np.allclose(b.to_numpy(), [2.0, 3.0, 4.0])

    def test_sub_scalar_right(self):
        a = Tensor.from_numpy(np.array([3.0, 4.0, 5.0], dtype=np.float32))
        b = a - 1.0
        assert np.allclose(b.to_numpy(), [2.0, 3.0, 4.0])

    def test_sub_scalar_left(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = 10.0 - a
        assert np.allclose(b.to_numpy(), [9.0, 8.0, 7.0])

    def test_mul_scalar_right(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = a * 2.0
        assert np.allclose(b.to_numpy(), [2.0, 4.0, 6.0])

    def test_mul_scalar_left(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = 2.0 * a
        assert np.allclose(b.to_numpy(), [2.0, 4.0, 6.0])

    def test_div_scalar_right(self):
        a = Tensor.from_numpy(np.array([4.0, 6.0, 8.0], dtype=np.float32))
        b = a / 2.0
        assert np.allclose(b.to_numpy(), [2.0, 3.0, 4.0])

    def test_neg(self):
        a = Tensor.from_numpy(np.array([1.0, -2.0, 3.0], dtype=np.float32))
        b = -a
        assert np.allclose(b.to_numpy(), [-1.0, 2.0, -3.0])


class TestTensorReshape:
    def test_reshape_returns_new_tensor(self, tensor_1d):
        reshaped = tensor_1d.reshape([1, 5])
        assert reshaped.shape == [1, 5]
        assert tensor_1d.shape == [5]

    def test_reshape_2d_to_1d(self):
        t = Tensor.from_numpy(np.arange(6.0, dtype=np.float32).reshape(2, 3))
        reshaped = t.reshape([6])
        assert reshaped.shape == [6]

    def test_reshape_inplace(self):
        t = Tensor.from_numpy(np.arange(6.0, dtype=np.float32).reshape(2, 3))
        t.reshape_inplace([3, 2])
        assert t.shape == [3, 2]


class TestTensorTranspose:
    def test_transpose_returns_new(self):
        t = Tensor.from_numpy(np.arange(6.0, dtype=np.float32).reshape(2, 3))
        transposed = t.transpose()
        assert transposed.shape == [3, 2]
        assert t.shape == [2, 3]

    def test_transpose_inplace(self):
        t = Tensor.from_numpy(np.arange(6.0, dtype=np.float32).reshape(2, 3))
        t.transpose_inplace()
        assert t.shape == [3, 2]

    def test_transpose_values(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        t = Tensor.from_numpy(arr)
        transposed = t.transpose()
        assert np.allclose(transposed.to_numpy(), arr.T)


class TestTensorFlatten:
    def test_flatten_returns_new(self):
        t = Tensor.from_numpy(np.arange(12.0, dtype=np.float32).reshape(3, 4))
        flat = t.flatten()
        assert flat.shape == [12]
        assert t.shape == [3, 4]

    def test_flatten_inplace(self):
        t = Tensor.from_numpy(np.arange(12.0, dtype=np.float32).reshape(3, 4))
        t.flatten_inplace()
        assert t.shape == [12]

    def test_flatten_preserves_values(self):
        arr = np.arange(6.0, dtype=np.float32).reshape(2, 3)
        t = Tensor.from_numpy(arr)
        flat = t.flatten()
        assert np.allclose(flat.to_numpy(), arr.flatten())


class TestTensorDot:
    def test_matmul_2d(self):
        a = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        b = Tensor.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
        c = a.dot(b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(c.to_numpy(), expected)

    def test_matmul_shapes(self):
        a = Tensor.from_numpy(np.random.randn(4, 3).astype(np.float32))
        b = Tensor.from_numpy(np.random.randn(3, 5).astype(np.float32))
        c = a.dot(b)
        assert c.shape == [4, 5]


class TestTensorConcatenation:
    def test_cat_1d(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
        b = Tensor.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        c = Tensor.cat([a, b], axis=0)
        assert np.allclose(c.to_numpy(), [1.0, 2.0, 3.0, 4.0])
        assert c.shape == [4]

    def test_cat_2d_axis0(self):
        a = Tensor.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        b = Tensor.from_numpy(np.array([[3.0, 4.0]], dtype=np.float32))
        c = Tensor.cat([a, b], axis=0)
        assert c.shape == [2, 2]

    def test_push_1d(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
        b = Tensor.from_numpy(np.array([3.0], dtype=np.float32))
        result = a.push(b, axis=0)
        assert result is None
        assert np.allclose(a.to_numpy(), [1.0, 2.0, 3.0])

    def test_push_row(self):
        a = Tensor.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        row = Tensor.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        a.push_row(row)
        assert a.shape == [2, 2]


class TestTensorNumpyInterop:
    def test_to_numpy(self, tensor_2d):
        arr = tensor_2d.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4)

    def test_array_protocol(self):
        t = Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
        arr = np.asarray(t)
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, [1.0, 2.0])

    def test_roundtrip(self):
        original = np.random.randn(3, 4).astype(np.float32)
        t = Tensor.from_numpy(original)
        restored = t.to_numpy()
        assert np.allclose(original, restored)


class TestTensorIndexing:
    def test_getitem_1d(self):
        t = Tensor([3], [10.0, 20.0, 30.0])
        assert t[0] == 10.0
        assert t[1] == 20.0
        assert t[2] == 30.0

    def test_setitem_1d(self):
        t = Tensor([3], [1.0, 2.0, 3.0])
        t[0] = 99.0
        assert t[0] == 99.0

    def test_getitem_2d(self):
        t = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
        assert t[0, 0] == 1.0
        assert t[1, 1] == 4.0

    def test_setitem_2d(self):
        t = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
        t[0, 1] = 99.0
        assert t[0, 1] == 99.0


class TestTensorRepr:
    def test_repr(self):
        t = Tensor([2], [1.0, 2.0])
        assert isinstance(repr(t), str)

    def test_str(self):
        t = Tensor([2], [1.0, 2.0])
        assert isinstance(str(t), str)
