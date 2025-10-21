import unittest
from neomatrix.core import Tensor


class TensorMethodsTest(unittest.TestCase):
    def test_tensor_new(self):
        t = Tensor([1, 2, 3], [1, 2, 3, 4, 5, 6])
        self.assertEqual(t.dimension, 3)
        self.assertEqual(t.shape, [1, 2, 3])
        print(t.data)

    def test_tensor_zeros(self ):
        t = Tensor.zeros([2, 3])
        self.assertEqual(t.dimension, 2)
        self.assertEqual(t.shape, [2, 3])
        print(t.data)

    def test_tensor_random(self ):
        t = Tensor.random([2,3])
        self.assertEqual(t.dimension, 2)
        self.assertEqual(t.shape, [2, 3])
        print(t.data)

    def test_tensor_from_numpy(self):
        import numpy as np
        arr = np.array([[1, 2, 3]], dtype=np.float64)
        t = Tensor.from_numpy(arr)
        self.assertEqual(t.dimension, 2)
        self.assertEqual(t.shape, [1, 3])
        print(t.data)

    def test_tensor_dot_transpose(self):
        t_1 = Tensor([2, 4], [1, 3, 5, 7, 9, 11, 13, 15])
        t_2 = Tensor([4], [2, 4, 6, 8])

        result_1 = t_2.dot(t_2)
        self.assertEqual(result_1.dimension, 0)
        self.assertEqual(result_1.shape, [])
        print(result_1.data)

        result_2 = t_2.dot(t_1.transpose())
        self.assertEqual(result_2.dimension, 1)
        self.assertEqual(result_2.shape, [2])
        print(result_2.data)

        result_3 = t_1.dot(t_2)
        self.assertEqual(result_3.dimension, 1)
        self.assertEqual(result_3.shape, [2])
        print(result_3.data)

        result_4 = t_1.dot(t_1.transpose())
        self.assertEqual(result_4.dimension, 2)
        self.assertEqual(result_4.shape, [2, 2])
        print(result_4.data)

    def test_tensor_sum(self):
        t_1 = Tensor([4], [2, 4, 6, 8])
        t_2 = Tensor([4], [1, 3, 5, 7])

        result = t_1.tensor_sum(t_2)
        self.assertEqual(result.dimension, 1)
        self.assertEqual(result.shape, [4])
        print(result.data)

    def test_tensor_subtraction(self):
        t_1 = Tensor([4], [2, 4, 6, 8])
        t_2 = Tensor([4], [1, 3, 5, 7])

        result = t_1.tensor_subtraction(t_2)
        self.assertEqual(result.dimension, 1)
        self.assertEqual(result.shape, [4])
        print(result.data)

    def test_tensor_multiplication(self):
        t_1 = Tensor([4], [2, 4, 6, 8])
        t_2 = Tensor([4], [1, 3, 5, 7])

        result = t_1.tensor_multiplication(t_2)
        self.assertEqual(result.dimension, 1)
        self.assertEqual(result.shape, [4])
        print(result.data)

    def test_tensor_division(self):
        t_1 = Tensor([4], [2, 4, 6, 8])
        t_2 = Tensor([4], [1, 3, 5, 7])

        result = t_1.tensor_division(t_2)
        self.assertEqual(result.dimension, 1)
        self.assertEqual(result.shape, [4])
        print(result.data)

    def test_tensor_scalar_sum(self):
        t_1 = Tensor([4, 2], [2, 4, 6, 8, 1, 3, 5, 7])
        scalar = 2.0

        result = t_1.scalar_sum(scalar)
        self.assertEqual(result.dimension, 2)
        self.assertEqual(result.shape, [4, 2])
        print(result.data)

    def test_tensor_scalar_subtraction(self):
        t_1 = Tensor([4, 2], [2, 4, 6, 8, 1, 3, 5, 7])
        scalar = 2.0

        result = t_1.scalar_subtraction(scalar)
        self.assertEqual(result.dimension, 2)
        self.assertEqual(result.shape, [4, 2])
        print(result.data)

    def test_tensor_scalar_multiplication(self):
        t_1 = Tensor([4, 2], [2, 4, 6, 8, 1, 3, 5, 7])
        scalar = 2.0

        result = t_1.scalar_multiplication(scalar)
        self.assertEqual(result.dimension, 2)
        self.assertEqual(result.shape, [4, 2])
        print(result.data)

    def test_tensor_scalar_division(self):
        t_1 = Tensor([4, 2], [2, 4, 6, 8, 1, 3, 5, 7])
        scalar = 2.0

        result = t_1.scalar_division(scalar)
        self.assertEqual(result.dimension, 2)
        self.assertEqual(result.shape, [4, 2])
        print(result.data)

    def test_tensor_length(self):
        t = Tensor([3, 4], [1,2,3,4,5,6,7,8,9,10,11,12])
        len = t.length()
        self.assertEqual(12, len)

    def test_tensor_reshape(self):
        t = Tensor([4, 2], [2, 4, 6, 8, 1, 3, 5, 7])

        t.reshape([2, 4])
        self.assertEqual(t.dimension, 2)
        self.assertEqual(t.shape, [2, 4])
        print(t.data)

        t.reshape([8])
        self.assertEqual(t.dimension, 1)
        self.assertEqual(t.shape, [8])
        print(t.data)

        t.reshape([2, 2, 2])
        self.assertEqual(t.dimension, 3)
        self.assertEqual(t.shape, [2, 2, 2])
        print(t.data)

    def test_tensor_flatten(self):
        t = Tensor([2, 2, 2], [2, 4, 6, 8, 1, 3, 5, 7])

        t.flatten()
        self.assertEqual(t.dimension, 1)
        self.assertEqual(t.shape, [8])
        print(t.data)

    def test_tensor_push_cat(self):
        t_1 = Tensor([2, 2], [2, 4, 6, 8])
        t_2 = Tensor([2, 2], [1, 3, 5, 7])

        t_1.push_cat(t_2, 1)
        self.assertEqual(t_1.dimension, 2)
        self.assertEqual(t_1.shape, [2, 4])
        print(t_1.data)

        t_1 = Tensor([2, 2], [2, 4, 6, 8])
        t_1.push_cat(t_2, 0)
        self.assertEqual(t_1.dimension, 2)
        self.assertEqual(t_1.shape, [4, 2])
        print(t_1.data)

        t_1 = Tensor([2, 2], [2, 4, 6, 8])
        t_1.flatten()
        t_2.flatten()
        t_3 = Tensor([3], [9, 8, 7])

        result = t_1.cat([t_2, t_3], 0)

        self.assertEqual(result.dimension, 1)
        self.assertEqual(result.shape, [11])
        print(result.data)

    def test_tensor_push_row(self):
        t = Tensor([2, 2], [2, 4, 6, 8])
        row = Tensor([2], [1, 3])

        t.push_row(row)
        self.assertEqual(t.dimension, 2)
        self.assertEqual(t.shape, [3, 2])
        print(t.data)

    def test_tensor_push_column(self):
        t = Tensor([2, 2], [2, 4, 6, 8])
        col = Tensor([2], [1, 3])

        t.push_column(col)
        self.assertEqual(t.dimension, 2)
        self.assertEqual(t.shape, [2, 3])
        print(t.data)

if __name__ == '__main__':
    unittest.main()
