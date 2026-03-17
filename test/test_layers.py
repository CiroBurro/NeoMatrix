import numpy as np

from neomatrix import layers
from neomatrix._backend import Tensor


class TestDenseLayer:
    def test_creation(self):
        dense = layers.Dense(10, 5, layers.Init.He)
        assert dense is not None

    def test_forward_shape(self, dense_layer, tensor_batch):
        output = dense_layer.forward(tensor_batch, training=True)
        assert output.shape == [32, 5]

    def test_backward_shape(self, dense_layer, tensor_batch):
        dense_layer.forward(tensor_batch, training=True)
        grad_output = Tensor.from_numpy(np.random.randn(32, 5).astype(np.float32))
        grad_input = dense_layer.backward(grad_output)
        assert grad_input.shape == [32, 10]

    def test_get_parameters(self, dense_layer):
        params = dense_layer.get_parameters()
        assert params is not None

    def test_trainable_layer_protocol(self, dense_layer):
        assert isinstance(dense_layer, layers.TrainableLayer)


class TestReLU:
    def test_forward_shape(self, relu_activation, tensor_batch):
        output = relu_activation.forward(tensor_batch, training=True)
        assert output.shape == [32, 10]

    def test_forward_values(self):
        relu = layers.ReLU()
        t = Tensor.from_numpy(np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32))
        output = relu.forward(t, training=True)
        expected = np.array([[0.0, 2.0], [0.0, 4.0]])
        assert np.allclose(output.to_numpy(), expected)

    def test_backward_shape(self, relu_activation, tensor_batch):
        relu_activation.forward(tensor_batch, training=True)
        grad = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        grad_input = relu_activation.backward(grad)
        assert grad_input.shape == [32, 10]

    def test_layer_protocol(self, relu_activation):
        assert isinstance(relu_activation, layers.Layer)


class TestSigmoid:
    def test_forward_shape(self, sigmoid_activation, tensor_batch):
        output = sigmoid_activation.forward(tensor_batch, training=True)
        assert output.shape == [32, 10]

    def test_forward_range(self):
        sigmoid = layers.Sigmoid()
        t = Tensor.from_numpy(np.array([[-10.0, 0.0, 10.0]], dtype=np.float32))
        output = sigmoid.forward(t, training=True)
        data = output.to_numpy()
        assert np.all(data >= 0.0) and np.all(data <= 1.0)

    def test_backward_shape(self, sigmoid_activation, tensor_batch):
        sigmoid_activation.forward(tensor_batch, training=True)
        grad = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        grad_input = sigmoid_activation.backward(grad)
        assert grad_input.shape == [32, 10]

    def test_fused_layer_protocol(self, sigmoid_activation):
        assert isinstance(sigmoid_activation, layers.FusedLayer)

    def test_backward_optimized(self, sigmoid_activation, tensor_batch):
        sigmoid_activation.forward(tensor_batch, training=True)
        grad = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        grad_input = sigmoid_activation.backward_optimized(grad)
        assert grad_input.shape == [32, 10]


class TestTanh:
    def test_forward_shape(self, tanh_activation, tensor_batch):
        output = tanh_activation.forward(tensor_batch, training=True)
        assert output.shape == [32, 10]

    def test_forward_range(self):
        tanh = layers.Tanh()
        t = Tensor.from_numpy(np.array([[-5.0, 0.0, 5.0]], dtype=np.float32))
        output = tanh.forward(t, training=True)
        data = output.to_numpy()
        assert np.all(data >= -1.0) and np.all(data <= 1.0)

    def test_backward_shape(self, tanh_activation, tensor_batch):
        tanh_activation.forward(tensor_batch, training=True)
        grad = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        grad_input = tanh_activation.backward(grad)
        assert grad_input.shape == [32, 10]


class TestSoftmax:
    def test_forward_shape(self, softmax_activation, tensor_batch):
        output = softmax_activation.forward(tensor_batch, training=True)
        assert output.shape == [32, 10]

    def test_forward_sums_to_one(self):
        softmax = layers.Softmax()
        t = Tensor.from_numpy(
            np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
        )
        output = softmax.forward(t, training=True)
        sums = output.to_numpy().sum(axis=1)
        assert np.allclose(sums, 1.0)

    def test_backward_shape(self, softmax_activation, tensor_batch):
        softmax_activation.forward(tensor_batch, training=True)
        grad = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        grad_input = softmax_activation.backward(grad)
        assert grad_input.shape == [32, 10]

    def test_fused_layer_protocol(self, softmax_activation):
        assert isinstance(softmax_activation, layers.FusedLayer)

    def test_backward_optimized(self, softmax_activation, tensor_batch):
        softmax_activation.forward(tensor_batch, training=True)
        grad = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        grad_input = softmax_activation.backward_optimized(grad)
        assert grad_input.shape == [32, 10]


class TestInit:
    def test_xavier(self):
        dense = layers.Dense(10, 5, layers.Init.Xavier)
        assert dense is not None

    def test_he(self):
        dense = layers.Dense(10, 5, layers.Init.He)
        assert dense is not None

    def test_random(self):
        dense = layers.Dense(10, 5, layers.Init.Random)
        assert dense is not None

    def test_default_init(self):
        dense = layers.Dense(10, 5)
        assert dense is not None
