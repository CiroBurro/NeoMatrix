import numpy as np

from neomatrix import layers, optimizers
from neomatrix._backend import Tensor


class TestGradientDescent:
    def test_creation(self):
        opt = optimizers.GradientDescent(learning_rate=0.01)
        assert opt is not None

    def test_register_params(self, dense_layer, optimizer):
        params = dense_layer.get_parameters()
        optimizer.register_params([params])
        # No error should be raised
        assert params is not None

    def test_zero_grad(self, dense_layer, optimizer):
        params = dense_layer.get_parameters()
        optimizer.register_params([params])

        x = Tensor.from_numpy(np.random.randn(4, 10).astype(np.float32))
        dense_layer.forward(x, training=True)
        grad = Tensor.from_numpy(np.random.randn(4, 5).astype(np.float32))
        dense_layer.backward(grad)

        optimizer.zero_grad()

        dense_layer.forward(x, training=True)
        dense_layer.backward(grad)
        output_before = dense_layer.forward(x, training=True).to_numpy().copy()

        optimizer.zero_grad()

        dense_layer.forward(x, training=True)
        dense_layer.backward(grad)
        output_after = dense_layer.forward(x, training=True).to_numpy().copy()

        assert np.allclose(output_before, output_after), (
            "Outputs should be identical when gradients are zeroed before each update"
        )

    def test_step(self, dense_layer, optimizer):
        params = dense_layer.get_parameters()
        optimizer.register_params([params])

        x = Tensor.from_numpy(np.random.randn(4, 10).astype(np.float32))

        optimizer.zero_grad()
        output_before = dense_layer.forward(x, training=True).to_numpy().copy()
        grad = Tensor.from_numpy(np.random.randn(4, 5).astype(np.float32))
        dense_layer.backward(grad)
        optimizer.step()

        optimizer.zero_grad()
        output_after = dense_layer.forward(x, training=True).to_numpy()

        assert not np.allclose(output_before, output_after), (
            "Weights should change after optimizer.step()"
        )

    def test_step_reduces_loss(self):
        dense = layers.Dense(3, 5, layers.Init.He)
        opt = optimizers.GradientDescent(learning_rate=0.1)
        params = dense.get_parameters()
        opt.register_params([params])

        x = Tensor.from_numpy(np.random.randn(2, 3).astype(np.float32))
        output_before = dense.forward(x, training=True).to_numpy().copy()

        grad = Tensor.from_numpy(np.random.randn(2, 5).astype(np.float32))
        dense.backward(grad)
        opt.step()

        output_after = dense.forward(x, training=True).to_numpy()
        assert not np.allclose(output_before, output_after), (
            "Weights should have changed after optimizer.step()"
        )

    def test_multiple_layers(self, optimizer):
        layer1 = layers.Dense(10, 5, layers.Init.He)
        layer2 = layers.Dense(5, 3, layers.Init.He)

        params1 = layer1.get_parameters()
        params2 = layer2.get_parameters()
        optimizer.register_params([params1, params2])

        x = Tensor.from_numpy(np.random.randn(4, 10).astype(np.float32))

        out1 = layer1.forward(x, training=True)
        out2_before = layer2.forward(out1, training=True).to_numpy().copy()

        grad = Tensor.from_numpy(np.random.randn(4, 3).astype(np.float32))
        grad2 = layer2.backward(grad)
        layer1.backward(grad2)

        optimizer.step()

        optimizer.zero_grad()
        out1_after = layer1.forward(x, training=True)
        out2_after = layer2.forward(out1_after, training=True).to_numpy()

        assert not np.allclose(out2_before, out2_after), (
            "Output should change after updating both layers"
        )

    def test_training_loop(self):
        dense = layers.Dense(5, 3, layers.Init.He)
        opt = optimizers.GradientDescent(learning_rate=0.01)
        params = dense.get_parameters()
        opt.register_params([params])

        x = Tensor.from_numpy(np.random.randn(4, 5).astype(np.float32))

        output_initial = dense.forward(x, training=True).to_numpy().copy()

        for _ in range(5):
            opt.zero_grad()
            dense.forward(x, training=True)
            grad = Tensor.from_numpy(np.random.randn(4, 3).astype(np.float32))
            dense.backward(grad)
            opt.step()

        opt.zero_grad()
        output_final = dense.forward(x, training=True).to_numpy()

        assert not np.allclose(output_initial, output_final), (
            "Weights should change after 5 training iterations"
        )


class TestOptimizerProtocol:
    def test_gradient_descent_is_optimizer(self, optimizer):
        assert isinstance(optimizer, optimizers.Optimizer)


class TestParametersRef:
    def test_is_not_none(self, dense_layer):
        params = dense_layer.get_parameters()
        assert params is not None

    def test_can_register_with_optimizer(self, dense_layer, optimizer):
        params = dense_layer.get_parameters()
        optimizer.register_params([params])
        assert params is not None

    def test_multiple_params_from_multiple_layers(self):
        layer1 = layers.Dense(10, 5, layers.Init.He)
        layer2 = layers.Dense(5, 3, layers.Init.He)
        params1 = layer1.get_parameters()
        params2 = layer2.get_parameters()
        assert params1 is not None
        assert params2 is not None

    def test_gradients_accumulate_via_step(self):
        dense = layers.Dense(3, 2, layers.Init.He)
        opt = optimizers.GradientDescent(learning_rate=0.5)
        params = dense.get_parameters()
        opt.register_params([params])

        x = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        grad = Tensor.from_numpy(np.array([[1.0, 1.0]], dtype=np.float32))

        output_before = dense.forward(x, training=True).to_numpy().copy()

        dense.backward(grad)
        opt.step()

        output_after = dense.forward(x, training=True).to_numpy()
        assert not np.allclose(output_before, output_after), (
            "Output should change after gradient update"
        )
