import numpy as np
import pytest

from neomatrix import layers, losses, optimizers, metrics, Model
from neomatrix._backend import Tensor


class TestModelCreation:
    def test_create_empty_model(self):
        model = Model([])
        assert model.layers == []

    def test_create_model_with_layers(self, dense_layer, relu_activation):
        model = Model([dense_layer, relu_activation])
        assert len(model.layers) == 2


class TestModelCompile:
    def test_compile_basic(self):
        model = Model([layers.Dense(10, 5, layers.Init.He)])
        model.compile(
            loss_function=losses.MSE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
        )
        assert model.loss_function is not None
        assert model.optimizer is not None

    def test_compile_with_metrics(self):
        model = Model([layers.Dense(10, 5, layers.Init.He)])
        model.compile(
            loss_function=losses.MSE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
            metrics=[metrics.Accuracy(), metrics.MSE()],
        )
        assert len(model.metrics) == 2


class TestModelPredict:
    def test_predict_single_layer(self):
        model = Model([layers.Dense(10, 5, layers.Init.He)])
        model.compile(
            loss_function=losses.MSE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
        )
        x = Tensor.from_numpy(np.random.randn(4, 10).astype(np.float32))
        y_pred = model.predict(x)
        assert y_pred.shape == [4, 5]

    def test_predict_multiple_layers(self):
        model = Model(
            [
                layers.Dense(10, 5, layers.Init.He),
                layers.ReLU(),
                layers.Dense(5, 3, layers.Init.He),
            ]
        )
        model.compile(
            loss_function=losses.MSE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
        )
        x = Tensor.from_numpy(np.random.randn(4, 10).astype(np.float32))
        y_pred = model.predict(x)
        assert y_pred.shape == [4, 3]


class TestModelBackward:
    def test_backward_basic(self):
        model = Model([layers.Dense(10, 5, layers.Init.He)])
        model.compile(
            loss_function=losses.MSE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
        )
        x = Tensor.from_numpy(np.random.randn(4, 10).astype(np.float32))
        y_true = Tensor.from_numpy(np.random.randn(4, 5).astype(np.float32))

        output_before = model.predict(x, training=True).to_numpy().copy()
        y_pred = model.predict(x, training=True)
        model.backward(y_true, y_pred)
        model.optimizer.step()

        model.optimizer.zero_grad()
        output_after = model.predict(x, training=True).to_numpy()

        assert not np.allclose(output_before, output_after), (
            "Weights should change after backward + optimizer.step()"
        )


class TestModelFit:
    def test_fit_regression(self):
        model = Model([layers.Dense(10, 1, layers.Init.He)])
        model.compile(
            loss_function=losses.MSE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
            logger_enabled=False,
        )

        x = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        y = Tensor.from_numpy(np.random.randn(32, 1).astype(np.float32))

        model.fit(
            training_x=x,
            training_y=y,
            val_x=None,  # type: ignore[arg-type]
            val_y=None,  # type: ignore[arg-type]
            epochs=2,
            batch_size=8,  # type: ignore[arg-type]
        )

    def test_fit_classification(self):
        model = Model(
            [
                layers.Dense(20, 10, layers.Init.He),
                layers.ReLU(),
                layers.Dense(10, 3, layers.Init.Xavier),
                layers.Softmax(),
            ]
        )
        model.compile(
            loss_function=losses.CCE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
            metrics=[metrics.Accuracy()],
            logger_enabled=False,
        )

        x = Tensor.from_numpy(np.random.randn(32, 20).astype(np.float32))
        y = Tensor.from_numpy(np.eye(3)[np.random.randint(0, 3, 32)].astype(np.float32))

        model.fit(
            training_x=x,
            training_y=y,
            val_x=None,  # type: ignore[arg-type]
            val_y=None,  # type: ignore[arg-type]
            epochs=2,
            batch_size=8,  # type: ignore[arg-type]
        )

    def test_fit_requires_compile(self):
        model = Model([layers.Dense(10, 1, layers.Init.He)])
        x = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        y = Tensor.from_numpy(np.random.randn(32, 1).astype(np.float32))

        with pytest.raises((ValueError, AttributeError)):
            model.fit(
                training_x=x,
                training_y=y,
                val_x=None,  # type: ignore[arg-type]
                val_y=None,  # type: ignore[arg-type]
                epochs=1,
                batch_size=8,
            )


class TestOptimizedGradient:
    def test_softmax_cce_optimization(self):
        model = Model([layers.Dense(10, 3, layers.Init.Xavier), layers.Softmax()])
        model.compile(
            loss_function=losses.CCE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
        )
        assert model.use_optimized_gradient is True

    def test_sigmoid_bce_optimization(self):
        model = Model([layers.Dense(10, 1, layers.Init.He), layers.Sigmoid()])
        model.compile(
            loss_function=losses.BCE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
        )
        assert model.use_optimized_gradient is True

    def test_no_optimization(self):
        model = Model([layers.Dense(10, 3, layers.Init.He), layers.ReLU()])
        model.compile(
            loss_function=losses.MSE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01),
        )
        assert model.use_optimized_gradient is False


class TestLogisticRegression:
    def test_creation(self):
        from neomatrix.model import LogisticRegression

        model = LogisticRegression(input_nodes=10, learning_rate=0.01)
        assert len(model.layers) == 2
        assert isinstance(model.layers[0], layers.Dense)
        assert isinstance(model.layers[1], layers.Sigmoid)

    def test_fit(self):
        from neomatrix.model import LogisticRegression

        model = LogisticRegression(input_nodes=10, learning_rate=0.01)

        x = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        y = Tensor.from_numpy(
            np.random.randint(0, 2, 32).reshape(-1, 1).astype(np.float32)
        )

        model.fit(
            training_x=x,
            training_y=y,
            val_x=None,  # type: ignore[arg-type]
            val_y=None,  # type: ignore[arg-type]
            epochs=2,
            batch_size=8,  # type: ignore[arg-type]
        )


class TestSoftmaxRegression:
    def test_creation(self):
        from neomatrix.model import SoftmaxRegression

        model = SoftmaxRegression(input_nodes=10, output_nodes=3, learning_rate=0.01)
        assert len(model.layers) == 2
        assert isinstance(model.layers[0], layers.Dense)
        assert isinstance(model.layers[1], layers.Softmax)

    def test_fit(self):
        from neomatrix.model import SoftmaxRegression

        model = SoftmaxRegression(input_nodes=10, output_nodes=3, learning_rate=0.01)

        x = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))
        y = Tensor.from_numpy(np.eye(3)[np.random.randint(0, 3, 32)].astype(np.float32))

        model.fit(
            training_x=x,
            training_y=y,
            val_x=None,  # type: ignore[arg-type]
            val_y=None,  # type: ignore[arg-type]
            epochs=2,
            batch_size=8,  # type: ignore[arg-type]
        )
