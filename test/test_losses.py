import numpy as np

from neomatrix import losses
from neomatrix._backend import Tensor


class TestMSE:
    def test_call(self, mse_loss):
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.1, 1.9, 3.2]], dtype=np.float32))
        loss = mse_loss.call(y_true, y_pred)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_backward_shape(self, mse_loss):
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.1, 1.9, 3.2]], dtype=np.float32))
        grad = mse_loss.backward(y_true, y_pred)
        assert grad.shape == [1, 3]

    def test_zero_loss_on_perfect_prediction(self, mse_loss):
        # IMPORTANT: Must use two separate Tensor objects with identical data.
        # Passing the same Tensor object as both args deadlocks (Arc<Mutex> double-lock).
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        loss = mse_loss.call(y_true, y_pred)
        assert np.isclose(loss, 0.0, atol=1e-6)


class TestMAE:
    def test_call(self, mae_loss):
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.5, 1.8, 3.5]], dtype=np.float32))
        loss = mae_loss.call(y_true, y_pred)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_backward_shape(self, mae_loss):
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.5, 1.8, 3.5]], dtype=np.float32))
        grad = mae_loss.backward(y_true, y_pred)
        assert grad.shape == [1, 3]


class TestBCE:
    def test_call(self, bce_loss):
        y_true = Tensor.from_numpy(np.array([[0.0, 1.0, 1.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[0.1, 0.9, 0.8]], dtype=np.float32))
        loss = bce_loss.call(y_true, y_pred)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_backward_shape(self, bce_loss):
        y_true = Tensor.from_numpy(np.array([[0.0, 1.0, 1.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[0.1, 0.9, 0.8]], dtype=np.float32))
        grad = bce_loss.backward(y_true, y_pred)
        assert grad.shape == [1, 3]

    def test_backward_optimized_shape(self, bce_loss):
        y_true = Tensor.from_numpy(np.array([[0.0, 1.0, 1.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[0.1, 0.9, 0.8]], dtype=np.float32))
        grad = bce_loss.backward_optimized(y_true, y_pred)
        assert grad.shape == [1, 3]

    def test_fused_loss_function_protocol(self, bce_loss):
        assert isinstance(bce_loss, losses.FusedLossFunction)


class TestCCE:
    def test_call(self, cce_loss):
        y_true = Tensor.from_numpy(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        y_pred = Tensor.from_numpy(
            np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
        )
        loss = cce_loss.call(y_true, y_pred)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_backward_shape(self, cce_loss):
        y_true = Tensor.from_numpy(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[0.7, 0.2, 0.1]], dtype=np.float32))
        grad = cce_loss.backward(y_true, y_pred)
        assert grad.shape == [1, 3]

    def test_backward_optimized_shape(self, cce_loss):
        y_true = Tensor.from_numpy(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[0.7, 0.2, 0.1]], dtype=np.float32))
        grad = cce_loss.backward_optimized(y_true, y_pred)
        assert grad.shape == [1, 3]

    def test_fused_loss_function_protocol(self, cce_loss):
        assert isinstance(cce_loss, losses.FusedLossFunction)


class TestHuberLoss:
    def test_call(self):
        loss_fn = losses.HuberLoss(1.0)
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.1, 1.9, 10.0]], dtype=np.float32))
        loss = loss_fn.call(y_true, y_pred)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_backward_shape(self):
        loss_fn = losses.HuberLoss(1.0)
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.1, 1.9, 10.0]], dtype=np.float32))
        grad = loss_fn.backward(y_true, y_pred)
        assert grad.shape == [1, 3]


class TestHingeLoss:
    def test_call(self, hinge_loss):
        y_true = Tensor.from_numpy(np.array([[-1.0, 1.0, 1.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[-0.5, 2.0, 0.3]], dtype=np.float32))
        loss = hinge_loss.call(y_true, y_pred)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_backward_shape(self, hinge_loss):
        y_true = Tensor.from_numpy(np.array([[-1.0, 1.0, 1.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[-0.5, 2.0, 0.3]], dtype=np.float32))
        grad = hinge_loss.backward(y_true, y_pred)
        assert grad.shape == [1, 3]


class TestLossFunctionProtocol:
    def test_mse_is_loss_function(self, mse_loss):
        assert isinstance(mse_loss, losses.LossFunction)

    def test_mae_is_loss_function(self, mae_loss):
        assert isinstance(mae_loss, losses.LossFunction)

    def test_bce_is_loss_function(self, bce_loss):
        assert isinstance(bce_loss, losses.LossFunction)

    def test_cce_is_loss_function(self, cce_loss):
        assert isinstance(cce_loss, losses.LossFunction)
