import numpy as np

from neomatrix import metrics
from neomatrix._backend import Tensor


class TestAccuracy:
    def test_perfect_predictions(self):
        metric = metrics.Accuracy()
        y_true = Tensor.from_numpy(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        )
        y_pred = Tensor.from_numpy(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        )
        metric.update(y_true, y_pred)
        assert metric.compute() == 1.0

    def test_zero_predictions(self):
        metric = metrics.Accuracy()
        y_true = Tensor.from_numpy(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        y_pred = Tensor.from_numpy(
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        )
        metric.update(y_true, y_pred)
        assert metric.compute() == 0.0

    def test_partial_predictions(self):
        metric = metrics.Accuracy()
        y_true = Tensor.from_numpy(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        )
        y_pred = Tensor.from_numpy(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
        )
        metric.update(y_true, y_pred)
        assert metric.compute() == 0.5

    def test_reset(self):
        metric = metrics.Accuracy()
        y_true = Tensor.from_numpy(np.array([[1.0, 0.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.0, 0.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        metric.reset()
        assert metric.compute() == 0.0


class TestPrecision:
    def test_perfect_precision(self):
        metric = metrics.Precision()
        y_true = Tensor.from_numpy(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        )
        y_pred = Tensor.from_numpy(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        )
        metric.update(y_true, y_pred)
        assert metric.compute() == 1.0

    def test_no_predictions(self):
        metric = metrics.Precision()
        y_true = Tensor.from_numpy(np.array([[1.0, 0.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[0.0, 1.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        assert metric.compute() == 0.0


class TestRecall:
    def test_perfect_recall(self):
        metric = metrics.Recall()
        y_true = Tensor.from_numpy(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        )
        y_pred = Tensor.from_numpy(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        )
        metric.update(y_true, y_pred)
        assert metric.compute() == 1.0


class TestF1Score:
    def test_perfect_f1(self):
        metric = metrics.F1Score()
        y_true = Tensor.from_numpy(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        y_pred = Tensor.from_numpy(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        metric.update(y_true, y_pred)
        assert metric.compute() == 1.0

    def test_zero_f1(self):
        metric = metrics.F1Score()
        y_true = Tensor.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        assert metric.compute() == 0.0


class TestMSEMetric:
    def test_zero_mse(self):
        metric = metrics.MSE()
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        assert metric.compute() == 0.0

    def test_nonzero_mse(self):
        metric = metrics.MSE()
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[2.0, 3.0, 4.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        result = metric.compute()
        assert result > 0.0

    def test_accumulates_across_batches(self):
        metric = metrics.MSE()

        y_true_1 = Tensor.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        y_pred_1 = Tensor.from_numpy(np.array([[1.5, 2.5]], dtype=np.float32))
        metric.update(y_true_1, y_pred_1)

        y_true_2 = Tensor.from_numpy(np.array([[3.0, 4.0]], dtype=np.float32))
        y_pred_2 = Tensor.from_numpy(np.array([[3.5, 4.5]], dtype=np.float32))
        metric.update(y_true_2, y_pred_2)

        result = metric.compute()
        assert result > 0.0

    def test_reset(self):
        metric = metrics.MSE()
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[2.0, 3.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        metric.reset()
        assert metric.compute() == 0.0


class TestMAEMetric:
    def test_zero_mae(self):
        metric = metrics.MAE()
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        assert metric.compute() == 0.0

    def test_nonzero_mae(self):
        metric = metrics.MAE()
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[2.0, 3.0, 4.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        result = metric.compute()
        assert result > 0.0

    def test_reset(self):
        metric = metrics.MAE()
        y_true = Tensor.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        y_pred = Tensor.from_numpy(np.array([[2.0, 3.0]], dtype=np.float32))
        metric.update(y_true, y_pred)
        metric.reset()
        assert metric.compute() == 0.0


class TestMetricBase:
    def test_base_class_exists(self):
        assert hasattr(metrics, "Metric")

    def test_all_metrics_inherit(self):
        for cls in [
            metrics.Accuracy,
            metrics.Precision,
            metrics.Recall,
            metrics.F1Score,
            metrics.MSE,
            metrics.MAE,
        ]:
            assert issubclass(cls, metrics.Metric)
