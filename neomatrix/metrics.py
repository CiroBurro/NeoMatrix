from typing import TYPE_CHECKING

import neomatrix

if TYPE_CHECKING:
    # from neomatrix import layers, losses, optimizers, utils
    from neomatrix._backend import Tensor

__all__ = [
    "Metric",
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "MAE",
    "MSE",
]


class Metric:
    """Base class for evaluation metrics.

    All metrics expose ``reset()``, ``update(y_true, y_pred)``, and ``compute()``.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def update(self, y_true: Tensor, y_pred: Tensor):
        pass

    def compute(self) -> float:
        return 0.0


class Accuracy(Metric):
    """Classification accuracy metric.

    Computes the fraction of correct predictions over total predictions.

    **Formula:** accuracy = correct / total
    """

    def __init__(self):
        super().__init__()
        self.correct: int = 0
        self.total: int = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, y_true: Tensor, y_pred: Tensor):
        class_pred = y_pred.to_numpy().argmax(axis=1)
        class_true = y_true.to_numpy().argmax(axis=1)

        self.correct += int((class_pred == class_true).sum())
        self.total += len(class_pred)

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class Precision(Metric):
    def __init__(self):
        self.tp_per_class: dict[int, int] = {}
        self.fp_per_class: dict[int, int] = {}
        super().__init__()

    def reset(self):
        self.tp_per_class.clear()
        self.fp_per_class.clear()

    def update(self, y_true: Tensor, y_pred: Tensor):
        class_pred = y_pred.to_numpy().argmax(axis=1)
        class_true = y_true.to_numpy().argmax(axis=1)

        for pred, true in zip(class_pred, class_true):
            if pred == true:
                self.tp_per_class[pred] = self.tp_per_class.get(pred, 0) + 1
            else:
                self.fp_per_class[pred] = self.fp_per_class.get(pred, 0) + 1

    def compute(self) -> float:
        all_classes = set(self.tp_per_class.keys()) | set(self.fp_per_class.keys())
        if not all_classes:
            return 0.0

        precisions = []
        for cls in all_classes:
            tp = self.tp_per_class.get(cls, 0)
            fp = self.fp_per_class.get(cls, 0)
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
            else:
                precisions.append(0.0)

        return sum(precisions) / len(precisions)


class Recall(Metric):
    def __init__(self):
        self.tp_per_class: dict[int, int] = {}
        self.fn_per_class: dict[int, int] = {}
        super().__init__()

    def reset(self):
        self.tp_per_class.clear()
        self.fn_per_class.clear()

    def update(self, y_true: Tensor, y_pred: Tensor):
        class_pred = y_pred.to_numpy().argmax(axis=1)
        class_true = y_true.to_numpy().argmax(axis=1)

        for pred, true in zip(class_pred, class_true):
            if pred == true:
                self.tp_per_class[pred] = self.tp_per_class.get(pred, 0) + 1
            else:
                self.fn_per_class[true] = self.fn_per_class.get(true, 0) + 1

    def compute(self) -> float:
        all_classes = set(self.tp_per_class.keys()) | set(self.fn_per_class.keys())
        if not all_classes:
            return 0.0

        recalls = []
        for cls in all_classes:
            tp = self.tp_per_class.get(cls, 0)
            fn = self.fn_per_class.get(cls, 0)
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
            else:
                recalls.append(0.0)

        return sum(recalls) / len(recalls)


class F1Score(Metric):
    def __init__(self):
        self.tp_per_class: dict[int, int] = {}
        self.fp_per_class: dict[int, int] = {}
        self.fn_per_class: dict[int, int] = {}
        super().__init__()

    def reset(self):
        self.tp_per_class.clear()
        self.fp_per_class.clear()
        self.fn_per_class.clear()

    def update(self, y_true: Tensor, y_pred: Tensor):
        class_pred = y_pred.to_numpy().argmax(axis=1)
        class_true = y_true.to_numpy().argmax(axis=1)

        for pred, true in zip(class_pred, class_true):
            if pred == true:
                self.tp_per_class[pred] = self.tp_per_class.get(pred, 0) + 1
            else:
                self.fp_per_class[pred] = self.fp_per_class.get(pred, 0) + 1
                self.fn_per_class[true] = self.fn_per_class.get(true, 0) + 1

    def compute(self) -> float:
        all_classes = (
            set(self.tp_per_class.keys())
            | set(self.fp_per_class.keys())
            | set(self.fn_per_class.keys())
        )
        if not all_classes:
            return 0.0

        f1_scores = []
        for cls in all_classes:
            tp = self.tp_per_class.get(cls, 0)
            fp = self.fp_per_class.get(cls, 0)
            fn = self.fn_per_class.get(cls, 0)
            if tp + fp + fn > 0:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if precision + recall > 0:
                    f1_scores.append(2 * precision * recall / (precision + recall))
                else:
                    f1_scores.append(0.0)
            else:
                f1_scores.append(0.0)

        return sum(f1_scores) / len(f1_scores)


class MSE(Metric):
    """Mean Squared Error metric.

    MSE = (1/n) * sum((y_true - y_pred)^2)
    Measures average squared difference between predictions and targets.
    """

    def __init__(self):
        super().__init__()
        self.mse = 0.0
        self.count = 0

    def reset(self):
        self.mse = 0.0
        self.count = 0

    def update(self, y_true: Tensor, y_pred: Tensor):
        self.mse += neomatrix.MSE().call(y_true, y_pred)
        self.count += 1

    def compute(self) -> float:
        return self.mse / self.count if self.count > 0 else 0.0


class MAE(Metric):
    """Mean Absolute Error metric.

    MAE = (1/n) * sum(|y_true - y_pred|)
    Measures average absolute difference between predictions and targets.
    """

    def __init__(self):
        super().__init__()
        self.mae = 0.0
        self.count = 0

    def reset(self):
        self.mae = 0.0
        self.count = 0

    def update(self, y_true: Tensor, y_pred: Tensor):
        self.mae += neomatrix.MAE().call(y_true, y_pred)
        self.count += 1

    def compute(self) -> float:
        return self.mae / self.count if self.count > 0 else 0.0
