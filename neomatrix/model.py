"""
Module for ML models. It provides classes for the most common ML algorithms:
- Neural Network
- Linear Regression
- Logistic Regression
- Softmax Regression
"""

from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from neomatrix import layers, losses, metrics, optimizers, utils
    from neomatrix._backend import Tensor


__all__ = [
    "Model",
    "LinearRegression",
    "LogisticRegression",
    "SoftmaxRegression",
]


class Model:
    """
    High-level neural network model for training and inference.

    Provides Keras-style API with ``compile()``, ``fit()``, and ``predict()``
    methods. Automatically detects Softmax+CCE and Sigmoid+BCE for optimized
    gradient computation during backpropagation.

    **Constructor:**
        ```python
        Model(layers: list[Layer])
        ```

    **Parameters:**
        - `layers` (list[Layer]): Sequential list of layer instances (Dense, ReLU, Softmax, etc.)

    **Example:**
        ```python
        from neomatrix import Model, layers, losses, optimizers
        from neomatrix._backend import Tensor
        import numpy as np

        # Build model architecture
        model = Model([
            layers.Dense(128, 784, layers.Init.He),
            layers.ReLU(),
            layers.Dense(10, 128, layers.Init.Xavier),
            layers.Softmax(),
        ])

        # Compile with loss and optimizer
        model.compile(
            loss_function=losses.CCE(),
            optimizer=optimizers.GradientDescent(learning_rate=0.01)
        )

        # Prepare data
        x_train = Tensor.from_numpy(np.random.randn(1000, 784).astype(np.float32))
        y_train = Tensor.from_numpy(np.eye(10)[np.random.randint(0, 10, 1000)])

        # Train model
        model.fit(
            training_x=x_train,
            training_y=y_train,
            val_x=None,
            val_y=None,
            epochs=10,
            batch_size=32
        )

        # Make predictions
        x_test = Tensor.from_numpy(np.random.randn(5, 784).astype(np.float32))
        y_pred = model.predict(x_test)
        ```

    **Workflow:**
        1. **Build**: Create Model with list of layers
        2. **Compile**: Set loss function and optimizer (registers parameters)
        3. **Fit**: Train on data (handles batching, forward/backward/update automatically)
        4. **Predict**: Run inference on new data

    **Optimized Gradients (Automatic):**
        - Detects **Softmax + CCE** → Uses `∂L/∂z = softmax(z) - y` (simplified gradient)
        - Detects **Sigmoid + BCE** → Uses `∂L/∂z = σ(z) - y` (simplified gradient)
        - Other combinations → Uses standard chain rule derivatives

    **Attributes:**
        - `layers` (list[Layer]): Layer sequence
        - `loss_function` (LossFunction): Loss function set by compile()
        - `optimizer` (Optimizer): Optimizer set by compile()
        - `metrics` (list): Metrics to track during training (optional)
        - `use_optimized_gradient` (bool): Whether optimized gradient is active
    """

    def __init__(self, layers: list[layers.Layer]):
        self.layers = layers
        self._use_optimized = None

    def _detect_optimization(self):
        """Internal: Detect if Softmax+CCE or Sigmoid+BCE fusion is possible."""
        if self._use_optimized is not None:
            return

        last_layer = self.layers[-1]
        if isinstance(last_layer, layers.Softmax) and isinstance(
            self.loss_function, losses.CCE
        ):
            self._use_optimized = True
        elif isinstance(last_layer, layers.Sigmoid) and isinstance(
            self.loss_function, losses.BCE
        ):
            self._use_optimized = True
        else:
            self._use_optimized = False

    @property
    def use_optimized_gradient(self) -> bool:
        """Check if optimized gradient computation is active."""
        if self._use_optimized is None:
            self._detect_optimization()
        return self._use_optimized or False

    def compile(
        self,
        loss_function: losses.LossFunction,
        optimizer: optimizers.Optimizer,
        metrics: Optional[list[metrics.Metric]] = None,
        logger_enabled: bool = True,
    ):
        """
        Configure the model for training.

        Registers layer parameters with the optimizer and detects opportunities
        for optimized gradient computation (Softmax+CCE or Sigmoid+BCE).

        **Must be called before fit().**

        **Parameters:**
            - `loss_function` (LossFunction): Loss function instance (MSE, CCE, BCE, etc.)
            - `optimizer` (Optimizer): Optimizer instance (GradientDescent, Adam, etc.)
            - `metrics` (list, optional): List of metric instances to track (Accuracy, MSE, etc.)
            - `logger_enabled` (bool): Enable logging during training. Default: True.

        **Example:**
            ```python
            from neomatrix.metrics import Accuracy, MSE

            model.compile(
                loss_function=losses.CCE(),
                optimizer=optimizers.GradientDescent(learning_rate=0.01),
                metrics=[Accuracy(), MSE()]
            )
            ```

        **Side Effects:**
            - Sets `self.loss_function`, `self.optimizer`, `self.metrics`
            - Calls `optimizer.register_params()` with all layer parameters
            - Detects and enables optimized gradients if applicable
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics or []
        self._logger_enabled = logger_enabled
        self._detect_optimization()

        params = []
        for layer in self.layers:
            if isinstance(layer, layers.TrainableLayer):
                p = layer.get_parameters()
                if p:
                    params.append(p)

        # Register parameters with optimizer
        self.optimizer.register_params(params)

    def predict(self, x: Tensor, training: bool = False) -> Tensor:
        """
        Run forward pass through all layers (inference).

        **Parameters:**
            - `x` (Tensor): Input data tensor
            - `training` (bool): Whether to run in training mode (affects Dropout, BatchNorm). Default: False.

        **Returns:**
            - `Tensor`: Model predictions (output of final layer)

        **Example:**
            ```python
            x_test = Tensor.from_numpy(np.random.randn(10, 784).astype(np.float32))
            y_pred = model.predict(x_test)  # Shape: [10, num_classes]
            ```
        """
        for layer in self.layers:
            x = layer.forward(x, training)

        return x

    def backward(self, y_true: Tensor, y_pred: Tensor):
        """
        Run backward pass through all layers (backpropagation).

        Computes gradients for all layer parameters. If optimized gradient is
        enabled (Softmax+CCE or Sigmoid+BCE), skips the last layer's backward pass.

        **Parameters:**
            - `y_true` (Tensor): Ground truth labels
            - `y_pred` (Tensor): Model predictions (output from predict())

        **Side Effects:**
            - Accumulates gradients in all layer parameters (via Arc<Mutex<Tensor>>)
            - If optimized gradient active, temporarily removes last layer from self.layers

        **Note:**
            Call `optimizer.zero_grad()` before each forward pass to reset gradients.

        **Example:**
            ```python
            # Manual training step
            optimizer.zero_grad()
            y_pred = model.predict(x_batch)
            loss_val = loss_fn.call(y_true, y_pred)
            model.backward(y_true, y_pred)  # Computes and accumulates gradients
            optimizer.step()  # Updates weights using gradients
            ```
        """
        if self.use_optimized_gradient and isinstance(
            self.loss_function, losses.FusedLossFunction
        ):
            output_grads = self.loss_function.backward_optimized(
                y_true=y_true, y_pred=y_pred
            )
            layers_to_backprop = self.layers[:-1]
        else:
            output_grads = self.loss_function.backward(y_true=y_true, y_pred=y_pred)
            layers_to_backprop = self.layers

        for layer in reversed(layers_to_backprop):
            output_grads = layer.backward(output_grads)

        return

    def fit(
        self,
        training_x: Tensor,
        training_y: Tensor,
        val_x: Tensor,
        val_y: Tensor,
        epochs: int,
        batch_size: int,
    ):
        """
        Train the model on provided data.

        Automatically handles batching, forward pass, loss computation, backpropagation,
        parameter updates, and metric tracking for the specified number of epochs.

        **Parameters:**
            - `training_x` (Tensor): Training input data
            - `training_y` (Tensor): Training labels (ground truth)
            - `val_x` (Tensor, optional): Validation input data
            - `val_y` (Tensor, optional): Validation labels
            - `epochs` (int): Number of training epochs
            - `batch_size` (int): Number of samples per batch

        **Raises:**
            - `ValueError`: If `compile()` has not been called before fit()

        **Example:**
            ```python
            from neomatrix.metrics import Accuracy, MSE

            model.compile(
                loss_function=losses.MSE(),
                optimizer=optimizers.GradientDescent(learning_rate=0.01),
                metrics=[Accuracy(), MSE()]
            )

            model.fit(
                training_x=x_train,
                training_y=y_train,
                val_x=x_val,
                val_y=y_val,
                epochs=50,
                batch_size=32
            )
            ```

        **Training Loop (Per Epoch):**
            1. Split data into batches
            2. For each batch:
                - Zero gradients
                - Forward pass (predict)
                - Compute loss
                - Backward pass (compute gradients)
                - Update parameters (optimizer.step)
            3. Compute training metrics
            4. If validation data provided: compute validation metrics
            5. Log epoch results

        **Logging:**
            - Training logs show epoch, train_loss, and metrics
            - Validation logs show val_loss and validation metrics
            - Disable with `logger_enabled=False` in compile()
        """

        if self.loss_function is None or self.optimizer is None:
            raise ValueError(
                "Call model.compile(loss_function=..., optimizer=...) first"
            )

        if training_x.ndim == 1:
            training_x = training_x.reshape([1, len(training_x)])
            training_y = training_y.reshape([1, len(training_y)])

        if self._logger_enabled:
            logger.info("Training started:")
            logger.info(f"  Epochs: {epochs}")
            logger.info(f"  Batch size: {batch_size}")

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_val_loss = 0
            batches_x = utils.get_batches(training_x, batch_size)
            batches_y = utils.get_batches(training_y, batch_size)

            for metric in self.metrics:
                metric.reset()

            for i, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
                self.optimizer.zero_grad()
                pred = self.predict(x=batch_x, training=True)
                loss = self.loss_function.call(y_true=batch_y, y_pred=pred)
                self.backward(y_true=batch_y, y_pred=pred)
                self.optimizer.step()

                epoch_loss += loss
                for metric in self.metrics:
                    metric.update(batch_y, pred)

            train_loss = epoch_loss / len(batches_x) if batches_x else 0
            train_metrics = {m.__class__.__name__: m.compute() for m in self.metrics}

            for metric in self.metrics:
                metric.reset()

            if val_x is not None and val_y is not None:
                batches_x = utils.get_batches(val_x, batch_size)
                batches_y = utils.get_batches(val_y, batch_size)

                for i, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
                    pred = self.predict(x=batch_x, training=False)
                    val_loss = self.loss_function.call(y_true=batch_y, y_pred=pred)
                    epoch_val_loss += val_loss
                    for metric in self.metrics:
                        metric.update(batch_y, pred)
            else:
                val_metrics = {}

            val_loss = epoch_val_loss / len(batches_x) if batches_x else 0
            val_metrics = {m.__class__.__name__: m.compute() for m in self.metrics}

            if self._logger_enabled:
                msg = f"Epoch {epoch + 1}/{epochs} | train_loss: {train_loss:.4f}"
                for name, value in train_metrics.items():
                    msg += f" | {name}: {value:.4f}"
                if val_x is not None and val_y is not None:
                    msg += f" | val_loss: {val_loss:.4f}"
                    for name, value in val_metrics.items():
                        msg += f" | val_{name}: {value:.4f}"
                logger.info(msg)

        return


class LinearRegression(Model):
    """
    Pre-configured linear regression model: ``y = Wx + b``.

    A single Dense layer with MSE loss and GradientDescent optimizer.
    Equivalent to building a Model manually with one Dense layer and
    calling ``compile()`` with ``MSE`` and ``GradientDescent``.

    **Constructor:**
        ```python
        LinearRegression(input_nodes, output_nodes, learning_rate)
        ```

    **Parameters:**
        - `input_nodes` (int): Number of input features.
        - `output_nodes` (int): Number of output targets.
        - `learning_rate` (float): Learning rate for gradient descent.

    **Example:**
        ```python
        from neomatrix.model import LinearRegression
        from neomatrix._backend import Tensor
        import numpy as np

        model = LinearRegression(input_nodes=3, output_nodes=1, learning_rate=0.01)

        x = Tensor.from_numpy(np.random.randn(100, 3).astype(np.float32))
        y = Tensor.from_numpy(np.random.randn(100, 1).astype(np.float32))

        model.fit(
            training_x=x, training_y=y,
            val_x=None, val_y=None,
            epochs=50, batch_size=32,
        )

        predictions = model.predict(x)
        ```
    """

    def __init__(self, input_nodes: int, output_nodes: int, learning_rate: float):
        super().__init__(layers=[layers.Dense(input_nodes, output_nodes)])
        self.compile(
            loss_function=losses.MSE(),
            optimizer=optimizers.GradientDescent(learning_rate=learning_rate),
        )


class LogisticRegression(Model):
    """Logistic regression model for binary classification.

    A single Dense layer with Sigmoid activation and BCE loss.
    Equivalent to: Sigmoid(Dense(1, input_nodes)) with BCE loss.

    **Constructor:**
        ```python
        LogisticRegression(input_nodes: int, learning_rate: float)
        ```

    **Parameters:**
        - `input_nodes` (int): Number of input features
        - `learning_rate` (float): Learning rate for gradient descent

    **Example:**
        ```python
        from neomatrix.model import LogisticRegression

        model = LogisticRegression(input_nodes=10, learning_rate=0.01)
        ```
    """

    def __init__(self, input_nodes: int, learning_rate: float):
        super().__init__(layers=[layers.Dense(input_nodes, 1), layers.Sigmoid()])
        self.compile(
            loss_function=losses.BCE(),
            optimizer=optimizers.GradientDescent(learning_rate=learning_rate),
        )


class SoftmaxRegression(Model):
    """Softmax regression model for multi-class classification.

    A Dense layer followed by Softmax activation with CCE loss.
    Equivalent to: Softmax(Dense(output_nodes, input_nodes)) with CCE loss.

    **Constructor:**
        ```python
        SoftmaxRegression(input_nodes: int, output_nodes: int, learning_rate: float)
        ```

    **Parameters:**
        - `input_nodes` (int): Number of input features
        - `output_nodes` (int): Number of output classes
        - `learning_rate` (float): Learning rate for gradient descent

    **Example:**
        ```python
        from neomatrix.model import SoftmaxRegression

        model = SoftmaxRegression(input_nodes=784, output_nodes=10, learning_rate=0.01)
        ```
    """

    def __init__(self, input_nodes: int, output_nodes: int, learning_rate: float):
        super().__init__(
            layers=[layers.Dense(input_nodes, output_nodes), layers.Softmax()]
        )
        self.compile(
            loss_function=losses.CCE(),
            optimizer=optimizers.GradientDescent(learning_rate=learning_rate),
        )
