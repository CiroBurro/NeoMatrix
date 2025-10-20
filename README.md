# NeoMatrix

NeoMatrix is a tensorflow-inspired machine learning library, written in Rust with an high-level Python API. It aims to combine the computational speed of Rust with the simplicity and flexibility of Python for building and training neural networks.

## How It Works

The core of NeoMatrix is a Rust library named `rustybrain`, which handles all computationally intensive operations:
- **Tensor Operations**: Manipulation of multi-dimensional arrays (tensors).
- **Linear Algebra**: Optimized dot and matrix products with support for parallelism.
- **Propagation**: Calculation of forward and backward propagation passes.

A Python wrapper exposes these features through a simple API inspired by libraries like Keras. This allows users to define, train, and evaluate deep learning models without writing any Rust code.

## Features and Functionality

- **Rust Backend**: Leverages the speed and safety of Rust for high-performance numerical computations.
- **Pythonic API**: A clean and simple interface for model building.
- **Flexible Structures**:
    - `Tensor`: A multi-dimensional tensor compatible with NumPy for data manipulation.
    - `Layer`: The basic building block for neural network layers, with customizable weights, biases, and activation functions.
- **Included Activation Functions**:
    - `Linear`
    - `Sigmoid`
    - `ReLU`
    - `Tanh`
    - `Softmax`
- **Cost Functions**:
    - `MeanSquaredError`
    - `MeanAbsoluteError`
    - `BinaryCrossEntropy` (optimized for `Sigmoid`)
    - `CategoricalCrossEntropy` (optimized for `Softmax`)
    - `HuberLoss`
    - `HingeLoss`
- **Optimizers**:
    - `SGD` (Stochastic Gradient Descent)
    - `BatchGD` (Batch Gradient Descent)
    - `MiniBatchGD` (Mini-Batch Gradient Descent)
- **Parallel Computing**: Support for parallel execution of complex operations to maximize efficiency.

## Usage

The NeoMatrix API is designed to be intuitive. Here is an example of how to define, compile, and train a neural network.

### 1. Create a Model

You can build a `NeuralNetwork` model by assembling a list of `Layer`s. Each layer requires the number of neurons, the input length, and an activation function.

```python
import neomatrix.core as core
import neomatrix.core.optimizer as opt
from neomatrix.utils.dataset import get_batches # Utility function for data handling
import numpy as np

# Define the network architecture
layers = [
    core.Layer(nodes=10, input_len=2, activation=core.Activation.Relu),
    core.Layer(nodes=5, input_len=10, activation=core.Activation.Relu),
    core.Layer(nodes=1, input_len=5, activation=core.Activation.Sigmoid)
]

# Create the model
model = core.NeuralNetwork(
    layers=layers,
    cost_function=core.Cost.MeanSquaredError(),
    learning_rate=0.01
)
```

### 2. Prepare the Data

Input data must be NeoMatrix `Tensor` objects. You can easily create them from NumPy arrays.

```python
# Sample data (training)
X_train_np = np.random.rand(100, 2)
y_train_np = np.random.randint(0, 2, (100, 1))

# Sample data (validation)
X_val_np = np.random.rand(50, 2)
y_val_np = np.random.randint(0, 2, (50, 1))

# Convert to NeoMatrix Tensors
training_set = core.Tensor.from_numpy(X_train_np)
training_targets = core.Tensor.from_numpy(y_train_np)
val_set = core.Tensor.from_numpy(X_val_np)
val_targets = core.Tensor.from_numpy(y_val_np)
```

### 3. Train the Model

Use the `fit` method to train the network. Select an optimizer and the number of epochs.

```python
# Choose an optimizer
optimizer = opt.MiniBatchGD(training_batch_size=10, validation_batch_size=5)

# Train the model
model.fit(
    training_set=training_set,
    training_targets=training_targets,
    val_set=val_set,
    val_targets=val_targets,
    optimizer=optimizer,
    epochs=20,
    parallel=True  # Enable parallel computation
)
```

### 4. Make Predictions

After training, you can use the `predict` method to get the model's predictions.

```python
# Test data
X_test_np = np.array([[0.1, 0.9], [0.8, 0.2]])
test_data = core.Tensor.from_numpy(X_test_np)

# Make predictions
predictions = model.predict(test_data)

print("Predictions:", predictions.get_data())
```

## TODO List
- [ ] **Implement more optimizers**: Adam, RMSprop, Adagrad.
- [ ] **Add examples**: Implementation linear, logistic, and softmax regression, and a simple neuralnetwork
- [ ] **Save and Load Models**: Functionality to serialize and deserialize trained models.
- [ ] **Advanced Documentation**: Create more detailed documentation with examples and tutorials.
- [ ] **Add evaluation metrics**: Accuracy, Precision, Recall, F1-score.
