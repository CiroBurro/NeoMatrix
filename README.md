# NeoMatrix

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**NeoMatrix** is a Keras-inspired machine learning library that combines the computational performance of Rust with the simplicity and flexibility of Python for building and training neural networks.

## Architecture Overview

NeoMatrix is structured as a multi-layered system:

```
┌─────────────────────────────────────────┐
│   Python API (neomatrix)                │  ← High-level model building
│   • NeuralNetwork, Layer, Optimizer     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   PyO3 Bindings (neomatrix-python-      │  ← Python ↔ Rust bridge
│   wrapper)                               │     (In Development)
│   • Expose Rust types to Python         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   Rust Core (neomatrix-core)            │  ← High-performance backend
│   • Tensor ops, layers, math            │     ✅ Fully Implemented
│   • Parallel computation (Rayon)        │
└─────────────────────────────────────────┘
```

### Components

#### 1. **neomatrix-core** (Rust Library) ✅

The computational engine of NeoMatrix, written in pure Rust with comprehensive documentation.

**Features:**
- **Tensors**: Multi-dimensional arrays (`f32`) with automatic shape inference and broadcasting
- **Layers**: Composable neural network building blocks
  - `Dense`: Fully-connected layer with learnable weights and biases
  - Activation layers: `ReLu`, `Sigmoid`, `Tanh`, `Softmax`
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, Linear
- **Loss Functions**:
  - Regression: MSE, MAE, Huber
  - Classification: Binary Cross-Entropy, Categorical Cross-Entropy, Hinge
- **Weight Initialization**:
  - Xavier/Glorot (optimal for Sigmoid/Tanh)
  - He (optimal for ReLU)
  - Random uniform
- **Parallel Computing**: Rayon-powered matrix multiplication for multi-core performance
- **Error Handling**: Comprehensive `Result`-based error types (no panics in library code)


#### 2. **neomatrix-python-wrapper** (PyO3 Bindings) 🚧

*Status: In Development*

Python bindings exposing the Rust core to Python via PyO3. This crate will provide:
- Zero-copy tensor operations between Python and Rust
- Pythonic API for all Rust types (Tensor, Layer, Dense, etc.)
- Seamless conversion between NumPy arrays and NeoMatrix tensors
- Full access to parallel computation features


#### 3. **neomatrix** (Python Package) 🚧

*Status: In Development*

High-level Python API inspired by Keras, built on top of the Rust backend. Will provide:
- `NeuralNetwork`: Sequential model container
- `Layer`: High-level layer abstraction
- `Optimizer`: Training algorithms (SGD, MiniBatchGD, BatchGD)
- `Cost`: Loss function wrappers
- Model persistence (save/load trained models)
- Dataset utilities for batch processing

---

## Features

### Current Implementation ✅

- [x] **Rust Backend**: Complete core library with tensor operations and neural network layers
- [x] **Comprehensive Documentation**: All modules, functions, and mathematical operations documented in English
- [x] **Mathematical Rigor**: Backpropagation algorithms with gradient formulas
- [x] **Parallel Computation**: Multi-threaded matrix multiplication via Rayon
- [x] **Type Safety**: `Result`-based error handling throughout
- [x] **Operator Overloading**: Pythonic tensor arithmetic (`+`, `-`, `*`, `/`)
- [x] **Iterator Support**: Tensors implement Rust's `Iterator` trait
- [x] **216 Unit Tests**: Comprehensive test coverage for all modules

### Activation Functions

| Function | Formula | Derivative | Use Case |
|----------|---------|------------|----------|
| **ReLU** | `max(0, x)` | `1 if x > 0, else 0` | Hidden layers (most common) |
| **Sigmoid** | `1 / (1 + e^(-x))` | `σ(x) · (1 - σ(x))` | Binary classification output |
| **Tanh** | `(e^x - e^(-x)) / (e^x + e^(-x))` | `1 - tanh²(x)` | Hidden layers (zero-centered) |
| **Softmax** | `e^(x_i) / Σ_j e^(x_j)` | Jacobian matrix | Multi-class classification output |
| **Linear** | `x` | `1` | Regression output |

### Loss Functions

| Function | Formula | Best For |
|----------|---------|----------|
| **MSE** | `(1/n) Σ(y - ŷ)²` | Regression |
| **MAE** | `(1/n) Σ|y - ŷ|` | Robust regression |
| **Huber** | Smooth MSE/MAE transition | Outlier-resistant regression |
| **BCE** | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]` | Binary classification (with Sigmoid) |
| **CCE** | `-Σ y_i · log(ŷ_i)` | Multi-class classification (with Softmax) |
| **Hinge** | `max(0, 1 - y·ŷ)` | SVM-style classification |

### Weight Initialization

| Strategy | Formula | Recommended For |
|----------|---------|-----------------|
| **Xavier** | `W ~ N(0, √(2/(n_in + n_out)))` | Sigmoid, Tanh activations |
| **He** | `W ~ N(0, √(2/n_in))` | ReLU, Leaky ReLU activations |
| **Random** | `U(a, b)` uniform | Legacy (not recommended) |

---

## Installation

### Building from Source

**Prerequisites:**
- Rust 1.70+ ([Install Rust](https://www.rust-lang.org/tools/install))
- Python 3.8+
- maturin (`pip install maturin` or `uv add maturin`)

**Build the Rust core:**
```bash
cd neomatrix-core
cargo build --release
cargo test  # Run 216 unit tests
cargo doc --no-deps --open  # View documentation
```

**Build the Python extension (when wrapper is ready):**
```bash
# Development build
maturin develop

# Release wheel
maturin build --release
```

---

## Usage Examples

### Current Status (Rust Core)

The Rust core is fully functional and can be used directly:

```rust
use neomatrix_core::tensor::Tensor;
use neomatrix_core::layers::{dense::Dense, init::Init, Layer};
use neomatrix_core::math::activations::{Relu, ActivationFunction};

// Create tensors
let input = Tensor::new(vec![1, 784], vec![/* ... */]).unwrap();

// Build a dense layer with He initialization
let mut layer = Dense::new(784, 128, Some(Init::He), None);

// Forward pass
let output = layer.forward(&input, true).unwrap();

// Compute loss and backward pass
let grad = /* compute gradient from loss */;
let input_grad = layer.backward(&grad).unwrap();

// Access parameters for optimization
if let Some(params) = layer.get_params_and_grads() {
    for (weight, gradient) in params {
        // Update weights: w = w - lr * grad
    }
}
```


## Project Structure

```
NeoMatrix/
├── neomatrix-core/          # ✅ Rust core library (COMPLETE)
│   ├── src/
│   │   ├── lib.rs          # Crate root
│   │   ├── tensor/         # Tensor operations
│   │   ├── math/           # Activations, losses, matmul
│   │   ├── layers/         # Dense, activation layers, init
│   │   ├── errors.rs       # Error types
│   │   └── test/           # 216 unit tests
│   └── Cargo.toml
├── neomatrix-python-wrapper/ # 🚧 PyO3 bindings (IN DEVELOPMENT)
│   ├── src/
│   │   ├── lib.rs          # Module registration
│   │   ├── tensor.rs       # Python Tensor class
│   │   └── ...             # Layer, Dense, etc. (planned)
│   └── Cargo.toml
├── neomatrix/              # 🚧 Python package (IN DEVELOPMENT)
│   ├── core/               # High-level API
│   │   ├── model.py        # NeuralNetwork class
│   │   └── optimizer.py    # Training algorithms
│   └── utils/              # Utilities (dataset, metrics)
├── examples/               # Usage examples
│   ├── NeuralNetwork.py
│   └── LinearRegression.py
├── pyproject.toml          # Python project config (maturin)
├── Cargo.toml              # Rust workspace
└── README.md               # This file
```

---

## Development Status

### Completed ✅

- [x] Complete Rust core implementation (`neomatrix-core`)
- [x] Comprehensive English documentation for all modules
- [x] Tensor operations with operator overloading
- [x] Neural network layers (Dense, Activation)
- [x] 5 activation functions with backpropagation
- [x] 6 loss functions with numerical stability
- [x] 3 weight initialization strategies
- [x] Parallel matrix multiplication (Rayon)
- [x] 216 unit tests with 100% pass rate
- [x] Error handling with `Result` types (no panics)

### In Progress 🚧

- [ ] PyO3 Python bindings (`neomatrix-python-wrapper`)
  - [x] Basic Tensor class (760 lines, iterator support)
  - [ ] Layer, Dense, Activation bindings
  - [ ] Loss function bindings
  - [ ] NumPy interoperability
- [ ] High-level Python API (`neomatrix` package)
  - [ ] NeuralNetwork class
  - [ ] Optimizer implementations
  - [ ] Model save/load
  - [ ] Dataset utilities

### Planned 📋

- [ ] Additional optimizers: Adam, RMSprop, Adagrad
- [ ] Evaluation metrics: Accuracy, Precision, Recall, F1-score
- [ ] Regularization: L1, L2, Dropout
- [ ] Batch normalization layer
- [ ] Convolutional layers (Conv2D, MaxPool2D)
- [ ] Recurrent layers (LSTM, GRU)
- [ ] GPU acceleration (CUDA/Metal)

---

## Documentation

### Rust Core Documentation

Full API documentation is available via `cargo doc`:

```bash
cd neomatrix-core
cargo doc --no-deps --open
```

Key documentation sections:
- **Module-level docs** (`//!`) describe the purpose of each module
- **Struct/Enum docs** (`///`) explain data structures and their fields
- **Function docs** (`///`) detail parameters, returns, and usage examples
- **Mathematical formulas** (`//`) accompany all activation/loss computations
- **Implementation notes** (`//`) clarify complex algorithms (e.g., backpropagation)

### Examples

See the `examples/` directory for usage demonstrations:
- `LinearRegression.py` - Simple linear regression model
- `NeuralNetwork.py` - Multi-layer neural network for classification

---

## Performance

NeoMatrix leverages Rust's performance and safety guarantees:

- **Zero-cost abstractions**: Generic types compile to optimal machine code
- **Memory safety**: No garbage collection, no null pointers, no data races
- **Parallel execution**: Rayon automatically parallelizes matrix operations across CPU cores
- **SIMD optimization**: ndarray uses vectorized operations where possible

Benchmarks (coming soon): Training performance comparisons with NumPy, PyTorch, and TensorFlow.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Roadmap

### Phase 1: Core Implementation ✅ (COMPLETE)
- Rust core library with full documentation
- Tensor operations and linear algebra
- Dense layers and activation functions
- Loss functions with numerical stability
- Weight initialization strategies
- Parallel computation support

### Phase 2: Python Integration 🚧 (IN PROGRESS)
- Complete PyO3 bindings for all Rust types
- NumPy interoperability
- High-level Python API
- Model persistence (save/load)

### Phase 3: Advanced Features 📋 (PLANNED)
- Additional optimizers (Adam, RMSprop, etc.)
- Regularization techniques
- Evaluation metrics
- Batch normalization

### Phase 4: Expansion 🔮 (FUTURE)
- Convolutional layers for image processing
- Recurrent layers for sequence modeling
- GPU acceleration
- Distributed training

---

**Status**: Active development | Core complete, Python integration in progress
