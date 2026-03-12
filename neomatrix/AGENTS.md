# NEOMATRIX — PYTHON PACKAGE

## OVERVIEW

High-level Python API for NeoMatrix. Provides Keras-inspired interface (`Model`, `fit()`, `compile()`) built on top of Rust backend via PyO3 bindings.

**Philosophy**: Hide complexity, expose power. Users write Keras-like code, get Rust performance.

---

## STRUCTURE

```
neomatrix/
├── __init__.py          # Package entry point
├── model.py             # Model class with compile/fit
├── layers.py            # Re-exports Dense/ReLU/etc + Layer protocols
├── optimizers.py        # Re-exports GradientDescent + Optimizer protocol
├── losses.py            # Re-exports MSE/BCE/etc + LossFunction protocols
└── utils.py             # get_batches utility
```

---

## FILE DESCRIPTIONS

### `model.py` — Model Class

**Status**: ⚠️ **HAS BUGS** — needs complete rewrite

**Purpose**: Sequential model container with `compile()` and `fit()` methods

**Current Interface**:
```python
class Model:
    def __init__(self, layers: list[Layer])
    def compile(self, loss_function: LossFunction, optimizer: Optimizer, metrics=None)
    def predict(self, x: Tensor, training: bool = False) -> Tensor
    def backward(self, y_true: Tensor, y_pred: Tensor)
    def fit(self, training_x, training_y, val_x, val_y, epochs, batch_size)
```

**Known Bugs** (10 total — see root AGENTS.md):
1. Typo `_use_optimize` vs `_use_optimized` (inconsistent naming)
2. Property references wrong attribute
3. Variable `layer` undefined (should be `l`)
4. Missing `optimizer.register_params()` call
5. `self.layers.reverse()` returns `None` (should be `reversed(self.layers)`)
6. References `self.loss` but attr is `self.loss_function`
7. Wrong `zip` usage in `enumerate((batches_x, batches_y))`
8. Calls `self.forward()` but method is `self.predict()`
9. Calls `self.backward(grad, ...)` but signature is `backward(y_true, y_pred)`
10. Missing `optimizer.step()` call after backward pass

**Correct Training Loop Pattern**:
```python
# In compile()
params = [layer.get_parameters() for layer in self.layers 
          if hasattr(layer, 'get_parameters')]
optimizer.register_params(params)

# In fit() — one training step
optimizer.zero_grad()                              # 1. Reset gradients
y_pred = self.predict(x, training=True)            # 2. Forward pass
loss_val = loss_fn.call(y_true, y_pred)            # 3. Compute loss
grad = loss_fn.backward(y_true, y_pred)            # 4. Loss gradient
self.backward(y_true, y_pred)                      # 5. Backprop (accumulates grads)
optimizer.step()                                   # 6. Update weights
```

---

### `layers.py` — Layer Re-exports + Protocols

**Status**: ✅ Complete

**Purpose**: Re-export Rust layer bindings from `neomatrix._backend` and define structural protocols

**Exported Types**:
- `Dense` — Fully-connected layer (from `_backend.Dense`)
- `ReLU`, `Sigmoid`, `Tanh`, `Softmax` — Activation layers (from `_backend`)
- `Init` — Weight initialization enum: `Init.Xavier`, `Init.He`, `Init.Random` (from `_backend`)

**Protocols** (runtime checkable):
```python
@runtime_checkable
class Layer(Protocol):
    """All layers expose forward() and backward()"""
    def forward(self, input: Tensor, training: bool) -> Tensor: ...
    def backward(self, output_gradient: Tensor) -> Tensor: ...

@runtime_checkable
class TrainableLayer(Layer, Protocol):
    """Layers with learnable parameters (e.g., Dense)"""
    def get_parameters(self) -> ParametersRef: ...

@runtime_checkable
class FusedLayer(Layer, Protocol):
    """Activation layers supporting fused loss gradients"""
    def backward_optimized(self, output_gradient: Tensor) -> Tensor: ...
```

**Usage**:
```python
from neomatrix import layers

# Create layers
dense = layers.Dense(128, 784, layers.Init.He)
relu = layers.ReLU()

# Check capabilities
if isinstance(dense, layers.TrainableLayer):
    params = dense.get_parameters()  # Returns ParametersRef

if isinstance(relu, layers.FusedLayer):
    grad = relu.backward_optimized(loss_grad)
```

---

### `optimizers.py` — Optimizer Re-exports + Protocol

**Status**: ✅ Complete

**Purpose**: Re-export Rust optimizer bindings and define structural protocol

**Exported Types**:
- `GradientDescent` — Standard SGD optimizer (from `_backend.GradientDescent`)
- `ParametersRef` — Shared parameter reference (from `_backend.ParametersRef`)

**Protocol** (runtime checkable):
```python
@runtime_checkable
class Optimizer(Protocol):
    """Structural protocol for any optimizer"""
    def register_params(self, params: list[ParametersRef]): ...
    def step(self): ...
    def zero_grad(self): ...
```

**GradientDescent Interface**:
```python
class GradientDescent:
    def __init__(self, learning_rate: float)
    def register_params(self, params: list[ParametersRef])
    def step()  # Updates all registered parameters: θ = θ - lr·∇θ
    def zero_grad()  # Resets all gradients to zero
```

**Usage**:
```python
from neomatrix import optimizers

# Create optimizer
opt = optimizers.GradientDescent(learning_rate=0.01)

# Register layer parameters
params = [layer.get_parameters() for layer in layers 
          if hasattr(layer, 'get_parameters')]
opt.register_params(params)

# Training loop
opt.zero_grad()
# ... forward pass, loss, backward ...
opt.step()
```

**ParametersRef Structure**:
```python
class ParametersRef:
    weights: Tensor   # Layer weights (Arc<Mutex<Tensor>> in Rust)
    biases: Tensor    # Layer biases
    w_grads: Tensor   # Weight gradients (accumulated during backward)
    b_grads: Tensor   # Bias gradients
```

---

### `losses.py` — Loss Function Re-exports + Protocols

**Status**: ✅ Complete

**Purpose**: Re-export Rust loss function bindings and define structural protocols

**Exported Types**:
- `MSE` — Mean Squared Error (from `_backend.MSE`)
- `MAE` — Mean Absolute Error (from `_backend.MAE`)
- `BCE` — Binary Cross-Entropy (from `_backend.BCE`)
- `CCE` — Categorical Cross-Entropy (from `_backend.CCE`)
- `HuberLoss` — Huber Loss (from `_backend.HuberLoss`)
- `HingeLoss` — Hinge Loss (from `_backend.HingeLoss`)

**Protocols** (runtime checkable):
```python
@runtime_checkable
class LossFunction(Protocol):
    """All loss functions expose call() and backward()"""
    def call(self, y_true: Tensor, y_pred: Tensor) -> float: ...
    def backward(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...

@runtime_checkable
class FusedLossFunction(LossFunction, Protocol):
    """Loss functions supporting fused gradients with activations"""
    def backward_optimized(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...
```

**Fused Gradient Optimization**:
- `CCE.backward_optimized()` — Returns `softmax(z) - y_true` (use with Softmax activation)
- `BCE.backward_optimized()` — Returns `sigmoid(z) - y_true` (use with Sigmoid activation)

**Usage**:
```python
from neomatrix import losses

# Create loss function
loss_fn = losses.CCE()

# Forward pass (compute loss)
loss_val = loss_fn.call(y_true, y_pred)  # Returns scalar

# Backward pass (compute gradients)
grad = loss_fn.backward(y_true, y_pred)  # Returns Tensor

# Optimized gradient (if last layer is Softmax)
if isinstance(loss_fn, losses.FusedLossFunction):
    grad = loss_fn.backward_optimized(y_true, y_pred)
```

---

### `utils.py` — Utility Functions

**Status**: ✅ Complete

**Purpose**: Helper functions for data processing

**Exported Functions**:
```python
def get_batches(data: Tensor, batch_size: int) -> list[Tensor]:
    """Split data tensor into batches of specified size"""
```

**Usage**:
```python
from neomatrix import utils

# Split training data into mini-batches
batches_x = utils.get_batches(train_x, batch_size=32)
batches_y = utils.get_batches(train_y, batch_size=32)

for batch_x, batch_y in zip(batches_x, batches_y):
    # Train on mini-batch
    ...
```

---

## BACKEND INTEGRATION

All Rust types are accessed via `neomatrix._backend` module (built by PyO3).

**Import Pattern**:
```python
# In Python package files (layers.py, optimizers.py, etc.)
from neomatrix._backend import (
    Tensor,
    Dense, ReLU, Sigmoid, Tanh, Softmax,
    GradientDescent, ParametersRef,
    MSE, MAE, BCE, CCE, HuberLoss, HingeLoss,
    Init,
)

# Users import from neomatrix package
from neomatrix import layers, optimizers, losses
```

---

## DESIGN PATTERNS

### 1. Protocol-Based Interfaces

We use `Protocol` classes (PEP 544) for structural typing, not inheritance:

**Why**:
- Duck typing — types from `_backend` don't need to inherit from protocols
- Runtime checking via `isinstance(obj, Protocol)`
- Type hints for IDEs and mypy

**Example**:
```python
@runtime_checkable
class Layer(Protocol):
    def forward(self, input: Tensor, training: bool) -> Tensor: ...
    def backward(self, output_gradient: Tensor) -> Tensor: ...

# _backend.Dense implements these methods → isinstance(dense, Layer) is True
```

### 2. Re-export Pattern

Python package re-exports Rust types to hide internal structure:

```python
# neomatrix/layers.py
from neomatrix._backend import Dense, ReLU
__all__ = ["Dense", "ReLU", "Layer", "TrainableLayer"]

# User code
from neomatrix.layers import Dense  # Clean import path
```

### 3. Fused Gradient Optimization

Detect Softmax+CCE or Sigmoid+BCE combinations to use optimized gradients:

```python
# In Model._detect_optimization()
last_layer = self.layers[-1]
if isinstance(last_layer, layers.Softmax) and isinstance(self.loss_function, losses.CCE):
    self._use_optimized = True  # Use CCE.backward_optimized() + Softmax.backward_with_logits()
```

---

## CONVENTIONS

### Type Hints
- All public methods have type hints
- Use `TYPE_CHECKING` imports for forward references to avoid circular imports

### Protocols
- All protocols are `@runtime_checkable`
- Protocol methods have `...` bodies (no implementation)

### Error Handling
- Rust errors are converted to Python exceptions by PyO3
- `TensorError` → `RuntimeError`
- `LayerError` → `RuntimeError`

### Naming
- Python naming: `snake_case` for functions/methods, `PascalCase` for classes
- Match Keras conventions where possible: `fit()`, `compile()`, `predict()`

---

## KNOWN ISSUES

### High Priority
1. **`model.py` is broken** — 10 bugs documented above; needs complete rewrite
2. **No metrics support** — `Model.compile(metrics=...)` exists but does nothing
3. **No callbacks support** — No EarlyStopping, ModelCheckpoint, etc.

### Medium Priority
4. **No validation during fit** — `val_x`, `val_y` arguments ignored
5. **No progress reporting** — Silent training loop (no tqdm, no epoch logs)
6. **No model persistence** — No `save()`/`load()` methods

### Low Priority
7. **No docstrings** — Zero documentation in Python files
8. **No type stubs** — No `.pyi` files for `_backend` module

---

## NEXT STEPS

### Immediate (Fix Broken Code)
1. **Rewrite `model.py`**:
   - Fix all 10 bugs
   - Implement correct `compile()` with `optimizer.register_params()`
   - Implement correct `fit()` with `zero_grad()` → `predict()` → `backward()` → `step()`
   - Add fused gradient detection

### Short-term (Complete API)
2. **Implement Metrics**:
   - Base `Metric` class with `update()`, `compute()`, `reset()`
   - `Accuracy`, `MAE`, `Precision`, `Recall`
3. **Implement Callbacks**:
   - Base `Callback` class with hooks (`on_epoch_start`, `on_batch_end`, etc.)
   - `EarlyStopping`, `ModelCheckpoint`, `LearningRateScheduler`
4. **Add validation support** in `fit()`
5. **Add progress reporting** (tqdm or custom)

### Long-term (Polish)
6. **Add docstrings** to all public classes/methods
7. **Create type stubs** (`.pyi`) for `_backend` module
8. **Implement model persistence** (`save()`/`load()`)
9. **Add pytest suite** for Python API

---

## TESTING

### Current Status
- No pytest suite
- Only ad-hoc `test.py` script in project root
- `test_optimizer_refactor.py` validates new optimizer pattern

### Test Coverage Needed
- `Model.compile()` parameter registration
- `Model.fit()` training loop correctness
- `Model.predict()` forward pass
- `Model.backward()` gradient propagation
- Fused gradient optimization detection
- Layer protocol conformance
- Optimizer protocol conformance
- Loss function protocol conformance

---

## ANTI-PATTERNS

**DO NOT**:
- Inherit from `Protocol` classes in `_backend` types (they're structural, not nominal)
- Use `exit(1)` in library code (raise exceptions)
- Use `os.system('clear')` in `fit()` (remove or make optional)
- Use `print()` for progress (use `logging` or callbacks)
- Suppress type errors with `# type: ignore` (fix the underlying issue)

**DO**:
- Use `isinstance(obj, Protocol)` for runtime checks
- Raise descriptive exceptions (`ValueError`, `TypeError`, `RuntimeError`)
- Provide clean import paths (re-export from package root)
- Follow Keras conventions for method names
- Add type hints to all public methods

---

## INTEGRATION WITH RUST BACKEND

### Shared Ownership Pattern (ParametersRef)

The optimizer and layers share references to the same tensors via `Arc<Mutex<Tensor>>` in Rust:

```python
# In Python (high-level view)
dense = layers.Dense(128, 784)
params = dense.get_parameters()  # Returns ParametersRef

# params.weights points to same memory as dense's internal weights
# Calling dense.backward() modifies params.w_grads
# Calling optimizer.step() modifies params.weights
# No copying — shared ownership via Arc
```

**Critical**: This is why `optimizer.register_params()` must be called in `compile()` before training.

### Training Flow (Rust + Python)

```
Python: optimizer.zero_grad()
  → Rust: GradientDescent::zero_grad()
    → Parallel iteration over Vec<ParametersRef>
    → Lock each Arc<Mutex<Tensor>>, set to zeros

Python: y_pred = model.predict(x, training=True)
  → Rust: Dense::forward() for each layer
    → Compute activations, cache for backward

Python: loss_fn.backward(y_true, y_pred)
  → Rust: CCE::backward()
    → Compute ∂L/∂y_pred

Python: model.backward(y_true, y_pred)
  → Rust: Layer::backward() in reverse order
    → Accumulate gradients in Arc<Mutex<Tensor>> (shared with optimizer)

Python: optimizer.step()
  → Rust: GradientDescent::step()
    → Parallel iteration over Vec<ParametersRef>
    → Lock each Arc<Mutex<Tensor>>, update: θ = θ - lr·∇θ
```

---

## REFERENCES

- **Root AGENTS.md**: Project-wide overview, optimizer architecture, Rust core details
- **neomatrix-core/AGENTS.md**: Rust library internals
- **PyO3 docs**: https://pyo3.rs/
- **PEP 544 (Protocols)**: https://www.python.org/dev/peps/pep-0544/
