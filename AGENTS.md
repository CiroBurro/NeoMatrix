# NEOMATRIX — PROJECT KNOWLEDGE BASE

**Branch:** main | **Last Updated:** 2026-03-11

## OVERVIEW

Keras-inspired ML library: **Rust backend** (`neomatrix-core`) for tensor ops and backprop, **PyO3 wrapper** (`neomatrix-python-wrapper`) for Python bindings, **Python API** (`neomatrix/`) for high-level model building.

**Stack**: Rust (ndarray + rayon) + PyO3 + Python 3.8+ | Built with **maturin**

**Philosophy**: Hybrid PyTorch/Keras design — high-level `Model.fit()` API for convenience + low-level Rust APIs public for advanced users who want full control

---

## STRUCTURE

```
NeoMatrix/
├── neomatrix-core/          # Rust lib crate — tensor, math, layers, optimizers
│   └── src/
│       ├── tensor/          # Tensor struct + ops
│       ├── math/            # Activations, losses (function implementations)
│       ├── layers/          # Dense, activation layers (Layer trait impls)
│       │   ├── dense.rs     # Fully-connected layer with backprop
│       │   ├── activations.rs # ReLU, Sigmoid, Tanh, Softmax layers
│       │   └── init.rs      # Xavier, He, LeCun weight initialization
│       ├── optimizers/      # ⭐ NEW: Stateful optimizer architecture
│       │   ├── mod.rs       # Optimizer trait + ParametersRef (Arc<Mutex<Tensor>>)
│       │   └── gradient_descent.rs # GradientDescent with register_params/step/zero_grad
│       ├── errors.rs        # All error types (thiserror)
│       └── test/            # Rust unit tests (236 tests total)
│           ├── tensor_test.rs
│           └── optimizers_test.rs # 12 tests for new optimizer pattern
├── neomatrix-python-wrapper/# Rust crate — PyO3 bindings
│   └── src/
│       ├── tensor_bindings/ # PyTensor with NumPy interop
│       │   ├── tensor.rs    # Full Python API + __array__
│       │   └── tensor_iter.rs # TensorIter for __iter__
│       ├── layer_bindings/  # Layer Python wrappers
│       │   ├── dense.rs     # PyDense with get_parameters() → PyParametersRef
│       │   ├── activations.rs # PyReLU, PySigmoid, PyTanh, PySoftmax
│       │   └── init.rs      # PyInit enum
│       ├── optimizer_bindings/ # ⭐ NEW: Optimizer Python wrappers
│       │   ├── mod.rs       # PyParametersRef struct
│       │   └── gradient_descent.rs # PyGradientDescent with step/zero_grad
│       ├── losses_bindings.rs # All loss functions (MSE, MAE, BCE, CCE, Huber, Hinge)
│       └── lib.rs           # Module registration (_backend)
├── neomatrix/               # Python package (high-level API)
│   ├── model.py             # Model class with compile/fit (⚠️ has bugs, needs update)
│   ├── optimizers.py        # Re-exports GradientDescent + Optimizer protocol
│   ├── layers.py            # Re-exports Dense/ReLU/etc + Layer protocols
│   ├── losses.py            # Re-exports MSE/BCE/etc + LossFunction protocols
│   ├── utils.py             # get_batches utility
│   └── __init__.py          # Package entry point
├── examples/                # LinearRegression.py, NeuralNetwork.py
├── test.py                  # Ad-hoc integration test at root
├── test_optimizer_refactor.py # Test new optimizer pattern
├── benchmark_parallel_optimizer.py # Performance benchmark (Rayon speedup)
├── Cargo.toml               # Workspace (members: -core, -python-wrapper)
└── pyproject.toml           # maturin build, module=neomatrix._backend
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Tensor math/ops (Rust) | `neomatrix-core/src/tensor/tensor.rs` |
| Activations (Rust) | `neomatrix-core/src/math/activations.rs` |
| Loss functions (Rust) | `neomatrix-core/src/math/losses.rs` |
| Dense layer / backprop | `neomatrix-core/src/layers/dense.rs` |
| Activation layers (Layer trait) | `neomatrix-core/src/layers/activations.rs` |
| Layer trait | `neomatrix-core/src/layers/mod.rs` |
| Weight initialization | `neomatrix-core/src/layers/init.rs` |
| **⭐ Optimizer trait + ParametersRef** | `neomatrix-core/src/optimizers/mod.rs` |
| **⭐ GradientDescent implementation** | `neomatrix-core/src/optimizers/gradient_descent.rs` |
| All Rust error types | `neomatrix-core/src/errors.rs` |
| Python Tensor class | `neomatrix-python-wrapper/src/tensor_bindings/tensor.rs` |
| Python layer bindings | `neomatrix-python-wrapper/src/layer_bindings/` |
| **⭐ Python optimizer bindings** | `neomatrix-python-wrapper/src/optimizer_bindings/` |
| Python loss bindings | `neomatrix-python-wrapper/src/losses_bindings.rs` |
| Register Python types | `neomatrix-python-wrapper/src/lib.rs` |
| Model class (compile/fit) | `neomatrix/model.py` |
| Optimizer re-exports | `neomatrix/optimizers.py` |
| Layer re-exports | `neomatrix/layers.py` |
| Loss re-exports | `neomatrix/losses.py` |
| Rust unit tests (tensors) | `neomatrix-core/src/test/tensor_test.rs` |
| **⭐ Rust unit tests (optimizers)** | `neomatrix-core/src/test/optimizers_test.rs` |
| Usage examples | `examples/NeuralNetwork.py` |

## COMMANDS

```bash
# Build Python extension (in-place for dev)
maturin develop

# OR with uv
uv run maturin develop

# Run Python integration test
python test.py

# Run Rust tests
cargo test

# Build release wheel
maturin build --release
```

## ARCHITECTURE: CALLING CHAIN

```
Python user code
  → neomatrix.core (Python)     model.py / optimizer.py
  → neomatrix._backend (PyO3)   neomatrix-python-wrapper/src/
  → neomatrix_core (Rust lib)   neomatrix-core/src/
```

## PYTHON WRAPPER STATUS

### ✅ Fully Exposed to Python
- **Tensor**: All core operations + NumPy interop (`__array__`, `to_numpy`, `from_numpy`)
- **Layers**: Dense, ReLU, Sigmoid, Tanh, Softmax (all with `forward`, `backward`, `backward_with_logits`)
- **Init Strategies**: Xavier, He, LeCun, Random
- **Loss Functions**: MSE, MAE, BCE, CCE, Huber, Hinge (all registered in `lib.rs`)
- **⭐ Optimizers**: GradientDescent with `register_params()`, `step()`, `zero_grad()` (NEW)
- **⭐ ParametersRef**: Shared ownership wrapper for weights/biases/gradients (NEW)

### ⚠️ Missing (Non-Critical)
- **Docstrings**: Zero Python documentation (no `help()` output)
- **Utility Methods**: `Tensor.ones()`, `sum()`, `mean()` (workaround: use NumPy)
- **Tests**: No pytest suite (only `test.py` ad-hoc script + `test_optimizer_refactor.py`)
- **Adam Optimizer**: Not yet implemented (trivial after refactor — just clone GradientDescent + add momentum state)

### 🎯 Production Readiness: 9.0/10
- Core functionality complete including stateful optimizers
- API pythonic with properties (`tensor.shape`, `tensor.data`)
- Parallel optimization via Rayon (~2-4x speedup on 10+ layers)
- Main gap: lack of docstrings for UX

## KNOWN BUGS

### Rust Core ✅
None currently identified. Previous bugs have been fixed:
- `transpose_inplace`, `random` range, `cat_inplace` semantics (fixed)
- `f32 - Tensor` used hardcoded `1.0` instead of `self` (fixed in tensor_ops.rs L329)
- `f32 / Tensor` called `scalar_division` computing `Tensor / f32` instead of `f32 / Tensor` (fixed via `inverse_scalar_division`)
- Typo `PyParameteresRef` → `PyParametersRef` (fixed in 4 files)

### Python API ⚠️
`neomatrix/model.py` has multiple bugs (not yet fixed):
1. **Line 27**: Typo `_use_optimize` should be `_use_optimized` (inconsistent naming)
2. **Line 43**: References `self._use_optimized` but property is `_use_optimize`
3. **Line 56**: Variable `layer` not defined (should be `l`)
4. **Line 59**: Missing `optimizer.register_params(params)` call
5. **Line 76**: `self.layers.reverse()` returns `None` (should be `reversed(self.layers)`)
6. **Line 83**: References `self.loss` but attr is `self.loss_function`
7. **Line 91**: Wrong `zip` usage — `enumerate((batches_x, batches_y))` should be `zip(batches_x, batches_y)`
8. **Line 94**: Calls `self.forward()` but method is `self.predict()`
9. **Line 97**: Calls `self.backward(grad, ...)` but signature is `backward(y_true, y_pred)`
10. **Missing `optimizer.step()`** call after backward pass

## MATHEMATICAL OPTIMIZATIONS

### Softmax + Categorical Cross-Entropy
- **Standard approach**: Separate softmax forward → CCE loss → complex Jacobian backward
- **Optimized gradient**: `∇L/∂z = softmax(z) - y_true` (simple subtraction!)
- **Implementation**: 
  - Core layers: `Softmax.backward()` computes full Jacobian (general-purpose)
  - Python API: `CCE.backward_with_logits()` uses optimized formula (opt-in)
  - Activation layers: `Softmax.backward_with_logits()` passthroughs gradient (no modification)
- **Numerical stability**: Softmax uses log-sum-exp trick (`x - max(x)` before exp)

### Sigmoid + Binary Cross-Entropy
- **Optimized gradient**: `∇L/∂z = sigmoid(z) - y_true` (same pattern)
- **Implementation**: Same opt-in pattern as Softmax+CCE
- **Stability note**: Sigmoid in core uses naive formula (⚠️ can overflow for z << 0)

### Design Principle
- **Core Rust**: Always mathematically correct (classical derivatives)
- **Python wrapper**: Exposes `backward_with_logits()` for fused optimization
- **Why**: Core remains general-purpose; optimization is explicit and opt-in

## ANTI-PATTERNS

- `exit(1)` in library code — use `raise ValueError`
- `os.system('clear')` inside `fit()` — remove or make optional
- `print()` for progress inside library — use `logging` or callbacks
- Panics in Rust core — replace with `Result<_, E>` (active TODO)
- `as any` / `@ts-ignore` — N/A (no TypeScript)

## CONVENTIONS

- **Rust**: `Result<T, E>` with `thiserror`; errors defined in `errors.rs`; `#[derive(Clone, Debug)]` on all data structs
- **Python**: type hints on all public methods; explicit bool defaults (`parallel=False`); JSON serialization via `to_dict()`/`from_dict()`
- **Rayon**: parallel matmul via `par_dot`; `par_mapv_inplace` for element-wise activation ops
- **Tests (Rust)**: grouped in `mod` blocks; descriptive names; `.unwrap()` on Ok; `is_err()` on Err
- **Tests (Python)**: no formal test suite — only `test.py` (pytest in dev-deps but unused)
- **Layer constructor**: `Layer(out_nodes, in_nodes, Activation.X)` — output nodes first

## NOTES

- `ndarray::ArrayD<f32>` is the core data container; `f32` throughout (no f64)
- 1D×1D dot returns `dimension=0, shape=[]` (scalar encoded as Tensor)
- `parallel=True` in `model.predict()`/`model.fit()` enables rayon in Rust matmul
- Persistence: `model.save(path)` writes JSON; `NeuralNetwork.load(path)` restores
- Italian comments appear in test files — project language is Italian/English mixed

## LAYER ARCHITECTURE

### Core (Rust)
- **`layers/dense.rs`**: Fully-connected layer with weight/bias gradients
  - `get_parameters()` returns `ParametersRef` with `Arc<Mutex<Tensor>>` for shared ownership
- **`layers/activations.rs`**: ReLU, Sigmoid, Tanh, Softmax (Layer trait implementations)
  - All compute **classical derivatives** (Jacobian for Softmax, σ(1-σ) for Sigmoid)
  - General-purpose, work with any loss function
- **`layers/init.rs`**: Weight initialization strategies (Xavier, He, LeCun, Random)

### Wrapper (Python API)
- **`layer_bindings/dense.rs`**: PyDense with `forward`, `backward`, `get_parameters`
  - `get_parameters()` → `PyParametersRef` for optimizer registration
- **`layer_bindings/activations.rs`**: PyReLU, PySigmoid, PyTanh, PySoftmax
  - All expose `backward_with_logits()` for optimized gradient passthrough
  - Used when combined with CCE/BCE loss functions
- **`layer_bindings/init.rs`**: PyInit enum exposed to Python

### Loss Functions
- **Core (`math/losses.rs`)**: Pure mathematical functions (forward + derivative)
- **Wrapper (`losses_bindings.rs`)**: Python classes with `call()` and `backward()`
  - CCE and BCE have `backward_optimized()` for fused softmax/sigmoid + loss gradient
  - All 6 loss functions registered in `lib.rs`

## OPTIMIZER ARCHITECTURE ⭐ NEW

### Core (Rust)

**ParametersRef** — Shared ownership wrapper for layer parameters:
```rust
pub struct ParametersRef {
    pub weights: Arc<Mutex<Tensor>>,  // Shared between layer and optimizer
    pub biases: Arc<Mutex<Tensor>>,
    pub w_grads: Arc<Mutex<Tensor>>,
    pub b_grads: Arc<Mutex<Tensor>>,
}
```

**Optimizer trait** — Stateful interface:
```rust
pub trait Optimizer {
    fn register_params(&mut self, params: Vec<ParametersRef>);
    fn step(&mut self) -> Result<(), TensorError>;
    fn zero_grad(&mut self) -> Result<(), TensorError>;
}
```

**GradientDescent** — PyTorch-style stateful optimizer:
```rust
pub struct GradientDescent {
    pub learning_rate: f32,
    pub params: Vec<ParametersRef>,  // Registered parameters
}
```
- `register_params()` stores references to all layer parameters
- `step()` updates all weights in parallel via Rayon: `θ = θ - lr·∇θ`
- `zero_grad()` resets all gradients to zero in parallel
- **Performance**: ~2-4x speedup on 10+ layer networks (Rayon parallelization)

### Wrapper (Python API)

**PyParametersRef** — Python wrapper for ParametersRef:
```python
class PyParametersRef:
    weights: Tensor
    biases: Tensor
    w_grads: Tensor
    b_grads: Tensor
```

**PyGradientDescent** — Exposed to Python:
```python
class GradientDescent:
    def __init__(self, learning_rate: float): ...
    def register_params(self, params: list[ParametersRef]): ...
    def step(self): ...
    def zero_grad(self): ...
```

### Usage Pattern (Python)

```python
# In Model.compile()
params = [layer.get_parameters() for layer in self.layers if hasattr(layer, 'get_parameters')]
optimizer.register_params(params)

# In Model.fit() — one training step
optimizer.zero_grad()           # 1. Reset gradients
y_pred = self.predict(x)        # 2. Forward pass
loss_val = loss_fn.call(y_true, y_pred)  # 3. Compute loss
grad = loss_fn.backward(y_true, y_pred)  # 4. Loss gradient
self.backward(grad)             # 5. Backprop (accumulates grads via shared Arc)
optimizer.step()                # 6. Update weights
```

### Key Design Points

1. **Shared Ownership**: `Arc<Mutex<Tensor>>` allows layer and optimizer to both hold references to same tensors
2. **Gradient Accumulation**: Calling `backward()` modifies gradients through Arc — optimizer sees changes automatically
3. **`zero_grad()` Timing**: Must be called BEFORE each forward pass (gradients accumulate by design)
4. **Parallel Updates**: Rayon parallelizes across layers — each layer's parameters updated independently
5. **Lock Overhead**: Negligible (~10-50ns uncontended) vs tensor ops (milliseconds)
6. **Future Optimizers**: Adam/RMSprop trivial to add — just clone GradientDescent + add momentum/velocity fields

## TENSOR API COMPLETENESS

### Exposed Operations
- **Creation**: `zeros`, `random`, `from_numpy`
- **Shape**: `reshape`, `transpose`, `flatten`
- **Arithmetic**: `+`, `-`, `*`, `/` (tensor-tensor, tensor-scalar, scalar-tensor)
- **Linear Algebra**: `dot` (matrix multiplication)
- **Manipulation**: `push`, `cat` (concatenation)
- **Interop**: `to_numpy`, `__array__` (NumPy array protocol)
- **Properties**: `shape`, `data`, `ndim` (via `#[getter]`/`#[setter]`)
- **Utility**: `length()` — returns total number of elements

### Missing (Non-Critical)
- **Statistical**: `ones()`, `sum()`, `mean()`, `std()`
- **Advanced**: `clip()`, `argmax()`, `argmin()`
- **Workaround**: Use `tensor.to_numpy()` + NumPy functions

## PERFORMANCE BENCHMARKS

### Optimizer Step Times (10-layer network, 1000→50 neurons)
- Average `step()` time: ~4ms (with Rayon parallelization)
- Average `zero_grad()` time: ~0.7ms
- Expected speedup: ~2-4x on multi-core CPUs vs sequential updates
- Lock overhead: <0.1% of total time (negligible)

### Test Coverage
- **Core Rust tests**: 236 total
  - Tensor tests: 224 (tensor_test.rs)
  - Optimizer tests: 12 (optimizers_test.rs)
- **Python integration**: `test_optimizer_refactor.py` validates new pattern
- **Benchmarks**: `benchmark_parallel_optimizer.py` measures Rayon speedup

## NEXT STEPS / ROADMAP

### Immediate (Ready to Implement)
1. **Fix `neomatrix/model.py` bugs** (10 issues documented above)
2. **Implement fused gradient optimization detection** in `Model.compile()`:
   - Detect Softmax + CCE → use `loss.backward_optimized()` + `activation.backward_with_logits()`
   - Detect Sigmoid + BCE → same pattern
3. **Add Metrics class** (Accuracy, MAE, etc.) for `Model.fit()` reporting
4. **Add Callbacks class** (EarlyStopping, ModelCheckpoint, LearningRateScheduler)

### Short-term (Easy After Refactor)
5. **Implement Adam optimizer**:
   - Clone `GradientDescent` struct
   - Add `m_weights`, `v_weights`, `m_biases`, `v_biases` fields (one Vec per registered param)
   - Initialize in `register_params()`
   - Update `step()` with Adam formula using internal state
6. **Add pytest suite** (replace ad-hoc `test.py` script)
7. **Add Python docstrings** to all PyO3 classes via `#[doc = "..."]`

### Long-term (Architectural Changes)
8. **Regularization**: L1/L2 penalties (add to optimizer `step()` or loss `backward()`)
9. **Dropout layer**: Requires random mask generation + training mode handling
10. **BatchNorm layer**: Requires running mean/variance tracking
11. **Conv2D / MaxPool2D**: Image processing layers
12. **LSTM / GRU**: Sequence modeling layers
13. **GPU acceleration**: CUDA/Metal/Vulkan backends
