# NEOMATRIX — PROJECT KNOWLEDGE BASE

**Branch:** main | **Commit:** cd37636

## OVERVIEW

Keras-inspired ML library: **Rust backend** (`neomatrix-core`) for tensor ops and backprop, **PyO3 wrapper** (`neomatrix-python-wrapper`) for Python bindings, **Python API** (`neomatrix/`) for high-level model building.

**Stack**: Rust (ndarray + rayon) + PyO3 + Python 3.8+ | Built with **maturin**

---

## STRUCTURE

```
NeoMatrix/
├── neomatrix-core/          # Rust lib crate — tensor, math, layers
│   └── src/
│       ├── tensor/          # Tensor struct + ops
│       ├── math/            # Activations, losses (function implementations)
│       ├── layers/          # Dense, activation layers (Layer trait impls)
│       │   ├── dense.rs     # Fully-connected layer with backprop
│       │   ├── activations.rs # ReLU, Sigmoid, Tanh, Softmax layers
│       │   └── init.rs      # Xavier, He, LeCun weight initialization
│       ├── errors.rs        # All error types (thiserror)
│       └── test/            # Rust unit tests
├── neomatrix-python-wrapper/# Rust crate — PyO3 bindings
│   └── src/
│       ├── tensor_bindings/ # PyTensor with NumPy interop
│       │   ├── tensor.rs    # Full Python API + __array__
│       │   └── tensor_iter.rs # TensorIter for __iter__
│       ├── layer_bindings/  # Layer Python wrappers
│       │   ├── dense.rs     # PyDense
│       │   ├── activations.rs # PyReLU, PySigmoid, PyTanh, PySoftmax
│       │   └── init.rs      # PyInit enum
│       ├── losses_bindings.rs # All loss functions (MSE, MAE, BCE, CCE, Huber, Hinge)
│       └── lib.rs           # Module registration (_backend)
├── neomatrix/               # Python package (high-level API)
│   ├── core/                # Model, optimizer, __init__ (re-exports)
│   └── utils/               # dataset.py (get_batches)
├── examples/                # LinearRegression.py, NeuralNetwork.py
├── test.py                  # Ad-hoc integration test at root
├── test_array_interface.py  # NumPy array protocol tests
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
| All Rust error types | `neomatrix-core/src/errors.rs` |
| Python Tensor class | `neomatrix-python-wrapper/src/tensor_bindings/tensor.rs` |
| Python layer bindings | `neomatrix-python-wrapper/src/layer_bindings/` |
| Python loss bindings | `neomatrix-python-wrapper/src/losses_bindings.rs` |
| Register Python types | `neomatrix-python-wrapper/src/lib.rs` |
| NeuralNetwork / models | `neomatrix/core/model.py` |
| SGD / BatchGD / MiniBatchGD | `neomatrix/core/optimizer.py` |
| Rust unit tests | `neomatrix-core/src/test/tensor_test.rs` |
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

### ⚠️ Missing (Non-Critical)
- **Docstrings**: Zero Python documentation (no `help()` output)
- **Utility Methods**: `Tensor.ones()`, `sum()`, `mean()` (workaround: use NumPy)
- **Tests**: No pytest suite (only `test.py` ad-hoc script)

### 🎯 Production Readiness: 8.5/10
- Core functionality complete
- API pythonic with properties (`tensor.shape`, `tensor.data`)
- Main gap: lack of docstrings for UX

## KNOWN BUGS

None currently identified. Previous bugs have been fixed:
- `transpose_inplace`, `random` range, `cat_inplace` semantics (fixed)
- `f32 - Tensor` used hardcoded `1.0` instead of `self` (fixed in tensor_ops.rs L329)
- `f32 / Tensor` called `scalar_division` computing `Tensor / f32` instead of `f32 / Tensor` (fixed via `inverse_scalar_division`)

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
- **`layers/activations.rs`**: ReLU, Sigmoid, Tanh, Softmax (Layer trait implementations)
  - All compute **classical derivatives** (Jacobian for Softmax, σ(1-σ) for Sigmoid)
  - General-purpose, work with any loss function
- **`layers/init.rs`**: Weight initialization strategies (Xavier, He, LeCun, Random)

### Wrapper (Python API)
- **`layer_bindings/dense.rs`**: PyDense with `forward`, `backward`, `get/set_parameters`
- **`layer_bindings/activations.rs`**: PyReLU, PySigmoid, PyTanh, PySoftmax
  - All expose `backward_with_logits()` for optimized gradient passthrough
  - Used when combined with CCE/BCE loss functions
- **`layer_bindings/init.rs`**: PyInit enum exposed to Python

### Loss Functions
- **Core (`math/losses.rs`)**: Pure mathematical functions (forward + derivative)
- **Wrapper (`losses_bindings.rs`)**: Python classes with `call()` and `backward()`
  - CCE and BCE have `backward_with_logits()` for fused softmax/sigmoid + loss gradient
  - All 6 loss functions registered in `lib.rs`

## TENSOR API COMPLETENESS

### Exposed Operations
- **Creation**: `zeros`, `random`, `from_numpy`
- **Shape**: `reshape`, `transpose`, `flatten`
- **Arithmetic**: `+`, `-`, `*`, `/` (tensor-tensor, tensor-scalar, scalar-tensor)
- **Linear Algebra**: `dot` (matrix multiplication)
- **Manipulation**: `push`, `cat` (concatenation)
- **Interop**: `to_numpy`, `__array__` (NumPy array protocol)
- **Properties**: `shape`, `data`, `ndim` (via `#[getter]`/`#[setter]`)

### Missing (Non-Critical)
- **Statistical**: `ones()`, `sum()`, `mean()`, `std()`
- **Advanced**: `clip()`, `argmax()`, `argmin()`
- **Workaround**: Use `tensor.to_numpy()` + NumPy functions
