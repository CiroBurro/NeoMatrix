# NEOMATRIX PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-05 (current session)  
**Project:** NeoMatrix - Keras-inspired ML library with Rust compute backend

## OVERVIEW

Hybrid Python+Rust ML library. Keras-like API (Python) with fast tensor operations (Rust/ndarray+rayon). Build via maturin (pyo3 bindings). Extension module "rustybrain" wraps Rust compute, "neomatrix" package wraps Python API.

## STRUCTURE

```
NeoMatrix/
├── src/                    # Rust compute backend (rustybrain cdylib)
│   ├── structures/         # tensor.rs (703L), layer.rs (531L) - COMPLEXITY HOTSPOTS
│   ├── functions/          # activation.rs, cost.rs
│   └── utils/              # matmul.rs (parallel dot), weights_biases.rs
├── neomatrix/              # Python API wrapper
│   ├── core/               # model.py (NeuralNetwork), optimizer.py
│   └── utils/              # dataset.py
├── examples/               # NeuralNetwork.py, LinearRegression.py
├── tests/                  # tensor_methods.py (Python unittest)
├── test.py                 # ⚠️ No __main__ guard, executes on import
├── pyproject.toml          # Maturin build config
└── Cargo.toml              # Rust cdylib crate
```

**⚠️ CRITICAL BUG:** `src/structures/tenosor_iter.rs` has typo (should be `tensor_iter.rs`)

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Shape mismatch bugs** | `src/structures/layer.rs` | backward/forward logic, batch transposes |
| **Tensor arithmetic bugs** | `src/structures/tensor.rs` | dot(), operators (__add__/__mul__), reshape |
| **Optimize matmul** | `src/utils/matmul.rs` | par_dot (rayon parallel 2D*2D) |
| **Add activation/cost** | `src/functions/` | activation.rs, cost.rs + update layer.rs select_activation |
| **Model init/serialization** | `src/structures/*.rs`, `neomatrix/core/model.py` | to_dict/from_dict, NeuralNetwork class |
| **Python API changes** | `src/structures/{tensor,layer}.rs` | pyo3 #[pymethods], neomatrix/core/*.py |
| **Entry point** | `neomatrix/__init__.py` → `core/__init__.py` | Re-exports from rustybrain module |
| **Dev build** | maturin develop | NOT pip install -e |
| **Tests** | `tests/`, `src/tests.rs`, `test.py` | Python unittest + Rust cargo test |

## CODE MAP

**Rust Core (rustybrain):**
- `Tensor` (src/structures/tensor.rs): Multi-dimensional arrays, NumPy interop, arithmetic ops, reshape/transpose/concat, iteration
- `Layer` (src/structures/layer.rs): forward/backward propagation, batch processing, activation/cost integration
- `Activation` (src/functions/activation.rs): Relu, Sigmoid, Softmax, Tanh, Linear + derivatives
- `Cost` (src/functions/cost.rs): MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy, HuberLoss, HingeLoss + derivatives
- `par_dot` (src/utils/matmul.rs): Parallel 2D*2D matrix multiplication via rayon

**Python API (neomatrix):**
- `NeuralNetwork` (neomatrix/core/model.py): fit(), predict(), save/load
- `Optimizer` (neomatrix/core/optimizer.py): SGD, BatchGD, MiniBatchGD
- Public imports: `from neomatrix.core import Tensor, Layer, Activation, Cost, NeuralNetwork`

## CONVENTIONS

**NAMING:**
- Classes: CamelCase (Tensor, Layer, NeuralNetwork)
- Functions/params: snake_case (training_set, batch_processing)
- Type hints: Required for public Python API
- Booleans: Explicit defaults (batch_processing=True, parallel=False)

**BUILD (NON-STANDARD):**
- Build backend: **maturin** (NOT pip/setuptools)
- Extension module name: **"rustybrain"** ≠ package name "neomatrix"
- Dev install: `maturin develop` (NOT `pip install -e`)
- Rust crate type: cdylib (FFI for Python)
- Python min: >=3.8
- Dependencies: numpy>=1.24.4, pandas>=2.0.3, dataclasses>=0.8
- ndarray features: rayon (parallel ops enabled)

**IMPORTS:**
- User imports: `from neomatrix.core import Tensor, Layer, NeuralNetwork`
- neomatrix.core re-exports from rustybrain extension module
- neomatrix.core.__init__.py maintains __all__ for stable API

**ERROR HANDLING:**
- Rust library code: Return `PyResult<T>`, propagate errors
- Python bindings: Raise PyErr on invalid input
- Tests: unwrap()/expect() ALLOWED for conciseness

## ANTI-PATTERNS (THIS PROJECT)

**FORBIDDEN in src/** (library code):
1. ❌ `panic!()` - Must return Result/PyResult (present in: tensor.rs, layer.rs, matmul.rs)
2. ❌ `unwrap()`/`expect()` on runtime paths - Propagate errors (present in: tensor.rs, cost.rs, layer.rs)
3. ❌ `println!()` in API - Use logging (present in: tensor.rs from_dict)
4. ❌ Panicking during deserialization - Return PyErr on invalid data (tensor.rs, layer.rs)
5. ❌ Hardcoded dimension limits - dot() only supports 1D/2D, needs explicit check

**ALLOWED in tests/:**
- ✅ unwrap()/expect() for test conciseness

**TODO (from README):**
- Custom arithmetic: Implement std::ops traits (Add/Sub/Mul/Div) instead of methods
- Add optimizers: Adam, RMSprop, Adagrad
- Add examples: linear regression, logistic regression
- Add metrics: Accuracy, Precision, Recall, F1

**ENFORCEMENT:**
```bash
# Audit library code for forbidden patterns
grep -r "panic!\|unwrap()\|expect()\|println!" src/ --exclude-dir=tests
```

## COMMANDS

```bash
# Development
maturin develop                    # Build + install (editable, local dev)

# Build
maturin build --release            # Production wheel

# Test
python -m unittest discover -v     # Python tests
cargo test                         # Rust tests
python test.py                     # ⚠️ Ad-hoc test (no __main__ guard)

# Lint
cargo fmt --all                    # Format Rust code
cargo clippy -- -D warnings        # Lint Rust code

# Publish
maturin publish                    # Upload to PyPI
```

**CI Steps (no CI config present):**
```bash
pip install maturin==1.8.*
maturin build --release
pip install target/wheels/*.whl
python -m unittest discover -v
cargo test
cargo clippy -- -D warnings
```

## NOTES

**Gotchas:**
- `test.py` executes on import (missing `if __name__ == "__main__"`)
- Typo: `src/structures/tenosor_iter.rs` → should be `tensor_iter.rs`
- Extension module name "rustybrain" ≠ package name "neomatrix" (intentional design)
- Private Rust modules (tensor_ops, tenosor_iter) - only lib.rs exports are public API
- dot() only supports 1D/2D tensors (explicit check prevents extension to 3D+)
- UV lock present (uv.lock + [tool.uv] in pyproject.toml)

**Key Design:**
- Rust handles: tensor ops, linear algebra (parallel dot/matmul), forward/backward propagation
- Python handles: high-level API, model orchestration, training loops, data loading
- pyo3 bridges: Tensor/Layer serialization (to_dict/from_dict), NumPy interop

---

## PROJECT ASSESSMENT & ROADMAP

**Current Status:** 5.5/10 (good architecture, poor implementation)
- Architecture: 7/10 (clean separation Rust/Python)
- Implementation: 4/10 (88 panic!/unwrap, memory waste)
- Scalability: 3/10 (CPU-only, no BLAS, f64-only)
- Production-ready: 2/10 (crashes on invalid input)

### CRITICAL ISSUES (88 occurrences)

**Error Handling Catastrophic:**
```
matmul.rs:22    - panic! on dimension mismatch
tensor.rs:71,448 - panic! on shape/content mismatch
layer.rs:109+   - 15+ panic! on shape checks
cost.rs:22+     - 14+ panic! on operations
```
**Impact:** Invalid input aborts entire Python process instead of raising PyErr.

**Memory Inefficiency:**
```rust
// par_dot takes Array2 by-value → mandatory clone
pub fn par_dot(a: Array2<f64>, b: Array2<f64>)

// tensor.dot clones before every operation
let a_2d = self.data.clone().into_dimensionality::<Ix2>().unwrap();
```
**Impact:** 2x memory usage for every matrix multiplication.

**Scalability Limits:**
- ❌ CPU-only (no GPU)
- ❌ f64-only (2x memory vs f32)
- ❌ No BLAS (10-100x slower than OpenBLAS/MKL)
- ❌ dot() 1D/2D only (no 3D+ tensors)
- ❌ JSON serialization (impractical for large models)
- ❌ Optimizer stateless (no momentum/Adam state)

**Test Coverage:**
- ✅ tests/tensor_methods.py (basic tensor ops)
- ❌ No layer forward/backward tests
- ❌ No gradient checking
- ❌ No training convergence tests
- ❌ No serialization tests

---

## DEVELOPMENT ROADMAP

### 🔥 PHASE 1: STABILIZATION (Week 1-2) - CRITICAL

**Priority: Fix crashes + hygiene**

```bash
# 1. Fix typos (5 min)
git mv src/structures/tenosor_iter.rs src/structures/tensor_iter.rs

# 2. Fix test.py (2 min)
# Add: if __name__ == "__main__": before execution

# 3. Replace panic! → PyResult (2-3 days)
# Files: tensor.rs, layer.rs, matmul.rs, cost.rs, activation.rs
# Pattern:
#   Before: panic!("Incompatible dimensions")
#   After:  return Err(PyValueError::new_err("Incompatible dimensions"))

# 4. Add test suite (3-5 days)
tests/test_layer.py          # Layer forward/backward correctness
tests/test_gradients.py      # Numerical gradient checking
tests/test_serialization.py  # Round-trip save/load
tests/test_training.py       # XOR convergence test

# 5. CI/CD (1 day)
.github/workflows/ci.yml     # cargo test + pytest + clippy
```

**Deliverable:** Stable library that raises errors instead of crashing.  
**After Phase 1:** 7.0/10 (production-ready for small CPU workloads)

---

### ⚡ PHASE 2: PERFORMANCE (Week 3-4)

**Priority: Memory efficiency + speed**

```rust
// 1. Fix par_dot signature (1 day)
pub fn par_dot(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64>
// Impact: Remove mandatory clone, 50% memory reduction

// 2. tensor.dot use views (1 day)
let a_view = self.data.view().into_dimensionality::<Ix2>()?;
// Impact: No clone on every operation

// 3. Binary serialization (2 days)
model.save_npz("model.npz")  // NumPy compressed format
// Impact: 10-100x faster save/load

// 4. Benchmark suite (1 day)
benches/matmul.rs            # Criterion benchmarks
benches/training.rs          # Training loop benchmarks
```

**Deliverable:** 2x memory efficiency, 10x faster serialization.  
**After Phase 2:** 7.5/10 (competitive with small NumPy-based libraries)

---

### 🚀 PHASE 3: FEATURES (Month 2)

**Priority: Modern ML capabilities**

```python
# 1. Advanced optimizers (1 week)
neomatrix/core/optimizer.py:
  class Adam(Optimizer):
      def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
          self.m, self.v = {}, {}  # Momentum state
  class RMSprop(Optimizer): ...
# Impact: 2-10x faster training convergence

# 2. Layer trait + new layer types (1 week)
src/structures/layer_trait.rs:
  trait Layer {
      fn forward(&self, input: &Tensor) -> PyResult<Tensor>;
      fn backward(&mut self, delta: &Tensor) -> PyResult<Tensor>;
  }
  impl Layer for Dense { ... }
  impl Layer for Dropout { ... }
  impl Layer for BatchNorm { ... }
# Impact: Modern architectures (dropout, batchnorm)

# 3. f32 dtype support (3 days)
src/structures/tensor.rs:
  pub enum Dtype { F32, F64 }
  pub struct Tensor { data: ArrayD<f32>, ... }
# Impact: 50% memory, 2x SIMD speed

# 4. Examples (3 days)
examples/mnist.py            # MNIST classification
examples/linear_reg.py       # Linear regression
examples/logistic_reg.py     # Logistic regression
```

**Deliverable:** Feature parity with basic scikit-learn/Keras.  
**After Phase 3:** 8.0/10 (solid educational/prototyping tool)

---

### 🌟 PHASE 4: SCALING (Month 3-6)

**Priority: Performance at scale**

```toml
# 1. Integrate BLAS (1 week)
[dependencies]
ndarray = { version = "0.16", features = ["rayon", "blas"] }
blas-src = { version = "0.10", features = ["openblas"] }
# Impact: 10-100x matmul speedup

# 2. Convolutional layers (2 weeks)
src/structures/conv2d.rs:
  pub struct Conv2D {
      filters: Array4<f32>,  // (out_ch, in_ch, h, w)
      stride: usize,
      padding: usize,
  }
# Impact: Computer vision support (CNN)

# 3. Recurrent layers (2 weeks)
src/structures/lstm.rs:
  pub struct LSTM {
      Wf, Wi, Wo, Wc: Array2<f32>,  // Gate weights
      hidden_state: Array1<f32>,
  }
# Impact: Sequence modeling (NLP)

# 4. Autograd engine (4-6 weeks)
src/autograd/:
  mod graph;       // Computation graph
  mod ops;         // Operations with grad fns
  mod backward;    // Automatic differentiation
# Impact: PyTorch-like experience, no manual backward
```

**Deliverable:** Competitive with TensorFlow/PyTorch for CPU workloads.  
**After Phase 4:** 8.5/10 (mature ML library)

---

### 🔮 PHASE 5: GPU & DISTRIBUTED (Month 6+)

**Priority: Cutting-edge capabilities**

```rust
// 1. Device abstraction (2-3 months)
pub enum Device {
    CPU,
    CUDA(usize),  // GPU id
    ROCm(usize),
}
impl Tensor {
    pub fn to_device(&self, device: Device) -> PyResult<Tensor>;
}

// 2. Distributed training (2 months)
model.fit(X, y, distributed=True, num_workers=4)

// 3. ONNX export (1 month)
model.save_onnx("model.onnx")  // Deploy anywhere
```

**Deliverable:** Production-grade deep learning framework.  
**After Phase 5:** 9.0/10 (competitive with PyTorch/JAX)

---

## EFFORT ESTIMATES

| Phase | Duration | Effort | Complexity |
|-------|----------|--------|------------|
| Phase 1: Stabilization | 1-2 weeks | 40-60h | Low (mechanical fixes) |
| Phase 2: Performance | 1-2 weeks | 40-60h | Medium (refactor) |
| Phase 3: Features | 1 month | 80-120h | Medium (new code) |
| Phase 4: Scaling | 3-6 months | 300-500h | High (advanced ML) |
| Phase 5: GPU | 6+ months | 500-1000h | Very High (systems) |

**Recommendation:**
- Solo developer: Stop at Phase 3 (good educational tool)
- Small team: Aim for Phase 4 (production CPU library)
- Serious project: Consider contributing to Burn/Candle instead (1-2 years mature)

---

## ALTERNATIVE PATHS

**If goal = learning Rust + ML:**
✅ Continue - excellent learning project
→ Focus on Phase 1-3, skip GPU complexity

**If goal = production ML library:**
⚠️ Evaluate alternatives first:
- Burn (https://github.com/tracel-ai/burn) - mature, GPU, autograd
- Candle (https://github.com/huggingface/candle) - HuggingFace, production-ready
- Contributing > reinventing

**If goal = research/experiments:**
✅ Fix Phase 1 issues, then iterate fast
→ Stability > completeness for prototyping
