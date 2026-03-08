# NEOMATRIX — PROJECT KNOWLEDGE BASE

**Branch:** main | **Commit:** 6e99245

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
│       ├── math/            # Activations, losses, matmul
│       ├── layers/          # Dense, activation layers, init
│       ├── errors.rs        # All error types (thiserror)
│       └── test/            # Rust unit tests
├── neomatrix-python-wrapper/# Rust crate — PyO3 bindings
│   └── src/
│       ├── tensor.rs        # PyTensor (760L) — full Python API
│       ├── tensor_iter.rs   # TensorIter for __iter__
│       └── lib.rs           # TODO: register Layer, Cost etc.
├── neomatrix/               # Python package (high-level API)
│   ├── core/                # Model, optimizer, __init__ (re-exports)
│   └── utils/               # dataset.py (get_batches)
├── examples/                # LinearRegression.py, NeuralNetwork.py
├── test.py                  # Ad-hoc integration test at root
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
| Layer trait | `neomatrix-core/src/layers/mod.rs` |
| Weight initialization | `neomatrix-core/src/layers/init.rs` |
| All Rust error types | `neomatrix-core/src/errors.rs` |
| Python Tensor class | `neomatrix-python-wrapper/src/tensor.rs` |
| Register new Python types | `neomatrix-python-wrapper/src/lib.rs` |
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
  → neomatrix._backend (PyO3)   neomatrix-python-wrapper/src/tensor.rs
  → neomatrix_core (Rust lib)   neomatrix-core/src/
```

⚠️ `neomatrix/core/__init__.py` currently imports from `rustybrain` (old module name). Will switch to `neomatrix._backend` once wrapper registration is complete (`lib.rs` is still TODO).

## KNOWN BUGS

1. `Tensor.transpose_inplace()` (both Rust and Python): result discarded — transposition never applied
2. `Tensor.random()` in Python wrapper: range is `0..100`, not `-1..1` like Rust core
3. `Tensor.cat_inplace()` in Python wrapper: returns new Tensor instead of mutating self

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
