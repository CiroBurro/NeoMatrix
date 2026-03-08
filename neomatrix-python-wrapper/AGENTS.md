# NEOMATRIX-PYTHON-WRAPPER — PYO3 BINDINGS

## OVERVIEW

Rust crate that exposes `neomatrix-core` types to Python via PyO3. Compiles to `neomatrix._backend` (cdylib). Currently only `Tensor` is registered; `Layer`, `Cost`, `Activation` are TODO.

## STRUCTURE

```
src/
├── lib.rs          # Module registration — TODO: register Layer, Cost, Activation
├── tensor.rs       # PyTensor — full Python API (~760L)
└── tensor_iter.rs  # TensorIter — __iter__ support
```

**Build target**: `neomatrix._backend` (set in `pyproject.toml` → `module-name`)

## PyTensor API (tensor.rs)

```python
# Construction
Tensor(shape: list[int], content: list[float])    # raises RuntimeError on shape mismatch
Tensor.zeros(shape) -> Tensor                      # [staticmethod]
Tensor.random(shape) -> Tensor                     # [staticmethod] ⚠️ range 0..100, not -1..1
Tensor.from_numpy(array: np.ndarray) -> Tensor     # [staticmethod]

# Properties (#[pyo3(get,set)])
tensor.dimension: int
tensor.shape: list[int]
tensor.data: np.ndarray    # getter returns numpy array; setter accepts ndarray

# Math
tensor.dot(t: Tensor) -> Tensor          # 1D×1D, 1D×2D, 2D×1D, 2D×2D only
tensor + other   # TensorOrScalar: Tensor or float
tensor - other
tensor * other
tensor / other

# Shape
tensor.length() -> int
tensor.transpose() -> Tensor             # 2D only
tensor.transpose_inplace()              # ⚠️ BUG: no-op — result discarded
tensor.reshape(shape) -> Tensor
tensor.reshape_inplace(shape)
tensor.flatten() -> Tensor
tensor.flatten_inplace()

# Concatenation
tensor.push(t: Tensor, axis: int)        # mutates self
tensor.push_row(t: Tensor)               # t must be 1D; promotes self 1D→2D
tensor.push_column(t: Tensor)            # same constraints
tensor.cat_inplace(tensors, axis) -> Tensor   # ⚠️ BUG: returns new, doesn't mutate self
Tensor.cat(tensors, axis) -> Tensor      # [staticmethod]

# Iteration / Serialization
iter(tensor)                              # via TensorIter
tensor.to_dict() -> dict                  # {dimension, shape, data: list[float]}
Tensor.from_dict(d: dict) -> Tensor      # [staticmethod]
repr(tensor)                              # "Tensor(dimension=N, shape=[...])"
```

## KNOWN BUGS

1. **`transpose_inplace()`** — calls `.reversed_axes()` on a clone but never saves result. Self is unchanged.
2. **`random()`** — generates values in `0.0..100.0`, not `-1.0..1.0` (Rust core default).
3. **`cat_inplace()`** — misleading name: returns a **new** Tensor instead of mutating `self`.

## REGISTERING NEW TYPES

`lib.rs` is currently just a `// TODO`. To expose a new type:
```rust
#[pymodule]
fn neomatrix_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_class::<YourNewType>()?;  // add here
    Ok(())
}
```

## CONVENTIONS

- All Python-exposed structs: `#[pyclass(module="neomatrix")]`
- Constructors use `#[new]`; static methods use `#[staticmethod]`
- Errors map to PyO3 exceptions: `TensorError` → `PyRuntimeError`
- Operator overloads accept `TensorOrScalar` (via `#[derive(FromPyObject)]` union type)
- `#[pyo3(get,set)]` only on fields safe to expose (dimension, shape — not data directly)

## BUILD

```bash
# Dev build (editable install)
maturin develop
# or
uv run maturin develop

# Release wheel
maturin build --release
```
