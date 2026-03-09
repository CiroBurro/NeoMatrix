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
Tensor.random(shape, start=-1.0, end=1.0) -> Tensor  # [staticmethod] fixed: default -1..1
Tensor.from_numpy(array: np.ndarray) -> Tensor     # [staticmethod]

# Properties (#[pyo3(get,set)])
tensor.dimension: int
tensor.ndim: int               # alias of dimension (NumPy convention)
tensor.shape: list[int]
tensor.data: np.ndarray        # getter returns numpy array; setter accepts ndarray

# Math
tensor.dot(t: Tensor) -> Tensor          # 1D×1D, 1D×2D, 2D×1D, 2D×2D only
# Operators (support TensorOrScalar: Tensor or float)
tensor + other   # __add__, __radd__ (commutative)
tensor - other   # __sub__, __rsub__ (non-commutative: other - tensor)
-tensor          # __neg__ (unary negation)
tensor * other   # __mul__, __rmul__ (commutative)
tensor / other   # __truediv__, __rtruediv__ (non-commutative: other / tensor)

# Shape
len(tensor) -> int              # __len__, returns total element count
tensor.to_numpy() -> np.ndarray # explicit conversion to numpy array (same as .data)
tensor.transpose() -> Tensor    # 2D only
tensor.transpose_inplace()      # mutates self (fixed)
tensor.reshape(shape) -> Tensor
tensor.reshape_inplace(shape)
tensor.flatten() -> Tensor
tensor.flatten_inplace()

# Concatenation
tensor.push(t: Tensor, axis: int)        # mutates self
tensor.push_row(t: Tensor)               # t must be 1D; promotes self 1D→2D
tensor.push_column(t: Tensor)            # same constraints
tensor.cat_inplace(tensors, axis)        # mutates self (fixed)
Tensor.cat(tensors, axis) -> Tensor      # [staticmethod]

# Iteration / Indexing
iter(tensor)                              # via TensorIter
len(tensor)                               # __len__, total element count
tensor[i]                                 # __getitem__, flat indexing (1D: int)
tensor[i, j, ...]                         # __getitem__, multi-dim indexing (tuple of ints)
tensor[i] = value                         # __setitem__, supports both flat and multi-dim
repr(tensor)                              # "Tensor(dimension=N, shape=[...])"
```

## KNOWN BUGS

None currently identified. All previously reported bugs have been fixed:
- `transpose_inplace()` now correctly mutates self
- `random()` now defaults to `-1.0..1.0` range
- `cat_inplace()` now properly mutates self instead of returning new Tensor

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
