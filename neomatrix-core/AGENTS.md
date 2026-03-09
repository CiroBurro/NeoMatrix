# NEOMATRIX-CORE — RUST LIBRARY

## OVERVIEW

Pure Rust ML library crate. Provides `Tensor`, `Layer` trait, dense/activation layers, math ops (activations, losses, matmul). No Python dependencies — used by `neomatrix-python-wrapper` and testable standalone.

## STRUCTURE

```
src/
├── lib.rs              # pub mod: errors, layers, math, tensor; #[cfg(test)] mod test
├── errors.rs           # All error types (thiserror)
├── tensor/
│   ├── mod.rs          # pub use tensor::Tensor
│   ├── tensor.rs       # Tensor struct + all methods (~400L)
│   └── tensor_ops.rs   # Add/Sub/Mul/Div operator overloads
├── math/
│   ├── mod.rs
│   ├── activations.rs  # ActivationFunction trait + Relu/Sigmoid/Tanh/Linear/Softmax
│   ├── losses.rs       # CostFunction trait + Cost enum + get_cost() dispatch (~350L)
│   └── matmul.rs       # par_dot() — rayon parallel 2D matmul
├── layers/
│   ├── mod.rs          # Layer trait
│   ├── dense.rs        # Dense layer (weights, biases, forward, backward)
│   ├── activations.rs  # Relu/Sigmoid/Tanh/Softmax layer wrappers with backprop caching
│   └── init.rs         # Init enum: Random / Xavier / He
└── test/
    └── tensor_test.rs  # ~759L comprehensive Tensor tests
```

## KEY TYPES

### Tensor
```rust
#[derive(Clone, Debug)]
pub struct Tensor { pub dimension: usize, pub shape: Vec<usize>, pub data: ArrayD<f32> }
```
- `new(shape, content) -> Result<Tensor, TensorError>` — validates shape×content size
- `zeros(shape)`, `random(shape, rg: Range<f32>)` — constructors
- `dot(&self, t) -> Result<Tensor, TensorError>` — 1D×1D, 1D×2D, 2D×1D, 2D×2D only
- `transpose() -> Result<Tensor, TensorError>` — 2D only
- `transpose_inplace()` — mutates self (fixed)
- `reshape(shape)`, `flatten()` — non-mutating (return new Tensor)
- `reshape_inplace()`, `flatten_inplace()` — mutating variants
- `push_row(&mut self, t)`, `push_column(&mut self, t)` — append 1D tensor
- `cat(tensors, axis) -> Result<Tensor, _>` — static concat

### Layer Trait
```rust
pub trait Layer {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError>;
    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError>;
    fn get_params_and_grads(&mut self) -> Option<Vec<(&mut Tensor, &Tensor)>> { None }
}
```
- `Dense::new(in_feat, out_feat, init: Option<Init>, rg: Option<Range<f32>>)` — default Xavier
- Activation layers (Relu/Sigmoid/Tanh/Softmax): cache input or output for backprop
- Sigmoid/Tanh cache **output** (not input); Relu/Softmax cache **input**

### Cost enum (losses.rs)
`MeanSquaredError | MeanAbsoluteError | BinaryCrossEntropy | CategoricalCrossEntropy | HuberLoss{delta:f32} | HingeLoss`
- `get_cost(cost, t, z, batch_processing: bool) -> Result<f32, MathError>`
- `TryFrom<&str>` implemented for all except `HuberLoss` (not deserializable)

### Init enum (init.rs)
`Random | Xavier | He` — all produce shape `[in_feat, out_feat]` Tensor

## ERRORS

All in `errors.rs` using `#[derive(Error, Debug)]` from `thiserror`:
- `TensorError` (+PartialEq) — shape/dim/dot/transpose/concat errors
- `LayerError` — wraps TensorError + MathError via `#[from]`; NotInitialized
- `MathError` — activation/loss shape mismatches, unknown fn names, HuberLoss missing delta

## CONVENTIONS

- `f32` only — no f64 anywhere
- `Result<T, E>` everywhere — never `unwrap()` in library code (only in tests)
- Errors defined once in `errors.rs`, converted via `#[from]`
- Rayon: `par_mapv_inplace` for element-wise ops; `par_dot` for 2D matmul
- 1D×1D dot → scalar Tensor with `dimension=0, shape=[]`, accessed via `data[[]]`

## TESTS

```bash
cargo test
```
- All in `test/tensor_test.rs` grouped by feature in `mod` blocks
- Italian comments are intentional — mixed IT/EN project
- `.unwrap()` on Ok results; `assert!(result.is_err())` on error cases

## ANTI-PATTERNS

- Panics in library code — replace with `Result<_, E>` (active TODO)
- Adding `unwrap()` outside test code
- New error types outside `errors.rs`
- f64 data types (keep everything f32)
