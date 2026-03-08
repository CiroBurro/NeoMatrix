//! Tensor module: multi-dimensional arrays and arithmetic operations.
//!
//! Provides the `Tensor` struct for numerical computing, supporting creation, manipulation,
//! and arithmetic operations with NumPy-style broadcasting.

mod tensor;
mod tensor_ops;

pub use tensor::Tensor;
