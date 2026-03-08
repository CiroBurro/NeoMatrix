//! # NeoMatrix Core
//!
//! A Keras-inspired machine learning library implemented in Rust.
//!
//! This crate provides the core Rust implementation of NeoMatrix, including:
//! - Tensor operations and linear algebra
//! - Neural network layers (Dense, Activation)
//! - Activation functions (ReLU, Sigmoid, Tanh, Softmax, Linear)
//! - Loss functions (MSE, MAE, BCE, CCE, Huber, Hinge)
//! - Weight initialization strategies (Random, Xavier, He)
//! - Parallel matrix multiplication using Rayon
//!
//! ## Architecture
//!
//! The library is organized into the following modules:
//! - `tensor`: Multi-dimensional array operations
//! - `math`: Mathematical functions (activations, losses, matrix multiplication)
//! - `layers`: Neural network building blocks
//! - `errors`: Error types and handling
//!
//! ## Example
//!
//! ```rust
//! use neomatrix_core::tensor::Tensor;
//!
//! let t1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let t2 = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
//! let result = t1.dot(&t2).unwrap();
//! ```

extern crate openblas_src;

pub mod errors;
pub mod layers;
pub mod math;
pub mod tensor;

#[cfg(test)]
mod test;
