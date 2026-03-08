//! Mathematical operations for neural networks.
//!
//! This module provides core mathematical components used throughout the library:
//!
//! - **`activations`**: Activation functions (ReLU, Sigmoid, Tanh, Softmax, Linear) with derivatives
//! - **`losses`**: Loss functions (MSE, MAE, BCE, CCE, Huber, Hinge) for training
//! - **`matmul`**: Parallel 2D matrix multiplication using Rayon
//!
//! All functions are optimized for performance with parallel computation where applicable.

pub mod activations;
pub mod losses;
pub mod matmul;
