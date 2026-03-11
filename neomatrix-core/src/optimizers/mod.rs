//! Optimizer algorithms for neural network training.
//!
//! This module defines the [`Optimizer`] trait and provides concrete implementations
//! for gradient-based parameter update strategies. Optimizers are responsible for
//! adjusting a layer's weights and biases given the gradients computed during
//! backpropagation.
//!
//! # Trait Design
//!
//! All optimizers share a single [`Optimizer::update`] method. The difference between
//! strategies (Batch GD, SGD, Mini-Batch GD) lies entirely in **how the gradients are
//! produced** by the training loop — not in the optimizer itself. This keeps the
//! optimizer stateless with respect to batch size.
//!
//! # Available Optimizers
//!
//! | Struct | Description |
//! |--------|-------------|
//! | [`gradient_descent::GradientDescent`] | Standard gradient descent with fixed learning rate |
//!
//! # Example
//!
//! ```rust,ignore
//! use neomatrix_core::optimizers::{gradient_descent::GradientDescent, Optimizer};
//! use neomatrix_core::tensor::Tensor;
//!
//! let mut opt = GradientDescent { learning_rate: 0.01 };
//!
//! // Weights and gradients computed by a Dense layer
//! let mut weights = Tensor::new(vec![3, 2], vec![1.0; 6]).unwrap();
//! let mut biases  = Tensor::new(vec![2],    vec![0.0; 2]).unwrap();
//! let w_grads     = Tensor::new(vec![3, 2], vec![0.1; 6]).unwrap();
//! let b_grads     = Tensor::new(vec![2],    vec![0.05; 2]).unwrap();
//!
//! opt.update(&mut weights, &mut biases, &w_grads, &b_grads, 0).unwrap();
//! ```

pub mod gradient_descent;

use std::sync::{Arc, Mutex};

use crate::{errors::TensorError, tensor::Tensor};

#[derive(Clone, Debug)]
pub struct ParametersRef {
    pub weights: Arc<Mutex<Tensor>>,
    pub biases: Arc<Mutex<Tensor>>,
    pub w_grads: Arc<Mutex<Tensor>>,
    pub b_grads: Arc<Mutex<Tensor>>,
}

/// Common interface for all parameter update strategies.
///
/// Implementors receive the current parameters (weights and biases) of a layer
/// alongside the gradients computed during backpropagation, and update the parameters
/// in place.
///
/// # Parameters
///
/// The `step` argument is the current update iteration (0-indexed). Stateless
/// optimizers like [`GradientDescent`](gradient_descent::GradientDescent) ignore it;
/// adaptive optimizers (e.g., Adam) use it for bias correction.
///
/// # Errors
///
/// Returns a [`TensorError`] if the arithmetic operations on the tensors fail
/// (e.g., incompatible shapes).
pub trait Optimizer {
    /// Updates `weights` and `biases` in place using the provided gradients.
    ///
    /// # Arguments
    ///
    /// - `weights`: Mutable reference to the layer's weight tensor.
    /// - `biases`: Mutable reference to the layer's bias tensor.
    /// - `w_grads`: Gradient of the loss w.r.t. the weights (same shape as `weights`).
    /// - `b_grads`: Gradient of the loss w.r.t. the biases (same shape as `biases`).
    /// - `step`: Current update step (0-indexed). Used by adaptive optimizers for
    ///   bias correction; ignored by stateless ones.
    ///
    /// # Returns
    ///
    /// - `Ok(())` on success.
    /// - `Err(TensorError)` if the update arithmetic fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// opt.update(&mut layer_weights, &mut layer_biases, &grad_w, &grad_b, step)?;
    /// ```
    fn update(
        &mut self,
        weights: &mut Tensor,
        biases: &mut Tensor,
        w_grads: &Tensor,
        b_grads: &Tensor,
        step: usize,
    ) -> Result<(), TensorError>;

    fn register_params(&mut self, params: Vec<ParametersRef>);
    fn step(&mut self) -> Result<(), TensorError>;
    fn zero_grad(&mut self) -> Result<(), TensorError>;
}
