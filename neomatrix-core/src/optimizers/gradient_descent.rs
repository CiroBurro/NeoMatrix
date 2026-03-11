//! Standard gradient descent optimizer.
//!
//! This module provides [`GradientDescent`], the simplest first-order optimizer.
//! It updates each parameter by subtracting the gradient scaled by a fixed learning rate.
//!
//! # Mathematical Operation
//!
//! For each trainable parameter `θ` and its gradient `∇θ`:
//! ```text
//! θ_new = θ - lr · ∇θ
//! ```
//!
//! # Batch Strategy Independence
//!
//! [`GradientDescent`] does not know whether the gradients were computed from the full
//! dataset (Batch GD), a single sample (SGD), or a mini-batch (Mini-Batch GD).
//! That distinction belongs to the training loop. This optimizer applies the same
//! update rule regardless of how the gradients were produced.
//!
//! # Example
//!
//! ```rust,ignore
//! use neomatrix_core::optimizers::{gradient_descent::GradientDescent, Optimizer};
//! use neomatrix_core::tensor::Tensor;
//!
//! let mut gd = GradientDescent { learning_rate: 0.01 };
//!
//! let mut w = Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
//! let mut b = Tensor::new(vec![3],    vec![0.0; 3]).unwrap();
//! let dw    = Tensor::new(vec![2, 3], vec![0.5; 6]).unwrap();
//! let db    = Tensor::new(vec![3],    vec![0.1; 3]).unwrap();
//!
//! // step is ignored by this stateless optimizer
//! gd.update(&mut w, &mut b, &dw, &db, 0).unwrap();
//! // w is now [[0.995, 0.995, 0.995], [0.995, 0.995, 0.995]]
//! ```

use std::ops::{Deref, DerefMut};

use rayon::prelude::*;

use crate::{
    errors::TensorError,
    optimizers::{Optimizer, ParametersRef},
    tensor::Tensor,
};

/// Standard gradient descent optimizer with a fixed learning rate.
///
/// Updates parameters using the rule `θ_new = θ - lr · ∇θ`, applied independently
/// to both weights and biases of a layer.
///
/// This optimizer is **stateless**: it holds no accumulators or moment estimates.
/// The `step` argument passed to [`update`](GradientDescent::update) is therefore
/// ignored.
///
/// # Fields
///
/// - `learning_rate`: Scalar that controls the size of each parameter update.
///   Typical values are in the range `1e-4` to `1e-1`. A value of `0.0` results
///   in no update.
///
/// # Example
///
/// ```rust,ignore
/// let mut opt = GradientDescent { learning_rate: 0.001 };
/// opt.update(&mut weights, &mut biases, &grad_w, &grad_b, step)?;
/// ```
pub struct GradientDescent {
    /// Step size for each parameter update. Must be non-negative.
    pub learning_rate: f32,
    pub params: Vec<ParametersRef>,
}

impl GradientDescent {
    pub fn new(learning_rate: f32, params: Vec<ParametersRef>) -> Self {
        Self {
            learning_rate,
            params,
        }
    }
}

impl Optimizer for GradientDescent {
    /// Updates `weights` and `biases` in place:
    /// ```text
    /// weights = weights - learning_rate * w_grads
    /// biases  = biases  - learning_rate * b_grads
    /// ```
    ///
    /// # Arguments
    ///
    /// - `weights`: Weight tensor to update (mutated in place).
    /// - `biases`: Bias tensor to update (mutated in place).
    /// - `w_grads`: Gradient of the loss w.r.t. weights — same shape as `weights`.
    /// - `b_grads`: Gradient of the loss w.r.t. biases — same shape as `biases`.
    /// - `_step`: Ignored. Gradient descent is stateless and does not use iteration count.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if the subtraction or scalar multiplication fails
    /// due to incompatible shapes.
    fn update(
        &mut self,
        weights: &mut Tensor,
        biases: &mut Tensor,
        w_grads: &Tensor,
        b_grads: &Tensor,
        _step: usize,
    ) -> Result<(), TensorError> {
        // w_new = w - lr * grad_w
        *weights = (&*weights - w_grads * self.learning_rate)?;
        // b_new = b - lr * grad_b
        *biases = (&*biases - b_grads * self.learning_rate)?;
        Ok(())
    }

    fn register_params(&mut self, params: Vec<ParametersRef>) {
        self.params = params;
    }

    fn step(&mut self) -> Result<(), TensorError> {
        // Parallelize updates across layers using Rayon
        // Each layer's parameters are updated independently
        self.params.par_iter().try_for_each(|param| {
            let mut weights = param
                .weights
                .lock()
                .map_err(|e| TensorError::MemoryError(e.to_string()))?;
            let mut biases = param
                .biases
                .lock()
                .map_err(|e| TensorError::MemoryError(e.to_string()))?;
            let w_grads = param
                .w_grads
                .lock()
                .map_err(|e| TensorError::MemoryError(e.to_string()))?;
            let b_grads = param
                .b_grads
                .lock()
                .map_err(|e| TensorError::MemoryError(e.to_string()))?;

            *weights = (weights.deref_mut() - w_grads.deref() * self.learning_rate)?;
            *biases = (biases.deref_mut() - b_grads.deref() * self.learning_rate)?;

            Ok::<(), TensorError>(())
        })?;
        Ok(())
    }

    fn zero_grad(&mut self) -> Result<(), TensorError> {
        // Parallelize zeroing gradients across layers
        self.params.par_iter().try_for_each(|param| {
            let mut w_grads = param
                .w_grads
                .lock()
                .map_err(|e| TensorError::MemoryError(e.to_string()))?;
            let mut b_grads = param
                .b_grads
                .lock()
                .map_err(|e| TensorError::MemoryError(e.to_string()))?;
            *w_grads = Tensor::new(w_grads.shape.clone(), vec![0.0; w_grads.length()])?;
            *b_grads = Tensor::new(b_grads.shape.clone(), vec![0.0; b_grads.length()])?;

            Ok::<(), TensorError>(())
        })?;
        Ok(())
    }
}
