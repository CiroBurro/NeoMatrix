//! Optimizer algorithms for neural network training.
//!
//! This module defines the [`Optimizer`] trait and provides concrete implementations
//! for gradient-based parameter update strategies. Optimizers are responsible for
//! adjusting a layer's weights and biases using the gradients computed during
//! backpropagation.
//!
//! # Architecture
//!
//! NeoMatrix uses a **stateful optimizer architecture** inspired by PyTorch:
//!
//! 1. **Shared Ownership**: Layer parameters (weights, biases, gradients) are wrapped in
//!    [`ParametersRef`], which uses `Arc<Mutex<Tensor>>` for shared ownership between
//!    layers and optimizers.
//!
//! 2. **Registration**: Before training, layers register their parameters with the
//!    optimizer via [`Optimizer::register_params`].
//!
//! 3. **Training Loop**: During training, the optimizer operates on all registered
//!    parameters:
//!    - [`Optimizer::zero_grad`] resets all gradients to zero
//!    - Forward pass + backward pass accumulate gradients in the shared tensors
//!    - [`Optimizer::step`] updates all parameters using the accumulated gradients
//!
//! # Parallel Execution
//!
//! All optimizers use **Rayon** to parallelize updates across layers. Each layer's
//! parameters are updated independently, providing ~2-4x speedup on multi-core CPUs
//! for networks with 10+ layers.
//!
//! # Available Optimizers
//!
//! | Struct | Description | State |
//! |--------|-------------|-------|
//! | [`gradient_descent::GradientDescent`] | Standard gradient descent with fixed learning rate | Stateful (stores params) |
//! | [`momentum_gd::MomentumGD`] | Momentum gradient descent - accelerates convergence by accumulating velocity | Stateful (stores params + velocities) |
//!
//! # Example
//!
//! ```rust,ignore
//! use neomatrix_core::optimizers::{gradient_descent::GradientDescent, Optimizer, ParametersRef};
//! use neomatrix_core::layers::{dense::Dense, Layer};
//! use neomatrix_core::tensor::Tensor;
//! use std::sync::{Arc, Mutex};
//!
//! // Create layers
//! let mut layer1 = Dense::new(784, 128, None, None);
//! let mut layer2 = Dense::new(128, 10, None, None);
//!
//! // Collect parameter references
//! let params = vec![
//!     layer1.get_parameters(),
//!     layer2.get_parameters(),
//! ];
//!
//! // Create and register optimizer
//! let mut optimizer = GradientDescent::new(0.01, vec![]);
//! optimizer.register_params(params);
//!
//! // Training step
//! optimizer.zero_grad().unwrap();               // 1. Reset gradients
//! let output1 = layer1.forward(&input, true).unwrap(); // 2. Forward pass
//! let output2 = layer2.forward(&output1, true).unwrap();
//! // ... compute loss ...
//! let grad2 = layer2.backward(&loss_grad).unwrap(); // 3. Backward pass
//! let grad1 = layer1.backward(&grad2).unwrap();
//! optimizer.step().unwrap();                    // 4. Update parameters
//! ```

pub mod adagrad;
pub mod gradient_descent;
pub mod momentum_gd;

use std::sync::{Arc, Mutex};

use crate::{errors::TensorError, tensor::Tensor};

/// Shared reference to a layer's trainable parameters and their gradients.
///
/// This struct provides shared ownership of a layer's weights, biases, and their
/// corresponding gradients using `Arc<Mutex<Tensor>>`. It enables both the layer
/// and optimizer to hold references to the same underlying tensors.
///
/// # Architecture
///
/// - **Layer side**: Holds `ParametersRef` to access weights/biases during forward
///   pass and to accumulate gradients during backward pass.
/// - **Optimizer side**: Receives `ParametersRef` via [`Optimizer::register_params`]
///   and uses it to update parameters during [`Optimizer::step`].
///
/// # Fields
///
/// - `weights`: Layer's weight matrix (typically shape `[in_features, out_features]`)
/// - `biases`: Layer's bias vector (typically shape `[out_features]`)
/// - `w_grads`: Accumulated gradient for weights (same shape as `weights`)
/// - `b_grads`: Accumulated gradient for biases (same shape as `biases`)
///
/// # Example
///
/// ```rust,ignore
/// use neomatrix_core::optimizers::ParametersRef;
/// use neomatrix_core::tensor::Tensor;
/// use std::sync::{Arc, Mutex};
///
/// // Create shared parameters
/// let weights = Arc::new(Mutex::new(Tensor::random(vec![784, 128], 0.0..1.0).unwrap()));
/// let biases = Arc::new(Mutex::new(Tensor::zeros(vec![128])));
/// let w_grads = Arc::new(Mutex::new(Tensor::zeros(vec![784, 128])));
/// let b_grads = Arc::new(Mutex::new(Tensor::zeros(vec![128])));
///
/// let params = ParametersRef { weights, biases, w_grads, b_grads };
///
/// // Both layer and optimizer can now access these tensors
/// let w = params.weights.lock().unwrap();
/// println!("Weight shape: {:?}", w.shape);
/// ```
#[derive(Clone, Debug)]
pub struct ParametersRef {
    /// Shared reference to layer weights
    pub weights: Arc<Mutex<Tensor>>,
    /// Shared reference to layer biases
    pub biases: Arc<Mutex<Tensor>>,
    /// Shared reference to weight gradients
    pub w_grads: Arc<Mutex<Tensor>>,
    /// Shared reference to bias gradients
    pub b_grads: Arc<Mutex<Tensor>>,
}

/// Common interface for all parameter update strategies.
///
/// All optimizers follow the **stateful, PyTorch-style** training pattern:
///
/// 1. **Register parameters** via [`register_params`](Optimizer::register_params)
///    before training begins
/// 2. **Reset gradients** via [`zero_grad`](Optimizer::zero_grad) at the start of
///    each training iteration
/// 3. **Accumulate gradients** during the backward pass (layers write to shared
///    `Arc<Mutex<Tensor>>` via [`ParametersRef`])
/// 4. **Update parameters** via [`step`](Optimizer::step) using the accumulated
///    gradients
///
/// # Stateful vs. Stateless Optimizers
///
/// - **Simple optimizers** (e.g., [`GradientDescent`](gradient_descent::GradientDescent)):
///   Only store `params: Vec<ParametersRef>` and `learning_rate`
/// - **Adaptive optimizers** (e.g., Adam, RMSprop): Store additional state like
///   momentum vectors and velocity estimates, one per registered parameter
///
/// # Parallel Execution
///
/// All optimizers use **Rayon** to parallelize operations across layers. Each layer's
/// parameters are independent, enabling multi-threaded updates with ~2-4x speedup
/// on multi-core CPUs.
///
/// # Errors
///
/// Returns a [`TensorError`] if:
/// - Mutex locks fail (rare, indicates poisoned mutex)
/// - Tensor arithmetic operations fail (e.g., shape mismatches — should never happen
///   if gradients were computed correctly)
///
/// # Example Implementation
///
/// ```rust,ignore
/// use neomatrix_core::optimizers::{Optimizer, ParametersRef};
/// use neomatrix_core::errors::TensorError;
/// use rayon::prelude::*;
///
/// struct MyOptimizer {
///     learning_rate: f32,
///     params: Vec<ParametersRef>,
/// }
///
/// impl Optimizer for MyOptimizer {
///     fn register_params(&mut self, params: Vec<ParametersRef>) {
///         self.params = params;
///         // Adaptive optimizers would initialize momentum/velocity here
///     }
///
///     fn step(&mut self) -> Result<(), TensorError> {
///         // Parallel update across all registered parameters
///         self.params.par_iter().try_for_each(|param| {
///             let mut w = param.weights.lock().unwrap();
///             let w_grad = param.w_grads.lock().unwrap();
///             // Update rule: θ = θ - lr·∇θ
///             *w = (&*w - &*w_grad * self.learning_rate)?;
///             Ok::<(), TensorError>(())
///         })
///     }
///
///     fn zero_grad(&mut self) -> Result<(), TensorError> {
///         // Parallel zeroing of all gradients
///         self.params.par_iter().try_for_each(|param| {
///             let mut w_grad = param.w_grads.lock().unwrap();
///             *w_grad = Tensor::zeros(w_grad.shape.clone());
///             Ok::<(), TensorError>(())
///         })
///     }
/// }
/// ```
pub trait Optimizer {
    /// Register layer parameters with the optimizer.
    ///
    /// Must be called **once** before training begins. Stores references to all
    /// trainable parameters (weights, biases, gradients) that will be updated
    /// during training.
    ///
    /// # Parameters
    ///
    /// - `params`: Vector of [`ParametersRef`] obtained from layers via
    ///   `layer.get_parameters()`
    ///
    /// # Note
    ///
    /// Adaptive optimizers (Adam, RMSprop) initialize their internal state
    /// (momentum, velocity) in this method based on the shapes of the registered
    /// parameters.
    fn register_params(&mut self, params: Vec<ParametersRef>);

    /// Update all registered parameters using accumulated gradients.
    ///
    /// Applies the optimizer's update rule (e.g., `θ = θ - lr·∇θ` for gradient
    /// descent) to all weights and biases. Runs in parallel across layers via
    /// Rayon.
    ///
    /// # Training Loop Position
    ///
    /// Call **after** the backward pass:
    ///
    /// ```text
    /// zero_grad() → forward() → compute_loss() → backward() → step()
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if:
    /// - Mutex locks fail (indicates poisoned mutex)
    /// - Tensor operations fail (shape mismatches — should never occur)
    fn step(&mut self) -> Result<(), TensorError>;

    /// Reset all gradients to zero.
    ///
    /// Must be called **before each forward pass** to clear gradients from the
    /// previous iteration. Since gradients accumulate by default (+=), forgetting
    /// to zero them will cause incorrect updates.
    ///
    /// # Training Loop Position
    ///
    /// Call **before** the forward pass:
    ///
    /// ```text
    /// zero_grad() → forward() → compute_loss() → backward() → step()
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if:
    /// - Mutex locks fail
    /// - Tensor allocation fails (extremely rare)
    fn zero_grad(&mut self) -> Result<(), TensorError>;
}
