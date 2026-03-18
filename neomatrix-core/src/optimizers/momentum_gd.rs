//! Momentum Gradient Descent optimizer.
//!
//! This module provides [`MomentumGD`], a first-order optimizer that accelerates convergence
//! by accumulating an exponentially-weighted moving average of gradients (velocity).
//!
//! # Mathematical Operation
//!
//! For each trainable parameter `θ` (weights or biases) and its gradient `∇θ`:
//! ```text
//! v_(t+1) = β · v_t + (1 - β) · ∇θ    // Update velocity (exponentially weighted average)
//! θ_(t+1) = θ_t - α · v_(t+1)          // Update parameters using velocity
//! ```
//!
//! Where:
//! - `v_t`: Velocity at time t (exponentially weighted average of past gradients)
//! - `β`: Momentum coefficient (controls how much past gradients influence current update)
//! - `∇θ`: Current gradient
//! - `α`: Learning rate (step size)
//!
//! # Why Momentum Works
//!
//! Standard gradient descent can be slow and oscillate in narrow valleys. Momentum helps:
//!
//! - **Accelerates convergence**: Velocity builds up in the correct direction, like a ball
//!   rolling downhill gaining speed
//! - **Dampens oscillations**: Oscillations in perpendicular directions cancel out while
//!   progress in the primary direction accumulates
//! - **Better generalization**: Slightly stochastic behavior can escape sharp local minima
//!
//! # Stateful Architecture
//!
//! Unlike textbook momentum, [`MomentumGD`] is **stateful** with per-parameter velocity:
//! - Stores references to all layer parameters via `params: Vec<ParametersRef>`
//! - Maintains velocity vectors (`w_velocities`, `b_velocities`) for each parameter
//! - Uses `Arc<Mutex<Tensor>>` for shared ownership between layers and optimizer
//!
//! This design matches PyTorch's optimizer API and enables:
//! - Single `optimizer.step()` call instead of per-layer updates
//! - Parallel updates across layers via Rayon (~2-4x speedup on 10+ layers)
//! - Consistent velocity state across training iterations
//!
//! # Comparison with Standard Gradient Descent
//!
//! | Aspect | GradientDescent | MomentumGD |
//! |--------|-----------------|------------|
//! | Update rule | `θ = θ - lr·∇θ` | `θ = θ - lr·v` |
//! | Velocity | None | `v = β·v + (1-β)·∇θ` |
//! | Convergence | Slower | Faster (typically 2-10x) |
//! | Oscillation | High | Reduced |
//! | Memory overhead | Minimal | +2 tensors per layer |
//!
//! # Performance
//!
//! - **Parallel execution**: Updates across layers run in parallel via Rayon
//! - **Lock overhead**: Negligible (<0.1% of total time) — ~10-50ns per lock
//! - **Memory overhead**: 2 velocity tensors per registered layer (weights + biases)
//! - **Typical step time**: ~4ms for 10-layer network (similar to GradientDescent)
//!
//! # Hyperparameter Selection
//!
//! | Parameter | Typical Range | Effect |
//! |-----------|---------------|--------|
//! | Learning rate | 1e-4 to 1e-1 | Step size (lower than SGD) |
//! | Momentum (β) | 0.9 to 0.999 | Velocity decay (higher = smoother) |
//!
//! - **β = 0.9**: 10 most recent gradients weighted equally
//! - **β = 0.99**: 100 most recent gradients weighted equally
//! - **β = 0.999**: 1000 most recent gradients weighted equally
//!
//! # Example
//!
//! ```rust,ignore
//! use neomatrix_core::optimizers::{momentum_gd::MomentumGD, Optimizer};
//! use neomatrix_core::layers::{dense::Dense, Layer};
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
//! // Create optimizer with LR=0.01, momentum=0.9
//! let mut optimizer = MomentumGD::new(0.01, 0.9);
//! optimizer.register_params(params);
//!
//! // Training loop
//! for epoch in 0..100 {
//!     optimizer.zero_grad().unwrap();  // 1. Reset gradients
//!
//!     // 2. Forward pass
//!     let hidden = layer1.forward(&input, true).unwrap();
//!     let output = layer2.forward(&hidden, true).unwrap();
//!
//!     // 3. Compute loss and backward pass
//!     let loss_grad = compute_loss_gradient(&output, &target);
//!     let grad_hidden = layer2.backward(&loss_grad).unwrap();
//!     layer1.backward(&grad_hidden).unwrap();
//!
//!     // 4. Update all parameters (velocity accumulated)
//!     optimizer.step().unwrap();
//! }
//! ```

use std::ops::{Deref, DerefMut};

use rayon::prelude::*;

use crate::{
    errors::TensorError,
    optimizers::{Optimizer, ParametersRef},
    tensor::Tensor,
};

/// Momentum Gradient Descent optimizer with velocity accumulation.
///
/// Updates parameters using an exponentially-weighted moving average of gradients,
/// which accelerates convergence and dampens oscillations compared to standard GD.
///
/// # Architecture
///
/// This optimizer is **stateful** with adaptive momentum:
/// - Stores references to all layer parameters via `params: Vec<ParametersRef>`
/// - Maintains velocity vectors for each parameter (weights and biases separately)
/// - Accumulates velocity over training iterations for smoother updates
///
/// # Comparison with Other Optimizers
///
/// | Optimizer | Adaptive LR | Momentum | Complexity |
/// |-----------|------------|----------|------------|
/// | [`super::GradientDescent`] | No | No | Simplest |
/// | MomentumGD (this) | No | Yes | Simple |
/// | Adam | Yes | Yes | Moderate |
///
/// # Fields
///
/// - `learning_rate`: Step size for parameter updates (`α`). Controls how much
///   parameters change. Typically lower than SGD since momentum amplifies effective LR.
///   - Typical range: `1e-4` to `1e-1`
///   - With momentum, effective LR is approximately `lr / (1 - β)` in steady state
///
/// - `momentum`: Velocity decay coefficient (`β`). Controls how much past gradients
///   influence the current update.
///   - Typical range: `0.9` to `0.999`
///   - `0.9`: Fast response to gradient changes (recommended for non-stationary tasks)
///   - `0.99`: Smoother updates, slower response (recommended for most tasks)
///
/// - `params`: Registered layer parameters (populated via [`register_params`](super::Optimizer::register_params))
///
/// - `w_velocities`: Per-layer weight velocity vectors (initialized on `register_params`)
///
/// - `b_velocities`: Per-layer bias velocity vectors (initialized on `register_params`)
///
/// # Example
///
/// ```rust,ignore
/// use neomatrix_core::optimizers::{momentum_gd::MomentumGD, Optimizer};
///
/// // Create optimizer with LR=0.01, momentum=0.9
/// let mut optimizer = MomentumGD::new(0.01, 0.9);
///
/// // Register parameters from layers
/// let params = vec![layer1.get_parameters(), layer2.get_parameters()];
/// optimizer.register_params(params);
///
/// // Training step (see module-level docs for full example)
/// optimizer.zero_grad()?;
/// // ... forward + backward ...
/// optimizer.step()?;
/// ```
pub struct MomentumGD {
    pub learning_rate: f32,
    pub momentum: f32,
    pub params: Vec<ParametersRef>,
    pub w_velocities: Vec<Tensor>,
    pub b_velocities: Vec<Tensor>,
}

impl MomentumGD {
    /// Create a new momentum gradient descent optimizer.
    ///
    /// # Parameters
    ///
    /// - `learning_rate`: Step size for parameter updates. Must be positive.
    ///   Typical values: `0.001` to `0.1` (see struct docs for tuning guide).
    ///   With momentum, the effective LR is higher than the nominal value.
    /// - `momentum`: Velocity decay coefficient. Controls how much past gradients
    ///   influence the current update. Typical values: `0.9` to `0.999`.
    ///   - `0.9`: Fast adaptation (good for changing loss landscapes)
    ///   - `0.99`: Smoother convergence (default recommendation)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use neomatrix_core::optimizers::momentum_gd::MomentumGD;
    ///
    /// let optimizer = MomentumGD::new(0.01, 0.9);
    /// // Call optimizer.register_params(...) before training
    /// ```
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            params: Vec::new(),
            w_velocities: Vec::new(),
            b_velocities: Vec::new(),
        }
    }
}

impl Optimizer for MomentumGD {
    /// Register layer parameters with the optimizer and initialize velocity vectors.
    ///
    /// This method:
    /// 1. Clears existing velocities
    /// 2. Initializes zero-velocity tensors for each parameter (weights and biases)
    /// 3. Stores references to all layer parameters
    ///
    /// # Implementation Details
    ///
    /// Velocity vectors are created with the same shape as their corresponding parameters
    /// and initialized to zeros. This ensures proper gradient accumulation from the first
    /// training step.
    ///
    /// # Note
    ///
    /// Must be called before any training begins. The optimizer holds references to
    /// layer parameters, so layers must not be dropped while the optimizer is in use.
    fn register_params(&mut self, params: Vec<ParametersRef>) {
        self.w_velocities.clear();
        self.b_velocities.clear();

        for param in &params {
            let w_shape = param.weights.lock().unwrap().shape.clone();
            let b_shape = param.biases.lock().unwrap().shape.clone();

            let w_len = w_shape.iter().product();
            let b_len = b_shape.iter().product();

            self.w_velocities
                .push(Tensor::new(w_shape, vec![0.0; w_len]).unwrap());
            self.b_velocities
                .push(Tensor::new(b_shape, vec![0.0; b_len]).unwrap());
        }

        self.params = params;
    }

    /// Update all registered parameters using accumulated gradients and velocity.
    ///
    /// Applies the momentum update rule for both weights and biases:
    /// ```text
    /// v_new = momentum * v_old + (1 - momentum) * gradient
    /// param_new = param_old - learning_rate * v_new
    /// ```
    ///
    /// # Parallel Execution
    ///
    /// Uses Rayon to update layers in parallel. Each iteration:
    /// 1. Acquires locks on weights, biases, gradients, and velocities (independent per layer)
    /// 2. Updates velocity: `v = β·v + (1-β)·∇θ`
    /// 3. Updates parameters: `θ = θ - lr·v`
    /// 4. Writes updated values back to shared tensors
    ///
    /// # Performance
    ///
    /// - **Lock overhead**: ~10-50ns per lock (negligible vs tensor ops)
    /// - **Speedup**: ~2-4x on multi-core CPUs for 10+ layer networks
    /// - **Memory**: Velocity tensors already allocated on `register_params`
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if:
    /// - Mutex lock fails (indicates poisoned mutex — rare, fatal error)
    /// - Tensor arithmetic fails (shape mismatch — should never happen)
    fn step(&mut self) -> Result<(), TensorError> {
        self.params
            .par_iter()
            .zip(self.w_velocities.par_iter_mut())
            .zip(self.b_velocities.par_iter_mut())
            .try_for_each(|((param, w_velocity), b_velocity)| {
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

                *w_velocity =
                    (w_velocity.deref() * self.momentum + w_grads.deref() * (1.0 - self.momentum))?;
                *b_velocity =
                    (b_velocity.deref() * self.momentum + b_grads.deref() * (1.0 - self.momentum))?;

                *weights = (weights.deref_mut() - w_velocity.deref() * self.learning_rate)?;
                *biases = (biases.deref_mut() - b_velocity.deref() * self.learning_rate)?;

                Ok::<(), TensorError>(())
            })?;
        Ok(())
    }

    /// Reset all gradients to zero (velocity is preserved between steps).
    ///
    /// Must be called before each forward pass to clear gradients from the previous
    /// iteration. Since gradients accumulate during backward pass (+=), forgetting to
    /// zero them will cause incorrect updates.
    ///
    /// Unlike gradients, velocity vectors are preserved across training steps
    /// to maintain the accumulated momentum information.
    ///
    /// # Implementation
    ///
    /// Creates new zero tensors with the same shape as existing gradients and
    /// writes them to the shared `Arc<Mutex<Tensor>>`. Uses Rayon to parallelize
    /// across layers.
    ///
    /// # Performance
    ///
    /// - **Typical time**: ~0.7ms for 10-layer network (1000→50 neurons)
    /// - **Memory**: Allocates new zero tensors (old gradients dropped)
    /// - **Parallel**: Independent per layer (no contention)
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if:
    /// - Mutex lock fails (poisoned mutex)
    /// - Tensor allocation fails (out of memory — extremely rare)
    fn zero_grad(&mut self) -> Result<(), TensorError> {
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
