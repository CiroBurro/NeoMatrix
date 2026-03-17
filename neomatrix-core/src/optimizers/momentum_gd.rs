use std::ops::{Deref, DerefMut};

use rayon::prelude::*;

use crate::{
    errors::TensorError,
    optimizers::{Optimizer, ParametersRef},
    tensor::Tensor,
};

/// Momentum Gradient Descent optimizer.
///
/// Accelerates convergence by accumulating a velocity vector in directions of persistent
/// gradient descent, dampening oscillations in high-curvature directions.
///
/// # Mathematical Operation
///
/// For each parameter θ (weights or biases):
/// ```text
/// v_(t+1) = β·v_t + (1-β)·∇θ        // Update velocity
/// θ_(t+1) = θ_t - α·v_(t+1)         // Update parameters
/// ```
///
/// Where:
/// - `v_t`: Velocity at time t (exponentially weighted average of gradients)
/// - `β`: Momentum coefficient (typically 0.9)
/// - `∇θ`: Current gradient
/// - `α`: Learning rate
pub struct MomentumGD {
    pub learning_rate: f32,
    pub momentum: f32,
    pub params: Vec<ParametersRef>,
    pub w_velocities: Vec<Tensor>,
    pub b_velocities: Vec<Tensor>,
}

impl MomentumGD {
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
