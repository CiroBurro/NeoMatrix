//! Activation layer wrappers for neural networks.
//!
//! This module provides layer implementations that wrap activation functions from
//! [`crate::math::activations`], adding backpropagation support through input/output caching.
//!
//! # Available Activation Layers
//!
//! - **[`ReLu`]**: Rectified Linear Unit (caches **input** for derivative computation)
//! - **[`Sigmoid`]**: Logistic sigmoid (caches **output** for efficient gradient computation)
//! - **[`Tanh`]**: Hyperbolic tangent (caches **output** for efficient gradient computation)
//! - **[`Softmax`]**: Softmax normalization (simplified gradient passthrough)
//!
//! # Caching Strategy
//!
//! Different activation layers cache different values based on their derivative formulas:
//! - **Input caching** (ReLU, Softmax): Derivative depends on the input value
//! - **Output caching** (Sigmoid, Tanh): Derivative can be computed from the output value,
//!   avoiding redundant computation during backpropagation
//!
//! # Example
//!
//! ```rust,ignore
//! use neomatrix_core::layers::{activations::ReLu, Layer};
//! use neomatrix_core::tensor::Tensor;
//!
//! let mut relu_layer = ReLu::default();
//! let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![1, 4]);
//!
//! // Forward: max(0, x)
//! let output = relu_layer.forward(&input, true).unwrap();
//!
//! // Backward: gradient * (input > 0)
//! let grad = Tensor::ones(&[1, 4]).unwrap();
//! let input_grad = relu_layer.backward(&grad).unwrap();
//! ```

use crate::{
    errors::LayerError,
    layers::Layer,
    math::activations::{self, ActivationFunction},
    tensor::Tensor,
};

/// ReLU activation layer with backpropagation support.
///
/// Applies the Rectified Linear Unit activation: `f(x) = max(0, x)`.
/// Caches the **input** during forward pass for computing the derivative during backprop.
///
/// # Derivative
///
/// `f'(x) = 1 if x > 0, else 0`
///
/// The derivative depends on the input value, so this layer caches the input tensor
/// rather than the output.
pub struct ReLu {
    inner: activations::Relu,
    /// Cached input from forward pass, used for derivative computation in backward pass.
    input_cache: Option<Tensor>,
}
impl ReLu {
    pub fn new() -> Self {
        Self {
            inner: activations::Relu,
            input_cache: None,
        }
    }
}
impl Layer for ReLu {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        // Cache input for computing derivative during backward pass
        if training {
            self.input_cache = Some(input.clone());
        }

        self.inner.function(input).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        // Element-wise multiply: ∇L/∂x = ∇L/∂y · f'(x)
        // where f'(x) = 1 if x > 0, else 0
        output_gradient
            .dot(
                &self
                    .inner
                    .derivative(
                        self.input_cache
                            .as_ref()
                            .ok_or(LayerError::NotInitialized)?,
                    )
                    .map_err(LayerError::from)?,
            )
            .map_err(LayerError::from)
    }
}

/// Sigmoid activation layer with backpropagation support.
///
/// Applies the logistic sigmoid function: `f(x) = 1 / (1 + e^(-x))`.
/// Caches the **output** during forward pass for efficient derivative computation.
///
/// # Derivative
///
/// `f'(x) = f(x) · (1 - f(x))`
///
/// Since the derivative can be expressed in terms of the output value, this layer
/// caches the output rather than the input, avoiding redundant sigmoid computation.
pub struct Sigmoid {
    inner: activations::Sigmoid,
    /// Cached output from forward pass, used for derivative computation in backward pass.
    output_cache: Option<Tensor>,
}
impl Sigmoid {
    pub fn new() -> Self {
        Self {
            inner: activations::Sigmoid,
            output_cache: None,
        }
    }
}
impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        let output = self.inner.function(input).map_err(LayerError::from);

        // Cache output for computing derivative during backward pass
        if training {
            self.output_cache = Some(output?);
        }

        self.inner.function(input).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        let prev_output = self
            .output_cache
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // ∇L/∂x = ∇L/∂y · f'(x), where f'(x) = f(x) · (1 - f(x))
        (output_gradient * ((prev_output * (1.0 - prev_output)).map_err(LayerError::from)?))
            .map_err(LayerError::from)
    }
}

/// Hyperbolic tangent activation layer with backpropagation support.
///
/// Applies the tanh activation: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`.
/// Caches the **output** during forward pass for efficient derivative computation.
///
/// # Derivative
///
/// `f'(x) = 1 - f(x)²`
///
/// Similar to Sigmoid, the derivative can be computed from the output, avoiding
/// redundant computation of the exponential functions.
pub struct Tanh {
    inner: activations::Tanh,
    /// Cached output from forward pass, used for derivative computation in backward pass.
    output_cache: Option<Tensor>,
}
impl Tanh {
    pub fn new() -> Self {
        Self {
            inner: activations::Tanh,
            output_cache: None,
        }
    }
}
impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        let output = self.inner.function(input).map_err(LayerError::from);

        // Cache output for computing derivative during backward pass
        if training {
            self.output_cache = Some(output?);
        }

        self.inner.function(input).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        let prev_output = self
            .output_cache
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // ∇L/∂x = ∇L/∂y · f'(x), where f'(x) = 1 - f(x)²
        (output_gradient * (1.0 - (prev_output * prev_output).map_err(LayerError::from)?))
            .map_err(LayerError::from)
    }
}

/// Softmax activation layer with backpropagation support.
///
/// Applies the softmax normalization: `f(x_i) = e^(x_i) / Σ_j e^(x_j)`.
/// Caches the **output** during forward pass for derivative computation.
///
/// # Derivative
///
/// The Jacobian matrix of softmax is:
/// - Diagonal: `s_i(1 - s_i)`  
/// - Off-diagonal: `-s_i · s_j`
///
/// This implementation computes the full Jacobian-vector product during backpropagation.
///
/// # Optimization with Cross-Entropy
///
/// When used with categorical cross-entropy loss, the combined gradient simplifies to
/// `softmax(logits) - y_true`. Use the `backward_with_logits` method in the Python API
/// for this optimized path.
pub struct Softmax {
    inner: activations::Softmax,
    /// Cached output from forward pass, used for derivative computation in backward pass.
    output_cache: Option<Tensor>,
}
impl Softmax {
    pub fn new() -> Self {
        Self {
            inner: activations::Softmax,
            output_cache: None,
        }
    }
}
impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        let output = self.inner.function(input).map_err(LayerError::from)?;

        if training {
            self.output_cache = Some(output.clone());
        }

        Ok(output)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        let softmax_output = self
            .output_cache
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let jacobian = self
            .inner
            .derivative(softmax_output)
            .map_err(LayerError::from)?;

        jacobian.dot(output_gradient).map_err(LayerError::from)
    }
}
