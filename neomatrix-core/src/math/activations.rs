//! Activation functions for neural network layers.
//!
//! This module provides the `ActivationFunction` trait and implementations for common
//! activation functions used in deep learning. Each activation function includes both
//! the forward computation (`function`) and its derivative (`derivative`) for backpropagation.
//!
//! # Available Activations
//!
//! - **ReLU** (`Relu`): Rectified Linear Unit - introduces non-linearity while being computationally efficient
//! - **Sigmoid** (`Sigmoid`): Squashes values to (0, 1) - commonly used for binary classification
//! - **Tanh** (`Tanh`): Squashes values to (-1, 1) - zero-centered alternative to Sigmoid
//! - **Softmax** (`Softmax`): Converts logits to probability distribution - used for multi-class classification
//!
//! # Performance
//!
//! All activation functions leverage parallel computation via Rayon's `par_mapv_inplace`
//! for element-wise operations, providing significant speedup on multi-core systems.
//!
//! # Example
//!
//! ```ignore
//! use neomatrix_core::math::activations::{ActivationFunction, Relu};
//! use neomatrix_core::tensor::Tensor;
//!
//! let relu = Relu;
//! let input = Tensor::new(vec![2, 2], vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
//! let output = relu.function(&input).unwrap(); // [0.0, 2.0, 0.0, 4.0]
//! ```

use crate::errors::MathError;
use crate::tensor::Tensor;
use ndarray::{s, Array2, Array3, Axis};

/// Trait defining the interface for activation functions.
///
/// All activation functions must implement this trait to be used in neural network layers.
/// Both methods take immutable references to tensors and return new tensors, ensuring
/// thread-safety and predictable ownership semantics.
///
/// # Methods
///
/// - `function`: Applies the activation function element-wise to the input tensor
/// - `derivative`: Computes the derivative of the activation function for backpropagation
///
/// # Thread Safety
///
/// The `Send + Sync` bounds allow activation functions to be safely shared across threads,
/// enabling parallel batch processing during training.
///
/// # Implementation Notes
///
/// Implementations should:
/// 1. Clone the input tensor's data once
/// 2. Apply transformations in-place using parallel operations where possible
/// 3. Return appropriate errors for unsupported tensor dimensions
pub(crate) trait ActivationFunction: Send + Sync {
    /// Applies the activation function to the input tensor.
    ///
    /// # Parameters
    ///
    /// - `t`: Input tensor of any dimension
    ///
    /// # Returns
    ///
    /// A new tensor with the activation function applied element-wise, or an error
    /// if the operation is not supported for the input tensor's shape.
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError>;

    /// Computes the derivative of the activation function.
    ///
    /// # Parameters
    ///
    /// - `t`: Input tensor (for some activations) or output tensor (for others)
    ///
    /// # Returns
    ///
    /// A new tensor containing the derivative values, or an error if the operation
    /// is not supported for the input tensor's shape.
    ///
    /// # Implementation Variance
    ///
    /// - **ReLU, Softmax**: Expect the original input tensor
    /// - **Sigmoid, Tanh**: Expect the activation output (for numerical efficiency)
    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError>;
}

/// Rectified Linear Unit (ReLU) activation function.
///
/// # Mathematical Definition
///
/// - **Function**: `f(x) = max(0, x)`
/// - **Derivative**: `f'(x) = 1 if x > 0, else 0`
///
/// # Properties
///
/// - Non-saturating activation that helps mitigate vanishing gradients
/// - Computationally efficient (simple thresholding operation)
/// - Introduces sparsity (outputs exactly 0 for negative inputs)
/// - Most commonly used activation in modern deep learning
///
/// # Range
///
/// - Input: `(-∞, +∞)`
/// - Output: `[0, +∞)`
pub(crate) struct Relu;

impl ActivationFunction for Relu {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let mut data = t.data.to_owned();
        // f(x) = max(0, x)
        data.par_mapv_inplace(|x| x.max(0.0));
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }

    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError> {
        // f'(x) = 1 if x > 0, else 0
        let data = t.data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
}

/// Sigmoid activation function.
///
/// # Mathematical Definition
///
/// - **Function**: `f(x) = 1 / (1 + e^(-x))`
/// - **Derivative**: `f'(σ) = σ(1 - σ)` where `σ = f(x)`
///
/// # Properties
///
/// - Squashes input to range (0, 1), making it suitable for probability outputs
/// - Smooth, differentiable everywhere
/// - Saturates for large |x|, which can cause vanishing gradients
/// - Commonly used in binary classification output layers
///
/// # Range
///
/// - Input: `(-∞, +∞)`
/// - Output: `(0, 1)`
///
/// # Implementation Note
///
/// The derivative expects the **output** of the sigmoid function (not the input),
/// which is more numerically efficient since `σ' = σ(1 - σ)`.
pub(crate) struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let mut data = t.data.to_owned();
        data.par_mapv_inplace(|x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        });
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }

    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError> {
        // f'(σ) = σ(1 - σ) where σ is the sigmoid output
        let data = t.data.mapv(|x| x * (1.0 - x));
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
}

/// Hyperbolic tangent activation function.
///
/// # Mathematical Definition
///
/// - **Function**: `f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
/// - **Derivative**: `f'(tanh(x)) = 1 - tanh²(x)`
///
/// # Properties
///
/// - Squashes input to range (-1, 1), making it zero-centered
/// - Zero-centered output helps with gradient flow in deep networks
/// - Saturates for large |x|, similar to Sigmoid
/// - Often preferred over Sigmoid in hidden layers due to zero-centering
///
/// # Range
///
/// - Input: `(-∞, +∞)`
/// - Output: `(-1, 1)`
///
/// # Implementation Note
///
/// The derivative expects the **input** (not output), computing `1 - tanh²(x)` directly.
pub(crate) struct Tanh;

impl ActivationFunction for Tanh {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let mut data = t.data.to_owned();
        // f(x) = tanh(x)
        data.par_mapv_inplace(|x| x.tanh());
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }

    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError> {
        // f'(x) = 1 - tanh²(x)
        let data = t.data.mapv(|x| 1.0 - x.tanh().powf(2.0));
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
}

/// Softmax activation function.
///
/// # Mathematical Definition
///
/// - **Function**: `f(x_i) = e^(x_i) / Σ(e^(x_j))` for all j
/// - **Derivative**: Jacobian matrix where:
///   - `J[i,i] = f(x_i)(1 - f(x_i))` (diagonal)
///   - `J[i,k] = -f(x_i)f(x_k)` (off-diagonal)
///
/// # Properties
///
/// - Converts logits into a probability distribution (outputs sum to 1)
/// - Each output is in range (0, 1)
/// - Numerically stabilized using max subtraction: `e^(x_i - max(x))`
/// - Used exclusively in multi-class classification output layers
///
/// # Supported Dimensions
///
/// - **1D**: Single sample, returns Jacobian matrix (n×n)
/// - **2D**: Batch of samples, applies row-wise, returns 3D Jacobian (batch×n×n)
/// - **3D+**: Not supported (returns `SoftmaxUnsupportedDimension` error)
///
/// # Range
///
/// - Input: `(-∞, +∞)`
/// - Output: `(0, 1)` with `Σf(x_i) = 1`
pub(crate) struct Softmax;

impl ActivationFunction for Softmax {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let dimension = t.dimension;
        let shape = match dimension {
            1 | 2 => t.shape.clone(),
            _ => return Err(MathError::SoftmaxUnsupportedDimension),
        };
        let mut data = t.data.to_owned();

        if dimension == 1 {
            // Numerically stable softmax: subtract max before exp
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            data.par_mapv_inplace(|x| (x - max).exp());
            // f(x_i) = e^(x_i) / Σ(e^(x_j))
            let denom = data.sum();
            data.par_mapv_inplace(|x| x / denom);
        } else {
            // 2D: apply softmax row-wise for batch processing
            for mut row in data.axis_iter_mut(Axis(0)) {
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                row.mapv_inplace(|x| (x - max).exp());
                let denom = row.sum();
                row.mapv_inplace(|x| x / denom);
            }
        }

        Ok(Tensor {
            dimension,
            shape,
            data,
        })
    }

    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError> {
        // Computes full Jacobian matrix for softmax derivative
        let jacobian = |s: &ndarray::ArrayD<f32>| -> Array2<f32> {
            let n = s.len();
            let mut j = Array2::<f32>::zeros((n, n));
            for i in 0..n {
                for k in 0..n {
                    // J[i,k] = s_i * (δ_ik - s_k)
                    // where δ_ik is Kronecker delta (1 if i==k, else 0)
                    j[[i, k]] = if i == k {
                        s[i] * (1.0 - s[i]) // Diagonal
                    } else {
                        -s[i] * s[k] // Off-diagonal
                    };
                }
            }
            j
        };

        if t.dimension == 1 {
            // 1D input: compute softmax, then return n×n Jacobian
            let s = self.function(t)?.data;
            let n = s.len();
            let j = jacobian(&s);
            Ok(Tensor {
                dimension: 2,
                shape: vec![n, n],
                data: j.into_dyn(),
            })
        } else if t.dimension == 2 {
            // 2D input: compute batch×n×n Jacobian (one per sample)
            let batch_size = t.shape[0];
            let n = t.shape[1];
            let mut data = Array3::<f32>::zeros((batch_size, n, n));
            for (k, row) in t.data.axis_iter(Axis(0)).enumerate() {
                let row_t = Tensor {
                    dimension: 1,
                    shape: vec![n],
                    data: row.to_owned(),
                };
                let s = self.function(&row_t)?.data;
                data.slice_mut(s![k, .., ..]).assign(&jacobian(&s));
            }
            Ok(Tensor {
                dimension: 3,
                shape: vec![batch_size, n, n],
                data: data.into_dyn(),
            })
        } else {
            Err(MathError::SoftmaxDerivativeUnsupportedDimension)
        }
    }
}
