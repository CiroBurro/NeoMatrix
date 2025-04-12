/// This module defines activation functions for neural networks.
/// It implements various activation functions like ReLU, Sigmoid, Tanh and Softmax,
/// providing both regular and parallel computation methods.
/// Necessary imports
use crate::structures::tensor::Tensor;
use ndarray::Array2;
use pyo3::prelude::*;

/// Trait defining the interface for activation functions
/// 
/// Methods:
/// - function: Regular forward computation
/// - par_function: Parallel forward computation
/// - derivative: Derivative computation for backpropagation
pub trait ActivationFunction: Send + Sync {
    fn function(&self, t: Tensor) -> Tensor;
    fn par_function(&self, t: Tensor) -> Tensor;
    fn derivative(&self, t: Tensor) -> Tensor;
}

/// Python-accessible enum for activation function selection
#[pyclass]
#[derive(Clone, Debug)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}

/// Rectified Linear Unit (ReLU) activation function
/// f(x) = max(0, x)
pub struct Relu;
impl ActivationFunction for Relu {
    fn function(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.mapv_inplace(|x| x.max(0.0));

        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
    fn par_function(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.par_mapv_inplace(|x| x.max(0.0));

        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
    fn derivative(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
}

/// Sigmoid activation function
/// f(x) = 1 / (1 + e^(-x))
pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn function(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
    fn par_function(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
    fn derivative(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.mapv_inplace(|x| x * (1.0 - x));
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
}

/// Hyperbolic tangent activation function
/// f(x) = tanh(x)
pub struct Tanh;
impl ActivationFunction for Tanh {
    fn function(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.mapv_inplace(|x| x.tanh());
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
    fn par_function(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.par_mapv_inplace(|x| x.tanh());
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
    fn derivative(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        t.data.mapv_inplace(|x| 1.0 - x.tanh().powf(2.0));
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
}

/// Softmax activation function
/// f(x_i) = e^(x_i) / Σ(e^(x_j))
pub struct Softmax;
impl ActivationFunction for Softmax {
    fn function(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        let denom = t.data.mapv(|x| x.exp()).sum();
        t.data.mapv_inplace(|x| x.exp() / denom);
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
    fn par_function(&self, mut t: Tensor) -> Tensor {
        let dimension = 1;
        let shape = t.shape;
        let denom = t.data.mapv(|x| x.exp()).sum();
        t.data.par_mapv_inplace(|x| x.exp() / denom);
        Tensor {
            dimension,
            shape,
            data: t.data,
        }
    }
    fn derivative(&self, t: Tensor) -> Tensor {
        let s = self.function(t.clone()).data;
        let n = s.len();
        let mut jacobian_data = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    jacobian_data[[i, j]] = s[i] * (1.0 - s[i]);
                } else {
                    jacobian_data[[i, j]] = -s[i] * s[j];
                }
            }
        }

        Tensor {
            dimension: 2,
            shape: vec![n, n],
            data: jacobian_data.into_dyn(),
        }
    }
}
