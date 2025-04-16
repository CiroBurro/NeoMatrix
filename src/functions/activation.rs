/// This module defines activation functions for neural networks.
/// It implements various activation functions like ReLU, Sigmoid, Tanh and Softmax,
/// providing both regular and parallel computation methods.
/// Necessary imports
use crate::structures::tensor::Tensor;
use ndarray::{s, Array2, Array3, Axis};
use pyo3::prelude::*;

/// Trait defining the interface for activation functions
/// 
/// Methods:
/// - function: Regular forward computation
/// - par_function: Parallel forward computation
/// - derivative: Derivative computation for backpropagation
pub trait ActivationFunction: Send + Sync {
    fn function(&self, t: &mut Tensor) -> Tensor;
    fn par_function(&self, t: &mut Tensor) -> Tensor;
    fn derivative(&self, t: &mut Tensor) -> Tensor;
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
    fn function(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.mapv_inplace(|x| x.max(0.0));

        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
    fn par_function(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.par_mapv_inplace(|x| x.max(0.0));

        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
    fn derivative(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
}

/// Sigmoid activation function
/// f(x) = 1 / (1 + e^(-x))
pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn function(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
    fn par_function(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
    fn derivative(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.mapv_inplace(|x| x * (1.0 - x));
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
}

/// Hyperbolic tangent activation function
/// f(x) = tanh(x)
pub struct Tanh;
impl ActivationFunction for Tanh {
    fn function(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.mapv_inplace(|x| x.tanh());
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
    fn par_function(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.par_mapv_inplace(|x| x.tanh());
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
    fn derivative(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        t.data.mapv_inplace(|x| 1.0 - x.tanh().powf(2.0));
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
}

/// Softmax activation function
/// f(x_i) = e^(x_i) / Σ(e^(x_j))
pub struct Softmax;
impl ActivationFunction for Softmax {
    fn function(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        let denom = t.data.mapv(|x| x.exp()).sum();
        t.data.mapv_inplace(|x| x.exp() / denom);
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
    fn par_function(&self, t: &mut Tensor) -> Tensor {
        let dimension = t.dimension;
        let shape = vec![t.shape[0], t.shape[1]];
        let denom = t.data.mapv(|x| x.exp()).sum();
        t.data.par_mapv_inplace(|x| x.exp() / denom);
        Tensor {
            dimension,
            shape,
            data: t.data.to_owned(),
        }
    }
    fn derivative(&self, t: &mut Tensor) -> Tensor {

        if t.dimension == 1 {
            let s = self.function(t).data;
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
        } else if t.dimension == 2 {
            let batch_size = t.shape[0];

            let mut data:Array3<f64> = Array3::zeros((batch_size, t.shape[1], t.shape[1]));

            for (k, axe) in t.data.axis_iter(Axis(0)).enumerate() {
                let mut row_t = Tensor {
                    dimension: 1,
                    shape: vec![t.shape[1]],
                    data: axe.to_owned(),
                };
                let s = self.function(&mut row_t).data;
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

                data.slice_mut(s![k, .., ..]).assign(&jacobian_data);
            }

            Tensor {
                dimension: 3,
                shape: vec![batch_size, t.shape[1], t.shape[1]],
                data: data.into_dyn(),
            }
        } else {
            panic!("Softmax derivative is only implemented for 1D and 2D tensors");
        }
    }
}
