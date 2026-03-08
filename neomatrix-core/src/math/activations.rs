use crate::errors::MathError;
use crate::tensor::Tensor;
use ndarray::{s, Array2, Array3, Axis};

/// Trait defining the interface for activation functions
///
/// Both methods take `&Tensor` (immutable) and return a new `Tensor`.
/// Internally, each implementation clones the data once and applies
/// parallel operations on the owned copy.
pub trait ActivationFunction: Send + Sync {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError>;
    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError>;
}

/// Rectified Linear Unit (ReLU) — `f(x) = max(0, x)`
pub struct Relu;
impl ActivationFunction for Relu {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let mut data = t.data.to_owned();
        data.par_mapv_inplace(|x| x.max(0.0));
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let data = t.data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
}

/// Sigmoid — `f(x) = 1 / (1 + e^(-x))`
pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let mut data = t.data.to_owned();
        data.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let data = t.data.mapv(|x| x * (1.0 - x));
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
}

/// Hyperbolic tangent — `f(x) = tanh(x)`
pub struct Tanh;
impl ActivationFunction for Tanh {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let mut data = t.data.to_owned();
        data.par_mapv_inplace(|x| x.tanh());
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let data = t.data.mapv(|x| 1.0 - x.tanh().powf(2.0));
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
}

/// Softmax — `f(x_i) = e^(x_i) / Σ(e^(x_j))`
pub struct Softmax;
impl ActivationFunction for Softmax {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let dimension = t.dimension;
        let shape = match dimension {
            1 | 2 => t.shape.clone(),
            _ => return Err(MathError::SoftmaxUnsupportedDimension),
        };
        let mut data = t.data.to_owned();
        if dimension == 1 {
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            data.par_mapv_inplace(|x| (x - max).exp());
            let denom = data.sum();
            data.par_mapv_inplace(|x| x / denom);
        } else {
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
        let jacobian = |s: &ndarray::ArrayD<f32>| -> Array2<f32> {
            let n = s.len();
            let mut j = Array2::<f32>::zeros((n, n));
            for i in 0..n {
                for k in 0..n {
                    j[[i, k]] = if i == k {
                        s[i] * (1.0 - s[i])
                    } else {
                        -s[i] * s[k]
                    };
                }
            }
            j
        };

        if t.dimension == 1 {
            let s = self.function(t)?.data;
            let n = s.len();
            let j = jacobian(&s);
            Ok(Tensor {
                dimension: 2,
                shape: vec![n, n],
                data: j.into_dyn(),
            })
        } else if t.dimension == 2 {
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

/// Linear (identity) — `f(x) = x`
pub struct Linear;
impl ActivationFunction for Linear {
    fn function(&self, t: &Tensor) -> Result<Tensor, MathError> {
        Ok(t.clone())
    }
    fn derivative(&self, t: &Tensor) -> Result<Tensor, MathError> {
        let data = t.data.mapv(|_| 1.0);
        Ok(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        })
    }
}
