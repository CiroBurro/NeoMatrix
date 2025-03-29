use pyo3::prelude::*;
use crate::structures::tensor::Tensor;

pub trait ActivationFunction: Send + Sync {
    fn function(&self, t: Tensor) -> Tensor;
    fn derivative(&self, t: Tensor) -> Tensor;
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Softmax
}

pub struct Relu;
impl ActivationFunction for Relu {
    fn function(&self, t: Tensor) -> Tensor {
        t
    }
    fn derivative(&self, t: Tensor) -> Tensor {
        todo!()
    }
}

pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn function(&self, t: Tensor) -> Tensor {
        t
    }
    fn derivative(&self, t: Tensor) -> Tensor {
        todo!()
    }
}

pub struct Tanh;
impl ActivationFunction for Tanh {
    fn function(&self, t: Tensor) -> Tensor {
        todo!()
    }
    fn derivative(&self, t: Tensor) -> Tensor {
        todo!()
    }
}

pub struct Softmax;
impl ActivationFunction for Softmax {
    fn function(&self, t: Tensor) -> Tensor {
        todo!()
    }
    fn derivative(&self, t: Tensor) -> Tensor {
        todo!()
    }
}