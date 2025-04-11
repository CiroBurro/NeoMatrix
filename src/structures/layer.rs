/// This module defines the Layer class for a neural network
/// and provides methods for the forward and backward propagation in each layer.
/// It uses the Tensor struct for manipulating the data

/// Necessary imports
use ndarray::Ix2;
use pyo3::prelude::*;
use crate::structures::tensor::Tensor;
use crate::utils::weights_biases::{random_weights, random_biases};
use crate::functions::activation::*;

/// Layer class definition
///
/// Fields:
/// -nodes: number of neurons in the layer
/// -input: a 1D tensor storing the input data
/// -output: a 1D tensor storing the output data
/// -weights: a 2D tensor storing the weights of the layer
/// -biases: a 1D tensor storing the biases of the layer
/// -activation: struct to specify the activation function of the layer
#[pyclass(module = "neomatrix", get_all, set_all)]
#[derive(Clone, Debug)]
pub struct Layer {
    pub nodes: usize,
    pub input: Tensor,
    pub output: Tensor,
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: Activation,
}

/// Layer struct implementation
#[pymethods]
impl Layer {
    /// Constructor method for Layer class in python
    ///
    /// Parameters:
    /// -nodes: number of neurons in the layer
    /// -input: a 1D tensor storing the input data
    /// activation: struct to specify the activation function of the layer
    ///
    /// Python usage:
    /// ```python
    /// from neomatrix import Tensor, Layer, Activation
    /// input = Tensor([5], [1, 2, 3, 4, 5])
    /// layer = Layer(4, input, Activation.Relu)
    /// ```
    #[new]
    fn new(nodes: usize, input: Tensor, activation: Activation) -> Self {
        let weights = random_weights(input.data.len(), nodes.clone(), (-1.0, 1.0));
        let biases = random_biases(nodes.clone(), (-1.0, 1.0));
        let output_placeholder = Tensor::zeros(vec![nodes]);

        Self {
            nodes,
            input,
            output: output_placeholder,
            weights,
            biases,
            activation,
        }
    }

    /// Forward propagation method
    ///
    /// Python usage:
    ///```python
    /// output = layer.forward()
    /// ```
    fn forward(&mut self, parallel: bool) -> PyResult<Tensor> {
        // Check compatibility between dimensions
        if self.input.dimension != 1 || self.weights.dimension != 2 || self.biases.dimension != 1 {
            panic!("Dimensioni non valide layer");
        }
        let p = self.input.shape[0];
        let (m, _) = self.weights.data.clone().into_dimensionality::<Ix2>().unwrap().dim();
        if m != p {
            panic!("Dimensioni non valide n e p");
        }

        // Selection of the activation function
        let f: Box<dyn ActivationFunction> = match self.activation.clone() {
            Activation::Relu => Box::new(Relu),
            Activation::Sigmoid => Box::new(Sigmoid),
            Activation::Softmax => Box::new(Softmax),
            Activation::Tanh => Box::new(Tanh),
        };

        // forward prop algorithm
        if parallel {
            self.output = f.par_function(self.input.dot(&self.weights)?.add(&self.biases)?);
        } else {
            self.output = f.function(self.input.dot(&self.weights)?.add(&self.biases)?);
        }

        Ok(self.output.clone())
    }

    /// Backward propagation method
    fn backward(&self) -> Tensor {
        todo!()
    }

    fn __repr__(&self) -> String {
        format!("Layer (nodes: {})", self.nodes)
    }
}