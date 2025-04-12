/// This module defines the Layer class for a neural network
/// and provides methods for the forward and backward propagation in each layer.
/// It uses the Tensor struct for manipulating the data
/// Necessary imports
use ndarray::{Axis, Ix2};
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
        let weights = random_weights(input.data.len(), nodes, (-1.0, 1.0));
        let biases = random_biases(nodes, (-1.0, 1.0));
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
    /// output = layer.forward(parallel=True)
    /// ```
    fn forward(&mut self, parallel: bool) -> PyResult<Tensor> {
        // Check compatibility between dimensions
        if self.input.dimension != 1 || self.weights.dimension != 2 || self.biases.dimension != 1 {
            panic!("Layer field have wrong dimensions, expected: input (1D), weights (2D), biases (1D)");
        }
        let p = self.input.shape[0];
        let (m, _) = self.weights.data.clone().into_dimensionality::<Ix2>().unwrap().dim();
        if m != p {
            panic!("Inputs and weights have incompatible shapes for forward propagation, the length of the input vector should be equal to the number of rows of the weights matrix");
        }

        // Selection of the activation function
        let f: Box<dyn ActivationFunction> = match self.activation.clone() {
            Activation::Relu => Box::new(Relu),
            Activation::Sigmoid => Box::new(Sigmoid),
            Activation::Softmax => Box::new(Softmax),
            Activation::Tanh => Box::new(Tanh),
        };

        // Forward prop algorithm
        if parallel {
            self.output = f.par_function(self.input.dot(&self.weights)?.add(&self.biases)?);
        } else {
            self.output = f.function(self.input.dot(&self.weights)?.add(&self.biases)?);
        }

        Ok(self.output.clone())
    }

    /// Backward propagation method
    ///
    /// Python usage:
    ///```python
    /// (w_gradients, b_gradients, deltas) = layer.forward(out_layer=False, deltas=deltas, next_weights=next_weights, all_outputs=all_outputs)
    /// ```
    fn backward(&self, out_layer: bool, deltas: Tensor, next_weights: Tensor, all_outputs: Option<Tensor>) -> PyResult<(Tensor, Tensor, Tensor)> {
        // Deltas dimension can be 1 (single sample processing) or 2 (batch processing)
        if deltas.dimension > 2 || deltas.dimension == 0{
            panic!("Deltas tensor has to be 1D or 2D for backpropagation");
        }

        if out_layer {

            if deltas.dimension == 1 {

                // Each node has its own delta
                if deltas.shape[0] != self.nodes {
                    panic!("If out_layer is True and deltas tensor is 1D, the number of deltas should be equal to the number of nodes in the output layer");
                }

                // Biases gradients are equal to the output deltas
                let biases_gradients = deltas.clone();

                // Inputs and deltas tensors have to be reshaped to 2D since 1D tensors multiplication returns a scalar
                let inputs = Tensor {
                    dimension: 2,
                    shape: vec![self.input.shape[0], 1],
                    data: self.input.data.clone(),
                };
                let out_deltas = Tensor {
                    dimension: 2,
                    shape: vec![1, deltas.shape[0]],
                    data: deltas.data.clone(),
                };

                // Weights gradients are calculated as the dot product between the inputs of the layer and the deltas
                let weights_gradients = inputs.dot(&out_deltas)?;
                Ok((weights_gradients, biases_gradients, deltas))

            } else  {

                if all_outputs.is_none() {
                    panic!("If deltas tensor is 2D, all_outputs must be defined");
                }
                let all_outputs = all_outputs.unwrap();

                // All_outputs matrix contains all outputs of the previous layer of the entire batch and has dimension (m, n)
                // Where m: number of nodes of the previous layer, n: number of samples

                // Deltas matrix contains all deltas of the current (output) layer of the entire batch and has dimension (n, p)
                // Where n: number of samples, p: number of nodes of the current layer
                if deltas.shape[1] != self.nodes || all_outputs.shape[0] != self.input.shape[0] || deltas.shape[0] != all_outputs.shape[1] {
                    panic!("If out_layer is True and deltas tensor is 2D, p should be equal to the number of nodes in the output layer, n should be equal to the number of samples in the batch and m should be equal to the number of nodes in the previous layer");
                };

                // Weights gradients are calculated as the dot product between the inputs of the layer and the deltas
                let mut weights_gradients = all_outputs.dot(&deltas)?;
                weights_gradients.data.par_mapv_inplace(|x| x / deltas.shape[0] as f64); // Each gradient is divided by the number of samples in the batch

                // Biases gradients are calculated as the mean of the deltas in the entire batch 
                let biases_gradients = Tensor {
                    dimension: 1,
                    shape: vec![self.nodes],
                    data: deltas.data.clone().mean_axis(Axis(0)).unwrap()
                };

                 Ok((weights_gradients, biases_gradients, deltas))
            }
        } else {

            // Selection of the activation function
            let f: Box<dyn ActivationFunction> = match self.activation.clone() {
                Activation::Relu => Box::new(Relu),
                Activation::Sigmoid => Box::new(Sigmoid),
                Activation::Softmax => Box::new(Softmax),
                Activation::Tanh => Box::new(Tanh),
            };

            if deltas.dimension == 1 {

                // Deltas vector contains all deltas of the next layer and has dimension p
                // Where p: number of nodes of the next layer
                
                // Next_weights matrix contains all weights of the next layer and has dimension (q, p)
                // Where q: number of nodes of the current layer, p: number of nodes of the next layer
                if deltas.shape[0] != next_weights.shape[1] || next_weights.shape[0] != self.nodes {
                    panic!("If out_layer is False and deltas tensor is 1D, the number of deltas should be equal to the number of nodes in the next layer and the number of rows of the next weights matrix should be equal to the number of nodes in the current layer");
                }

                // The total input of the layer is argument of the derivative of the activation function
                let inputs_derivative = f.derivative(self.input.dot(&self.weights)?.add(&self.biases)?);
                // The layer deltas are calculated as the dot product between the deltas of the next layer and the weights of the next layer, and then multiplied by the derivative of the activation function
                let layer_deltas = next_weights.dot(&deltas)?.multiplication(&inputs_derivative)?;

                // Biases gradients are equal to the output deltas
                let biases_gradients = layer_deltas.clone();

                // Inputs and deltas tensors have to be reshaped to 2D since 1D tensors multiplication returns a scalar
                let inputs = Tensor {
                    dimension: 2,
                    shape: vec![self.input.shape[0], 1],
                    data: self.input.data.clone(),
                };
                let out_deltas = Tensor {
                    dimension: 2,
                    shape: vec![1, layer_deltas.shape[0]],
                    data: layer_deltas.data.clone(),
                };

                // Weights gradients are calculated as the dot product between the inputs of the layer and the current (layer) deltas
                let weights_gradients = inputs.dot(&out_deltas)?;
                Ok((weights_gradients, biases_gradients, layer_deltas))
            } else  {

                if all_outputs.is_none() {
                    panic!("If deltas tensor is 2D, all_outputs must be defined");
                }
                let all_outputs = all_outputs.unwrap();

                // Deltas matrix contains all deltas of the next layer in the entire batch and has dimension (n, p)
                // Where p: number of nodes of the next layer, n: number of samples

                // Next_weights matrix contains all weights of the next layer and has dimension (q, p)
                // Where q: number of nodes of the current layer, p: number of nodes of the next layer

                // All_outputs matrix contains all outputs of the previous layer of the entire batch and has dimension (m, n)
                // Where m: number of nodes of the previous layer, n: number of samples
                if deltas.shape[1] != next_weights.shape[1] || all_outputs.shape[0] != self.input.shape[0] || deltas.shape[0] != all_outputs.shape[1] {
                    panic!("If out_layer is False and deltas tensor is 2D, p should be equal to the number of nodes in the next layer, n should be equal to the number of samples in the batch and m should be equal to the number of nodes in the previous layer");
                };

                // The total input of the layer is argument of the derivative of the activation function
                let inputs_derivative = f.derivative(self.input.dot(&self.weights)?.add(&self.biases)?);
                // The derivative of the activation function has to be reshaped to 2D since 1D tensors multiplication returns a scalar
                let inputs_derivative = Tensor {
                    dimension: 2,
                    shape: vec![deltas.shape[0], inputs_derivative.shape[0]],
                    data: inputs_derivative.data.clone(),
                };
                // The layer deltas are calculated as the dot product between the deltas of the next layer and the weights of the next layer, and then multiplied by the derivative of the activation function
                let layer_deltas = deltas.dot(&next_weights.transpose()?)?.multiplication(&inputs_derivative)?;

                // Weights gradients are calculated as the dot product between the inputs of the layer of the entire batch and the current (layer) deltas
                let mut weights_gradients = all_outputs.dot(&layer_deltas)?;
                // Each gradient is divided by the number of samples in the batch
                weights_gradients.data.par_mapv_inplace(|x| x / layer_deltas.shape[0] as f64);

                // Biases gradients are calculated as the mean of the deltas in the entire batch
                let biases_gradients = Tensor {
                    dimension: 1,
                    shape: vec![self.nodes],
                    data: layer_deltas.data.mean_axis(Axis(0)).unwrap()
                };

                Ok((weights_gradients, biases_gradients, layer_deltas))

            }
        }
    }

    fn __repr__(&self) -> String {
        format!("Layer (nodes: {})", self.nodes)
    }
}