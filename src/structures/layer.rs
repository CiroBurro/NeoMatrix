/// This module defines the Layer class for a neural network
/// and provides methods for the forward and backward propagation in each layer.
/// It uses the Tensor struct for manipulating the data
/// Necessary imports
use ndarray::{s, Axis, Ix2};
use pyo3::prelude::*;
use crate::structures::tensor::Tensor;
use crate::utils::weights_biases::{random_weights, random_biases};
use crate::functions::activation::*;
use crate::functions::cost::{BinaryCrossEntropy, Cost, CostFunction, HingeLoss, HuberLoss, MeanAbsoluteError, MeanSquaredError};

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

/// Function to select the activation function struct based on the layer's activation field
/// ! Not a Python function !
fn select_activation(l: &Layer) -> Box<dyn ActivationFunction> {
    match l.activation.clone() {
        Activation::Relu => Box::new(Relu),
        Activation::Sigmoid => Box::new(Sigmoid),
        Activation::Softmax => Box::new(Softmax),
        Activation::Tanh => Box::new(Tanh),
    }
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
    pub fn new(nodes: usize, input_len: usize, activation: Activation) -> Self {
        let weights = random_weights(input_len, nodes, (-1.0, 1.0));
        let biases = random_biases(nodes, (-1.0, 1.0));
        let input_placeholder = Tensor::zeros(vec![input_len]);
        let output_placeholder = Tensor::zeros(vec![nodes]);

        Self {
            nodes,
            input: input_placeholder,
            output: output_placeholder,
            weights,
            biases,
            activation,
        }
    }

    /// Forward propagation method
    /// 
    /// Parameters:
    /// - parallel: boolean value to specify if the forward propagation should be done in parallel
    ///
    /// Python usage:
    ///```python
    /// output = layer.forward(parallel=True)
    /// ```
    pub fn forward(&mut self, input: Tensor, parallel: bool) -> PyResult<Tensor> {
        self.input = input;
        
        // Check compatibility between dimensions
        if self.input.dimension != 1 || self.weights.dimension != 2 || self.biases.dimension != 1 {
            panic!("Layer field have wrong dimensions, expected: input (1D), weights (2D), biases (1D)");
        }
        
        // Input length must be equal to the number of rows of the weights matrix
        let p = self.input.shape[0];
        let (m, n) = self.weights.data.clone().into_dimensionality::<Ix2>().unwrap().dim();
        // If input length is equal to the number of columns of the weights matrix, transpose the weights matrix
        if m != p && p == n {
            self.weights = self.weights.transpose()?;
        }
        else if m != p && p != n {
            panic!("Inputs and weights have incompatible shapes for forward propagation:\
             the length of the input vector should be equal to the number of rows of the weights matrix");
        }

        // Selection of the activation function
        let f: Box<dyn ActivationFunction> = select_activation(self);

        // Forward prop algorithm
        if parallel {
            self.output = f.par_function(&mut self.input.dot(&self.weights).unwrap().tensor_sum(&self.biases)?);
        } else {
            self.output = f.function(&mut self.input.dot(&self.weights).unwrap().tensor_sum(&self.biases)?);
        }

        Ok(self.output.clone())
    }

    /// Backward propagation method
    ///
    /// Parameters:
    /// - out_layer: boolean value to specify if the layer is the output layer
    /// - deltas: a 1D or 2D tensor containing the deltas of the next layer (output layer if out_layer is True)
    /// - next_weights: a 2D tensor containing the weights of the next layer (None if out_layer is True)
    /// - all_outputs: a 2D tensor containing the outputs of the previous layer of the entire batch (None if deltas tensor is 1D)
    ///
    /// Python usage:
    ///```python
    /// (w_gradients, b_gradients, current_deltas) = layer.forward(out_layer=False, deltas=deltas, next_weights=next_weights, all_outputs=all_outputs)
    /// ```
    fn backward(&self, out_layer: bool, mut deltas: Tensor, next_weights: Option<Tensor>, all_outputs: Option<Tensor>) -> PyResult<(Tensor, Tensor, Tensor)> {
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
                    data: self.input.data.clone().to_shape(vec![self.input.shape[0], 1]).unwrap().into_owned(),
                };
                let out_deltas = Tensor {
                    dimension: 2,
                    shape: vec![1, deltas.shape[0]],
                    data: deltas.data.clone().to_shape(vec![1, deltas.shape[0]]).unwrap().into_owned(),
                };

                // Weights gradients are calculated as the dot product between the inputs of the layer and the deltas
                let weights_gradients = inputs.dot(&out_deltas)?;
                Ok((weights_gradients, biases_gradients, deltas))

            } else  {

                if all_outputs.is_none() {
                    panic!("If deltas tensor is 2D, all_outputs must be defined");
                }
                let mut all_outputs = all_outputs.unwrap();

                // All_outputs matrix contains all outputs of the previous layer of the entire batch and has dimension (m, n)
                // Where m: number of nodes of the previous layer, n: number of samples

                // Deltas matrix contains all deltas of the current (output) layer of the entire batch and has dimension (n, p)
                // Where n: number of samples, p: number of nodes of the current layer

                // If n is equal to the number of nodes of the current layer and p is equal to the number of samples, transpose the deltas matrix
                if deltas.shape[0] == self.nodes && deltas.shape[1] == all_outputs.shape[1] {
                    deltas = deltas.transpose()?;
                }

                // If m is equal to the number of samples and n is equal to the number of nodes of the previous layer, transpose the all_outputs matrix
                if all_outputs.shape[0] == deltas.shape[0] && all_outputs.shape[1] == self.input.shape[0] {
                    all_outputs = all_outputs.transpose()?;
                }

                if deltas.shape[1] != self.nodes || all_outputs.shape[0] != self.input.shape[0] || deltas.shape[0] != all_outputs.shape[1] {
                    panic!("If out_layer is True and deltas tensor is 2D:\
                     p should be equal to the number of nodes in the output layer,\
                     n should be equal to the number of samples in the batch,\
                     m should be equal to the number of nodes in the previous layer");
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

            if next_weights.is_none() {
                panic!("If out_layer is False, next_weights must be defined");
            }
            let mut next_weights = next_weights.unwrap();

            // Check weights tensor dimensions
            if next_weights.dimension != 2 {
                panic!("Next weights tensor has to be 2D for backpropagation");
            }

            // Selection of the activation function
            let f: Box<dyn ActivationFunction> = select_activation(self);

            if deltas.dimension == 1 {

                // Deltas vector contains all deltas of the next layer and has dimension p
                // Where p: number of nodes of the next layer
                
                // Next_weights matrix contains all weights of the next layer and has dimension (q, p)
                // Where q: number of nodes of the current layer, p: number of nodes of the next layer

                // If q is equal to the number of nodes of the next layer and p is equal to the number of nodes of the current layer, transpose the next_weights matrix
                if next_weights.shape[0] == deltas.shape[0] && next_weights.shape[1] == self.nodes {
                    next_weights = next_weights.transpose()?;
                }

                if deltas.shape[0] != next_weights.shape[1] || next_weights.shape[0] != self.nodes {
                    panic!("If out_layer is False and deltas tensor is 1D:\
                     the number of deltas should be equal to the number of nodes in the next layer,\
                     the number of rows of the next weights matrix should be equal to the number of nodes in the current layer");
                }

                // The total input of the layer is argument of the derivative of the activation function
                let inputs_derivative = f.derivative(&mut self.input.dot(&self.weights).unwrap().tensor_sum(&self.biases)?);
                // The layer deltas are calculated as the dot product between the deltas of the next layer and the weights of the next layer, and then multiplied by the derivative of the activation function
                let layer_deltas = next_weights.dot(&deltas)?.tensor_multiplication(&inputs_derivative)?;

                // Biases gradients are equal to the output deltas
                let biases_gradients = layer_deltas.clone();

                // Inputs and deltas tensors have to be reshaped to 2D since 1D tensors multiplication returns a scalar
                let inputs = Tensor {
                    dimension: 2,
                    shape: vec![self.input.shape[0], 1],
                    data: self.input.data.clone().to_shape(vec![self.input.shape[0], 1]).unwrap().into_owned(),
                };
                let out_deltas = Tensor {
                    dimension: 2,
                    shape: vec![1, layer_deltas.shape[0]],
                    data: layer_deltas.data.clone().to_shape(vec![1, layer_deltas.shape[0]]).unwrap().into_owned(),
                };

                // Weights gradients are calculated as the dot product between the inputs of the layer and the current (layer) deltas
                let weights_gradients = inputs.dot(&out_deltas)?;
                Ok((weights_gradients, biases_gradients, layer_deltas))
            } else  {

                if all_outputs.is_none() {
                    panic!("If deltas tensor is 2D, all_outputs must be defined");
                }
                let mut all_outputs = all_outputs.unwrap();

                // Deltas matrix contains all deltas of the next layer in the entire batch and has dimension (n, p)
                // Where p: number of nodes of the next layer, n: number of samples

                // Next_weights matrix contains all weights of the next layer and has dimension (q, p)
                // Where q: number of nodes of the current layer, p: number of nodes of the next layer

                // All_outputs matrix contains all outputs of the previous layer of the entire batch and has dimension (m, n)
                // Where m: number of nodes of the previous layer, n: number of samples

                // If n is equal to the number of nodes of the next layer and p is equal to the number of samples, transpose the deltas matrix
                if deltas.shape[0] == next_weights.shape[1] && deltas.shape[1] == all_outputs.shape[1] {
                    deltas = deltas.transpose()?;
                }

                // If q is equal to the number of nodes of the next layer and p is equal to the number of nodes of the current layer, transpose the next_weights matrix
                if next_weights.shape[0] == deltas.shape[1] && next_weights.shape[1] == self.nodes {
                    next_weights = next_weights.transpose()?;
                }

                // If m is equal to the number of samples and n is equal to the number of nodes of the previous layer, transpose the all_outputs matrix
                if all_outputs.shape[0] == deltas.shape[0] && all_outputs.shape[1] == self.input.shape[0] {
                    all_outputs = all_outputs.transpose()?;
                }

                if deltas.shape[1] != next_weights.shape[1] || all_outputs.shape[0] != self.input.shape[0] || deltas.shape[0] != all_outputs.shape[1] {
                    panic!("If out_layer is False and deltas tensor is 2D, p should be equal to the number of nodes in the next layer, n should be equal to the number of samples in the batch and m should be equal to the number of nodes in the previous layer");
                };

                // The total input of the layer is argument of the derivative of the activation function
                let inputs_derivative = f.derivative(&mut self.input.dot(&self.weights)?.tensor_sum(&self.biases)?);
                // The derivative of the activation function has to be reshaped to 2D since 1D tensors multiplication returns a scalar
                let inputs_derivative = Tensor {
                    dimension: 2,
                    shape: vec![deltas.shape[0], inputs_derivative.shape[0]],
                    data: inputs_derivative.data.clone(),
                };
                // The layer deltas are calculated as the dot product between the deltas of the next layer and the weights of the next layer, and then multiplied by the derivative of the activation function
                let layer_deltas = deltas.dot(&next_weights.transpose()?)?.tensor_multiplication(&inputs_derivative)?;

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

    /// Method to get the output deltas of the layer
    /// 
    /// Parameters:
    /// - cost: cost function structure used to calculate the error
    /// - t: a 1D or 2D tensor containing the output of the layer
    /// - z: a 1D or 2D tensor containing the expected output of the layer
    /// 
    /// Python usage:
    ///```python
    /// output_deltas = layer.get_output_deltas(Cost.MeanSquaredError, t, z)
    /// ```
    fn get_output_deltas(&self, cost: Cost, t: &mut Tensor, z: &Tensor) -> PyResult<Tensor> {

        // Optimization for the case of binary cross entropy with softmax activation
        if matches!(cost, Cost::BinaryCrossEntropy) && matches!(self.activation, Activation::Softmax) {
            return z.tensor_subtraction(t) // Deltas are just the difference
        }

        // Derivatives calculus
        let activation_derivative = select_activation(self).derivative(t);
        let cost_derivative = match cost {
            Cost::MeanSquaredError => MeanSquaredError.derivative(t, z),
            Cost::MeanAbsoluteError => MeanAbsoluteError.derivative(t, z),
            Cost::BinaryCrossEntropy => BinaryCrossEntropy.derivative(t, z),
            Cost::HuberLoss => HuberLoss.derivative(t, z),
            Cost::HingeLoss => HingeLoss.derivative(t, z),
        };

        // Softmax derivative of one sample returns a matrix, the result of an entire batch is a 3D tensor
        if matches!(self.activation, Activation::Softmax) && t.dimension == 1 {
            return activation_derivative.dot(&cost_derivative)
            
        } else if matches!(self.activation, Activation::Softmax) && t.dimension == 2 {
            let batch_size = t.shape[0];
            let mut all_deltas = Tensor::zeros(vec![batch_size, self.nodes]);
            
            // 3D dot product is not supported
            for i in 0..batch_size {
                let activation_derivative_i = activation_derivative.data.slice(s![i, .., ..]);
                let cost_derivative_i = cost_derivative.data.slice(s![i, ..]);
                let delta_i = activation_derivative_i.dot(&cost_derivative_i);
                all_deltas.data.slice_mut(s![i, ..]).assign(&delta_i);
            }
            return Ok(all_deltas)
        }

        // In all other combinations of cost and activation function deltas = cost function derivative * activation function derivative
        cost_derivative.tensor_multiplication(&activation_derivative)
    }
    
    fn __repr__(&self) -> String {
        format!("Layer (nodes: {})", self.nodes)
    }
}