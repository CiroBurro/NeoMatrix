//! Multi-dimensional array structure and core tensor operations.
//!
//! This module provides the `Tensor` struct, which wraps `ndarray::ArrayD<f32>` and serves as
//! the primary data structure for numerical computing in NeoMatrix. It supports creation,
//! manipulation, and mathematical operations on tensors of arbitrary dimensions.

use crate::errors::TensorError;
use crate::math::matmul::par_dot;
use ndarray::prelude::*;
use ndarray::{Ix1, Ix2};
use std::ops::Range;

use rand;

/// Multi-dimensional array for numerical computing.
///
/// A `Tensor` wraps an `ndarray::ArrayD<f32>` and maintains metadata about its shape and dimensionality.
/// Supports operations including creation, reshaping, transposition, concatenation, and element-wise arithmetic.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Number of dimensions in this tensor.
    pub dimension: usize,
    /// Shape vector describing the size along each dimension.
    pub shape: Vec<usize>,
    /// The underlying multi-dimensional array storing all data elements as f32 values.
    pub data: ArrayD<f32>,
}

/// `Tensor` struct methods

impl Tensor {
    /// Creates a new tensor from a shape and content vector.
    ///
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    /// * `content` - A vector of f32 representing the data elements
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - New tensor, or error if shape and content size mismatch
    ///
    /// # Example
    /// ```ignore
    /// let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// ```
    pub fn new(shape: Vec<usize>, content: Vec<f32>) -> Result<Tensor, TensorError> {
        let dimension = shape.len();
        let data = match Array::from_shape_vec(shape.clone(), content) {
            Ok(array) => array,
            Err(_) => return Err(TensorError::ShapeAndContentMismatch),
        };
        Ok(Tensor {
            dimension,
            shape,
            data,
        })
    }

    /// Creates a tensor with all elements initialized to zero.
    ///
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    ///
    /// # Returns
    /// * `Tensor` - A tensor with all elements set to 0.0
    ///
    /// # Example
    /// ```ignore
    /// let tensor = Tensor::zeros(vec![2, 3]);  // 2×3 matrix of zeros
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let dimension = shape.len();
        let data = Array::zeros(shape.clone());
        Self {
            dimension,
            shape,
            data,
        }
    }

    /// Creates a tensor with random elements within a specified range.
    ///
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    /// * `rg` - A `Range<f32>` specifying the range of random values (e.g., `-1.0..1.0`)
    ///
    /// # Returns
    /// * `Tensor` - A tensor with random elements distributed within the given range
    ///
    /// # Example
    /// ```ignore
    /// let tensor = Tensor::random(vec![2, 3], -1.0..1.0);  // Random 2×3 matrix
    /// ```
    pub fn random(shape: Vec<usize>, rg: Range<f32>) -> Self {
        let mut tensor = Tensor::zeros(shape);

        tensor
            .data
            .par_mapv_inplace(|_| rand::random_range(rg.clone()));

        tensor
    }

    /// Computes the dot product with another tensor.
    ///
    /// Supports dot products for 1D and 2D tensors:
    /// - 1D · 1D = scalar (returned as 0-dimensional tensor)
    /// - 1D · 2D = 1D
    /// - 2D · 1D = 1D
    /// - 2D · 2D = 2D (uses parallel computation when applicable)
    ///
    /// # Arguments
    /// * `t` - The tensor to multiply with
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Result tensor, or error if dimensions are incompatible
    ///
    /// # Errors
    /// Returns `TensorError::IncompatibleDimensionsForDotProduct` if inner dimensions don't match
    pub fn dot(&self, t: &Tensor) -> Result<Tensor, TensorError> {
        match (self.dimension, t.dimension) {
            (1, 1) => {
                if self.shape[0] != t.shape[0] {
                    return Err(TensorError::IncompatibleDimensionsForDotProduct(format!(
                        "Incompatible dimensions for dot product: [{}] · [{}]",
                        self.shape[0], t.shape[0]
                    )));
                }

                let result = self
                    .data
                    .view()
                    .into_dimensionality::<Ix1>()
                    .map_err(|_| TensorError::InvalidDimension)?
                    .dot(
                        &t.data
                            .view()
                            .into_dimensionality::<Ix1>()
                            .map_err(|_| TensorError::InvalidDimension)?,
                    );
                Ok(Tensor {
                    dimension: 0,
                    shape: Vec::<usize>::new(),
                    data: ArrayD::from_elem(vec![], result),
                })
            }
            (1, 2) => {
                if self.shape[0] != t.shape[0] {
                    return Err(TensorError::IncompatibleDimensionsForDotProduct(format!(
                        "Incompatible dimensions for dot product: [{}] · [{}, {}]",
                        self.shape[0], t.shape[0], t.shape[1]
                    )));
                }

                let result = self
                    .data
                    .view()
                    .into_dimensionality::<Ix1>()
                    .map_err(|_| TensorError::InvalidDimension)?
                    .dot(
                        &t.data
                            .view()
                            .into_dimensionality::<Ix2>()
                            .map_err(|_| TensorError::InvalidDimension)?,
                    );
                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn(),
                })
            }
            (2, 1) => {
                if self.shape[1] != t.shape[0] {
                    return Err(TensorError::IncompatibleDimensionsForDotProduct(format!(
                        "Incompatible dimensions for dot product: [{}, {}] · [{}]",
                        self.shape[0], self.shape[1], t.shape[0]
                    )));
                }

                let result = self
                    .data
                    .view()
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| TensorError::InvalidDimension)?
                    .dot(
                        &t.data
                            .view()
                            .into_dimensionality::<Ix1>()
                            .map_err(|_| TensorError::InvalidDimension)?,
                    );
                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn(),
                })
            }
            (2, 2) => {
                if self.shape[1] != t.shape[0] {
                    return Err(TensorError::IncompatibleDimensionsForDotProduct(format!(
                        "Incompatible dimensions for dot product: [{}, {}] · [{}, {}]",
                        self.shape[0], self.shape[1], t.shape[0], t.shape[1]
                    )));
                }

                let result = par_dot(
                    self.data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .map_err(|_| TensorError::InvalidDimension)?,
                    t.data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .map_err(|_| TensorError::InvalidDimension)?,
                );

                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn(),
                })
            }

            _ => Err(TensorError::TensorContractionNotYetDefined),
        }
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// # Returns
    /// * `usize` - The total count of elements (product of all shape dimensions)
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Transposes a 2D tensor (swaps rows and columns).
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Transposed tensor, or error if tensor is not 2D
    ///
    /// # Errors
    /// Returns `TensorError::TransposeOnly2D` if dimension != 2
    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        // Check if the shapes are compatible for addition
        if self.dimension != 2 {
            return Err(TensorError::TransposeOnly2D);
        }
        let result = self.data.clone().reversed_axes();
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }

    /// Transposes a 2D tensor in-place (swaps rows and columns).
    ///
    /// # Returns
    /// * `Result<(), TensorError>` - Success, or error if tensor is not 2D
    ///
    /// # Errors
    /// Returns `TensorError::TransposeOnly2D` if dimension != 2
    pub fn transpose_inplace(&mut self) -> Result<(), TensorError> {
        // Check if the shapes are compatible for addition
        if self.dimension != 2 {
            return Err(TensorError::TransposeOnly2D);
        }

        self.data = self.data.to_owned().reversed_axes();
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();
        Ok(())
    }

    /// Reshapes the tensor to a new shape without modifying the original.
    ///
    /// Reorganizes elements into a new shape; total element count must remain the same.
    ///
    /// # Arguments
    /// * `shape` - New shape as a vector of usize
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Reshaped tensor, or error if shapes are incompatible
    ///
    /// # Errors
    /// Returns `TensorError::IncompatibleShape` if the new shape has different total size
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, TensorError> {
        let dim = shape.len();
        let data = self
            .data
            .to_owned()
            .into_shape_with_order(shape.as_slice())
            .map_err(|_| TensorError::IncompatibleShape)?;

        Ok(Tensor {
            shape: shape,
            dimension: dim,
            data: data,
        })
    }

    /// Reshapes the tensor in-place to a new shape.
    ///
    /// # Arguments
    /// * `shape` - New shape as a vector of usize
    ///
    /// # Returns
    /// * `Result<(), TensorError>` - Success, or error if shapes are incompatible
    ///
    /// # Errors
    /// Returns `TensorError::IncompatibleShape` if the new shape has different total size
    pub fn reshape_inplace(&mut self, shape: Vec<usize>) -> Result<(), TensorError> {
        self.shape = shape.clone();
        self.dimension = shape.len();
        self.data = self
            .data
            .to_owned()
            .into_shape_with_order(shape)
            .map_err(|_| TensorError::IncompatibleShape)?;
        Ok(())
    }

    /// Flattens the tensor to a 1D shape without modifying the original.
    ///
    /// # Returns
    /// * `Tensor` - Flattened 1D tensor with all elements in row-major order
    pub fn flatten(&self) -> Tensor {
        let flattened_data = self.data.flatten();

        Tensor {
            shape: flattened_data.shape().to_vec(),
            dimension: flattened_data.ndim(),
            data: flattened_data.to_owned().into_dyn(),
        }
    }

    /// Flattens the tensor to 1D shape in-place.
    ///
    /// Mutates self to become a 1D tensor with all elements in row-major order.
    pub fn flatten_inplace(&mut self) {
        let flattened_data = self.data.flatten();
        self.shape = flattened_data.shape().to_vec();
        self.dimension = flattened_data.ndim();
        self.data = flattened_data.to_owned().into_dyn();
    }

    /// Appends another tensor along a specified axis (mutating operation).
    ///
    /// Concatenates `t` to `self` along the given axis. Self is modified in-place.
    ///
    /// # Arguments
    /// * `t` - Tensor to append
    /// * `axis` - Axis along which to concatenate (0-indexed)
    ///
    /// # Returns
    /// * `Result<(), TensorError>` - Success, or error if dimensions are incompatible
    ///
    /// # Errors
    /// Returns `TensorError::IncompatibleDimensionsForPushing` if shapes don't match
    pub fn push(&mut self, t: &Tensor, axis: usize) -> Result<(), TensorError> {
        let mut vec_data = Vec::new();
        self.data.flatten().for_each(|x| vec_data.push(*x));
        t.data.flatten().for_each(|x| vec_data.push(*x));

        let mut shape = self.data.shape().to_vec();
        shape[axis] += t.data.len_of(Axis(axis));

        let new_data = ArrayD::from_shape_vec(IxDyn(&shape), vec_data)
            .map_err(|_| TensorError::IncompatibleDimensionsForPushing)?;

        self.dimension = new_data.ndim();
        self.shape = new_data.shape().to_vec();
        self.data = new_data;

        Ok(())
    }

    /// Concatenates multiple tensors along a specified axis, without modifying the original.
    ///
    /// # Arguments
    /// * `tensors` - Vector of tensors to concatenate with self
    /// * `axis` - Axis along which to concatenate (0-indexed)
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Concatenated tensor, or error if dimensions are incompatible
    ///
    /// # Errors
    /// Returns `TensorError::IncompatibleDimensionsForPushing` if shapes don't match
    pub fn cat_inplace(&self, tensors: Vec<Tensor>, axis: usize) -> Result<Tensor, TensorError> {
        let mut new_tensor = self.clone();
        for t in tensors.iter() {
            new_tensor.push(t, axis)?;
        }

        Ok(new_tensor)
    }

    /// Concatenates multiple tensors along a specified axis (static method).
    ///
    /// # Arguments
    /// * `tensors` - Vector of tensors to concatenate
    /// * `axis` - Axis along which to concatenate (0-indexed)
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Concatenated tensor, or error if dimensions are incompatible
    ///
    /// # Errors
    /// Returns `TensorError::IncompatibleDimensionsForPushing` if shapes don't match
    pub fn cat(tensors: Vec<Tensor>, axis: usize) -> Result<Tensor, TensorError> {
        let mut new_tensor = tensors[0].clone();
        for t in &tensors[1..] {
            new_tensor.push(t, axis)?;
        }

        Ok(new_tensor)
    }
    /// Appends a 1D tensor as a new row to the end (in-place).
    ///
    /// Mutates self to add `t` as a new row. Self must be 1D or 2D; if 1D, it's first reshaped to 2D.
    ///
    /// # Arguments
    /// * `t` - 1D tensor representing the row to add
    ///
    /// # Returns
    /// * `Result<(), TensorError>` - Success, or error if `t` is not 1D or self is > 2D
    ///
    /// # Errors
    /// - `TensorError::RowMustBe1D` if `t` has dimension != 1
    /// - `TensorError::PushingRowOnly1D2D` if self has dimension > 2
    pub fn push_row(&mut self, t: &Tensor) -> Result<(), TensorError> {
        if t.dimension != 1 {
            return Err(TensorError::RowMustBe1D);
        }

        if self.dimension > 2 {
            return Err(TensorError::PushingRowOnly1D2D);
        }

        if self.dimension == 1 {
            self.dimension = 2;
            self.shape = vec![1, self.shape[0]];
            self.data = self
                .data
                .to_shape(self.shape.clone())
                .map_err(|_| TensorError::InvalidDimension)?
                .to_owned();
        }

        self.data
            .push(Axis(0), t.data.view())
            .map_err(|_| TensorError::InvalidDimension)?;
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();

        Ok(())
    }

    /// Appends a 1D tensor as a new column to the end (in-place).
    ///
    /// Mutates self to add `t` as a new column. Self must be 1D or 2D; if 1D, it's first reshaped to 2D.
    ///
    /// # Arguments
    /// * `t` - 1D tensor representing the column to add
    ///
    /// # Returns
    /// * `Result<(), TensorError>` - Success, or error if `t` is not 1D or self is > 2D
    ///
    /// # Errors
    /// - `TensorError::ColumnMustBe1D` if `t` has dimension != 1
    /// - `TensorError::PushingColumnOnly1D2D` if self has dimension > 2
    pub fn push_column(&mut self, t: &Tensor) -> Result<(), TensorError> {
        if t.dimension != 1 {
            return Err(TensorError::ColumnMustBe1D);
        }

        if self.dimension > 2 {
            return Err(TensorError::PushingColumnOnly1D2D);
        }

        if self.dimension == 1 {
            self.dimension = 2;
            self.shape = vec![self.shape[0], 1];
            self.data = self
                .data
                .to_shape(self.shape.clone())
                .map_err(|_| TensorError::InvalidDimension)?
                .to_owned();
        }

        self.data
            .push(Axis(1), t.data.view())
            .map_err(|_| TensorError::InvalidDimension)?;
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();

        Ok(())
    }
}
