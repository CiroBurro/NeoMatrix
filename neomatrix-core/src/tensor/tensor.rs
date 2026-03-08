use crate::errors::TensorError;
use crate::math::matmul::par_dot;
use ndarray::prelude::*;
use ndarray::{Ix1, Ix2};
use std::ops::Range;

use rand;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dimension: usize,
    pub shape: Vec<usize>,
    pub data: ArrayD<f32>,
}

/// `Tensor` struct methods

impl Tensor {
    /// Constructor method for the Tensor
    ///
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    /// * `content` - A vector of f32 representing the content of the tensor
    ///
    /// # Returns
    /// * `Tensor` - New tensor
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

    /// Constructor method for an empty tensor
    ///
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    ///
    /// # Returns
    /// * `Tensor` - An empty tensor
    pub fn zeros(shape: Vec<usize>) -> Self {
        let dimension = shape.len();
        let data = Array::zeros(shape.clone());
        Self {
            dimension,
            shape,
            data,
        }
    }

    /// Constructor method for a random tensor
    ///
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    ///
    /// # Returns
    /// * `Tensor` - A tensor with random data
    pub fn random(shape: Vec<usize>, rg: Range<f32>) -> Self {
        let mut tensor = Tensor::zeros(shape);

        tensor
            .data
            .par_mapv_inplace(|_| rand::random_range(rg.clone()));

        tensor
    }

    /// Dot product method for 1D and 2D tensors
    ///
    /// # Arguments
    /// * `t` - The tensor to be multiplied with
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Result of the dot product (tensor)
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

    /// Length method for tensor
    ///
    /// # Returns
    /// * `usize` - Length of the tensor
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Transpose method for 2D tensors
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Transposed tensor
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

    /// Transpose inplace method for 2D tensors
    pub fn transpose_inplace(&mut self) -> Result<(), TensorError> {
        // Check if the shapes are compatible for addition
        if self.dimension != 2 {
            return Err(TensorError::TransposeOnly2D);
        }

        self.data.to_owned().reversed_axes();
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();
        Ok(())
    }

    /// Reshape method for Tensor
    ///
    /// # Arguments
    /// * `shape` - Vector representing the new shape
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

    /// Reshape inplace method for Tensor
    ///
    /// # Arguments
    /// * `shape` - Vector representing the new shape
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

    /// Flatten method for Tensor
    pub fn flatten(&self) -> Tensor {
        let flattened_data = self.data.flatten();

        Tensor {
            shape: flattened_data.shape().to_vec(),
            dimension: flattened_data.ndim(),
            data: flattened_data.to_owned().into_dyn(),
        }
    }

    /// Flatten inplace method for Tensor
    pub fn flatten_inplace(&mut self) {
        let flattened_data = self.data.flatten();
        self.shape = flattened_data.shape().to_vec();
        self.dimension = flattened_data.ndim();
        self.data = flattened_data.to_owned().into_dyn();
    }

    /// Push method for 2 tensors
    /// Pushes a tensor into self
    ///
    /// # Arguments
    /// * `t` - Tensor to be pushed
    /// * `axis` - Index of the axe along which tensors should be concatenated
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

    /// Concatenate inplace method for multiple tensors
    ///
    /// # Arguments
    /// * `tensors` - Vector of tensors to be concatenated to self
    /// * `axis` - Index of the axe along which tensors should be concatenated
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Tensor containing all the input tensors concatenated
    pub fn cat_inplace(&self, tensors: Vec<Tensor>, axis: usize) -> Result<Tensor, TensorError> {
        let mut new_tensor = self.clone();
        for t in tensors.iter() {
            new_tensor.push(t, axis)?;
        }

        Ok(new_tensor)
    }

    /// Concatenate method for multiple tensors
    ///
    /// # Arguments
    /// * `tensors` - Vector of tensors to be concatenated
    /// * `axis` - Index of the axe along which tensors should be concatenated
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>` - Tensor containing all the input tensors concatenated
    pub fn cat(tensors: Vec<Tensor>, axis: usize) -> Result<Tensor, TensorError> {
        let mut new_tensor = tensors[0].clone();
        for t in &tensors[1..] {
            new_tensor.push(t, axis)?;
        }

        Ok(new_tensor)
    }
    /// Push row method for Tensor
    ///
    /// # Arguments
    /// * `t` - Tensor representing the row to be added
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

    /// Push column method for Tensor
    ///
    /// # Arguments
    /// * `t` - Tensor representing the column to be added
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
