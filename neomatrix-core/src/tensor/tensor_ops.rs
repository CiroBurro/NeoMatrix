use crate::errors::TensorError;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Zip};
use std::ops::{Add, Div, Mul, Sub};

impl Tensor {
    /// Computes the broadcast-compatible output shape for two arrays.
    /// Follows NumPy/ndarray broadcasting rules.
    /// Returns an error (with the given error variant) if the shapes are incompatible.
    fn broadcast_shape(
        a: &ArrayD<f32>,
        b: &ArrayD<f32>,
        err: TensorError,
    ) -> Result<Vec<usize>, TensorError> {
        // Try broadcasting b into a's shape, then vice-versa.
        // The result shape is whichever broadcast succeeds and has the larger ndim.
        let shape_a = a.broadcast(b.raw_dim()).map(|v| v.shape().to_vec());
        let shape_b = b.broadcast(a.raw_dim()).map(|v| v.shape().to_vec());

        match (shape_a, shape_b) {
            (_, Some(s)) => Ok(s), // a's shape is the "larger" one
            (Some(s), None) => Ok(s),
            (None, None) => Err(err),
        }
    }

    fn tensor_sum(&self, t: &Tensor) -> Result<Tensor, TensorError> {
        let out_shape = Self::broadcast_shape(
            &self.data,
            &t.data,
            TensorError::ShapesNotCompatibleForElementWiseAddition,
        )?;
        let mut result = ArrayD::<f32>::zeros(out_shape.as_slice());
        Zip::from(&mut result)
            .and_broadcast(&self.data)
            .and_broadcast(&t.data)
            .for_each(|r, &a, &b| *r = a + b);
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result,
        })
    }

    fn tensor_subtraction(&self, t: &Tensor) -> Result<Tensor, TensorError> {
        let out_shape = Self::broadcast_shape(
            &self.data,
            &t.data,
            TensorError::ShapesNotCompatibleForElementWiseSubtraction,
        )?;
        let mut result = ArrayD::<f32>::zeros(out_shape.as_slice());
        Zip::from(&mut result)
            .and_broadcast(&self.data)
            .and_broadcast(&t.data)
            .for_each(|r, &a, &b| *r = a - b);
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result,
        })
    }

    fn tensor_multiplication(&self, t: &Tensor) -> Result<Tensor, TensorError> {
        let out_shape = Self::broadcast_shape(
            &self.data,
            &t.data,
            TensorError::ShapesNotCompatibleForElementWiseMultiplication,
        )?;
        let mut result = ArrayD::<f32>::zeros(out_shape.as_slice());
        Zip::from(&mut result)
            .and_broadcast(&self.data)
            .and_broadcast(&t.data)
            .for_each(|r, &a, &b| *r = a * b);
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result,
        })
    }

    fn tensor_division(&self, t: &Tensor) -> Result<Tensor, TensorError> {
        if t.data.iter().any(|&x| x == 0.0) {
            return Err(TensorError::CannotDivideByZeroDivisorTensorContainsZeroElements);
        }
        let out_shape = Self::broadcast_shape(
            &self.data,
            &t.data,
            TensorError::ShapesNotCompatibleForElementWiseDivision,
        )?;
        let mut result = ArrayD::<f32>::zeros(out_shape.as_slice());
        Zip::from(&mut result)
            .and_broadcast(&self.data)
            .and_broadcast(&t.data)
            .for_each(|r, &a, &b| *r = a / b);
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result,
        })
    }

    /// Internal method that implements addition of a scalar to all elements.
    /// This method is not meant to be called directly but is used to implement
    /// the `+` operator between a tensor and a scalar.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to add to each element
    ///
    /// # Returns
    /// * `Tensor` - New tensor with scalar added to all elements
    fn scalar_sum(&self, scalar: f32) -> Tensor {
        let result = &self.data + scalar;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }
    /// Internal method that implements subtraction of a scalar from all elements.
    /// This method is not meant to be called directly but is used to implement
    /// the `-` operator between a tensor and a scalar.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract from each element
    ///
    /// # Returns
    /// * `Tensor` - New tensor with scalar subtracted from all elements
    fn scalar_subtraction(&self, scalar: f32) -> Tensor {
        let result = &self.data - scalar;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }
    /// Internal method that implements multiplication of all elements by a scalar.
    /// This method is not meant to be called directly but is used to implement
    /// the `*` operator between a tensor and a scalar.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to multiply each element by
    ///
    /// # Returns
    /// * `Tensor` - New tensor with all elements multiplied by scalar
    fn scalar_multiplication(&self, scalar: f32) -> Tensor {
        let result = &self.data * scalar;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }
    /// Internal method that implements division of all elements by a scalar.
    /// This method is not meant to be called directly but is used to implement
    /// the `/` operator between a tensor and a scalar.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide each element by
    ///
    /// # Returns
    /// * `Tensor` - New tensor with all elements divided by scalar
    fn scalar_division(&self, scalar: f32) -> Result<Tensor, TensorError> {
        if scalar == 0.0 {
            return Err(TensorError::CannotDivideByZero);
        }

        let result = &self.data / scalar;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}

/// Add trait implementation for Tensor struct
impl Add<Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, rhs: Tensor) -> Self::Output {
        self.tensor_sum(&rhs)
    }
}
impl Add<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, rhs: &Tensor) -> Self::Output {
        self.tensor_sum(rhs)
    }
}
impl Add<&Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, rhs: &Tensor) -> Self::Output {
        self.tensor_sum(rhs)
    }
}
impl Add<Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, rhs: Tensor) -> Self::Output {
        self.tensor_sum(&rhs)
    }
}
impl Add<&mut Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, rhs: &mut Tensor) -> Self::Output {
        self.tensor_sum(rhs)
    }
}
impl Add<&mut Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, rhs: &mut Tensor) -> Self::Output {
        self.tensor_sum(rhs)
    }
}
impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        self.scalar_sum(rhs)
    }
}
impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        self.scalar_sum(rhs)
    }
}

impl Add<Tensor> for f32 {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        rhs.scalar_sum(self)
    }
}
impl Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        rhs.scalar_sum(self)
    }
}
impl Add<&mut Tensor> for f32 {
    type Output = Tensor;

    fn add(self, rhs: &mut Tensor) -> Self::Output {
        rhs.scalar_sum(self)
    }
}

/// Sub trait implementation for Tensor struct
impl Sub<Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, rhs: Tensor) -> Self::Output {
        self.tensor_subtraction(&rhs)
    }
}
impl Sub<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        self.tensor_subtraction(rhs)
    }
}
impl Sub<&Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        self.tensor_subtraction(rhs)
    }
}
impl Sub<Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, rhs: Tensor) -> Self::Output {
        self.tensor_subtraction(&rhs)
    }
}
impl Sub<&mut Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, rhs: &mut Tensor) -> Self::Output {
        self.tensor_subtraction(rhs)
    }
}
impl Sub<&mut Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, rhs: &mut Tensor) -> Self::Output {
        self.tensor_subtraction(rhs)
    }
}
impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self.scalar_subtraction(rhs)
    }
}
impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self.scalar_subtraction(rhs)
    }
}

impl Sub<Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let data = 1.0 - rhs.data;
        Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        }
    }
}
impl Sub<&Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let data = 1.0 - &rhs.data;
        Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        }
    }
}
impl Sub<&mut Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, rhs: &mut Tensor) -> Self::Output {
        let data = 1.0 - &rhs.data;
        Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        }
    }
}

/// Mul trait implementation for Tensor struct
impl Mul<Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, rhs: Tensor) -> Self::Output {
        self.tensor_multiplication(&rhs)
    }
}
impl Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        self.tensor_multiplication(rhs)
    }
}
impl Mul<&Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        self.tensor_multiplication(rhs)
    }
}
impl Mul<Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, rhs: Tensor) -> Self::Output {
        self.tensor_multiplication(&rhs)
    }
}
impl Mul<&mut Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, rhs: &mut Tensor) -> Self::Output {
        self.tensor_multiplication(rhs)
    }
}
impl Mul<&mut Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, rhs: &mut Tensor) -> Self::Output {
        self.tensor_multiplication(rhs)
    }
}
impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scalar_multiplication(rhs)
    }
}
impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scalar_multiplication(rhs)
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        rhs.scalar_multiplication(self)
    }
}
impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        rhs.scalar_multiplication(self)
    }
}
impl Mul<&mut Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: &mut Tensor) -> Self::Output {
        rhs.scalar_multiplication(self)
    }
}

/// Div Trait implementation for Tensor struct
impl Div<Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: Tensor) -> Self::Output {
        self.tensor_division(&rhs)
    }
}
impl Div<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: &Tensor) -> Self::Output {
        self.tensor_division(rhs)
    }
}
impl Div<&Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: &Tensor) -> Self::Output {
        self.tensor_division(rhs)
    }
}
impl Div<Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: Tensor) -> Self::Output {
        self.tensor_division(&rhs)
    }
}
impl Div<&mut Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: &mut Tensor) -> Self::Output {
        self.tensor_division(rhs)
    }
}
impl Div<&mut Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: &mut Tensor) -> Self::Output {
        self.tensor_division(rhs)
    }
}
impl Div<f32> for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: f32) -> Self::Output {
        self.scalar_division(rhs)
    }
}
impl Div<f32> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: f32) -> Self::Output {
        self.scalar_division(rhs)
    }
}

impl Div<Tensor> for f32 {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: Tensor) -> Self::Output {
        rhs.scalar_division(self)
    }
}
impl Div<&Tensor> for f32 {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: &Tensor) -> Self::Output {
        rhs.scalar_division(self)
    }
}
impl Div<&mut Tensor> for f32 {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: &mut Tensor) -> Self::Output {
        rhs.scalar_division(self)
    }
}
