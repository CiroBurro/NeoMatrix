use crate::structures::tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div};
use pyo3::{PyErr, PyResult};


impl Tensor {
    /// Internal method that implements element-wise addition between tensors.
    /// This method is not meant to be called directly but is used to implement
    /// the `+` operator for tensors.
    ///
    /// # Arguments
    /// * `t` - Reference to the tensor to add
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - New tensor containing element-wise sum or error if shapes mismatch
	fn tensor_sum(&self, t: &Tensor) -> PyResult<Tensor> {
		// Check if the shapes are compatible for addition
		if self.shape != t.shape {
			return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
				"Tensor shapes are not compatible for element-wise addition"
			));
		}
		let result = &self.data + &t.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
    /// Internal method that implements element-wise subtraction between tensors.
    /// This method is not meant to be called directly but is used to implement
    /// the `-` operator for tensors.
    ///
    /// # Arguments
    /// * `t` - Reference to the tensor to subtract
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - New tensor containing element-wise difference or error if shapes mismatch
	fn tensor_subtraction(&self, t: &Tensor) -> PyResult<Tensor> {
		// Check if the shapes are compatible for addition
		if self.shape != t.shape {
			return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
				"Tensor shapes are not compatible for element-wise subtraction"
			));
		}
		let result = &self.data - &t.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
    /// Internal method that implements element-wise multiplication between tensors.
    /// This method is not meant to be called directly but is used to implement
    /// the `*` operator for tensors.
    ///
    /// # Arguments
    /// * `t` - Reference to the tensor to multiply
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - New tensor containing element-wise product or error if shapes mismatch
	fn tensor_multiplication(&self, t: &Tensor) -> PyResult<Tensor> {
		// Check if the shapes are compatible for addition
		if self.shape != t.shape {
			return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
				"Tensor shapes are not compatible for element-wise multiplication"
			));
		}
		let result = &self.data * &t.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
    /// Internal method that implements element-wise division between tensors.
    /// This method is not meant to be called directly but is used to implement
    /// the `/` operator for tensors.
    ///
    /// # Arguments
    /// * `t` - Reference to the tensor to divide by
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - New tensor containing element-wise division or error if shapes mismatch
	fn tensor_division(&self, t: &Tensor) -> PyResult<Tensor> {
		// Check if the shapes are compatible for addition
		if self.shape != t.shape {
			return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
				"Tensor shapes are not compatible for element-wise division"
			));
		}
		let result = &self.data / &t.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
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
	fn scalar_sum(&self, scalar: f64) -> Tensor {
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
 	fn scalar_subtraction(&self, scalar: f64) -> Tensor {
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
	fn scalar_multiplication(&self, scalar: f64) -> Tensor {
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
	fn scalar_division(&self, scalar: f64) -> Tensor {
		let result = &self.data / scalar;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
}


/// Add trait implementation for Tensor struct
impl Add<Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn add(self, rhs: Tensor) -> Self::Output {
		self.tensor_sum(&rhs)
	}
}
impl Add<&Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn add(self, rhs: &Tensor) -> Self::Output {
		self.tensor_sum(rhs)
	}
}
impl Add<&Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn add(self, rhs: &Tensor) -> Self::Output {
		self.tensor_sum(rhs)
	}
}
impl Add<Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn add(self, rhs: Tensor) -> Self::Output {
		self.tensor_sum(&rhs)
	}
}
impl Add<&mut Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn add(self, rhs: &mut Tensor) -> Self::Output {
		self.tensor_sum(rhs)
	}
}
impl Add<&mut Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn add(self, rhs: &mut Tensor) -> Self::Output {
		self.tensor_sum(rhs)
	}
}
impl Add<f64> for Tensor {
	type Output = Tensor;

	fn add(self, rhs: f64) -> Self::Output {
		self.scalar_sum(rhs)
	}
}
impl Add<f64> for &Tensor {
	type Output = Tensor;

	fn add(self, rhs: f64) -> Self::Output {
		self.scalar_sum(rhs)
	}
}

/// Sub trait implementation for Tensor struct
impl Sub<Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn sub(self, rhs: Tensor) -> Self::Output {
		self.tensor_subtraction(&rhs)
	}
}
impl Sub<&Tensor> for &Tensor {
    type Output = PyResult<Tensor>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
		self.tensor_subtraction(rhs)
    }
}
impl Sub<&Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn sub(self, rhs: &Tensor) -> Self::Output {
		self.tensor_subtraction(rhs)
	}
}
impl Sub<Tensor> for &Tensor {
    type Output = PyResult<Tensor>;

    fn sub(self, rhs: Tensor) -> Self::Output {
		self.tensor_subtraction(&rhs)
    }
}
impl Sub<&mut Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn sub(self, rhs: &mut Tensor) -> Self::Output {
		self.tensor_subtraction(rhs)
	}
}
impl Sub<&mut Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn sub(self, rhs: &mut Tensor) -> Self::Output {
		self.tensor_subtraction(rhs)
	}
}
impl Sub<f64> for Tensor {
	type Output = Tensor;

	fn sub(self, rhs: f64) -> Self::Output {
		self.scalar_subtraction(rhs)
	}
}
impl Sub<f64> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
		self.scalar_subtraction(rhs)
    }
}

/// Mul trait implementation for Tensor struct
impl Mul<Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn mul(self, rhs: Tensor) -> Self::Output {
		self.tensor_multiplication(&rhs)
	}
}
impl Mul<&Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn mul(self, rhs: &Tensor) -> Self::Output {
		self.tensor_multiplication(rhs)
	}
}
impl Mul<&Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn mul(self, rhs: &Tensor) -> Self::Output {
		self.tensor_multiplication(rhs)
	}
}
impl Mul<Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn mul(self, rhs: Tensor) -> Self::Output {
		self.tensor_multiplication(&rhs)
	}
}
impl Mul<&mut Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn mul(self, rhs: &mut Tensor) -> Self::Output {
		self.tensor_multiplication(rhs)
	}
}
impl Mul<&mut Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn mul(self, rhs: &mut Tensor) -> Self::Output {
		self.tensor_multiplication(rhs)
	}
}
impl Mul<f64> for Tensor {
	type Output = Tensor;

	fn mul(self, rhs: f64) -> Self::Output {
		self.scalar_multiplication(rhs)
	}
}
impl Mul<f64> for &Tensor {
	type Output = Tensor;

	fn mul(self, rhs: f64) -> Self::Output {
		self.scalar_multiplication(rhs)
	}
}

/// Div Trait implementation for Tensor struct
impl Div<Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn div(self, rhs: Tensor) -> Self::Output {
		self.tensor_division(&rhs)
	}
}
impl Div<&Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn div(self, rhs: &Tensor) -> Self::Output {
		self.tensor_division(rhs)
	}
}
impl Div<&Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn div(self, rhs: &Tensor) -> Self::Output {
		self.tensor_division(rhs)
	}
}
impl Div<Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn div(self, rhs: Tensor) -> Self::Output {
		self.tensor_division(&rhs)
	}
}
impl Div<&mut Tensor> for Tensor {
	type Output = PyResult<Tensor>;

	fn div(self, rhs: &mut Tensor) -> Self::Output {
		self.tensor_division(rhs)
	}
}
impl Div<&mut Tensor> for &Tensor {
	type Output = PyResult<Tensor>;

	fn div(self, rhs: &mut Tensor) -> Self::Output {
		self.tensor_division(rhs)
	}
}
impl Div<f64> for Tensor {
	type Output = Tensor;

	fn div(self, rhs: f64) -> Self::Output {
		self.scalar_division(rhs)
	}
}
impl Div<f64> for &Tensor {
	type Output = Tensor;

	fn div(self, rhs: f64) -> Self::Output {
		self.scalar_division(rhs)
	}
}
