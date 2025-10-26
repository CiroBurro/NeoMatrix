use crate::structures::tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div};
use pyo3::{PyErr, PyResult};


impl Tensor {
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
	
	fn scalar_sum(&self, scalar: f64) -> Tensor {
		let result = &self.data + scalar;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
 	fn scalar_subtraction(&self, scalar: f64) -> Tensor {
		let result = &self.data - scalar;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
	fn scalar_multiplication(&self, scalar: f64) -> Tensor {
		let result = &self.data * scalar;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
	fn scalar_division(&self, scalar: f64) -> Tensor {
		let result = &self.data / scalar;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
}


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

