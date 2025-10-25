use crate::structures::tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div};
use pyo3::PyResult;

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

