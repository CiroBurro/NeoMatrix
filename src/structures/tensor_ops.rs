use crate::structures::tensor::{Tensor, TensorOrScalar};
use std::ops::{Add, Sub, Mul, Div};
use ndarray::{ShapeError, ErrorKind};

impl Add<TensorOrScalar> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn add(self, rhs: TensorOrScalar) -> Self::Output {
		match rhs {
			TensorOrScalar::Tensor(t) => {
				// Check if the shapes are compatible for addition
				if self.shape != t.shape {
					return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
				}
				let result = &self.data + &t.data;
				Ok(Tensor {
					dimension: result.ndim(),
					shape: result.shape().to_vec(),
					data: result.into_dyn(),
				})
			},
			TensorOrScalar::Scalar(scalar) => {

				let result = &self.data + scalar;
				Ok(Tensor {
					dimension: result.ndim(),
					shape: result.shape().to_vec(),
					data: result.into_dyn(),
				})
			},
		}
	}
}
impl Sub<TensorOrScalar> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn sub(self, rhs: TensorOrScalar) -> Self::Output {
		match rhs {
			TensorOrScalar::Tensor(t) => {

				// Check if the shapes are compatible for subtraction
				if self.shape != t.shape {
					return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
				}
				let result = &self.data - &t.data;
				Ok(Tensor {
					dimension: result.ndim(),
					shape: result.shape().to_vec(),
					data: result.into_dyn(),
				})
			},
			TensorOrScalar::Scalar(scalar) => {

				let result = &self.data - scalar;
				Ok(Tensor {
					dimension: result.ndim(),
					shape: result.shape().to_vec(),
					data: result.into_dyn(),
				})
			},
		}
	}
}
impl Mul<TensorOrScalar> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn mul(self, rhs: TensorOrScalar) -> Self::Output {
		match rhs {
			TensorOrScalar::Tensor(t) => {

				// Check if the shapes are compatible for multiplication
				if self.shape != t.shape {
					return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
				}
				let result = &self.data * &t.data;
				Ok(Tensor {
					dimension: result.ndim(),
					shape: result.shape().to_vec(),
					data: result.into_dyn(),
				})
			},
			TensorOrScalar::Scalar(scalar) => {
				let result = &self.data * scalar;
				Ok(Tensor {
					dimension: result.ndim(),
					shape: result.shape().to_vec(),
					data: result.into_dyn(),
				})
			},
		}
	}
}
impl Div<TensorOrScalar> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn div(self, rhs: TensorOrScalar) -> Self::Output {
		match rhs {
			TensorOrScalar::Tensor(t) => {

				// Check if the shapes are compatible for division
				if self.shape != t.shape {
					return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
				}
				let result = &self.data / &t.data;
				Ok(Tensor {
					dimension: result.ndim(),
					shape: result.shape().to_vec(),
					data: result.into_dyn(),
				})
			},
			TensorOrScalar::Scalar(scalar) => {

				let result = &self.data / scalar;
				Ok(Tensor {
					dimension: result.ndim(),
					shape: result.shape().to_vec(),
					data: result.into_dyn(),
				})
			},
		}
	}
}
