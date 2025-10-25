use crate::structures::tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div};
use ndarray::{ShapeError, ErrorKind};

impl Add<Tensor> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn add(self, rhs: Tensor) -> Self::Output {
		if self.shape != rhs.shape {
			return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
		}
		let result = &self.data + rhs.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
}
impl Add<&Tensor> for &Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn add(self, rhs: &Tensor) -> Self::Output {
		if self.shape != rhs.shape {
			return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
		}
		let result = &self.data + &rhs.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
}
impl Add<&Tensor> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn add(self, rhs: &Tensor) -> Self::Output {
		if self.shape != rhs.shape {
			return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
		}
		let result = &self.data + &rhs.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
}
impl Add<Tensor> for &Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn add(self, rhs: Tensor) -> Self::Output {
		if self.shape != rhs.shape {
			return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
		}
		let result = &self.data + rhs.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
}
impl Add<f64> for Tensor {
	type Output = Tensor;

	fn add(self, rhs: f64) -> Self::Output {
		let result = &self.data + rhs;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
}
impl Add<f64> for &Tensor {
	type Output = Tensor;

	fn add(self, rhs: f64) -> Self::Output {
		let result = &self.data + rhs;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
}

impl Sub<Tensor> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn sub(self, rhs: Tensor) -> Self::Output {
		if self.shape != rhs.shape {
			return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
		}
		let result = &self.data - &rhs.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
}
impl Sub<&Tensor> for &Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data - &rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Sub<Tensor> for &Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn sub(self, rhs: Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data - rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Sub<&Tensor> for Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data - &rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Sub<f64> for Tensor {
	type Output = Tensor;

	fn sub(self, rhs: f64) -> Self::Output {
		let result = &self.data - rhs;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
}
impl Sub<f64> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        let result = &self.data - rhs;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }
}

impl Mul<Tensor> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn mul(self, rhs: Tensor) -> Self::Output {
		if self.shape != rhs.shape {
			return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
		}
		let result = &self.data * &rhs.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
}
impl Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data * &rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Mul<Tensor> for &Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn mul(self, rhs: Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data * rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Mul<&Tensor> for Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data * &rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Mul<f64> for Tensor {
	type Output = Tensor;

	fn mul(self, rhs: f64) -> Self::Output {
		let result = &self.data * rhs;
		Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
}
impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        let result = &self.data * rhs;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }
}

impl Div<Tensor> for Tensor {
	type Output = Result<Tensor, ShapeError>;

	fn div(self, rhs: Tensor) -> Self::Output {
		if self.shape != rhs.shape {
			return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
		}
		let result = &self.data / &rhs.data;
		Ok(Tensor {
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		})
	}
}
impl Div<&Tensor> for &Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn div(self, rhs: &Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data / &rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Div<Tensor> for &Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn div(self, rhs: Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data / rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Div<&Tensor> for Tensor {
    type Output = Result<Tensor, ShapeError>;

    fn div(self, rhs: &Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let result = &self.data / &rhs.data;
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }
}
impl Div<f64> for Tensor{
	type Output = Tensor;

	fn div(self, rhs: f64) -> Self::Output {
		let result = &self.data / rhs;
		Tensor{
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
}
impl Div<f64> for &Tensor{
	type Output = Tensor;

	fn div(self, rhs: f64) -> Self::Output {
		let result = &self.data / rhs;
		Tensor{
			dimension: result.ndim(),
			shape: result.shape().to_vec(),
			data: result.into_dyn(),
		}
	}
}
