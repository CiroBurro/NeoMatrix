/// This module defines the Tensor class, which represents a multidimensional array
/// and provides methods for creating and manipulating tensors.
/// It uses Ndarray along with Rust-Numpy to handle the underlying data structure and the bindings with python.
/// Necessary imports
use pyo3::prelude::*;
use pyo3::Bound;
use ndarray::prelude::*;
use ndarray::{Ix1, Ix2};
use numpy::{ToPyArray, PyArrayDyn};
use crate::utils::matmul::par_dot;

/// Tensor struct definition
/// 
/// Fields:
/// - dimension: The number of dimensions of the tensor
/// - shape: The shape of the tensor (e.g., [2, 3] for a 2D tensor with 2 rows and 3 columns)
/// - data: The underlying data (float64) of the tensor, stored as a dynamic array provided by Ndarray
#[pyclass(module = "neomatrix")]
#[derive(Clone, Debug)]
pub struct Tensor {
    #[pyo3(get, set)]
    pub dimension: usize,
    #[pyo3(get, set)]
    pub shape: Vec<usize>,
    pub data: ArrayD<f64>,
}

/// Tensor struct implementation
#[pymethods]
impl Tensor {
    ///Constructor method for the Tensor class in python
    /// 
    /// Parameters:
    /// - shape: A vector of usize representing the shape of the tensor
    /// - content: A vector of f64 representing the content of the tensor
    ///
    /// Python usage:
    /// ```python
    /// from neomatrix import Tensor
    /// t = Tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 12])
    /// ```
    #[new]
    fn new(shape: Vec<usize>, content: Vec<f64>) -> Self {
        let dimension = shape.len();
        let data = match Array::from_shape_vec(shape.clone(), content) {
            Ok(array) => array,
            Err(_) => panic!("Shape and content do not match"),
        };
        Self { dimension, shape, data }
    }

    /// Constructor method for an empty tensor
    /// 
    /// Parameters:
    /// - shape: A vector of usize representing the shape of the tensor
    /// 
    /// Python usage:
    /// ```python
    /// from neomatrix import Tensor
    /// t = Tensor.zeros([2, 2, 3])
    /// ```
    #[staticmethod]
    pub fn zeros(sh: Vec<usize>) -> Self {
        let dimension = sh.len();
        let data = Array::zeros(sh.clone());
        Self { dimension, shape: sh, data }
    }

    /// Getter method for the data field
    /// It converts the ndarray dynamic array into a numpy PyArrayDyn object for python
    #[getter]
    fn get_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.data.to_pyarray(py)
    }

    /// Dot product method for 1D and 2D tensors
    /// 
    /// Parameters:
    /// - t: The tensor to be multiplied with
    /// 
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([4], [2, 4, 6, 8])
    /// t_2 = Tensor([4, 2], [1, 3, 5, 7, 9, 11, 13, 15])
    /// result = t_1.dot(t_2)
    /// print(result)
    /// print(result.data)
    /// ```
    pub fn dot(&self, t: &Tensor) -> PyResult<Tensor> {
        // Check if the dimensions are compatible for dot product
        match (self.dimension, t.dimension) {
            // Vector product (1D * 1D)
            (1, 1) => {
                let result = self.data.clone().into_dimensionality::<Ix1>().unwrap() // self.data ArrayD has to be converted into a defined dimensionality 
                    .dot(&t.data.clone().into_dimensionality::<Ix1>().unwrap());
                Ok(Tensor {
                    dimension: 0,
                    shape: vec![],
                    data: ArrayD::from_elem(vec![], result),
                })
            },
            // Matrix product (1D * 2D)
            (1, 2) => {
                let result = self.data.clone().into_dimensionality::<Ix1>().unwrap()
                    .dot(&t.data.clone().into_dimensionality::<Ix2>().unwrap());
                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn() // Convert the result back to a dynamic array
                })
            },
            // Matrix product (2D * 1D)
            (2, 1) => {
                let result = self.data.clone().into_dimensionality::<Ix2>().unwrap()
                    .dot(&t.data.clone().into_dimensionality::<Ix1>().unwrap());
                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn(),
                })
            },
            // Matrix product (2D * 2D) in parallel (multithreading with par_dot() function)
            (2, 2) => {
                let result = par_dot(self.data.clone().into_dimensionality::<Ix2>().unwrap(), t.data.clone().into_dimensionality::<Ix2>().unwrap());
                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn(),
                })
            },
            
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "It's possible to multiply only 1D and 2D tensors (dot product)"
            ))
        }
    }

    /// Sum method for tensors
    /// 
    /// Parameters:
    /// - t: The tensor to be added
    /// 
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([4], [2, 4, 6, 8])
    /// t_2 = Tensor([4, 2], [1, 3, 5, 7, 9, 11, 13, 15])
    /// result = t_1.add(t_2)
    /// print(result)
    /// print(result.data)
    /// ```
    pub fn add(&self, t: &Tensor) -> PyResult<Tensor> {
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

    pub fn subtract(&self, t: &Tensor) -> PyResult<Tensor> {
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

    pub fn multiplication(&self, t: &Tensor) -> PyResult<Tensor> {
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

    pub fn division(&self, t: &Tensor) -> PyResult<Tensor> {
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

    pub fn transpose (&self) -> PyResult<Tensor> {
        // Check if the shapes are compatible for addition
        if self.dimension != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "It's possible to transpose only 2D tensors"
            ));
        }
        let result = self.data.clone().reversed_axes();
        Ok(Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        })
    }

    fn __repr__(&self) -> String {
        format!("Tensor(dimension={}, shape={:?})", self.dimension, self.shape)
    }
}