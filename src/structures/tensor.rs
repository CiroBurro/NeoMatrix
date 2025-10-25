/// This module defines the Tensor class, which represents a multidimensional array
/// and provides methods for creating and manipulating tensors.
/// It uses Ndarray along with Rust-Numpy to handle the underlying data structure and the bindings with python.
/// Necessary imports
use pyo3::prelude::*;
use pyo3::Bound;
use ndarray::prelude::*;
use ndarray::{Ix1, Ix2};
use rand;
use numpy::{prelude::*, PyArrayDyn, PyReadonlyArrayDyn};
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

#[derive(FromPyObject)]
enum TensorOrScalar {
    Tensor(Tensor),
    Scalar(f64),
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
    pub fn new(shape: Vec<usize>, content: Vec<f64>) -> Self {
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

    /// Constructor method for a random tensor
    ///
    /// Parameters:
    /// - shape: A vector of usize representing the shape of the tensor
    ///
    /// Python usage:
    /// ```python
    /// from neomatrix import Tensor
    /// t = Tensor.random([2, 2, 3])
    /// ```
    #[staticmethod]
    pub fn random(sh: Vec<usize>) -> Self {

        let mut tensor = Tensor::zeros(sh);

        tensor.data.par_mapv_inplace(|_| {
            rand::random_range(0.0..100.0)
        });

        tensor
    }

    /// Getter method for the data field
    /// It converts the ndarray dynamic array into a numpy PyArrayDyn object for python
    #[getter]
    fn get_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.data.to_pyarray(py)
    }

    /// Setter method for the data field
    /// It allows to set the data field from python with a numpy array
    #[setter]
    fn set_data<'py>(&mut self, arr: PyReadonlyArrayDyn<'py, f64>) -> PyResult<()> {
        let owned = arr.as_array().to_owned();
        self.data = owned;
        Ok(())
    }

    /// Constructor method for the Tensor class from a numpy array
    /// 
    /// Parameters:
    /// - arr: A dynamic numpy array 
    /// 
    /// Python usage:
    /// ```python
    /// import numpy as np
    /// from neomatrix import Tensor
    /// arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    /// t = Tensor.from_numpy(arr)
    /// ```
    #[staticmethod]
    pub fn from_numpy<'py>(arr: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Tensor> {
        let owned = arr.as_array().to_owned();
        let dimension = owned.ndim();
        let shape = owned.shape().to_vec();

        let t = Self {
            dimension,
            shape,
            data: owned
        };
        Ok(t)
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

    /// Element-wise sum method between tensors
    /// 
    /// Parameters:
    /// - t: The tensor to be added
    /// 
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([4], [2, 4, 6, 8])
    /// t_2 = Tensor([4], [1, 3, 5, 7])
    /// result = t_1.tensor_sum(t_2)
    /// print(result)
    /// print(result.data) -> result: [3, 7, 11, 15]
    /// ```
    pub fn tensor_sum(&self, t: &Tensor) -> PyResult<Tensor> {
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
    
    /// Element-wise subtraction method between tensors
    ///
    /// Parameters:
    /// - t: The tensor to be subtracted
    ///
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([4], [2, 4, 6, 8])
    /// t_2 = Tensor([4], [1, 3, 5, 7])
    /// result = t_1.add(t_2)
    /// print(result)
    /// print(result.data) -> result: [1, 1, 1, 1]
    /// ```
    pub fn tensor_subtraction(&self, t: &Tensor) -> PyResult<Tensor> {
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

    /// Element-wise multiplication method between tensors
    ///
    /// Parameters:
    /// - t: The tensor to be multiplied
    ///
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([4], [2, 4, 6, 8])
    /// t_2 = Tensor([4], [1, 3, 5, 7])
    /// result = t_1.add(t_2)
    /// print(result)
    /// print(result.data) -> result: [2, 12, 30, 56]
    /// ```
    pub fn tensor_multiplication(&self, t: &Tensor) -> PyResult<Tensor> {
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

    /// Element-wise division method between tensors
    ///
    /// Parameters:
    /// - t: The tensor to be divided
    ///
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([4], [2, 4, 6, 8])
    /// t_2 = Tensor([4], [1, 3, 5, 7])
    /// result = t_1.add(t_2)
    /// print(result)
    /// print(result.data) -> result: [2, 4/3, 6/5, 8/7]
    /// ```
    pub fn tensor_division(&self, t: &Tensor) -> PyResult<Tensor> {
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

    /// Element-wise sum method between a tensor and a scalar
    ///
    /// Parameters:
    /// - scalar: Scalar number to be added
    ///
    /// Python usage:
    /// ```python
    /// t = Tensor([4], [2, 4, 6, 8])
    /// result = t_1.tensor_sum(3)
    /// print(result)
    /// print(result.data) -> result: [5, 7, 9, 11]
    /// ```
    pub fn scalar_sum(&self, scalar: f64) -> Tensor {
        let result = &self.data + scalar;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }

    /// Element-wise subtraction method between a tensor and a scalar
    ///
    /// Parameters:
    /// - scalar: Scalar number to be subtracted
    ///
    /// Python usage:
    /// ```python
    /// t = Tensor([4], [2, 4, 6, 8])
    /// result = t_1.tensor_sum(3)
    /// print(result)
    /// print(result.data) -> result: [-1, 1, 3, 5]
    /// ```
    pub fn scalar_subtraction(&self, scalar: f64) -> Tensor {
        let result = &self.data - scalar;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }

    /// Element-wise multiplication method between a tensor and a scalar
    ///
    /// Parameters:
    /// - scalar: Scalar number to be multiplied
    ///
    /// Python usage:
    /// ```python
    /// t = Tensor([4], [2, 4, 6, 8])
    /// result = t_1.tensor_sum(3)
    /// print(result)
    /// print(result.data) -> result: [6, 12, 18, 24]
    /// ```
    pub fn scalar_multiplication(&self, scalar: f64) -> Tensor {
        let result = &self.data * scalar;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }

    /// Element-wise division method between a tensor and a scalar
    ///
    /// Parameters:
    /// - scalar: Scalar number to be divided
    ///
    /// Python usage:
    /// ```python
    /// t = Tensor([4], [2, 4, 6, 8])
    /// result = t_1.tensor_sum(3)
    /// print(result)
    /// print(result.data) -> result: [2/3, 4/3, 2, 8/3]
    /// ```
    pub fn scalar_division(&self, scalar: f64) -> Tensor {
        let result = &self.data / scalar;
        Tensor {
            dimension: result.ndim(),
            shape: result.shape().to_vec(),
            data: result.into_dyn(),
        }
    }

    fn __add__(&self, other: TensorOrScalar) -> PyResult<Tensor> {

        match other {
            TensorOrScalar::Tensor(t) => {
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

    fn __sub__(&self, other: TensorOrScalar) -> PyResult<Tensor> {

        match other {
            TensorOrScalar::Tensor(t) => {
                // Check if the shapes are compatible for subtraction
                if self.shape != t.shape {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Tensor shapes are not compatible for element-wise addition"
                    ));
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

    fn __mul__(&self, other: TensorOrScalar) -> PyResult<Tensor> {

        match other {
            TensorOrScalar::Tensor(t) => {
                // Check if the shapes are compatible for multiplication
                if self.shape != t.shape {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Tensor shapes are not compatible for element-wise addition"
                    ));
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

    fn __truediv__(&self, other: TensorOrScalar) -> PyResult<Tensor> {

        match other {
            TensorOrScalar::Tensor(t) => {
                // Check if the shapes are compatible for division
                if self.shape != t.shape {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Tensor shapes are not compatible for element-wise addition"
                    ));
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

    /// Length method for tensor
    ///
    /// Python usage:
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// length = t.length()
    /// ```
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Transpose method for 2D tensors
    /// 
    /// Python usage:
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// result = t.transpose()
    /// print(result) -> result Tensor(dimension=2, shape=[3, 2])
    /// print(result.data) -> result: [[1, 4], [2, 5], [3, 6]]
    /// ```
    pub fn transpose(&self) -> PyResult<Tensor> {
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

    /// Reshape method for Tensor
    /// 
    /// Parameters:
    /// - shape: Vector representing the new shape
    /// 
    /// Python usage:
    /// ```python
    /// from neomatrix import Tensor
    /// t = Tensor([2, 2], [1, 2, 3, 4])
    /// t.reshape([4])
    /// ```
    pub fn reshape(&mut self, sh: Vec<usize>) {
        self.shape = sh.clone();
        self.dimension = sh.len();
        self.data = self.data.to_owned().into_shape_with_order(sh).expect("Incompatible shape");
    }

    /// Flatten method for Tensor
    /// 
    /// Python usage:
    /// ```python
    /// from neomatrix import Tensor
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// t.flatten()
    /// ```
    pub fn flatten(&mut self) {
        let flattened_data = self.data.flatten();
        self.shape = flattened_data.shape().to_vec();
        self.dimension = flattened_data.ndim();
        self.data = flattened_data.to_owned().into_dyn();
    }

    /// Push method for 2 tensor
    /// One tensor is pushed into the other
    /// 
    /// Parameters:
    /// - t: tensor to be pushed
    /// - axis: index of the axe along which tensors should be concatenated
    /// 
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([4], [2, 4, 6, 8])
    /// t_2 = Tensor([4], [1, 3, 5, 7])
    /// t_1.push_cat(t_2, 0)
    /// ```
    pub fn push(&mut self, t: &Tensor, axis: usize) {
        let mut vec_data = Vec::new();
        self.data.flatten().for_each(|x| vec_data.push(*x));
        t.data.flatten().for_each(|x| vec_data.push(*x));
    
        let mut shape = self.data.shape().to_vec();
        shape[axis] += t.data.len_of(Axis(axis));
    
        let new_data = ArrayD::from_shape_vec(IxDyn(&shape), vec_data)
            .expect("Incompatible dimensions for pushing one tensor into another");
        
        self.dimension = new_data.ndim();
        self.shape = new_data.shape().to_vec();
        self.data = new_data;
    }

    /// Concatenate method for multiple tensors
    /// 
    /// Parameters:
    /// - tensors: vector of tensors to be concatenated
    /// - axis: index of the axe along which tensors should be concatenated
    /// 
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([4], [2, 4, 6, 8])
    /// t_2 = Tensor([4], [1, 3, 5, 7])
    /// t_3 = Tensor([4], [10, 12, 14, 16])
    /// t_4 = Tensor([4], [9, 11, 13, 15])
    /// 
    /// t = t_1.cat([t_2, t_3, t_4], 0)
    /// ```
    pub fn cat(&self, tensors: Vec<Tensor>, axis: usize) -> PyResult<Tensor> {
        let mut new_tensor = self.clone();
        for t in tensors.iter() {
            new_tensor.push(t, axis);
        }

        Ok(new_tensor)
    }

    /// Push row method for Tensor
    /// 
    /// Parameters:
    /// - t: tensor representing the row to be added
    /// 
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([3, 2], [2, 4, 6, 8, 10, 12])
    /// t_2 = Tensor([2], [1, 3])
    /// t_1.push_row(t_2)
    /// ```
    pub fn push_row(&mut self, t: &Tensor) {

        if t.dimension != 1 {
            panic!("A row must be 1D")
        }

        if self.dimension > 2 {
            panic!("Pushing a row is allowed only for 1D and 2D tensors")
        }

        if self.dimension == 1 {
            self.dimension = 2;
            self.shape = vec![1, self.shape[0]];
            self.data = self.data.to_shape(self.shape.clone()).unwrap().to_owned();
        }

        self.data.push(Axis(0), t.data.view()).unwrap();
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();
    }

    /// Push column method for Tensor
    /// 
    /// Parameters:
    /// - t: tensor representing the column to be added
    /// 
    /// Python usage:
    /// ```python
    /// t_1 = Tensor([3, 2], [2, 4, 6, 8, 10, 12])
    /// t_2 = Tensor([3], [1, 3, 5])
    /// t_1.push_column(t_2)
    /// ```
    pub fn push_column(&mut self, t: &Tensor) {

        if t.dimension != 1 {
            panic!("A column must be 1D")
        }

        if self.dimension > 2 {
            panic!("Pushing a column is allowed only for 1D and 2D tensors")
        }

        if self.dimension == 1 {
            self.dimension = 2;
            self.shape = vec![self.shape[0], 1];
            self.data = self.data.to_shape(self.shape.clone()).unwrap().to_owned();
        }

        self.data.push(Axis(1), t.data.view()).unwrap();
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        todo!()
    }

    fn __repr__(&self) -> String {
        format!("Tensor(dimension={}, shape={:?})", self.dimension, self.shape)
    }
}
