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
use pyo3::types::PyDict;
use crate::structures::tenosor_iter::TensorIter;
use crate::utils::matmul::par_dot;

/// Struct `Tensor`
/// 
/// # Fields:
/// * `dimension` - The number of dimensions of the tensor
/// * `shape` - The shape of the tensor (e.g., [2, 3] for a 2D tensor with 2 rows and 3 columns)
/// * `data` - The underlying data (float64) of the tensor, stored as a dynamic array provided by Ndarray
#[pyclass(module = "neomatrix")]
#[derive(Clone, Debug)]
pub struct Tensor {
    #[pyo3(get, set)]
    pub dimension: usize,
    #[pyo3(get, set)]
    pub shape: Vec<usize>,
    pub data: ArrayD<f64>,
}

/// Enum `TensorOrScalar`
///
/// This enum is used to represent either a `Tensor` or a scalar value (`f64`).
/// It is particularly useful for operations that can accept both types as input,
/// such as element-wise addition, subtraction, multiplication, or division.
///
/// # Variants
/// * `Tensor`- Represents a `Tensor` object.
/// * `Scalar`- Represents a scalar value of type `f64`.
#[derive(FromPyObject)]
enum TensorOrScalar {
    /// Variant for a `Tensor` object.
    Tensor(Tensor),
    /// Variant for a scalar value of type `f64`
    Scalar(f64),
}

/// `Tensor` struct methods
#[pymethods]
impl Tensor {
    /// Constructor method for the Tensor class in python
    /// 
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    /// * `content` - A vector of f64 representing the content of the tensor
    ///
    /// # Returns
    /// * `Tensor` - New tensor
    ///
    /// # Python usage
    ///     ```
    ///     from neomatrix.core import Tensor
    ///     t = Tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 12])
    ///     ```
    #[pyo3(signature = (shape, content))]
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
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    ///
    /// # Returns
    /// * `Tensor` - An empty tensor
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix import Tensor
    ///     t = Tensor.zeros([2, 2, 3])
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (shape))]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let dimension = shape.len();
        let data = Array::zeros(shape.clone());
        Self { dimension, shape, data }
    }

    /// Constructor method for a random tensor
    ///
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    ///
    /// # Returns
    /// * `Tensor` - A tensor with random data
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix import Tensor
    ///     t = Tensor.random([2, 2, 3])
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (shape))]
    pub fn random(shape: Vec<usize>) -> Self {

        let mut tensor = Tensor::zeros(shape);

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
    /// # Arguments
    /// * `arr` - A dynamic numpy array
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - Tensor created from NumPy array
    ///
    /// # Python usage
    ///     ```python
    ///     import numpy as np
    ///     from neomatrix import Tensor
    ///     arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    ///     t = Tensor.from_numpy(arr)
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (array))]
    pub fn from_numpy<'py>(array: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Tensor> {
        let owned = array.as_array().to_owned();
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
    /// # Arguments
    /// * `t` - The tensor to be multiplied with
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - Result of the dot product (tensor)
    ///
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     t_2 = Tensor([4, 2], [1, 3, 5, 7, 9, 11, 13, 15])
    ///     result = t_1.dot(t_2)
    ///     ```
    #[pyo3(signature = (t))]
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

    /// Element-wise sum between two tensor or a tensor and a scalar
    ///
    /// # Arguments
    /// * `other` - Tensor object or scalar to add
    ///
    /// # Returns
    /// * `Pyresult<Tensor>` - Result of the sum (tensor)
    ///
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     t_2 = Tensor([4], [1, 3, 5, 7])
    ///     result = t_1 + t_2
    ///     ```
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

    /// Element-wise subtraction between two tensor or a tensor and a scalar
    ///
    /// # Arguments
    /// * `other` - Tensor object or scalar to subtract
    ///
    /// # Returns
    /// * `Pyresult<Tensor>` - Result of the subtraction (tensor)
    ///
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     result = t_1 - 5.2
    ///     ```
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

    /// Element-wise multiplication between two tensor or a tensor and a scalar
    ///
    /// # Arguments
    /// * `other` - Tensor object or scalar to multiply
    ///
    /// # Returns
    /// * `Pyresult<Tensor>` - Result of the multiplication (tensor)
    ///
    /// # Python usage:
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     result = t_1 * 3.2
    ///     ```
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

    /// Element-wise division between two tensor or a tensor and a scalar
    ///
    /// # Arguments
    /// * `other` - Tensor object or scalar to divide
    ///
    /// # Returns
    /// * `Pyresult<Tensor>` - Result of the division (tensor)
    ///
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     t_2 = Tensor([4], [1, 3, 5, 7])
    ///     result = t_1 / t_2
    ///     ```
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
    /// # Returns
    /// * `usize` - Length of the tensor
    ///
    /// # Python usage
    ///     ```python
    ///     t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    ///     length = t.length()
    ///     ```
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Transpose method for 2D tensors
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - Transposed tensor
    /// 
    /// # Python usage:
    ///     ```python
    ///     t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    ///     result = t.transpose()
    ///     ```
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

    /// Transpose inplace method for 2D tensors
    ///
    /// # Python usage:
    ///     ```python
    ///     t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    ///     t.transpose_inplace()
    ///     ```
    pub fn transpose_inplace(&mut self) {
        // Check if the shapes are compatible for addition
        if self.dimension != 2 {
            panic!("It's possible to transpose only 2D tensors");
        }

        self.data.to_owned().reversed_axes();
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();
    }

    /// Reshape inplace method for Tensor
    /// 
    /// # Arguments
    /// * `shape` - Vector representing the new shape
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix import Tensor
    ///     t = Tensor([2, 2], [1, 2, 3, 4])
    ///     t.reshape([4])
    ///     ```
    #[pyo3(signature = (shape))]
    pub fn reshape(&mut self, shape: Vec<usize>) {
        self.shape = shape.clone();
        self.dimension = shape.len();
        self.data = self.data.to_owned().into_shape_with_order(shape).expect("Incompatible shape");
    }

    /// Flatten inplace method for Tensor
    /// 
    /// # Python usage
    ///     ```python
    ///     from neomatrix import Tensor
    ///     t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    ///     t.flatten()
    ///     ```
    pub fn flatten(&mut self) {
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
    ///
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     t_2 = Tensor([4], [1, 3, 5, 7])
    ///     t_1.push(t_2, 0)
    ///     ```
    #[pyo3(signature = (t, axis))]
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

    /// Concatenate inplace method for multiple tensors
    /// 
    /// # Arguments
    /// * `tensors` - Vector of tensors to be concatenated to self
    /// * `axis` - Index of the axe along which tensors should be concatenated
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - Tensor containing all the input tensors concatenated
    ///
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     t_2 = Tensor([4], [1, 3, 5, 7])
    ///     t_3 = Tensor([4], [10, 12, 14, 16])
    ///     t_4 = Tensor([4], [9, 11, 13, 15])
    ///
    ///     t = t_1.cat_inplace([t_2, t_3, t_4], 0)
    ///     ```
    #[pyo3(signature = (tensors, axis))]
    pub fn cat_inplace(&self, tensors: Vec<Tensor>, axis: usize) -> PyResult<Tensor> {
        let mut new_tensor = self.clone();
        for t in tensors.iter() {
            new_tensor.push(t, axis);
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
    /// * `PyResult<Tensor>` - Tensor containing all the input tensors concatenated
    ///
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     t_2 = Tensor([4], [1, 3, 5, 7])
    ///     t_3 = Tensor([4], [10, 12, 14, 16])
    ///     t_4 = Tensor([4], [9, 11, 13, 15])
    ///
    ///     t = Tensor.cat([t_1, t_2, t_3, t_4], 0)
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (tensors, axis))]
    pub fn cat(tensors: Vec<Tensor>, axis: usize) -> PyResult<Tensor> {

        let mut new_tensor = tensors[0].clone();
        for t in &tensors[1..] {
            new_tensor.push(t, axis);
        }

        Ok(new_tensor)
    }
    /// Push row method for Tensor
    /// 
    /// # Arguments
    /// * `t` - Tensor representing the row to be added
    /// 
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([3, 2], [2, 4, 6, 8, 10, 12])
    ///     t_2 = Tensor([2], [1, 3])
    ///     t_1.push_row(t_2)
    ///     ```
    #[pyo3(signature = (t))]
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
    /// # Arguments
    /// * `t` - Tensor representing the column to be added
    /// 
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([3, 2], [2, 4, 6, 8, 10, 12])
    ///     t_2 = Tensor([3], [1, 3, 5])
    ///     t_1.push_column(t_2)
    ///     ```
    #[pyo3(signature = (t))]
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

    /// Iter method for a tensor in python
    ///
    /// # Returns
    /// * `PyResult<Py<TensorIter>>` - New iterator over a tensor
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<TensorIter>> {
        let iter = TensorIter { inner: slf.data.clone().into_iter()};
        Py::new(slf.py(), iter)
    }

    /// to_dict method converts a Tensor structure into a python dictionary
    ///
    /// # Returns
    /// * `PyResult<Py<PyAny>>` - Python dictionary with all fields of the tensor
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix.core import Tensor
    ///     t = Tensor([2,2], [1,2,3,4])
    ///     d = t.to_dict()
    ///     ```
    pub fn to_dict(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py|{
            let d = PyDict::new(py);
            d.set_item("dimension", self.dimension)?;
            d.set_item("shape", self.shape.clone())?;
            d.set_item("data", self.data.clone().to_string())?;
            Ok(d.into())
        })
    }

    /// Repr method for a tensor in python
    fn __repr__(&self) -> String {
        format!("Tensor(dimension={}, shape={:?})", self.dimension, self.shape)
    }
}