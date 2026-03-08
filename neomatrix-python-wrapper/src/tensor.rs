use crate::tensor::tensor_iter::TensorIter;
use crate::utils::matmul::par_dot;
use ndarray::prelude::*;
use ndarray::{Ix1, Ix2};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, prelude::*};
use pyo3::Bound;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand;

#[pyclass(module = "neomatrix")]
#[derive(Clone, Debug)]
pub struct Tensor {
    #[pyo3(get, set)]
    pub dimension: usize,
    #[pyo3(get, set)]
    pub shape: Vec<usize>,
    pub data: ArrayD<f32>,
}

#[derive(FromPyObject)]
enum TensorOrScalar<'py> {
    Tensor(PyRef<'py, Tensor>),
    Scalar(f32),
}

/// `Tensor` struct methods
#[pymethods]
impl Tensor {
    /// Constructor method for the Tensor class in python
    ///
    /// # Arguments
    /// * `shape` - A vector of usize representing the shape of the tensor
    /// * `content` - A vector of f32 representing the content of the tensor
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
    pub fn new(shape: Vec<usize>, content: Vec<f32>) -> PyResult<Tensor> {
        let dimension = shape.len();
        let data = match Array::from_shape_vec(shape.clone(), content) {
            Ok(array) => array,
            Err(_) => {
                return Err(PyErr::new::<PyRuntimeError, _>(
                    "Shape and content do not match",
                ));
            }
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

        tensor
            .data
            .par_mapv_inplace(|_| rand::random_range(0.0..100.0));

        tensor
    }

    /// Getter method for the data field
    /// It converts the ndarray dynamic array into a numpy PyArrayDyn object for python
    #[getter]
    fn get_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f32>> {
        self.data.to_pyarray(py)
    }

    /// Setter method for the data field
    /// It allows to set the data field from python with a numpy array
    #[setter]
    fn set_data<'py>(&mut self, arr: PyReadonlyArrayDyn<'py, f32>) -> PyResult<()> {
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
    ///     arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ///     t = Tensor.from_numpy(arr)
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (array))]
    pub fn from_numpy<'py>(array: PyReadonlyArrayDyn<'py, f32>) -> PyResult<Tensor> {
        let owned = array.as_array().to_owned();
        let dimension = owned.ndim();
        let shape = owned.shape().to_vec();

        let t = Self {
            dimension,
            shape,
            data: owned,
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
        match (self.dimension, t.dimension) {
            (1, 1) => {
                if self.shape[0] != t.shape[0] {
                    return Err(PyValueError::new_err(format!(
                        "Incompatible dimensions for dot product: [{}] · [{}]",
                        self.shape[0], t.shape[0]
                    )));
                }

                let result = self
                    .data
                    .view()
                    .into_dimensionality::<Ix1>()
                    .map_err(|_| PyValueError::new_err("Invalid dimension"))?
                    .dot(
                        &t.data
                            .view()
                            .into_dimensionality::<Ix1>()
                            .map_err(|_| PyValueError::new_err("Invalid dimension"))?,
                    );
                Ok(Tensor {
                    dimension: 0,
                    shape: Vec::<usize>::new(),
                    data: ArrayD::from_elem(vec![], result),
                })
            }
            (1, 2) => {
                if self.shape[0] != t.shape[0] {
                    return Err(PyValueError::new_err(format!(
                        "Incompatible dimensions for dot product: [{}] · [{}, {}]",
                        self.shape[0], t.shape[0], t.shape[1]
                    )));
                }

                let result = self
                    .data
                    .view()
                    .into_dimensionality::<Ix1>()
                    .map_err(|_| PyValueError::new_err("Invalid dimension"))?
                    .dot(
                        &t.data
                            .view()
                            .into_dimensionality::<Ix2>()
                            .map_err(|_| PyValueError::new_err("Invalid dimension"))?,
                    );
                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn(),
                })
            }
            (2, 1) => {
                if self.shape[1] != t.shape[0] {
                    return Err(PyValueError::new_err(format!(
                        "Incompatible dimensions for dot product: [{}, {}] · [{}]",
                        self.shape[0], self.shape[1], t.shape[0]
                    )));
                }

                let result = self
                    .data
                    .view()
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| PyValueError::new_err("Invalid dimension"))?
                    .dot(
                        &t.data
                            .view()
                            .into_dimensionality::<Ix1>()
                            .map_err(|_| PyValueError::new_err("Invalid dimension"))?,
                    );
                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn(),
                })
            }
            (2, 2) => {
                if self.shape[1] != t.shape[0] {
                    return Err(PyValueError::new_err(format!(
                        "Incompatible dimensions for dot product: [{}, {}] · [{}, {}]",
                        self.shape[0], self.shape[1], t.shape[0], t.shape[1]
                    )));
                }

                let result = par_dot(
                    self.data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .map_err(|_| PyValueError::new_err("Invalid dimension"))?,
                    t.data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .map_err(|_| PyValueError::new_err("Invalid dimension"))?,
                );

                Ok(Tensor {
                    dimension: result.ndim(),
                    shape: result.shape().to_vec(),
                    data: result.into_dyn(),
                })
            }

            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "It's possible to multiply only 1D and 2D tensors (dot product). General Tensor Contraction is not yet defined.",
            )),
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
            TensorOrScalar::Tensor(t) => self + &*t,
            TensorOrScalar::Scalar(scalar) => Ok(self + scalar),
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
            TensorOrScalar::Tensor(t) => self - &*t,
            TensorOrScalar::Scalar(scalar) => Ok(self - scalar),
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
            TensorOrScalar::Tensor(t) => self * &*t,
            TensorOrScalar::Scalar(scalar) => Ok(self * scalar),
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
            TensorOrScalar::Tensor(t) => self / &*t,
            TensorOrScalar::Scalar(scalar) => self / scalar,
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
                "It's possible to transpose only 2D tensors",
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
    pub fn transpose_inplace(&mut self) -> PyResult<()> {
        // Check if the shapes are compatible for addition
        if self.dimension != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "It's possible to transpose only 2D tensors",
            ));
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
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix import Tensor
    ///     t = Tensor([2, 2], [1, 2, 3, 4])
    ///     reshaped = t.reshape([4])
    ///     ```
    #[pyo3(signature = (shape))]
    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<Tensor> {
        let dim = shape.len();
        let data = self
            .data
            .to_owned()
            .into_shape_with_order(shape.as_slice())
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Incompatible shape"))?;

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
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix import Tensor
    ///     t = Tensor([2, 2], [1, 2, 3, 4])
    ///     t.reshape_inplace([4])
    ///     ```
    #[pyo3(signature = (shape))]
    pub fn reshape_inplace(&mut self, shape: Vec<usize>) -> PyResult<()> {
        self.shape = shape.clone();
        self.dimension = shape.len();
        self.data = self
            .data
            .to_owned()
            .into_shape_with_order(shape)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Incompatible shape"))?;
        Ok(())
    }

    /// Flatten method for Tensor
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix import Tensor
    ///     t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    ///     flattened = t.flatten()
    ///     ```
    pub fn flatten(&self) -> Tensor {
        let flattened_data = self.data.flatten();

        Tensor {
            shape: flattened_data.shape().to_vec(),
            dimension: flattened_data.ndim(),
            data: flattened_data.to_owned().into_dyn(),
        }
    }

    /// Flatten inplace method for Tensor
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix import Tensor
    ///     t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    ///     t.flatten_inplace()
    ///     ```
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
    ///
    /// # Python usage
    ///     ```python
    ///     t_1 = Tensor([4], [2, 4, 6, 8])
    ///     t_2 = Tensor([4], [1, 3, 5, 7])
    ///     t_1.push(t_2, 0)
    ///     ```
    #[pyo3(signature = (t, axis))]
    pub fn push(&mut self, t: &Tensor, axis: usize) -> PyResult<()> {
        let mut vec_data = Vec::new();
        self.data.flatten().for_each(|x| vec_data.push(*x));
        t.data.flatten().for_each(|x| vec_data.push(*x));

        let mut shape = self.data.shape().to_vec();
        shape[axis] += t.data.len_of(Axis(axis));

        let new_data = ArrayD::from_shape_vec(IxDyn(&shape), vec_data).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Incompatible dimensions for pushing one tensor into another",
            )
        })?;

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
            new_tensor.push(t, axis)?;
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
    pub fn push_row(&mut self, t: &Tensor) -> PyResult<()> {
        if t.dimension != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "A row must be 1D",
            ));
        }

        if self.dimension > 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pushing a row is allowed only for 1D and 2D tensors",
            ));
        }

        if self.dimension == 1 {
            self.dimension = 2;
            self.shape = vec![1, self.shape[0]];
            self.data = self
                .data
                .to_shape(self.shape.clone())
                .map_err(|_| PyValueError::new_err("Invalid dimension"))?
                .to_owned();
        }

        self.data
            .push(Axis(0), t.data.view())
            .map_err(|_| PyValueError::new_err("Invalid dimension"))?;
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();

        Ok(())
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
    pub fn push_column(&mut self, t: &Tensor) -> PyResult<()> {
        if t.dimension != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "A column must be 1D",
            ));
        }

        if self.dimension > 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pushing a column is allowed only for 1D and 2D tensors",
            ));
        }

        if self.dimension == 1 {
            self.dimension = 2;
            self.shape = vec![self.shape[0], 1];
            self.data = self
                .data
                .to_shape(self.shape.clone())
                .map_err(|_| PyValueError::new_err("Invalid dimension"))?
                .to_owned();
        }

        self.data
            .push(Axis(1), t.data.view())
            .map_err(|_| PyValueError::new_err("Invalid dimension"))?;
        self.dimension = self.data.ndim();
        self.shape = self.data.shape().to_vec();

        Ok(())
    }

    /// Iter method for a tensor in python
    ///
    /// # Returns
    /// * `PyResult<Py<TensorIter>>` - New iterator over a tensor
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<TensorIter>> {
        let iter = TensorIter {
            inner: slf.data.clone().into_iter(),
        };
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
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("dimension", self.dimension)?;
            d.set_item("shape", self.shape.clone())?;
            d.set_item("data", self.data.clone().to_pyarray(py).to_vec()?)?;
            Ok(d.into())
        })
    }
    #[staticmethod]
    /// from_dict method converts a python dictionary into a Tensor structure
    ///
    /// # Arguments
    /// * `d` - Python dictionary with all fields of the tensor
    ///
    /// # Returns
    /// * `PyResult<Tensor>` - Tensor from the python dictionary
    ///
    /// # Python usage
    ///     ```python
    ///     from neomatrix.core import Tensor
    ///     d = {
    ///         "dimension": 2,
    ///         "shape": [2, 2],
    ///         "data": [1.0, 2.0, 3.0, 4.0]
    ///     }
    ///     t = Tensor.from_dict(d)
    ///     ```
    pub fn from_dict(d: Bound<PyAny>) -> PyResult<Tensor> {
        let d = d.downcast::<PyDict>()?;
        let shape = d
            .get_item("shape")?
            .ok_or(PyValueError::new_err("No field for shape deserialization"))?;
        let shape = shape.downcast::<PyList>()?.extract::<Vec<usize>>()?;

        let data = d
            .get_item("data")?
            .ok_or(PyValueError::new_err("No field for data deserialization"))?;
        let data = data.downcast::<PyList>()?.extract::<Vec<f32>>()?;
        Tensor::new(shape, data)
    }

    /// Repr method for a tensor in python
    fn __repr__(&self) -> String {
        format!(
            "Tensor(dimension={}, shape={:?})",
            self.dimension, self.shape
        )
    }
}
