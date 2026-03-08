use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum TensorError {
    #[error("Shape and content do not match")]
    ShapeAndContentMismatch,

    #[error("Invalid dimension")]
    InvalidDimension,

    #[error("{0}")]
    IncompatibleDimensionsForDotProduct(String),

    #[error("It's possible to multiply only 1D and 2D tensors (dot product). General Tensor Contraction is not yet defined.")]
    TensorContractionNotYetDefined,

    #[error("It's possible to transpose only 2D tensors")]
    TransposeOnly2D,

    #[error("Incompatible shape")]
    IncompatibleShape,

    #[error("Incompatible dimensions for pushing one tensor into another")]
    IncompatibleDimensionsForPushing,

    #[error("A row must be 1D")]
    RowMustBe1D,

    #[error("Pushing a row is allowed only for 1D and 2D tensors")]
    PushingRowOnly1D2D,

    #[error("A column must be 1D")]
    ColumnMustBe1D,

    #[error("Pushing a column is allowed only for 1D and 2D tensors")]
    PushingColumnOnly1D2D,

    #[error("Tensor shapes are not compatible for element-wise addition")]
    ShapesNotCompatibleForElementWiseAddition,

    #[error("Tensor shapes are not compatible for element-wise subtraction")]
    ShapesNotCompatibleForElementWiseSubtraction,

    #[error("Tensor shapes are not compatible for element-wise multiplication")]
    ShapesNotCompatibleForElementWiseMultiplication,

    #[error("Tensor shapes are not compatible for element-wise division")]
    ShapesNotCompatibleForElementWiseDivision,

    #[error("Cannot divide by zero: divisor tensor contains zero elements")]
    CannotDivideByZeroDivisorTensorContainsZeroElements,

    #[error("Cannot divide by zero")]
    CannotDivideByZero,
}

#[derive(Error, Debug)]
pub enum LayerError {
    #[error(transparent)]
    Tensor(#[from] TensorError),

    #[error(transparent)]
    Math(#[from] MathError),

    #[error("Layer not initialized: call forward() before backward()")]
    NotInitialized,
}

#[derive(Error, Debug)]
pub enum MathError {
    // --- Activation errors ---
    #[error("Unsupported tensor dimension for Softmax: expected 1D or 2D")]
    SoftmaxUnsupportedDimension,

    #[error("Softmax derivative is only implemented for 1D and 2D tensors")]
    SoftmaxDerivativeUnsupportedDimension,

    #[error("Unknown activation function: '{0}'")]
    UnknownActivation(String),

    // --- Cost function errors ---
    #[error("Tensors must have the same shape and be 1D for cost function computation")]
    CostFunctionShapeMismatch,

    #[error("Tensors must have the same shape and be 2D for batch cost function computation")]
    CostFunctionBatchShapeMismatch,

    #[error("Tensors must have the same shape for derivative computation")]
    DerivativeShapeMismatch,

    #[error("Tensor subtraction failed")]
    TensorSubtractionFailed,

    #[error("Failed to build gradient tensor: shape mismatch")]
    GradientShapeMismatch,

    #[error("Unknown cost function: '{0}'")]
    UnknownCostFunction(String),

    #[error("Missing 'delta' field for HuberLoss deserialization")]
    HuberLossMissingDelta,
}
