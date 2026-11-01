//! Owned, shape-checked tensor storage for the v2 inference API.
//!
//! Values are stored in row-major order. Cloning a tensor shares immutable
//! storage; constructing or copying a tensor never transfers ownership to Core ML.

use std::{fmt, sync::Arc};

/// An element supported by the owned tensor representation.
pub trait TensorElement: private::Sealed + Copy + Send + Sync + 'static {
    #[doc(hidden)]
    const DATA_TYPE: DataType;
}

/// Scalar representation in a Core ML multiarray.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Float16,
    Float32,
    Float64,
    Int32,
    /// Core ML support requires macOS 26 or later.
    Int8,
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl Sealed for i8 {}
    impl Sealed for half::f16 {}
}
impl TensorElement for f32 {
    const DATA_TYPE: DataType = DataType::Float32;
}
impl TensorElement for f64 {
    const DATA_TYPE: DataType = DataType::Float64;
}
impl TensorElement for i32 {
    const DATA_TYPE: DataType = DataType::Int32;
}
impl TensorElement for i8 {
    const DATA_TYPE: DataType = DataType::Int8;
}
impl TensorElement for half::f16 {
    const DATA_TYPE: DataType = DataType::Float16;
}

/// A failure to construct an owned tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// The product of the dimensions cannot be represented as a `usize`.
    ShapeOverflow,
    /// The number of elements does not match the dimensions.
    LengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => write!(f, "tensor shape overflows usize"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "tensor shape requires {expected} elements, got {actual}")
            }
        }
    }
}
impl std::error::Error for TensorError {}

/// A contiguous, immutable tensor with reference-counted storage.
#[derive(Debug, Clone)]
pub struct Tensor<T: TensorElement> {
    shape: Arc<[usize]>,
    data: Arc<Vec<T>>,
}

impl<T: TensorElement> Tensor<T> {
    /// Construct a tensor, checking its shape against the number of elements.
    /// An empty shape represents a scalar and therefore requires one element.
    pub fn from_vec(shape: impl Into<Vec<usize>>, data: Vec<T>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let expected = shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or(TensorError::ShapeOverflow)?;
        if expected != data.len() {
            return Err(TensorError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            shape: shape.into(),
            data: Arc::new(data),
        })
    }

    /// Dimensions in row-major order.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The contiguous elements in row-major order.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

#[cfg(feature = "ndarray")]
impl<T: TensorElement> TryFrom<ndarray::ArrayViewD<'_, T>> for Tensor<T> {
    type Error = TensorError;
    /// Copy an ndarray view in logical order, including non-contiguous views.
    fn try_from(view: ndarray::ArrayViewD<'_, T>) -> Result<Self, Self::Error> {
        Self::from_vec(view.shape().to_vec(), view.iter().copied().collect())
    }
}
#[cfg(feature = "ndarray")]
impl<T: TensorElement> Tensor<T> {
    /// Copy this tensor into a contiguous ndarray.
    pub fn to_ndarray(&self) -> ndarray::ArrayD<T> {
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(self.shape()), self.as_slice().to_vec())
            .expect("tensor constructor validated shape and length")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_shape_and_scalar() {
        assert_eq!(
            Tensor::from_vec(vec![2, 2], vec![1i32; 3]).unwrap_err(),
            TensorError::LengthMismatch {
                expected: 4,
                actual: 3
            }
        );
        assert_eq!(
            Tensor::from_vec(vec![usize::MAX, 2], Vec::<f32>::new()).unwrap_err(),
            TensorError::ShapeOverflow
        );
        assert_eq!(
            Tensor::from_vec(vec![], vec![1f32]).unwrap().as_slice(),
            &[1.0]
        );
        assert!(Tensor::from_vec(vec![2, 0], Vec::<i32>::new()).is_ok());
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn copies_noncontiguous_ndarray_in_logical_order() {
        use ndarray::{array, s};
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let tensor = Tensor::try_from(a.slice(s![.., ..;-1]).into_dyn()).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_slice(), &[3, 2, 1, 6, 5, 4]);
        assert_eq!(tensor.to_ndarray().shape(), &[2, 3]);
    }

    #[test]
    fn clones_retain_storage() {
        let tensor = Tensor::from_vec(vec![2], vec![1i32, 2]).unwrap();
        let clone = tensor.clone();
        assert_eq!(tensor.as_slice().as_ptr(), clone.as_slice().as_ptr());
        drop(tensor);
        assert_eq!(clone.as_slice(), &[1, 2]);
    }
}
