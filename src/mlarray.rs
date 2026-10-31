use half::f16;
use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr};

#[derive(Debug)]
pub enum MLArray {
    Float32Array(ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>),
    Float16Array(ArrayBase<OwnedRepr<f16>, Dim<IxDynImpl>>),
    Int32Array(ArrayBase<OwnedRepr<i32>, Dim<IxDynImpl>>),
    Int16Array(ArrayBase<OwnedRepr<i16>, Dim<IxDynImpl>>),
    Int8Array(ArrayBase<OwnedRepr<i8>, Dim<IxDynImpl>>),
    UInt32Array(ArrayBase<OwnedRepr<u32>, Dim<IxDynImpl>>),
    UInt16Array(ArrayBase<OwnedRepr<u16>, Dim<IxDynImpl>>),
    UInt8Array(ArrayBase<OwnedRepr<u8>, Dim<IxDynImpl>>),
}

impl MLArray {
    pub fn shape(&self) -> &[usize] {
        match self {
            MLArray::Float32Array(array_base) => array_base.shape(),
            MLArray::Float16Array(array_base) => array_base.shape(),
            MLArray::Int32Array(array_base) => array_base.shape(),
            MLArray::Int16Array(array_base) => array_base.shape(),
            MLArray::Int8Array(array_base) => array_base.shape(),
            MLArray::UInt32Array(array_base) => array_base.shape(),
            MLArray::UInt16Array(array_base) => array_base.shape(),
            MLArray::UInt8Array(array_base) => array_base.shape(),
        }
    }
}

pub fn mean_absolute_error_bytes<
    T: core::ops::Sub<Output = T>
        + PartialOrd
        + Copy
        + core::ops::Add<Output = T>
        + num::cast::AsPrimitive<f64>
        + bytemuck::Pod
        + core::fmt::Debug,
>(
    lhs: &[u8],
    rhs: &[u8],
) -> f64 {
    let lhs = bytemuck::cast_slice(lhs);
    let rhs = bytemuck::cast_slice(rhs);

    assert_eq!(lhs.len(), rhs.len(), "lhs and rhs have different lengths");
    mean_absolute_error::<T>(lhs, rhs)
}

pub fn mean_absolute_error<
    T: core::ops::Sub<Output = T>
        + PartialOrd
        + Copy
        + core::ops::Add<Output = T>
        + num::cast::AsPrimitive<f64>,
>(
    lhs: impl AsRef<[T]>,
    rhs: impl AsRef<[T]>,
) -> f64 {
    let (sum, count) = lhs
        .as_ref()
        .iter()
        .zip(rhs.as_ref())
        .map(|(&l, &r)| if l > r { l - r } else { r - l })
        .fold((0f64, 0usize), |(acc, count), x| (acc + x.as_(), count + 1));
    sum / count as f64
}

mod sealed {
    pub trait Sealed {}
}

/// An element type supported by `MLArray`.
///
/// This trait is sealed so downstream types cannot claim another type's layout.
pub trait MLType: sealed::Sealed + Sized {
    const TY: usize;
    fn into_ml_array(value: Array<Self, Dim<IxDynImpl>>) -> MLArray;
    fn try_from_ml_array(value: MLArray) -> Result<Array<Self, Dim<IxDynImpl>>, MLArray>;
}

macro_rules! impl_ml_type {
    ($ty:ty, $variant:ident, $id:expr) => {
        impl sealed::Sealed for $ty {}
        impl MLType for $ty {
            const TY: usize = $id;
            fn into_ml_array(value: Array<Self, Dim<IxDynImpl>>) -> MLArray {
                MLArray::$variant(value)
            }
            fn try_from_ml_array(value: MLArray) -> Result<Array<Self, Dim<IxDynImpl>>, MLArray> {
                match value {
                    MLArray::$variant(array) => Ok(array),
                    other => Err(other),
                }
            }
        }
    };
}

impl_ml_type!(f32, Float32Array, 0);
impl_ml_type!(f16, Float16Array, 1);
impl_ml_type!(i32, Int32Array, 2);
impl_ml_type!(u16, UInt16Array, 3);
impl_ml_type!(u8, UInt8Array, 4);
impl_ml_type!(i16, Int16Array, 5);
impl_ml_type!(i8, Int8Array, 6);
impl_ml_type!(u32, UInt32Array, 7);

impl<T: MLType> From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>> for MLArray {
    fn from(value: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>) -> Self {
        T::into_ml_array(value)
    }
}

impl MLArray {
    /// Extract the stored tensor without reinterpreting its element type.
    ///
    /// # Panics
    /// Panics if `T` does not match the stored variant. Use `try_extract_to_tensor`
    /// to recover the original array on a type mismatch.
    pub fn extract_to_tensor<T: MLType>(self) -> Array<T, Dim<IxDynImpl>> {
        self.try_extract_to_tensor()
            .expect("requested tensor type does not match MLArray variant")
    }

    pub fn try_extract_to_tensor<T: MLType>(self) -> Result<Array<T, Dim<IxDynImpl>>, Self> {
        T::try_from_ml_array(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_element_type_round_trips() {
        macro_rules! check {
            ($ty:ty, $value:expr) => {{
                let array = ndarray::Array::from_elem(vec![2, 3], $value);
                let expected = array.clone();
                let wrapped = MLArray::from(array);
                assert_eq!(wrapped.extract_to_tensor::<$ty>(), expected);
            }};
        }
        check!(f32, 3.5f32);
        check!(f16, f16::from_f32(1.5));
        check!(i32, -3i32);
        check!(u16, 3u16);
        check!(u8, 3u8);
        check!(i16, -3i16);
        check!(i8, -3i8);
        check!(u32, 3u32);
    }

    #[test]
    fn mismatch_preserves_original_tensor() {
        let original = ndarray::arr1(&[1.0f32, 2.0]).into_dyn();
        let recovered = MLArray::from(original.clone())
            .try_extract_to_tensor::<u8>()
            .unwrap_err();
        assert_eq!(recovered.extract_to_tensor::<f32>(), original);
    }

    #[test]
    #[should_panic(expected = "requested tensor type does not match")]
    fn mismatched_infallible_extraction_panics() {
        MLArray::from(ndarray::arr1(&[1.0f32]).into_dyn()).extract_to_tensor::<u8>();
    }

    #[test]
    fn conversion_preserves_nonstandard_layout() {
        let original = ndarray::arr2(&[[1i32, 2, 3], [4, 5, 6]])
            .reversed_axes()
            .into_dyn();
        let expected = original.clone();
        assert_eq!(MLArray::from(original).extract_to_tensor::<i32>(), expected);
    }
}
