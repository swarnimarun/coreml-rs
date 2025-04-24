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

pub trait MLType {
    const TY: usize;
}

impl MLType for f32 {
    const TY: usize = 0;
}
impl MLType for half::f16 {
    const TY: usize = 1;
}
impl MLType for i32 {
    const TY: usize = 2;
}
impl MLType for u16 {
    const TY: usize = 3;
}
impl MLType for u8 {
    const TY: usize = 4;
}
impl MLType for i16 {
    const TY: usize = 5;
}
impl MLType for i8 {
    const TY: usize = 6;
}
impl MLType for u32 {
    const TY: usize = 7;
}
// impl MLType for i8 {
//     const TY: usize = 8;
// }

impl<T: MLType> From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>> for MLArray {
    fn from(value: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>) -> Self {
        unsafe {
            match T::TY {
                0 => MLArray::Float32Array(std::mem::transmute(value)),
                1 => MLArray::Float16Array(std::mem::transmute(value)),
                2 => MLArray::Int32Array(std::mem::transmute(value)),
                3 => MLArray::UInt16Array(std::mem::transmute(value)),
                4 => MLArray::UInt8Array(std::mem::transmute(value)),
                5 => MLArray::Int16Array(std::mem::transmute(value)),
                6 => MLArray::Int8Array(std::mem::transmute(value)),
                7 => MLArray::UInt32Array(std::mem::transmute(value)),
                _ => panic!("not supported"),
            }
        }
    }
}

impl MLArray {
    pub fn extract_to_tensor<T: MLType>(self) -> Array<T, Dim<IxDynImpl>> {
        unsafe {
            match self {
                MLArray::Float32Array(fm) => std::mem::transmute(fm),
                MLArray::Float16Array(fm) => std::mem::transmute(fm),

                MLArray::Int32Array(im) => std::mem::transmute(im),
                MLArray::Int16Array(im) => std::mem::transmute(im),
                MLArray::Int8Array(im) => std::mem::transmute(im),

                MLArray::UInt32Array(um) => std::mem::transmute(um),
                MLArray::UInt16Array(um) => std::mem::transmute(um),
                MLArray::UInt8Array(um) => std::mem::transmute(um),
            }
        }
    }
}
