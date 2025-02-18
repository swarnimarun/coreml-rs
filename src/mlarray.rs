use half::{f16, vec::HalfFloatVecExt};
use ndarray::{
    Array, Array2, Array3, Array4, Array5, Array6, ArrayBase, Dim, IxDynImpl, OwnedRepr,
};

#[derive(Debug)]
pub enum FloatMLArray {
    Array(ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>),
    Array2(Array2<f32>),
    Array3(Array3<f32>),
    Array4(Array4<f32>),
    Array5(Array5<f32>),
    Array6(Array6<f32>),
}

#[derive(Debug)]
pub enum Int32MLArray {
    Array2(Array2<i32>),
    Array3(Array3<i32>),
    Array4(Array4<i32>),
    Array5(Array5<i32>),
    Array6(Array6<i32>),
}

#[derive(Debug)]
pub enum Float16MLArray {
    Array2(Array2<f16>),
    Array3(Array3<f16>),
    Array4(Array4<f16>),
    Array5(Array5<f16>),
    Array6(Array6<f16>),
}

#[derive(Debug)]
pub enum MLArray {
    FloatArray(FloatMLArray),
    Float16Array(Float16MLArray),
    Int32Array(Int32MLArray),
}

impl MLArray {
    pub fn is_f32(&self) -> bool {
        matches!(self, MLArray::FloatArray(_))
    }
    pub fn is_f16(&self) -> bool {
        matches!(self, MLArray::Float16Array(_))
    }
    pub fn is_i32(&self) -> bool {
        matches!(self, MLArray::Int32Array(_))
    }
    pub fn into_raw_vec_f32(self) -> Vec<f32> {
        match self {
            MLArray::FloatArray(FloatMLArray::Array(ab)) => ab.into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array2(ab)) => ab.into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array3(ab)) => ab.into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array4(ab)) => ab.into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array5(ab)) => ab.into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array6(ab)) => ab.into_raw_vec(),
            _ => {
                panic!("f32 not supported")
            }
        }
    }
    pub fn into_raw_vec_u16(&self) -> Vec<u16> {
        match self {
            MLArray::Float16Array(Float16MLArray::Array2(ab)) => {
                (ab.as_slice().unwrap()).to_vec().reinterpret_into()
            }
            MLArray::Float16Array(Float16MLArray::Array3(ab)) => {
                (ab.as_slice().unwrap()).to_vec().reinterpret_into()
            }
            MLArray::Float16Array(Float16MLArray::Array4(ab)) => {
                (ab.as_slice().unwrap()).to_vec().reinterpret_into()
            }
            MLArray::Float16Array(Float16MLArray::Array5(ab)) => {
                (ab.as_slice().unwrap()).to_vec().reinterpret_into()
            }
            MLArray::Float16Array(Float16MLArray::Array6(ab)) => {
                (ab.as_slice().unwrap()).to_vec().reinterpret_into()
            }
            _ => {
                panic!("f16 not supported")
            }
        }
    }
    pub fn into_raw_vec_i32(self) -> Vec<i32> {
        match self {
            MLArray::Int32Array(Int32MLArray::Array2(ab)) => ab.into_raw_vec(),
            MLArray::Int32Array(Int32MLArray::Array3(ab)) => ab.into_raw_vec(),
            MLArray::Int32Array(Int32MLArray::Array4(ab)) => ab.into_raw_vec(),
            MLArray::Int32Array(Int32MLArray::Array5(ab)) => ab.into_raw_vec(),
            MLArray::Int32Array(Int32MLArray::Array6(ab)) => ab.into_raw_vec(),
            _ => {
                panic!("i32 not supported")
            }
        }
    }
    pub fn shape(&self) -> &[usize] {
        match self {
            MLArray::FloatArray(FloatMLArray::Array(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array2(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array3(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array4(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array5(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array6(ab)) => ab.shape(),

            MLArray::Int32Array(Int32MLArray::Array2(ab)) => ab.shape(),
            MLArray::Int32Array(Int32MLArray::Array3(ab)) => ab.shape(),
            MLArray::Int32Array(Int32MLArray::Array4(ab)) => ab.shape(),
            MLArray::Int32Array(Int32MLArray::Array5(ab)) => ab.shape(),
            MLArray::Int32Array(Int32MLArray::Array6(ab)) => ab.shape(),

            MLArray::Float16Array(Float16MLArray::Array2(ab)) => ab.shape(),
            MLArray::Float16Array(Float16MLArray::Array3(ab)) => ab.shape(),
            MLArray::Float16Array(Float16MLArray::Array4(ab)) => ab.shape(),
            MLArray::Float16Array(Float16MLArray::Array5(ab)) => ab.shape(),
            MLArray::Float16Array(Float16MLArray::Array6(ab)) => ab.shape(),
        }
    }
}

impl From<Array3<f32>> for MLArray {
    fn from(value: Array3<f32>) -> Self {
        MLArray::FloatArray(FloatMLArray::Array3(value))
    }
}
impl From<Array4<f32>> for MLArray {
    fn from(value: Array4<f32>) -> Self {
        MLArray::FloatArray(FloatMLArray::Array4(value))
    }
}
impl From<Array5<f32>> for MLArray {
    fn from(value: Array5<f32>) -> Self {
        MLArray::FloatArray(FloatMLArray::Array5(value))
    }
}
impl From<Array6<f32>> for MLArray {
    fn from(value: Array6<f32>) -> Self {
        MLArray::FloatArray(FloatMLArray::Array6(value))
    }
}
impl From<Array3<i32>> for MLArray {
    fn from(value: Array3<i32>) -> Self {
        MLArray::Int32Array(Int32MLArray::Array3(value))
    }
}
impl From<Array4<i32>> for MLArray {
    fn from(value: Array4<i32>) -> Self {
        MLArray::Int32Array(Int32MLArray::Array4(value))
    }
}
impl From<Array5<i32>> for MLArray {
    fn from(value: Array5<i32>) -> Self {
        MLArray::Int32Array(Int32MLArray::Array5(value))
    }
}
impl From<Array6<i32>> for MLArray {
    fn from(value: Array6<i32>) -> Self {
        MLArray::Int32Array(Int32MLArray::Array6(value))
    }
}
impl From<Array3<f16>> for MLArray {
    fn from(value: Array3<f16>) -> Self {
        MLArray::Float16Array(Float16MLArray::Array3(value))
    }
}
impl From<Array4<f16>> for MLArray {
    fn from(value: Array4<f16>) -> Self {
        MLArray::Float16Array(Float16MLArray::Array4(value))
    }
}
impl From<Array5<f16>> for MLArray {
    fn from(value: Array5<f16>) -> Self {
        MLArray::Float16Array(Float16MLArray::Array5(value))
    }
}
impl From<Array6<f16>> for MLArray {
    fn from(value: Array6<f16>) -> Self {
        MLArray::Float16Array(Float16MLArray::Array6(value))
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

impl<T: MLType> From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>> for MLArray {
    fn from(value: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>) -> Self {
        unsafe {
            match T::TY {
                0 => MLArray::FloatArray(FloatMLArray::Array(std::mem::transmute(value))),
                _ => panic!("not supported"),
            }
        }
    }
}

impl<T: MLType> From<Array2<T>> for MLArray {
    fn from(value: Array2<T>) -> Self {
        unsafe {
            match T::TY {
                0 => MLArray::FloatArray(FloatMLArray::Array2(std::mem::transmute(value))),
                1 => MLArray::Float16Array(Float16MLArray::Array2(std::mem::transmute(value))),
                2 => MLArray::Int32Array(Int32MLArray::Array2(std::mem::transmute(value))),
                _ => panic!("not supported"),
            }
        }
    }
}

impl FloatMLArray {
    pub fn extract(self) -> Array<f32, Dim<IxDynImpl>> {
        match self {
            FloatMLArray::Array(array_base) => array_base,
            _ => todo!(),
        }
    }
}

impl Float16MLArray {
    pub fn extract(self) -> Array<f32, Dim<IxDynImpl>> {
        match self {
            _ => todo!(),
        }
    }
}
impl Int32MLArray {
    pub fn extract(self) -> Array<f32, Dim<IxDynImpl>> {
        match self {
            _ => todo!(),
        }
    }
}

impl MLArray {
    pub fn extract_to_tensor<T: MLType>(self) -> Array<T, Dim<IxDynImpl>> {
        unsafe {
            match self {
                MLArray::FloatArray(fm) => std::mem::transmute(fm.extract()),
                MLArray::Float16Array(fm) => std::mem::transmute(fm.extract()),
                MLArray::Int32Array(im) => std::mem::transmute(im.extract()),
            }
        }
    }
}
