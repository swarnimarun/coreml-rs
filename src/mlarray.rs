use half::f16;
use ndarray::{Array2, Array3, Array4, Array5, Array6};

#[derive(Clone, Debug)]
pub enum FloatMLArray {
    Array2(Array2<f32>),
    Array3(Array3<f32>),
    Array4(Array4<f32>),
    Array5(Array5<f32>),
    Array6(Array6<f32>),
}

#[derive(Clone, Debug)]
pub enum Float16MLArray {
    Array2(Array2<f16>),
    Array3(Array3<f16>),
    Array4(Array4<f16>),
    Array5(Array5<f16>),
    Array6(Array6<f16>),
}

#[derive(Clone, Debug)]
pub enum MLArray {
    FloatArray(FloatMLArray),
    // Float16Array(Float16MLArray),
}

impl MLArray {
    pub fn into_raw_vec(&self) -> Vec<f32> {
        match self {
            MLArray::FloatArray(FloatMLArray::Array2(ab)) => ab.clone().into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array3(ab)) => ab.clone().into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array4(ab)) => ab.clone().into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array5(ab)) => ab.clone().into_raw_vec(),
            MLArray::FloatArray(FloatMLArray::Array6(ab)) => ab.clone().into_raw_vec(),
        }
    }
    pub fn shape(&self) -> &[usize] {
        match self {
            MLArray::FloatArray(FloatMLArray::Array2(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array3(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array4(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array5(ab)) => ab.shape(),
            MLArray::FloatArray(FloatMLArray::Array6(ab)) => ab.shape(),
        }
    }
}

impl From<Array2<f32>> for MLArray {
    fn from(value: Array2<f32>) -> Self {
        MLArray::FloatArray(FloatMLArray::Array2(value))
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
