use std::collections::HashMap;

use swift::{ComputePlatform, ModelOutput};

use crate::mlarray::MLArray;

#[swift_bridge::bridge]
pub mod swift {
    enum ComputePlatform {
        Cpu,
        CpuAndANE,
        CpuAndGpu,
    }
    extern "Rust" {
        fn rust_vec_from_ptr_i32(ptr: *mut i32, len: usize) -> Vec<i32>;
        fn rust_vec_from_ptr_f32(ptr: *mut f32, len: usize) -> Vec<f32>;
        fn rust_vec_from_ptr_u16(ptr: *mut u16, len: usize) -> Vec<u16>;
        fn rust_vec_free_f32(ptr: *mut f32, len: usize);
        fn rust_vec_free_i32(ptr: *mut i32, len: usize);
        fn rust_vec_free_u16(ptr: *mut u16, len: usize);
    }

    extern "Swift" {
        type Model;

        #[swift_bridge(init)]
        fn compileModel(path: String, compute: ComputePlatform) -> Model;
        fn bindOutputF32(&self, shape: Vec<i32>, featureName: String, data: *mut f32, len: usize);
        fn bindInputF32(&self, shape: Vec<i32>, featureName: String, data: *mut f32, len: usize);
        fn bindInputI32(&self, shape: Vec<i32>, featureName: String, data: *mut i32, len: usize);
        fn bindInputU16(&self, shape: Vec<i32>, featureName: String, data: *mut u16, len: usize);
        #[swift_bridge(swift_name = "load")]
        fn modelLoad(&mut self);
        #[swift_bridge(swift_name = "description")]
        fn modelDescription(&self) -> ModelDescription;
        #[swift_bridge(swift_name = "predict")]
        fn modelRun(&self) -> ModelOutput;
    }

    extern "Swift" {
        type ModelDescription;

        fn inputs(&self) -> Vec<String>;
        fn outputs(&self) -> Vec<String>;
        fn output_names(&self) -> Vec<String>;
        fn output_type(&self, name: String) -> String;
        fn output_shape(&self, name: String) -> Vec<usize>;
    }

    extern "Swift" {
        type ModelOutput;

        fn outputDescription(&self) -> Vec<String>;
        fn outputF32(&self, name: String) -> Vec<f32>;
        fn outputU16(&self, name: String) -> Vec<u16>;
        fn outputI32(&self, name: String) -> Vec<i32>;
    }
}

impl std::default::Default for ComputePlatform {
    fn default() -> Self {
        ComputePlatform::CpuAndGpu
    }
}

/// performs a memcpy
fn rust_vec_from_ptr_f32(ptr: *mut f32, len: usize) -> Vec<f32> {
    unsafe { Vec::from_raw_parts(ptr, len, len) }
    // unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}
/// performs a memcpy
fn rust_vec_from_ptr_u16(ptr: *mut u16, len: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}
/// performs a memcpy
fn rust_vec_from_ptr_i32(ptr: *mut i32, len: usize) -> Vec<i32> {
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn rust_vec_free_f32(ptr: *mut f32, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}
fn rust_vec_free_u16(ptr: *mut u16, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}
fn rust_vec_free_i32(ptr: *mut i32, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}

pub struct MLModelOutput {
    pub model_output: ModelOutput,
    pub outputs: HashMap<String, MLArray>,
}
