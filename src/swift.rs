use std::collections::HashMap;

use swift::{BatchOutput, ComputePlatform, ModelOutput};

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
        fn rust_vec_free_u8(ptr: *mut u8, len: usize);
    }

    extern "Swift" {
        #[swift_bridge(swift_name = "initWithPath")]
        pub fn modelWithPath(path: String, compute: ComputePlatform, compiled: bool) -> Model;
        #[swift_bridge(swift_name = "initWithCompiledAsset")]
        pub fn modelWithAssets(ptr: *mut u8, len: isize, compute: ComputePlatform) -> Model;
        #[swift_bridge(swift_name = "initWithCompiledAssetBatch")]
        pub fn modelWithAssetsBatch(
            ptr: *mut u8,
            len: isize,
            compute: ComputePlatform,
        ) -> BatchModel;
        #[swift_bridge(swift_name = "initWithPathBatch")]
        pub fn modelWithPathBatch(
            path: String,
            compute: ComputePlatform,
            compiled: bool,
        ) -> BatchModel;
    }

    extern "Swift" {
        type BatchOutput;

        #[swift_bridge(swift_name = "getOutputAtIndex")]
        pub fn for_idx(&self, at: isize) -> ModelOutput;
        pub fn getError(&self) -> Option<String>;
        pub fn count(&self) -> isize;
    }

    extern "Swift" {
        type BatchModel;

        fn load(&mut self) -> bool;
        fn unload(&mut self) -> bool;
        fn description(&self) -> ModelDescription;
        fn predict(&self) -> BatchOutput;
        fn bindInputF32(
            &self,
            shape: Vec<usize>,
            featureName: String,
            data: *mut f32,
            len: usize,
            idx: isize,
        ) -> bool;
        #[swift_bridge(swift_name = "hasFailedToLoad")]
        fn failed(&self) -> bool;

    }

    extern "Swift" {
        type Model;

        #[must_use()]
        fn bindOutputF32(
            &self,
            shape: Vec<i32>,
            featureName: String,
            data: *mut f32,
            len: usize,
        ) -> bool;
        #[must_use()]
        fn bindInputF32(
            &self,
            shape: Vec<usize>,
            featureName: String,
            data: *mut f32,
            len: usize,
        ) -> bool;
        #[must_use()]
        fn bindInputI32(
            &self,
            shape: Vec<usize>,
            featureName: String,
            data: *mut i32,
            len: usize,
        ) -> bool;
        #[must_use()]
        fn bindInputU16(
            &self,
            shape: Vec<usize>,
            featureName: String,
            data: *mut u16,
            len: usize,
        ) -> bool;

        #[swift_bridge(swift_name = "getCompiledPath")]
        fn compiled_path(&self) -> Option<String>;

        fn load(&mut self) -> bool;
        fn unload(&mut self) -> bool;
        fn description(&self) -> ModelDescription;
        fn predict(&self) -> ModelOutput;
        #[swift_bridge(swift_name = "hasFailedToLoad")]
        fn failed(&self) -> bool;
    }

    extern "Swift" {
        type ModelDescription;

        fn inputs(&self) -> Vec<String>;
        fn outputs(&self) -> Vec<String>;
        fn output_names(&self) -> Vec<String>;
        fn output_type(&self, name: String) -> String;
        fn output_shape(&self, name: String) -> Vec<usize>;
        fn input_shape(&self, name: String) -> Vec<usize>;
    }

    extern "Swift" {
        type ModelOutput;

        fn outputDescription(&self) -> Vec<String>;
        fn outputF32(&self, name: String) -> Vec<f32>;
        fn outputU16(&self, name: String) -> Vec<u16>;
        fn outputI32(&self, name: String) -> Vec<i32>;
        fn getError(&self) -> Option<String>;
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

fn rust_vec_free_u8(ptr: *mut u8, len: usize) {
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
    pub outputs: HashMap<String, MLArray>,
}

pub struct MLBatchModelOutput {
    pub outputs: Vec<HashMap<String, MLArray>>,
}
