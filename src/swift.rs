#![allow(non_camel_case_types)]
use std::collections::HashMap;

use swift::ComputePlatform;

use crate::mlarray::MLArray;

#[swift_bridge::bridge]
pub mod swift {
    enum ComputePlatform {
        Cpu,
        CpuAndANE,
        CpuAndGpu,
    }
    extern "Swift" {
        #[swift_bridge(swift_name = "initWithPath")]
        pub fn modelWithPath(path: String, compute: ComputePlatform, compiled: bool) -> Model;
        #[swift_bridge(swift_name = "initWithCompiledAsset")]
        pub fn modelWithAssets(bytes: Vec<u8>, compute: ComputePlatform) -> Model;
        #[swift_bridge(swift_name = "initWithCompiledAssetBatch")]
        pub fn modelWithAssetsBatch(bytes: Vec<u8>, compute: ComputePlatform) -> BatchModel;
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
            data: Vec<f32>,
            idx: isize,
        ) -> bool;
        #[swift_bridge(swift_name = "hasFailedToLoad")]
        fn failed(&self) -> bool;

    }

    extern "Swift" {
        type Model;

        #[swift_bridge(swift_name = "setDisableExperimentalMLE")]
        fn setDisableExperimentalMLE(&mut self, disabled: bool);

        #[must_use()]
        fn bindOutputF32(&self, shape: Vec<i32>, featureName: String, data: Vec<f32>) -> bool;
        #[must_use()]
        fn bindInputF32(&self, shape: Vec<usize>, featureName: String, data: Vec<f32>) -> bool;
        #[must_use()]
        fn bindInputI32(&self, shape: Vec<usize>, featureName: String, data: Vec<i32>) -> bool;
        #[must_use()]
        fn bindInputU16(&self, shape: Vec<usize>, featureName: String, data: Vec<u16>) -> bool;

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

pub struct MLModelOutput {
    pub outputs: HashMap<String, MLArray>,
}

pub struct MLBatchModelOutput {
    pub outputs: Vec<HashMap<String, MLArray>>,
}
