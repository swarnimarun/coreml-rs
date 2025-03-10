use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use ndarray::Array;

use crate::{
    ffi::{modelWithAssets, ComputePlatform, Model},
    mlarray::MLArray,
    swift::modelWithPath,
};

pub use crate::swift::MLModelOutput;

#[derive(Default)]
pub struct CoreMLModelOptions {
    pub compute_platform: ComputePlatform,
}

impl std::fmt::Debug for CoreMLModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMLModelOptions")
            .field(
                "compute_platform",
                match self.compute_platform {
                    ComputePlatform::Cpu => &"CPU",
                    ComputePlatform::CpuAndANE => &"CpuAndAne",
                    ComputePlatform::CpuAndGpu => &"CpuAndGpu",
                },
            )
            .finish()
    }
}

#[derive(Debug)]
pub struct CoreMLModel {
    model: Option<Model>,
    path: Option<PathBuf>,
    opts: CoreMLModelOptions,
    save_path: Option<PathBuf>,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
    loaded: bool,
}

unsafe impl Send for CoreMLModel {}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

impl Drop for CoreMLModel {
    fn drop(&mut self) {
        if let Some(save_path) = &self.save_path {
            _ = std::fs::remove_dir_all(save_path);
        }
    }
}

impl CoreMLModel {
    pub fn new(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self {
            model: None,
            path: Some(path.as_ref().to_path_buf()),
            save_path: None,
            opts,
            outputs: Default::default(),
            loaded: false,
        }
    }
    pub fn new_compiled(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self {
            path: None,
            save_path: None,
            model: Some(modelWithPath(
                path.as_ref().display().to_string(),
                opts.compute_platform,
                true,
            )),
            opts,
            outputs: Default::default(),
            loaded: false,
        }
    }

    pub fn from_buf(mut buf: Vec<u8>, opts: CoreMLModelOptions) -> Self {
        let m = Self {
            path: None,
            save_path: None,
            model: Some(modelWithAssets(
                buf.as_mut_ptr(),
                buf.len() as isize,
                opts.compute_platform,
            )),
            opts,
            outputs: Default::default(),
            loaded: false,
        };
        std::mem::forget(buf);
        m
    }

    pub fn from_buf_indirect(buf: &[u8], save_path: PathBuf, opts: CoreMLModelOptions) -> Self {
        let _ = std::fs::write(&save_path, buf);
        let mut m = Self::new_compiled(&save_path, opts);
        m.save_path = Some(save_path);
        m
    }

    pub fn add_input(&mut self, tag: impl AsRef<str>, input: impl Into<MLArray>) {
        // route input correctly
        let input: MLArray = input.into();
        if input.is_f32() {
            self.add_input_f32(tag, input);
        } else if input.is_f16() {
            self.add_input_f16(tag, input);
        } else if input.is_i32() {
            self.add_input_i32(tag, input);
        } else {
            panic!("unreachable!")
        }
    }

    pub fn add_input_i32(&mut self, tag: impl AsRef<str>, input: impl Into<MLArray>) {
        debug_assert!(
            self.model.is_some(),
            "ensure model is compiled & loaded; before adding inputs"
        );
        let v: MLArray = input.into();
        let shape = v.shape().into_iter().map(|s| *s as i32).collect();
        let mut data = v.into_raw_vec_i32();
        let name = tag.as_ref().to_string();
        self.model
            .as_mut()
            .unwrap()
            .bindInputI32(shape, name, data.as_mut_ptr(), data.capacity());
        std::mem::forget(data);
    }

    pub fn add_input_f32(&mut self, tag: impl AsRef<str>, input: impl Into<MLArray>) {
        debug_assert!(
            self.model.is_some(),
            "ensure model is compiled & loaded; before adding inputs"
        );
        let v: MLArray = input.into();
        let shape = v.shape().into_iter().map(|s| *s as i32).collect();
        let mut data = v.into_raw_vec_f32();
        let name = tag.as_ref().to_string();
        self.model
            .as_mut()
            .unwrap()
            .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity());
        std::mem::forget(data);
    }

    pub fn add_input_f16(&mut self, tag: impl AsRef<str>, input: impl Into<MLArray>) {
        debug_assert!(
            self.model.is_some(),
            "ensure model is compiled & loaded; before adding inputs"
        );
        let v: MLArray = input.into();
        let shape = v.shape().into_iter().map(|s| *s as i32).collect();
        let mut data = v.into_raw_vec_u16();
        let name = tag.as_ref().to_string();
        self.model
            .as_mut()
            .unwrap()
            .bindInputU16(shape, name, data.as_mut_ptr(), data.capacity());
        std::mem::forget(data);
    }

    pub fn compile(&mut self) {
        self.model = Some(modelWithPath(
            self.path.as_ref().unwrap().display().to_string(),
            self.opts.compute_platform,
            false,
        ));
    }

    pub fn add_output_f32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) {
        debug_assert!(
            self.model.is_some(),
            "ensure model is compiled & loaded; before adding inputs"
        );
        let arr: MLArray = out.into();
        let shape = arr.shape();
        self.outputs
            .insert(tag.as_ref().to_string(), ("f32", shape.to_vec()));

        let shape: Vec<i32> = shape.into_iter().map(|i| *i as i32).collect();
        let mut data = arr.into_raw_vec_f32();
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        self.model
            .as_mut()
            .unwrap()
            .bindOutputF32(shape, name, ptr, len);
        std::mem::forget(data);
    }

    pub fn predict(&mut self) -> Option<MLModelOutput> {
        let desc = self.model.as_ref().unwrap().modelDescription();
        for name in desc.output_names() {
            let output_shape = desc.output_shape(name.clone());
            let ty = desc.output_type(name.clone());
            match ty.as_str() {
                "f32" => {
                    self.add_output_f32(name, Array::<f32, _>::zeros(output_shape));
                }
                _ => panic!("not supported"),
            }
        }
        let output = self.model.as_ref().unwrap().modelRun();
        Some(MLModelOutput {
            outputs: self
                .outputs
                .clone()
                .into_iter()
                .map(|(key, (ty, shape))| {
                    assert_eq!(ty, "f32", "non f32 types are currently not supported");
                    let name = key.clone();
                    let out = output.outputF32(name);
                    let array = Array::from_shape_vec(shape, out).unwrap();
                    (key, array.into())
                })
                .collect(),
            model_output: output,
        })
    }

    pub fn load(&mut self) {
        if let Some(model) = &mut self.model {
            model.modelLoad();
        }
        self.loaded = true;
    }

    pub fn unload(&mut self) {
        if let Some(model) = &mut self.model {
            model.modelUnload();
        }
        self.loaded = false;
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        let desc = self.model.as_ref().unwrap().modelDescription();
        let mut map = HashMap::new();
        map.insert("input", desc.inputs());
        map.insert("output", desc.outputs());
        map
    }
}
