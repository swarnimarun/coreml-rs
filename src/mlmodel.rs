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

#[derive(Default, Clone)]
pub struct CoreMLModelOptions {
    pub compute_platform: ComputePlatform,
    pub cache_dir: PathBuf,
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
pub enum CoreMLModelLoader {
    ModelPath(PathBuf),
    CompiledPath(PathBuf),
    Buffer(Vec<u8>),
    BufferPath(PathBuf),
}

#[derive(Debug)]
pub enum CoreMLModelWithState {
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    Loaded(CoreMLModel, CoreMLModelInfo, CoreMLModelLoader),
}

impl CoreMLModelWithState {
    pub fn new(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(
            CoreMLModelInfo { opts },
            CoreMLModelLoader::ModelPath(path.as_ref().to_path_buf()),
        )
    }
    pub fn new_compiled(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(
            CoreMLModelInfo { opts },
            CoreMLModelLoader::CompiledPath(path.as_ref().to_path_buf()),
        )
        // Self {
        //       path: None,
        //       save_path: None,
        //       model: Some(modelWithPath(
        //           path.as_ref().display().to_string(),
        //           opts.compute_platform,
        //           true,
        //       )),
        //       opts,
        //       outputs: Default::default(),
        //       loaded: false,
        //   }
    }

    pub fn from_buf(buf: Vec<u8>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(CoreMLModelInfo { opts }, CoreMLModelLoader::Buffer(buf))
    }

    pub fn load(self) -> Result<Self, Self> {
        if let Self::Unloaded(info, loader) = self {
            match loader {
                CoreMLModelLoader::ModelPath(path_buf) => {
                    // compile and load
                    todo!()
                }
                CoreMLModelLoader::CompiledPath(path_buf) => {
                    // assume compiled model path provided!
                    todo!()
                }
                CoreMLModelLoader::Buffer(vec) => {
                    println!("loading from buffer!");
                    if !info.opts.cache_dir.exists() || !info.opts.cache_dir.is_dir() {
                        _ = std::fs::remove_dir_all(&info.opts.cache_dir);
                        _ = std::fs::create_dir_all(&info.opts.cache_dir);
                        println!("creating directory for model cache!");
                    }
                    let m = info.opts.cache_dir.join("model_cache");
                    match std::fs::write(&m, &vec) {
                        Ok(_) => {}
                        Err(err) => {
                            println!("failed to write to model cache! {err}");
                            return Err(CoreMLModelWithState::Unloaded(
                                info,
                                CoreMLModelLoader::Buffer(vec),
                            ));
                        }
                    };
                    let mut coreml_model = CoreMLModel::load_buffer(vec, info.clone());
                    coreml_model.model.modelLoad();
                    let loader = CoreMLModelLoader::BufferPath(m);
                    Ok(Self::Loaded(coreml_model, info, loader))
                }
                CoreMLModelLoader::BufferPath(u) => {
                    let Ok(vec) = std::fs::read(&u) else {
                        return Err(CoreMLModelWithState::Unloaded(
                            info,
                            CoreMLModelLoader::BufferPath(u),
                        ));
                    };
                    let mut coreml_model = CoreMLModel::load_buffer(vec, info.clone());
                    coreml_model.model.modelLoad();
                    let loader = CoreMLModelLoader::BufferPath(u);
                    Ok(Self::Loaded(coreml_model, info, loader))
                }
            }
        } else {
            Ok(self)
        }
    }

    pub fn unload(self) -> Self {
        if let Self::Loaded(_, info, loader) = self {
            Self::Unloaded(info, loader)
        } else {
            self
        }
    }

    // pub fn from_buf_indirect(buf: &[u8], save_path: PathBuf, opts: CoreMLModelOptions) -> Self {
    //     let _ = std::fs::write(&save_path, buf);
    //     let mut m = Self::new_compiled(&save_path, opts);
    //     m.save_path = Some(save_path);
    //     m
    // }

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Default::default(),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.description(),
        }
    }

    pub fn add_input(&mut self, tag: impl AsRef<str>, input: impl Into<MLArray>) -> bool {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => false,
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                core_mlmodel.add_input(tag, input);
                true
            }
        }
    }

    pub fn predict(&mut self) -> Result<MLModelOutput, ()> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(()),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict().ok_or(()),
        }
    }
}

// Info required to create a coreml model
#[derive(Debug, Clone)]
pub struct CoreMLModelInfo {
    opts: CoreMLModelOptions,
}

#[derive(Debug)]
pub struct CoreMLModel {
    model: Model,
    save_path: Option<PathBuf>,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
}

unsafe impl Send for CoreMLModel {}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

// impl Drop for CoreMLModel {
//     fn drop(&mut self) {
//         if let Some(save_path) = &self.save_path {
//             _ = std::fs::remove_dir_all(save_path);
//         }
//     }
// }

impl CoreMLModel {
    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let coreml_model = Self {
            model: modelWithAssets(
                buf.as_mut_ptr(),
                buf.len() as isize,
                info.opts.compute_platform,
            ),
            save_path: None,
            outputs: Default::default(),
        };
        std::mem::forget(buf);
        coreml_model
    }

    pub fn add_input(&mut self, tag: impl AsRef<str>, input: impl Into<MLArray>) {
        // route input correctly
        let input: MLArray = input.into();
        let name = tag.as_ref().to_string();
        let shape = input.shape().into_iter().map(|s| *s as i32).collect();
        if input.is_f32() {
            let mut data = input.into_raw_vec_f32();
            self.model
                .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity());
            std::mem::forget(data);
        } else if input.is_f16() {
            let mut data = input.into_raw_vec_u16();
            self.model
                .bindInputU16(shape, name, data.as_mut_ptr(), data.capacity());
            std::mem::forget(data);
        } else if input.is_i32() {
            let mut data = input.into_raw_vec_i32();
            self.model
                .bindInputI32(shape, name, data.as_mut_ptr(), data.capacity());
            std::mem::forget(data);
        } else {
            panic!("unreachable!")
        }
    }

    pub fn add_output_f32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) {
        let arr: MLArray = out.into();
        let shape = arr.shape();
        self.outputs
            .insert(tag.as_ref().to_string(), ("f32", shape.to_vec()));
        let shape: Vec<i32> = shape.into_iter().map(|i| *i as i32).collect();
        let mut data = arr.into_raw_vec_f32();
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        self.model.bindOutputF32(shape, name, ptr, len);
        std::mem::forget(data);
    }

    pub fn predict(&mut self) -> Option<MLModelOutput> {
        let desc = self.model.modelDescription();
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
        let output = self.model.modelRun();
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

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        let desc = self.model.modelDescription();
        let mut map = HashMap::new();
        map.insert("input", desc.inputs());
        map.insert("output", desc.outputs());
        map
    }
}
