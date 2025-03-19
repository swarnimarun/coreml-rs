use crate::{
    ffi::{modelWithAssets, ComputePlatform, Model},
    mlarray::MLArray,
};
use flate2::Compression;
use ndarray::Array;
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};
// use lz4_flex::block::{compress_prepend_size, decompress_size_prepended, DecompressError};

pub use crate::swift::MLModelOutput;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreMLError {
    #[error("IoError: {0}")]
    IoError(std::io::Error),
    // #[error("Lz4 Decompression Error: {0}")]
    // Lz4DecompressError(DecompressError),
    #[error("UnknownError: {0}")]
    UnknownError(String),
    #[error("UnknownError: {0}")]
    UnknownErrorStatic(&'static str),
}

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
    }

    pub fn from_buf(buf: Vec<u8>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(CoreMLModelInfo { opts }, CoreMLModelLoader::Buffer(buf))
    }

    pub fn load(self) -> Result<Self, Self> {
        let Self::Unloaded(mut info, loader) = self else {
            return Ok(self);
        };
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
                if info.opts.cache_dir.as_os_str().is_empty() {
                    info.opts.cache_dir = PathBuf::from(".");
                }
                if !info.opts.cache_dir.exists() {
                    _ = std::fs::remove_dir_all(&info.opts.cache_dir);
                    _ = std::fs::create_dir_all(&info.opts.cache_dir);
                }
                // pick the file specified, if it's a folder/dir append model_cache
                let m = if !info.opts.cache_dir.is_dir() {
                    info.opts.cache_dir.clone()
                } else {
                    info.opts.cache_dir.join("model_cache")
                };
                match std::fs::File::create(&m)
                    .map_err(|io| CoreMLError::IoError(io))
                    .map(|file| {
                        flate2::write::ZlibEncoder::new(file, Compression::best())
                            .write_all(&vec)
                            .map_err(CoreMLError::IoError)
                    }) {
                    Ok(_) => {}
                    Err(err) => {
                        eprintln!("failed to load the model from the buffer: {err}");
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
                match std::fs::File::open(&u)
                    .map_err(|io| CoreMLError::IoError(io))
                    .and_then(|file| {
                        let mut vec = vec![];
                        _ = flate2::read::ZlibDecoder::new(file)
                            .read_to_end(&mut vec)
                            .map_err(|io| CoreMLError::IoError(io))?;
                        Ok(vec)
                    }) {
                    Ok(vec) => {
                        let mut coreml_model = CoreMLModel::load_buffer(vec, info.clone());
                        coreml_model.model.modelLoad();
                        let loader = CoreMLModelLoader::BufferPath(u);
                        Ok(Self::Loaded(coreml_model, info, loader))
                    }
                    Err(err) => {
                        eprintln!("failed to load the model from cached buffer: {err}");
                        Err(CoreMLModelWithState::Unloaded(
                            info,
                            CoreMLModelLoader::BufferPath(u),
                        ))
                    }
                }
            }
        }
    }

    pub fn unload(self) -> Self {
        if let Self::Loaded(_, info, loader) = self {
            Self::Unloaded(info, loader)
        } else {
            self
        }
    }

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
    // save_path: Option<PathBuf>,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
}

unsafe impl Send for CoreMLModel {}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

impl CoreMLModel {
    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let coreml_model = Self {
            model: modelWithAssets(
                buf.as_mut_ptr(),
                buf.len() as isize,
                info.opts.compute_platform,
            ),
            // save_path: None,
            outputs: Default::default(),
        };
        std::mem::forget(buf);
        coreml_model
    }

    pub fn add_input(&mut self, tag: impl AsRef<str>, input: impl Into<MLArray>) -> bool {
        // route input correctly
        let input: MLArray = input.into();
        let name = tag.as_ref().to_string();
        let shape = input.shape().into_iter().map(|s| *s as i32).collect();
        if input.is_f32() {
            let mut data = input.into_raw_vec_f32();
            if !self
                .model
                .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity())
            {
                return false;
            }
            std::mem::forget(data);
        } else if input.is_f16() {
            let mut data = input.into_raw_vec_u16();
            if !self
                .model
                .bindInputU16(shape, name, data.as_mut_ptr(), data.capacity())
            {
                return false;
            }
            std::mem::forget(data);
        } else if input.is_i32() {
            let mut data = input.into_raw_vec_i32();
            if !self
                .model
                .bindInputI32(shape, name, data.as_mut_ptr(), data.capacity())
            {
                return false;
            }
            std::mem::forget(data);
        } else {
            panic!("unreachable!")
        }
        true
    }

    pub fn add_output_f32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        let arr: MLArray = out.into();
        let shape = arr.shape();
        self.outputs
            .insert(tag.as_ref().to_string(), ("f32", shape.to_vec()));
        let shape: Vec<i32> = shape.into_iter().map(|i| *i as i32).collect();
        let mut data = arr.into_raw_vec_f32();
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        if !self.model.bindOutputF32(shape, name, ptr, len) {
            return false;
        }
        std::mem::forget(data);
        true
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
                .filter_map(|(key, (ty, shape))| {
                    if ty != "f32" {
                        eprintln!("warning: non-f32 types aren't supported, and will be skipped in the output");
                        return None;
                    }
                    let name = key.clone();
                    let out = output.outputF32(name);
                    let array = Array::from_shape_vec(shape, out).ok()?;
                    Some((key, array.into()))
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
