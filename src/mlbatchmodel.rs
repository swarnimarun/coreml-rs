use crate::{
    ffi::{modelWithAssetsBatch, modelWithPathBatch, BatchModel},
    mlarray::MLArray,
    mlmodel::{CoreMLError, CoreMLModelInfo, CoreMLModelLoader},
    swift::MLBatchModelOutput,
    CoreMLModelOptions,
};
use flate2::Compression;
use ndarray::Array;
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempdir::TempDir;

pub use crate::swift::MLModelOutput;

#[derive(Debug)]
pub enum CoreMLBatchModelWithState {
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    Loaded(CoreMLBatchModel, CoreMLModelInfo, CoreMLModelLoader),
}

impl CoreMLBatchModelWithState {
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

    pub fn load(self) -> Result<Self, CoreMLError> {
        let Self::Unloaded(info, loader) = self else {
            return Ok(self);
        };
        match loader {
            CoreMLModelLoader::ModelPath(path_buf) => {
                let mut coreml_model = CoreMLBatchModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    false,
                );
                coreml_model.model.load();
                let loader = CoreMLModelLoader::ModelPath(path_buf);
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML model file",
                        Self::Unloaded(info, loader),
                    ));
                }
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::CompiledPath(path_buf) => {
                let mut coreml_model = CoreMLBatchModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    true,
                );
                coreml_model.model.load();
                let loader = CoreMLModelLoader::CompiledPath(path_buf);
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML model file",
                        Self::Unloaded(info, loader),
                    ));
                }
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::Buffer(vec) => {
                let mut coreml_model = CoreMLBatchModel::load_buffer(vec.clone(), info.clone());
                coreml_model.model.load();
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML mlmodel file",
                        Self::Unloaded(info, CoreMLModelLoader::Buffer(vec)),
                    ));
                }
                let loader = CoreMLModelLoader::Buffer(vec);
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::BufferToDisk(u) => {
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
                        let mut coreml_model = CoreMLBatchModel::load_buffer(vec, info.clone());
                        coreml_model.model.load();
                        let loader = CoreMLModelLoader::BufferToDisk(u);
                        Ok(Self::Loaded(coreml_model, info, loader))
                    }
                    Err(err) => Err(CoreMLError::FailedToBatchLoad(
                        format!("failed to load the model from cached buffer path: {err}"),
                        CoreMLBatchModelWithState::Unloaded(
                            info,
                            CoreMLModelLoader::BufferToDisk(u),
                        ),
                    )),
                }
            }
        }
    }

    /// Doesn't unload the model buffer in case model is loaded from a buffer
    pub fn unload(self) -> Result<Self, CoreMLError> {
        if let Self::Loaded(_, info, loader) = self {
            Ok(Self::Unloaded(
                info,
                match loader {
                    CoreMLModelLoader::Buffer(v) => {
                        let t = TempDir::new("coreml").map_err(CoreMLError::IoError)?;
                        _ = std::fs::remove_dir_all(&t);
                        _ = std::fs::create_dir_all(&t);
                        let path = t.path().join("mlmodel_cache");
                        std::fs::write(&path, v).unwrap();
                        CoreMLModelLoader::Buffer(
                            std::fs::read(&path).map_err(CoreMLError::IoError)?,
                        )
                    }
                    x => x,
                },
            ))
        } else {
            Ok(self)
        }
    }

    /// Unloads the model buffer to the disk, at cache_dir
    pub fn unload_to_disk(self) -> Result<Self, CoreMLError> {
        match self {
            Self::Loaded(_, mut info, loader) | Self::Unloaded(mut info, loader) => {
                let loader = {
                    match loader {
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
                                    return Err(CoreMLError::FailedToBatchLoad(
                                        format!("failed to load the model from the buffer: {err}"),
                                        CoreMLBatchModelWithState::Unloaded(
                                            info,
                                            CoreMLModelLoader::Buffer(vec),
                                        ),
                                    ));
                                }
                            };
                            CoreMLModelLoader::BufferToDisk(m)
                        }
                        loader => loader,
                    }
                };
                Ok(Self::Unloaded(info, loader))
            }
        }
    }

    pub fn description(&self) -> Result<HashMap<&str, Vec<String>>, CoreMLError> {
        match self {
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => Ok(core_mlmodel.description()),
        }
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
        idx: isize,
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => {
                core_mlmodel.add_input(tag, input, idx)
            }
        }
    }

    pub fn predict(&mut self) -> Result<MLBatchModelOutput, CoreMLError> {
        match self {
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict(),
        }
    }
}

#[derive(Debug)]
pub struct CoreMLBatchModel {
    model: BatchModel,
    // save_path: Option<PathBuf>,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
}

unsafe impl Send for CoreMLBatchModel {}

impl std::fmt::Debug for BatchModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchModel").finish()
    }
}

impl CoreMLBatchModel {
    pub fn load_from_path(path: String, info: CoreMLModelInfo, compiled: bool) -> Self {
        let coreml_model = Self {
            model: modelWithPathBatch(path, info.opts.compute_platform, compiled),
            // save_path: None,
            outputs: Default::default(),
        };
        coreml_model
    }

    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let coreml_model = Self {
            model: modelWithAssetsBatch(
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

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
        idx: isize,
    ) -> Result<(), CoreMLError> {
        // route input correctly
        let input: MLArray = input.into();
        let name = tag.as_ref().to_string();
        let desc = self.model.description();
        let shape: Vec<usize> = input.shape().to_vec();
        let arr = desc.input_shape(name.clone());
        if arr.len() != shape.len() || !arr.iter().eq(shape.iter()) {
            if arr.len() == 0 {
                return Err(CoreMLError::BadInputShape(format!(
                    "Input feature name '{name}' not expected!"
                )));
            }
            return Err(CoreMLError::BadInputShape(format!(
                "expected shape {arr:?} found {shape:?}"
            )));
        }
        match input {
            MLArray::Float32Array(array_base) => {
                let mut data = array_base.into_raw_vec();
                if !self
                    .model
                    .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity(), idx)
                {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind input to model",
                    ));
                }
                std::mem::forget(data);
            }
            _ => {
                return Err(CoreMLError::UnknownErrorStatic(
                    "failed to bind input to model",
                ));
            }
        }
        Ok(())
    }

    pub fn predict(&mut self) -> Result<MLBatchModelOutput, CoreMLError> {
        let desc = self.model.description();
        for name in desc.output_names() {
            let shape = desc.output_shape(name.clone());
            let ty = desc.output_type(name.clone());
            match ty.as_str() {
                "f32" => {
                    self.outputs.insert(name, ("f32", shape.to_vec()));
                }
                _ => {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "non-f32 output types are not supported (yet)!",
                    ))
                }
            }
        }

        let output = self.model.predict();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }
        let n = output.count();
        Ok(MLBatchModelOutput {
            outputs: (0..n).into_iter().map(|i|
                    (i, self
                        .outputs
                        .clone()
                        .into_iter())
                )
                .map(|(i, s)| {
                    let output = output.for_idx(i);
                    s.flat_map(|(key, (ty, shape))| {
                        if ty != "f32" {
                            eprintln!("warning: non-f32 types aren't supported, and will be skipped in the output");
                            return None;
                        }
                        let name = key.clone();
                        let out = output.outputF32(name);
                        let array = Array::from_shape_vec(shape, out).ok()?;
                        Some((key, array.into()))
                    })
                    .collect::<HashMap<String, MLArray>>()
                }).collect()
        })
    }

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        let desc = self.model.description();
        let mut map = HashMap::new();
        map.insert("input", desc.inputs());
        map.insert("output", desc.outputs());
        map
    }
}
