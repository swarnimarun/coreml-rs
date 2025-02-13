use std::{
    collections::HashMap,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use ndarray::Array;

use crate::{
    ffi::{ComputePlatform, Model},
    mlarray::MLArray,
    swift::MLModelOutput,
};

pub struct Store(Arc<Mutex<Vec<MLArray>>>);

impl Store {
    pub fn empty_ref<'a>(&'a self) -> CoreMLInputRef<'a> {
        CoreMLInputRef(0, self)
    }
    pub fn new() -> Self {
        Store(Arc::new(Mutex::new(vec![])))
    }
    pub fn insert(&self, d: impl Into<MLArray>) -> usize {
        let mut guard = self.0.lock().unwrap();
        guard.push(d.into());
        guard.len() - 1
    }
    pub fn bind_ref<'a>(&'a self, idx: usize) -> CoreMLInputRef<'a> {
        CoreMLInputRef(idx, self)
    }
    pub fn get(
        &self,
    ) -> Result<
        std::sync::MutexGuard<'_, Vec<MLArray>>,
        std::sync::PoisonError<std::sync::MutexGuard<'_, Vec<MLArray>>>,
    > {
        self.0.lock()
    }
}

pub struct CoreMLInputRef<'a>(usize, &'a Store);

impl<'a> CoreMLInputRef<'a> {
    pub fn input_data_f32(&self) -> Option<(Vec<i32>, Vec<f32>)> {
        let Ok(mut v) = self.1.get() else {
            return None;
        };
        let v = v.remove(self.0);
        Some((
            v.shape().into_iter().map(|s| *s as i32).collect(),
            v.into_raw_vec_f32(),
        ))
    }
    pub fn input_data_f16(&self) -> Option<(Vec<i32>, Vec<u16>)> {
        let Ok(mut v) = self.1.get() else {
            return None;
        };
        let v = v.remove(self.0);
        Some((
            v.shape().into_iter().map(|s| *s as i32).collect(),
            v.into_raw_vec_u16(),
        ))
    }
    pub fn input_data_i32(&self) -> Option<(Vec<i32>, Vec<i32>)> {
        let Ok(mut v) = self.1.get() else {
            return None;
        };
        let v = v.remove(self.0);
        Some((
            v.shape().into_iter().map(|s| *s as i32).collect(),
            v.into_raw_vec_i32(),
        ))
    }
}

#[derive(Default)]
pub struct CoreMLModelOptions {
    pub compute_platform: ComputePlatform,
}

pub struct CoreMLModel<'a> {
    model: Option<Model>,
    path: PathBuf,
    opts: CoreMLModelOptions,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
    _p: PhantomData<CoreMLInputRef<'a>>,
}

impl<'a> CoreMLModel<'a> {
    pub fn new(path: impl AsRef<Path>, _: &'a Store, opts: CoreMLModelOptions) -> Self {
        Self {
            model: None,
            path: path.as_ref().to_path_buf(),
            _p: PhantomData::default(),
            opts,
            outputs: Default::default(),
        }
    }

    pub fn add_input_i32(&mut self, tag: impl AsRef<str>, input: CoreMLInputRef<'a>) {
        debug_assert!(
            self.model.is_some(),
            "ensure model is compiled & loaded; before adding inputs"
        );
        let Some((shape, mut data)) = input.input_data_i32() else {
            panic!("i32 welp")
        };
        let name = tag.as_ref().to_string();
        self.model
            .as_mut()
            .unwrap()
            .bindInputI32(shape, name, data.as_mut_ptr(), data.capacity());
        std::mem::forget(data);
    }

    pub fn add_input_f32(&mut self, tag: impl AsRef<str>, input: CoreMLInputRef<'a>) {
        debug_assert!(
            self.model.is_some(),
            "ensure model is compiled & loaded; before adding inputs"
        );
        let Some((shape, mut data)) = input.input_data_f32() else {
            panic!("f32 welp")
        };
        let name = tag.as_ref().to_string();
        self.model
            .as_mut()
            .unwrap()
            .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity());
        std::mem::forget(data);
    }

    pub fn add_input_f16(&mut self, tag: impl AsRef<str>, input: CoreMLInputRef<'a>) {
        debug_assert!(
            self.model.is_some(),
            "ensure model is compiled & loaded; before adding inputs"
        );
        let Some((shape, mut data)) = input.input_data_f16() else {
            panic!("f16 welp")
        };
        let name = tag.as_ref().to_string();
        self.model
            .as_mut()
            .unwrap()
            .bindInputU16(shape, name, data.as_mut_ptr(), data.capacity());
        std::mem::forget(data);
    }

    pub fn compile(&mut self) {
        self.model = Some(Model::compileModel(
            self.path.display().to_string(),
            self.opts.compute_platform,
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
        if let Some(model) = &mut self.model {
            let output = model.modelRun();
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
        } else {
            None
        }
    }

    pub fn load(&mut self) {
        if let Some(model) = &mut self.model {
            model.modelLoad();
        } else {
        }
    }

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        let desc = self.model.as_ref().unwrap().modelDescription();
        let mut map = HashMap::new();
        map.insert("input", desc.inputs());
        map.insert("output", desc.outputs());
        map
    }
}
