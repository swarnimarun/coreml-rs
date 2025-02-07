use std::{
    collections::HashMap,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use crate::{
    ffi::{Model, ModelOutput},
    mlarray::MLArray,
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
        let Ok(v) = self.1.get() else {
            return None;
        };
        let Some(v) = v.get(self.0) else {
            return None;
        };
        Some((
            v.shape().into_iter().map(|s| *s as i32).collect(),
            v.clone().into_raw_vec_f32(),
        ))
    }
    pub fn input_data_f16(&self) -> Option<(Vec<i32>, Vec<u16>)> {
        let Ok(v) = self.1.get() else {
            return None;
        };
        let Some(v) = v.get(self.0) else {
            return None;
        };
        Some((
            v.shape().into_iter().map(|s| *s as i32).collect(),
            v.clone().into_raw_vec_u16(),
        ))
    }
    pub fn input_data_i32(&self) -> Option<(Vec<i32>, Vec<i32>)> {
        let Ok(v) = self.1.get() else {
            return None;
        };
        let Some(v) = v.get(self.0) else {
            return None;
        };
        Some((
            v.shape().into_iter().map(|s| *s as i32).collect(),
            v.clone().into_raw_vec_i32(),
        ))
    }
}

pub struct CoreMLModel<'a> {
    model: Option<Model>,
    path: PathBuf,
    _p: PhantomData<CoreMLInputRef<'a>>,
}

impl<'a> CoreMLModel<'a> {
    pub fn new(path: impl AsRef<Path>, _: &'a Store) -> Self {
        Self {
            model: None,
            path: path.as_ref().to_path_buf(),
            _p: PhantomData::default(),
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
        self.model.as_mut().unwrap().bindInputI32(shape, name, data);
        // std::mem::forget(data);
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
            .bindInputF32(shape, name, data.as_mut_ptr());
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
        self.model.as_mut().unwrap().bindInputF16(shape, name, data);
        // std::mem::forget(data);
    }

    pub fn compile(&mut self) {
        self.model = Some(Model::compileModel(self.path.display().to_string()));
    }

    pub fn predict(&mut self) -> Option<ModelOutput> {
        if let Some(model) = &mut self.model {
            Some(model.modelRun())
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
