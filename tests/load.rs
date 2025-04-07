use coreml_rs::{mlmodel::CoreMLError, CoreMLModelOptions, CoreMLModelWithState};

#[test]
pub fn load_empty() {
    let m = CoreMLModelWithState::from_buf(vec![], CoreMLModelOptions::default());
    let res = m.load();
    assert!(matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
}
