use std::{path::PathBuf, str::FromStr};

use coreml_rs::{mlmodel::CoreMLError, CoreMLModelOptions, CoreMLModelWithState};
use sha2::{Digest, Sha256};

#[test]
pub fn load_empty() {
    let m = CoreMLModelWithState::from_buf(vec![], CoreMLModelOptions::default());
    let res = m.load();
    assert!(matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
}

pub fn unzip_to_path_from_hash(buf: &[u8]) -> Option<PathBuf> {
    fn get_cache_filename(model_buffer: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model_buffer);
        let hash = hasher.finalize();
        format!("{:x}.cache", hash)
    }
    let name = get_cache_filename(buf);

    // writing to tmp allows faster processing and usage as it's part of swap based in-memory tmpfs
    let path = PathBuf::from_str("/tmp/coreml-aftershoot/").ok()?;
    let path = path.join(name);

    let m = path.join("model.mlpackage");
    if m.exists() {
        return None;
    }

    _ = std::fs::remove_dir_all(&path);
    _ = std::fs::remove_file(&path);
    let mut res = zip::ZipArchive::new(std::io::Cursor::new(buf)).ok()?;
    res.extract(&path).ok()?;

    if m.exists() {
        Some(m)
    } else {
        None
    }
}

#[test]
pub fn reload_from_compiled_path() {
    let model_path = "./demo/model.zip";
    let buf = std::fs::read(model_path).unwrap();
    let path = unzip_to_path_from_hash(&buf).unwrap();
    let m = CoreMLModelWithState::new(&path, CoreMLModelOptions::default());
    let res = m.load();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
    let m = res.unwrap();
    let res = m.unload();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
    _ = std::fs::remove_dir_all(path);
    let m = res.unwrap();
    let res = m.load();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
}

#[test]
pub fn reload_from_buf() {
    let model_path = "./demo/model_3.mlmodel";
    let buf = std::fs::read(model_path).unwrap();
    let m = CoreMLModelWithState::from_buf(buf, CoreMLModelOptions::default());
    let res = m.load();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
    let m = res.unwrap();
    let res = m.unload();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
    let m = res.unwrap();
    let res = m.load();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
}
