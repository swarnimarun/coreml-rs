use std::{path::PathBuf, str::FromStr, sync::atomic::AtomicUsize};

use coreml_rs::{
    mlbatchmodel::CoreMLBatchModelWithState, ComputePlatform, CoreMLModelOptions,
    CoreMLModelWithState,
};
use libproc::pid_rusage::RUsageInfoV4;
use ndarray::Array4;
use sha2::Digest;

static LEVEL: AtomicUsize = AtomicUsize::new(0);

pub fn timeit<T>(t: impl AsRef<str>, f: impl FnOnce() -> T) -> T {
    LEVEL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let time = std::time::Instant::now();
    let o = f();
    let v = time.elapsed().as_millis();
    let level = LEVEL.load(std::sync::atomic::Ordering::SeqCst);
    println!("|{}> {} : {} ms", "\t|".repeat(level - 1), t.as_ref(), v);
    LEVEL.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    o
}

pub fn proc_mem_usage() -> String {
    let pid = std::process::id() as i32;
    let info = libproc::pid_rusage::pidrusage::<RUsageInfoV4>(pid).unwrap();
    format!(
        "RSS: {}, Physical Footprint: {}",
        info.ri_resident_size, info.ri_phys_footprint
    )
}

pub fn unzip_to_path_from_hash(buf: &[u8]) -> Option<PathBuf> {
    fn get_cache_filename(model_buffer: &[u8]) -> String {
        let mut hasher = sha2::Sha256::new();
        hasher.update(model_buffer);
        let hash = hasher.finalize();
        format!("{:x}.mlpackage", hash)
    }
    let name = get_cache_filename(buf);

    // writing to tmp allows faster processing and usage as it's part of swap based in-memory tmpfs
    let path = PathBuf::from_str("/tmp/coreml-aftershoot/").ok()?;
    let path = path.join(name);
    _ = std::fs::remove_dir_all(&path);
    _ = std::fs::remove_file(&path);

    let mut res = zip::ZipArchive::new(std::io::Cursor::new(buf)).ok()?;
    res.extract(&path).ok()?;

    let m = path.join("model.mlpackage");
    if m.exists() {
        Some(m)
    } else {
        None
    }
}

fn temp_buf_to_path<T>(buf: Vec<u8>, f: impl FnOnce(std::path::PathBuf) -> Option<T>) -> Option<T> {
    let path = unzip_to_path_from_hash(&buf)?;
    let res = f(path.clone());
    _ = dbg!(std::fs::remove_dir_all(&path));
    _ = dbg!(std::fs::remove_file(path));
    res
}

fn is_zip(s: &[u8]) -> bool {
    zip::ZipArchive::new(std::io::Cursor::new(s)).is_ok()
}

pub fn main() {
    dbg!("process init", proc_mem_usage());

    // let buf = std::fs::read("./demo/model_2.zip").unwrap();
    // if !is_zip(&buf) {
    //     panic!("unsupported");
    // }
    let buf = std::fs::read("./demo/model_3.mlmodel").unwrap();

    // let mut m = temp_buf_to_path(buf, |path| {
    // Some(
    let mut m = timeit("load and compile model", move || {
        let mut model_options = CoreMLModelOptions::default();
        model_options.compute_platform = ComputePlatform::CpuAndANE;
        // model_options.cache_dir = PathBuf::from(".");
        // let mut model = CoreMLModelWithState::new(PathBuf::from(path), model_options);
        let mut model = CoreMLBatchModelWithState::from_buf(buf, model_options);
        model = timeit("load model", || model.load().unwrap());

        return model;
    });
    // )
    // })
    // .unwrap();
    println!("model description:\n{:#?}", m.description());
    let input_name = "image";

    dbg!("model load", proc_mem_usage());

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    for i in 0..80 {
        _ = m.add_input(input_name, input.clone().into_dyn(), i);
    }

    dbg!("load input", proc_mem_usage());

    let output = timeit("predict", || {
        return m.predict();
    })
    .unwrap();
    let outs = &output.outputs;
    dbg!(outs.len());
    dbg!("predict output in mem", proc_mem_usage());

    // dbg!(&outs[0].get("mask"));

    drop(output);
    dbg!("predict output yeeted", proc_mem_usage());

    m = timeit("unload model", || {
        // println!("{m:#?}");
        let r = m.unload().unwrap();
        // let r = m.unload_to_disk().unwrap();
        // println!("{r:#?}");
        r
    });
    dbg!("unloaded", proc_mem_usage());

    m = timeit("load from cache model", || {
        let r = m.load().unwrap();
        // println!("{r:#?}");
        r
    });
    dbg!("loaded again", proc_mem_usage());

    _ = m.add_input(input_name, input.into_dyn(), 0);
    let _ = timeit("predict", || {
        return m.predict();
    })
    .unwrap();
    // dbg!("deallocate all of it", proc_mem_usage());
}
