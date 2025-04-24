use std::{path::PathBuf, sync::atomic::AtomicUsize};

use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use libproc::pid_rusage::RUsageInfoV4;
use ndarray::Array4;

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

pub fn main() {
    dbg!("process init", proc_mem_usage());

    let buf = std::fs::read("./demo/model_3.mlmodel").unwrap();
    let input_name = "image";

    let mut m = timeit("load and compile model", || {
        let mut model_options = CoreMLModelOptions::default();
        model_options.compute_platform = ComputePlatform::CpuAndANE;
        // model_options.cache_dir = PathBuf::from(".");
        // let mut model =
        //     CoreMLModelWithState::new(PathBuf::from("./demo/test.mlpackage"), model_options);
        let mut model = CoreMLModelWithState::from_buf(buf, model_options);
        model = timeit("load model", || model.load().unwrap());
        println!("model description:\n{:#?}", model.description());
        return model;
    });
    dbg!("model load", proc_mem_usage());

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    _ = m.add_input(input_name, input.clone().into_dyn());
    dbg!("load input", proc_mem_usage());

    let output = timeit("predict", || {
        return m.predict();
    })
    .unwrap();
    dbg!("predict output in mem", proc_mem_usage());

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

    _ = m.add_input(input_name, input.into_dyn());
    let _ = timeit("predict", || {
        return m.predict();
    })
    .unwrap();
    // dbg!("deallocate all of it", proc_mem_usage());
}
