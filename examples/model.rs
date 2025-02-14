use std::{io::Read, sync::atomic::AtomicUsize};

use coreml_rs::{
    ffi::ComputePlatform,
    mlarray::{self, FloatMLArray, MLArray},
    mlmodel::{CoreMLModel, CoreMLModelOptions, Store},
};
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

pub fn main() {
    let s = Store::new();
    let mut m = timeit("load and compile model", || {
        let mut model_options = CoreMLModelOptions::default();
        // this performs worse when cold starting
        model_options.compute_platform = ComputePlatform::CpuAndGpu;
        let mut model = CoreMLModel::new("./demo/test.mlpackage", &s, model_options);
        timeit("compile model", || {
            model.compile();
        });
        timeit("load model", || {
            model.load();
        });
        // println!("model description:\n{:#?}", model.description());
        return model;
    });

    timeit("create and add input", || {
        let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
        // fill input with 1.0
        input.fill(1.0f32);
        // let mut fs = std::fs::File::create("input.bin").unwrap();
        // _ = fs.write_all(bytemuck::cast_slice(input.as_slice().unwrap()));
        let idx = s.insert(input);
        m.add_input_f32("img", s.bind_ref(idx));
    });

    // initialize the output buffer on rust side
    for _ in 0..10 {
        let output = Array4::<f32>::zeros((1, 3, 2048, 2048));
        m.add_output_f32("add", output);

        // println!("{:?}", output.shape());
        let output = timeit("predict", || {
            return m.predict();
        })
        .unwrap();
        let _f = match output.outputs.get("add").unwrap() {
            MLArray::FloatArray(FloatMLArray::Array(array)) => array,
            _ => panic!("unreachable"),
        };
        // println!("{_f:#?}");
    }

    // very cheap doesn't need to be measured!
    // let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // let mut fs = std::fs::File::open("output.bin").unwrap();
    // let mut lhs = Vec::new();
    // _ = fs.read_to_end(&mut lhs);
    // let rhs: &[u8] = bytemuck::cast_slice(output.as_slice().unwrap());
    // dbg!(mlarray::mean_absolute_error_bytes::<f32>(rhs, &lhs));
    // let d = time.elapsed() - (p + d);
    // println!("transmute to shape {} ms", d.as_millis());
}
