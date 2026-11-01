//! Diagnostic microbenchmark using the same tiny ReLU network as `tensors`.
//! This is not representative of a large model. Run in release mode.
mod support;
use coreml_rs::{ComputeUnits, FeatureValue, Inputs, Model, ModelOptions, MultiArray, Tensor};
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("relu.mlmodel");
    std::fs::write(&path, support::MODEL)?;
    let start = Instant::now();
    let compiled = Model::compile(path)?;
    let model = compiled.load(ModelOptions {
        compute_units: ComputeUnits::CpuAndGpu,
        ..Default::default()
    })?;
    println!("compile + load: {:?}", start.elapsed());
    let make = || -> Result<Inputs, Box<dyn Error>> {
        let tensor = Tensor::from_vec(vec![3], vec![1f32, -2., 3.])?;
        Ok(Inputs::new().with(
            "input",
            FeatureValue::multi_array(&MultiArray::from_tensor(&tensor)?),
        ))
    };
    for _ in 0..10 {
        model.predict(&make()?)?;
    }
    let start = Instant::now();
    for _ in 0..1000 {
        let output = model.predict(&make()?)?;
        std::hint::black_box(output.multi_array("output")?.to_tensor::<f32>()?);
    }
    println!("v2 1000 fresh inputs, copied output: {:?}", start.elapsed());
    let input = make()?.prepare()?;
    for _ in 0..10 {
        model.predict_prepared(&input, None)?;
    }
    let start = Instant::now();
    for _ in 0..1000 {
        let output = model.predict_prepared(&input, None)?;
        std::hint::black_box(output.multi_array("output")?.to_tensor::<f32>()?);
    }
    println!(
        "v2 1000 prepared inputs, copied output: {:?}",
        start.elapsed()
    );
    Ok(())
}
