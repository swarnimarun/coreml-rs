//! Reusable tensor inputs, batch inference, metrics and caller-controlled fallback.
//! Run: cargo run --release --example tensors
use coreml_rs::{ComputeUnits, FeatureValue, Inputs, Model, ModelOptions, MultiArray, Tensor};
use std::error::Error;

mod support;
use support::MODEL;

fn main() -> Result<(), Box<dyn Error>> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("relu.mlmodel");
    std::fs::write(&path, MODEL)?;
    let compiled = Model::compile(path)?;
    let primary = compiled.load(ModelOptions::default())?;
    let fallback = compiled.load(ModelOptions {
        compute_units: ComputeUnits::CpuOnly,
        ..Default::default()
    })?;
    let observer = primary.metrics_handle(); // This handle can be sent to a monitoring thread.

    let mut array = MultiArray::from_tensor(&Tensor::from_vec(vec![3], vec![1.0f32, -2.0, 3.0])?)?;
    let features = Inputs::new().with("input", FeatureValue::multi_array(&array));
    let prepared = features.prepare()?;
    let output = match primary.predict_prepared(&prepared, None) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("primary failed: {error}; trying CPU");
            fallback.predict_prepared(&prepared, None)?
        }
    };
    println!(
        "output: {:?}",
        output.multi_array("output")?.to_tensor::<f32>()?.as_slice()
    );

    // Reuse the same Core ML input storage and native feature provider.
    array.update(&Tensor::from_vec(vec![3], vec![-1.0f32, 4.0, 5.0])?)?;
    let output = primary.predict_prepared(&prepared, None)?;
    println!(
        "updated output: {:?}",
        output.multi_array("output")?.to_tensor::<f32>()?.as_slice()
    );
    let batch = primary.predict_batch(&[features.clone(), features])?;
    println!(
        "batch items: {}, metrics: {:?}",
        batch.len(),
        observer.snapshot()
    );
    Ok(())
}
