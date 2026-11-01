//! Run a tiny stateful accumulator. Requires macOS 15 or later.
//! Run: cargo run --release --example stateful
#[path = "../tests/fixtures/stateful.rs"]
mod fixture;
use coreml_rs::{FeatureValue, Inputs, Model, ModelOptions, MultiArray, Tensor};
use half::f16;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let tmp = tempfile::tempdir()?;
    let package = tmp.path().join("Accumulator.mlpackage");
    let data = package.join("Data/com.apple.CoreML");
    std::fs::create_dir_all(data.join("weights"))?;
    std::fs::write(data.join("model.mlmodel"), fixture::MODEL)?;
    std::fs::write(
        package.join("Manifest.json"),
        r#"{"fileFormatVersion":"1.0.0","itemInfoEntries":{"model":{"author":"com.apple.CoreML","description":"model","name":"model.mlmodel","path":"com.apple.CoreML/model.mlmodel"},"weights":{"author":"com.apple.CoreML","description":"weights","name":"weights","path":"com.apple.CoreML/weights"}},"rootModelIdentifier":"model"}"#,
    )?;
    let compiled = Model::compile(package)?;
    let model = compiled.load(ModelOptions::default())?;
    let mut state = model.new_state()?;
    let input = Tensor::from_vec(vec![1], vec![f16::from_f32(1.0)])?;
    let input = MultiArray::from_tensor(&input)?;
    let inputs = Inputs::new().with("x", FeatureValue::multi_array(&input));
    for _ in 0..3 {
        let output = model.predict_with_state(&inputs, &mut state)?;
        println!(
            "accumulator: {}",
            output.multi_array("y")?.to_tensor::<f16>()?.as_slice()[0]
        );
    }
    Ok(())
}
