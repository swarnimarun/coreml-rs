//! Load a model specification and, optionally, one ML Program weight blob.
//! Run: cargo run --release --example in_memory -- model.mlmodel
//! Run: cargo run --release --example in_memory -- model.mlmodel weights/weight.bin weight.bin
use coreml_rs::{Capabilities, Model, ModelOptions, PlanStructure};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let specification_path = args
        .next()
        .ok_or("usage: in_memory <model.mlmodel> [<blob-key> <blob-file>]")?;
    let specification = std::fs::read(specification_path)?;
    let model = match (args.next(), args.next(), args.next()) {
        (None, None, None) => futures_executor::block_on(Model::load_from_memory(
            specification,
            ModelOptions::default(),
        ))?,
        (Some(key), Some(path), None) => {
            let blob = std::fs::read(path)?;
            futures_executor::block_on(Model::load_from_memory_with_blobs(
                specification,
                [(key, blob)],
                ModelOptions::default(),
            ))?
        }
        _ => return Err("usage: in_memory <model.mlmodel> [<blob-key> <blob-file>]".into()),
    };
    println!("model: {:?}", model.description());
    println!("load duration: {:?}", model.metrics().load_duration);
    if Capabilities::current().compute_plan {
        let plan = futures_executor::block_on(model.compute_plan())?;
        match plan.structure {
            PlanStructure::NeuralNetwork { layers } => println!("plan: {} layers", layers.len()),
            PlanStructure::Program { functions } => println!("plan: {} functions", functions.len()),
            PlanStructure::Pipeline { models } => println!("plan: {} submodels", models.len()),
            _ => println!("plan: another model structure"),
        }
    }
    Ok(())
}
