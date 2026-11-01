//! Inspect a model's anticipated device placement and observed request metrics.
//! Run: cargo run --release --example compute_plan -- MODEL
use coreml_rs::{Model, ModelOptions, PlanStructure};
use std::{error::Error, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: compute_plan <model.mlmodel|model.mlpackage|model.mlmodelc>")?;
    let compiled = if Path::new(&path)
        .extension()
        .is_some_and(|v| v == "mlmodelc")
    {
        None
    } else {
        Some(Model::compile(&path)?)
    };
    let model = match compiled.as_ref() {
        Some(compiled) => compiled.load(ModelOptions::default())?,
        None => Model::load(&path, ModelOptions::default())?,
    };
    // The library does not require a particular async runtime. This example
    // uses futures-executor, which is a development dependency of this repo.
    let plan = futures_executor::block_on(model.compute_plan())?;
    match plan.structure {
        PlanStructure::NeuralNetwork { layers } => {
            println!("{} neural-network layers", layers.len());
            for layer in layers.iter().take(3) {
                println!(
                    "{} ({}): {:?}",
                    layer.name, layer.layer_type, layer.device_usage
                );
            }
        }
        PlanStructure::Program { functions } => {
            for function in functions {
                println!(
                    "function {}: {} top-level operations",
                    function.name,
                    function.block.operations.len()
                );
                for operation in function.block.operations.iter().take(3) {
                    println!(
                        "{}: {:?}; relative cost {:?}",
                        operation.operator_name, operation.device_usage, operation.estimated_cost
                    );
                }
            }
        }
        PlanStructure::Pipeline { models } => println!("pipeline with {} submodels", models.len()),
        _ => println!("Core ML returned another model structure"),
    }
    println!("per-model metrics: {:?}", model.metrics());
    Ok(())
}
