//! Inspect a model without requiring application-specific input data.
//! Run: cargo run --example simple -- path/to/model.mlmodel
use coreml_rs::{Capabilities, Model, ModelOptions};
use std::{error::Error, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: simple <model.mlmodel|model.mlpackage|model.mlmodelc>")?;
    let compiled = if Path::new(&path)
        .extension()
        .is_some_and(|v| v == "mlmodelc")
    {
        None
    } else {
        Some(Model::compile(&path)?)
    };
    let path = compiled
        .as_ref()
        .map(|model| model.path())
        .unwrap_or_else(|| Path::new(&path));
    let model = Model::load(path, ModelOptions::default())?;
    println!("{:#?}", model.description());
    println!("{:#?}", Capabilities::current());
    Ok(())
}
