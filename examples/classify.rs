//! Classify an image with Apple's MobileNetV2 or FastViT models.
//! Run: cargo run --release --example classify -- MODEL IMAGE [ITERATIONS] [all|cpu|gpu|ane]
use coreml_rs::{
    Capabilities, ComputeUnits, DictionaryKey, FeatureValue, Inputs, Model, ModelOptions,
};
use image::imageops::FilterType;
use std::{error::Error, path::Path, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: classify <model> <image> [iterations] [all|cpu|gpu|ane]".into());
    }
    let iterations: usize = args.get(3).map_or(Ok(10), |s| s.parse())?;
    if iterations > u32::MAX as usize {
        return Err("too many iterations".into());
    }
    if iterations == 0 {
        return Err("iterations must be greater than zero".into());
    }
    let compute_units = match args.get(4).map(String::as_str).unwrap_or("all") {
        "all" => ComputeUnits::All,
        "cpu" => ComputeUnits::CpuOnly,
        "gpu" => ComputeUnits::CpuAndGpu,
        "ane" => ComputeUnits::CpuAndNeuralEngine,
        _ => return Err("compute units must be all, cpu, gpu, or ane".into()),
    };
    let start = Instant::now();
    let compiled = if Path::new(&args[1])
        .extension()
        .is_some_and(|s| s == "mlmodelc")
    {
        None
    } else {
        Some(Model::compile(&args[1])?)
    };
    println!("compile: {:?}", start.elapsed());
    let path = compiled
        .as_ref()
        .map(|c| c.path())
        .unwrap_or_else(|| Path::new(&args[1]));
    let start = Instant::now();
    let model = Model::load(
        path,
        ModelOptions {
            compute_units,
            display_name: (Capabilities::current().macos_version >= (13, 0, 0))
                .then(|| "coreml-rs image example".into()),
            ..Default::default()
        },
    )?;
    println!(
        "load: {:?}; capabilities: {:#?}",
        start.elapsed(),
        Capabilities::current()
    );
    let desc = model.description();
    let feature = desc
        .inputs
        .iter()
        .find(|feature| feature.image.is_some())
        .ok_or("model has no image input")?;
    let dimensions = feature.image.as_ref().unwrap();
    let width = u32::try_from(dimensions.width)?;
    let height = u32::try_from(dimensions.height)?;
    // This example resizes directly to the model's dimensions; production
    // applications should apply the preprocessing appropriate for their model.
    let image = image::open(&args[2])?
        .resize_exact(width, height, FilterType::Triangle)
        .to_rgba8();
    let input = FeatureValue::image_from_rgba8(width as usize, height as usize, image.as_raw())?;
    let prepared = Inputs::new().with(&feature.name, input).prepare()?;
    let warmup = Instant::now();
    model.predict_prepared(&prepared, None)?;
    println!("first prediction: {:?}", warmup.elapsed());
    let start = Instant::now();
    let mut last = None;
    for _ in 0..iterations {
        last = Some(model.predict_prepared(&prepared, None)?);
    }
    println!(
        "mean over {iterations} warm predictions: {:?}",
        start.elapsed() / iterations as u32
    );
    let outputs = last.unwrap();
    for (name, value) in outputs.iter() {
        if let Some(label) = value.as_string() {
            println!("{name}: {label}");
        }
        if let Some(mut entries) = value.as_dictionary()? {
            entries.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (key, probability) in entries.iter().take(5) {
                match key {
                    DictionaryKey::String(label) => println!("{name}: {label}: {probability:.4}"),
                    DictionaryKey::Integer(label) => println!("{name}: {label}: {probability:.4}"),
                }
            }
        }
    }
    println!("model metrics: {:?}", model.metrics());
    Ok(())
}
