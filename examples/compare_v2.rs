//! v2 runner for `benchmarks/compare.py`. Run through that driver.
#[path = "../benchmarks/common.rs"]
mod common;
use common::{BenchResult, Config, Data, Phase, Sample};
use coreml_rs::{
    ComputeUnits, DictionaryKey, FeatureValue, Inputs, Model, ModelOptions, MultiArray, Tensor,
};
use std::time::Instant;

fn main() -> BenchResult<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("compile") {
        if args.len() != 4 {
            return Err("usage: compare_v2 compile <source> <destination>".into());
        }
        Model::compile(&args[2])?.persist(&args[3])?;
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("validate-original") {
        if args.len() != 6 {
            return Err("usage: compare_v2 validate-original <original.mlmodelc> <array.mlmodelc> <labels> <cpu|gpu|ane>".into());
        }
        return validate_original(&args[2], &args[3], &args[4], &args[5]);
    }
    let config = Config::parse()?;
    let units = match config.compute.as_str() {
        "cpu" => ComputeUnits::CpuOnly,
        "gpu" => ComputeUnits::CpuAndGpu,
        "ane" => ComputeUnits::CpuAndNeuralEngine,
        _ => unreachable!(),
    };
    let model = Model::load(
        &config.model,
        ModelOptions {
            compute_units: units,
            ..Default::default()
        },
    )?;
    let data = Data::new(&config);
    let tensors: Vec<_> = data
        .inputs
        .iter()
        .map(|v| Tensor::from_vec(config.shape(), v.clone()))
        .collect::<Result<_, _>>()?;
    let mut reusable = MultiArray::from_tensor(&tensors[0])?;
    let prepared = Inputs::new()
        .with(config.input_name(), FeatureValue::multi_array(&reusable))
        .prepare()?;
    let mut samples = Vec::with_capacity(config.iterations);
    let total = config
        .warmup
        .checked_add(config.iterations)
        .ok_or("iteration count overflow")?;
    for i in 0..total {
        let id = if i < config.warmup {
            i % config.pool_size
        } else {
            data.order[i - config.warmup]
        };
        let prepare_start = Instant::now();
        let inputs = match config.phase {
            Phase::Predict => {
                let array = MultiArray::from_tensor(&tensors[id])?;
                Some(Inputs::new().with(config.input_name(), FeatureValue::multi_array(&array)))
            }
            Phase::EndToEnd => {
                let tensor = Tensor::from_vec(config.shape(), data.inputs[id].clone())?;
                let array = MultiArray::from_tensor(&tensor)?;
                Some(Inputs::new().with(config.input_name(), FeatureValue::multi_array(&array)))
            }
            Phase::Reuse => {
                reusable.update(&tensors[id])?;
                None
            }
        };
        let prepare = prepare_start.elapsed();
        let call_start = Instant::now();
        let output = if let Some(inputs) = inputs.as_ref() {
            model.predict(inputs)?
        } else {
            model.predict_prepared(&prepared, None)?
        };
        let call = call_start.elapsed();
        let copy_start = Instant::now();
        let result = output
            .multi_array(config.output_name())?
            .to_tensor::<f32>()?;
        let output_copy = copy_start.elapsed();
        let elapsed = if config.phase == Phase::Predict {
            call_start.elapsed()
        } else {
            prepare_start.elapsed()
        };
        if i >= config.warmup {
            samples.push(Sample {
                input: id,
                elapsed,
                prepare,
                call,
                output_copy,
                output: result.as_slice().to_vec(),
            });
        }
    }
    common::finish(&config, &data, &samples)
}

// Check that the converted model still agrees with the original image-input
// classifier on one deterministic RGB image. This is never timed.
fn validate_original(
    original: &str,
    derived: &str,
    labels: &str,
    compute: &str,
) -> BenchResult<()> {
    let compute_units = match compute {
        "cpu" => ComputeUnits::CpuOnly,
        "gpu" => ComputeUnits::CpuAndGpu,
        "ane" => ComputeUnits::CpuAndNeuralEngine,
        _ => return Err("unsupported compute units".into()),
    };
    let options = ModelOptions {
        compute_units,
        ..Default::default()
    };
    let original = Model::load(original, options.clone())?;
    let derived = Model::load(derived, options)?;
    let labels = std::fs::read_to_string(labels)?;
    let labels: Vec<_> = labels.lines().collect();
    if labels.len() != 1001 {
        return Err("expected 1001 class labels".into());
    }
    let pixels = 224 * 224;
    let mut rgba = vec![0u8; pixels * 4];
    let mut chw = vec![0.0f32; pixels * 3];
    for i in 0..pixels {
        for channel in 0..3 {
            let value = ((i * 37 + channel * 59 + 17) % 256) as u8;
            rgba[4 * i + channel] = value;
            chw[channel * pixels + i] = value as f32 * (2.0 / 255.0) - 1.0;
        }
        rgba[4 * i + 3] = 255;
    }
    let image = FeatureValue::image_from_rgba8(224, 224, &rgba)?;
    let output = original.predict(&Inputs::new().with("image", image))?;
    let dictionary = output
        .get("classLabelProbs")
        .ok_or("original probability output missing")?
        .as_dictionary()?
        .ok_or("original probability output is not a dictionary")?;
    let mut probabilities = std::collections::HashMap::new();
    for (key, value) in dictionary {
        match key {
            DictionaryKey::String(label) => {
                probabilities.insert(label, value);
            }
            _ => return Err("unexpected classifier key".into()),
        }
    }
    let array = MultiArray::from_tensor(&Tensor::from_vec(vec![3, 224, 224], chw)?)?;
    let output =
        derived.predict(&Inputs::new().with("image", FeatureValue::multi_array(&array)))?;
    let tensor = output.multi_array("classLabelProbs")?.to_tensor::<f32>()?;
    if tensor.as_slice().len() != labels.len() {
        return Err("classifier output size mismatch".into());
    }
    let mut counts = std::collections::HashMap::new();
    for label in &labels {
        *counts.entry(*label).or_insert(0usize) += 1;
    }
    if probabilities.len() != counts.len() {
        return Err("classifier dictionary size mismatch".into());
    }
    let mut max_difference = 0.0f32;
    let mut compared = 0;
    let mut total_difference = 0.0f64;
    for (label, &value) in labels.iter().zip(tensor.as_slice()) {
        // Duplicate class labels collapse to one dictionary key; that key does
        // not identify which corresponding vector element Core ML retained.
        if counts[label] != 1 {
            continue;
        }
        let expected = *probabilities
            .get(*label)
            .ok_or("classifier label missing")? as f32;
        if !expected.is_finite() || !value.is_finite() {
            return Err(format!("non-finite classifier output for {label}").into());
        }
        let difference = (expected - value).abs();
        max_difference = max_difference.max(difference);
        total_difference += f64::from(difference);
        compared += 1;
    }
    if compared == 0 || max_difference > 0.005 || total_difference > 0.05 {
        return Err(format!(
            "converted model differs from original: max {max_difference}, L1 {total_difference}"
        )
        .into());
    }
    println!("original image model vs derived array model: {compared} unique labels, max absolute difference {max_difference:.9}, L1 difference {total_difference:.9}");
    Ok(())
}
