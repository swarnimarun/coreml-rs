//! v1 runner. Copied into an isolated checkout by compare.py.
#[path = "../benchmarks/common.rs"]
mod common;
use common::{BenchResult, Config, Data, Phase, Sample};
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::{Array, IxDyn};
use std::time::Instant;

fn main() -> BenchResult<()> {
    let config = Config::parse()?;
    if config.phase == Phase::Reuse {
        return Err("v1 has no reusable input provider".into());
    }
    let units = match config.compute.as_str() {
        "cpu" => ComputePlatform::Cpu,
        "gpu" => ComputePlatform::CpuAndGpu,
        "ane" => ComputePlatform::CpuAndANE,
        _ => unreachable!(),
    };
    let mut model = CoreMLModelWithState::new_compiled(
        &config.model,
        CoreMLModelOptions {
            compute_platform: units,
            ..Default::default()
        },
    )
    .load()?;
    let data = Data::new(&config);
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
        let array = Array::from_shape_vec(IxDyn(config.shape()), data.inputs[id].clone())?;
        model.add_input(config.input_name(), array)?;
        let prepare = prepare_start.elapsed();
        let call_start = Instant::now();
        let mut output = model.predict()?;
        let call = call_start.elapsed();
        let copy_start = Instant::now();
        let value = output
            .outputs
            .remove(config.output_name())
            .ok_or("v1 omitted expected output")?;
        let output = value.extract_to_tensor::<f32>();
        let output_copy = copy_start.elapsed();
        let elapsed = if config.phase == Phase::Predict {
            call_start.elapsed()
        } else {
            prepare_start.elapsed()
        };
        if i >= config.warmup {
            let (output, offset) = output.into_raw_vec_and_offset();
            if !matches!(offset, Some(0) | None) {
                return Err("unexpected v1 output offset".into());
            }
            samples.push(Sample {
                input: id,
                elapsed,
                prepare,
                call,
                output_copy,
                output,
            });
        }
    }
    common::finish(&config, &data, &samples)
}
