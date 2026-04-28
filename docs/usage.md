# Usage Guide

## Core Concepts

- **[`CoreMLModelWithState`]** — State machine for single-inference models. Two states: `Unloaded` (holds model bytes/path, no GPU memory) and `Loaded` (model in memory, ready for inference).
- **[`CoreMLBatchModelWithState`]** — State machine for batch inference models. Accumulates multiple inputs and runs a single batch prediction.
- **[`MLArray`]** — Multi-dtype array wrapper around `ndarray::ArrayD`. Supports f32, f16, i32, i16, i8, u32, u16, u8. F32 and f16 are the most mature.
- **[`ComputePlatform`]** — Selects which Apple hardware to use: `Cpu`, `CpuAndGpu` (default), or `CpuAndANE`.

## Model Loading

### From a file path

```rust
use coreml_rs::{CoreMLModelWithState, CoreMLModelOptions, ComputePlatform};

let options = CoreMLModelOptions {
    compute_platform: ComputePlatform::CpuAndANE,
    ..Default::default()
};

let model = CoreMLModelWithState::new("./path/to/model.mlpackage", options)
    .load()
    .expect("failed to load model");
```

### From raw bytes (in-memory)

```rust
let buf = std::fs::read("./model.mlmodel").unwrap();
let model = CoreMLModelWithState::from_buf(buf, CoreMLModelOptions::default())
    .load()
    .expect("failed to load model");
```

### From a pre-compiled model cache

```rust
let model = CoreMLModelWithState::new_compiled("./path/to/model.mlmodelc", options)
    .load()
    .expect("failed to load compiled model");
```

## Compute Platform

```rust
// CPU only — slowest, most compatible
options.compute_platform = ComputePlatform::Cpu;

// CPU + GPU — default, good for most workloads
options.compute_platform = ComputePlatform::CpuAndGpu;

// CPU + Apple Neural Engine — best for power efficiency, may not support all ops
options.compute_platform = ComputePlatform::CpuAndANE;
```

### Disabling experimental MLE5

On macOS 15+, set this to avoid crashes with the MLE5 engine:

```rust
options.disable_experimental_mle = true;
```

## Single Inference

```rust
use ndarray::Array4;

let mut model = CoreMLModelWithState::new("model.mlpackage", options)
    .load()
    .unwrap();

// Create input — shape must match model's expected input
let input = Array4::<f32>::zeros((1, 3, 512, 512));

model.add_input("image", input.into_dyn())
    .expect("failed to add input");

let output = model.predict()
    .expect("prediction failed");

// Access output by name
let result: ndarray::ArrayD<f32> = output.outputs["output_1"]
    .extract_to_tensor();
```

### Input types

Supported input dtypes for single inference:

| Type     | Rust type     | Status     |
|----------|---------------|------------|
| f32      | `f32`         | Full       |
| f16      | `half::f16`   | Full       |
| i32      | `i32`         | Experimental |

Other dtypes (`i16`, `i8`, `u32`, `u16`, `u8`) are defined in `MLArray` but not yet wired to the inference pipeline.

### Output types

Outputs are returned as `HashMap<String, MLArray>`. Extract to typed ndarray with:

```rust
let values: ndarray::ArrayD<f32> = output.outputs["output_name"]
    .extract_to_tensor();
```

Currently, f32 outputs are fully supported. f16 outputs are decoded as u16 and reinterpreted.

## Batch Inference

Batch inference accumulates multiple inputs and runs them in a single batch — useful when processing many items at once.

```rust
use coreml_rs::CoreMLBatchModelWithState;

let mut model = CoreMLBatchModelWithState::new("model.mlpackage", options)
    .load()
    .unwrap();

let input = Array4::<f32>::zeros((1, 3, 512, 512));

// Add inputs with indices
for i in 0..10 {
    model.add_input("image", input.clone().into_dyn(), i as isize)
        .expect("failed to add batch input");
}

let output = model.predict()
    .expect("batch prediction failed");

// output.outputs is Vec<HashMap<String, MLArray>>, one per index
for (i, outputs) in output.outputs.iter().enumerate() {
    let result: ndarray::ArrayD<f32> = outputs["image"].extract_to_tensor();
    // ... process result
}
```

Batch inference currently only supports f32 inputs.

## Memory Management

Models hold GPU/ANE memory when loaded. You can unload and reload to manage memory pressure.

### Unload / reload

```rust
let mut model = CoreMLModelWithState::new("model.mlpackage", options)
    .load()
    .unwrap();

// ... run inference ...

// Unload — frees GPU memory, keeps model bytes in RAM
let unloaded = model.unload().unwrap();

// Later: reload from compiled cache
let model = unloaded.load().unwrap();
```

When unloading a model originally loaded from a file path, the compiled path from Core ML's compilation step is preserved and reused for fast reload.

### Persist buffer to disk

When loading from raw bytes, you can persist the compressed buffer to disk and reload later — useful for caching:

```rust
let model = CoreMLModelWithState::from_buf(buf, options)
    .load()
    .unwrap();

let unloaded = model.unload_to_disk().unwrap();
// Buffer is now written as zlib-compressed file at options.cache_dir/"model_cache"
// Reload later:
let model = unloaded.load().unwrap();
```

## Retry with Backoff

Prediction can fail transiently (e.g., GPU contention). Use retry with configurable backoff:

```rust
use coreml_rs::{PredictRetryOptions, RetryBackoff};
use std::time::Duration;

// Fixed delay: retry up to 3 times with 10ms between attempts
let options = PredictRetryOptions::fixed(3, Duration::from_millis(10));

let output = model.predict_with_retry(options)
    .expect("prediction failed after retries");
```

### Exponential backoff

```rust
let options = PredictRetryOptions::exponential(
    5,                              // max retries
    Duration::from_millis(10),      // initial delay
    2.0,                            // multiplier
    Duration::from_secs(5),         // max delay
);
```

### Conditional retry

Skip retry for specific error types:

```rust
let output = model.predict_with_retry_if(options, |err| {
    matches!(err, coreml_rs::mlmodel::CoreMLError::UnknownError(_))
}).expect("prediction failed");
```

Note: batch model (`CoreMLBatchModelWithState`) currently does not support retry.

## Model Introspection

Inspect a loaded model's inputs and outputs:

```rust
// Full description as text
let desc = model.description().unwrap();
for input in desc.get("input").into_iter().flatten() {
    println!("  {input}");
}
for output in desc.get("output").into_iter().flatten() {
    println!("  {output}");
}

// Structured — get shapes
let input_shapes = model.input_shapes().unwrap();
for (name, shape) in &input_shapes {
    println!("Input `{name}`: shape {shape:?}");
}

let output_shapes = model.output_shapes().unwrap();
for (name, shape) in &output_shapes {
    println!("Output `{name}`: shape {shape:?}");
}
```

## Working with MLArray

`MLArray` wraps `ndarray::ArrayD` across 8 dtypes:

```rust
use coreml_rs::mlarray::MLArray;
use ndarray::{Array, IxDyn};

// Create typed arrays, convert to MLArray
let f32_arr: MLArray = Array::<f32, _>::zeros(IxDyn(&[1, 3, 256, 256])).into();
let f16_arr: MLArray = Array::<half::f16, _>::zeros(IxDyn(&[1, 3, 256, 256])).into();

// Inspect shape
let shape: &[usize] = f32_arr.shape();

// Extract back to typed array
let typed: ndarray::ArrayD<f32> = f32_arr.extract_to_tensor();
```

## Error Handling

All public operations return `Result<T, CoreMLError>`:

| Variant                | When                                                  |
|------------------------|-------------------------------------------------------|
| `ModelNotLoaded`       | Operations on an `Unloaded` model                     |
| `FailedToLoad(...)`    | Model bytes are invalid or compilation failed         |
| `BadInputShape(...)`   | Input shape doesn't match model's expected shape      |
| `UnknownError(...)`    | Prediction or binding failure at runtime              |
| `IoError(...)`         | File I/O during load/unload cache operations          |

## Complete Example

A runnable CLI that loads a model, prepares an image, runs inference, and saves outputs is at `examples/run_local_mlpackage.rs`. Run with:

```sh
cargo run --example run_local_mlpackage -- path/to/model.mlpackage path/to/image.jpg --compute ane
```
