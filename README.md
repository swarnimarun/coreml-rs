## Core-ML Rust Bindings (Work in Progress)

`coreml-rs` is an experimental Rust library aimed at providing Rust bindings for Apple's Core ML framework.
Core ML is Apple's machine learning framework designed to integrate machine learning models into iOS, macOS, watchOS, and tvOS applications.

## Status

This project is currently work in progress.
The primary goal is to enable Rust developers to utilize Core ML models within their applications, leveraging Rust's performance and safety features for the rest of the ML infrastructure.

## Roadmap

- Cleanup & fix bugs with the types and allow more input formats.
- Build zerocopy types for more efficiently passing inputs and outputs.
- Provide more configuration options for models.

## Features

- **Model Loading**: Load Core ML models into Rust applications.
- **Inference**: Perform inference using loaded models.
- **Data Handling**: Manage input and output data for model inference.

## Installation

To include `coreml-rs` in your project, add the following to your `Cargo.toml` dependencies:
```toml
[dependencies]
coreml-rs = { version = "0.4", git = "https://github.com/swarnimarun/coreml-rs" }
```

## Usage

Here's a basic example of how to use `coreml-rs` to load a Core ML model and perform inference:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::{Array, Array4};

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;
    // model_options.cache_dir = PathBuf::from("."); // optional (generally not required, only use when you want to unload_to_disk)
    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    // load in the input -- for brevity we just fill it with 1.0f32 to avoid using zeroes
    input.fill(1.0f32);

    let Ok(w) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    // Perform inference
    let output = model.predict();

    // Process the output
    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // you can also unload/load model as needed to save in use memory
    // model.unload(); model.load();

    // you can also try to use unload_to_disk, to cache to disk the models to drastically reduce memory usage

    // .. use output how ever you like
}
```

**Note**: This is a simplified example. Actual implementation may vary based on the model's input and output specifications.

## Contributing

Contributions are welcome!
If you have experience with Core ML and Rust, consider helping to advance this project.
