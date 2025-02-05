## Core-ML Rust Bindings (Work in Progress)

`coreml-rs` is an experimental Rust library aimed at providing Rust bindings for Apple's Core ML framework. ore ML is Apple's machine learning framework designed to integrate machine learning models into iOS, macOS, watchOS, and tvOS applications.

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
coreml-rs = { version = "0.1", git = "https://github.com/swarnimarun/coreml-rs" }
```

## Usage

Here's a basic example of how to use `coreml-rs` to load a Core ML model and perform inference:

```rust
use coreml_rs::mlmodel::{CoreMLModel, Store};
use ndarray::{Array, Array4};

pub fn main() {
    // Store is currently a temporary type to hold inputs but it's largely to ensure in future we can provide more optimizations
    let s = Store::new();
    let mut m = model_load_and_compile(&s);
    // Prepare input data
    let idx = s.insert(Array::zeros((1, 3, 512, 512)));
    m.add_input("input_0", s.bind_ref(idx));

    // Perform inference
    let output = m.predict();

    // Process the output
    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // .. use output how ever you like
}

// Load the Core ML model, note you have to compile the model before load
fn model_load_and_compile<'a>(s: &'a Store) -> CoreMLModel<'a> {
    let mut model = CoreMLModel::new("./demo/test.mlpackage", s);
    model.compile();
    model.load();
    // use model.description to figure out input and output feature names and formats
    // println!("model description:\n{:#?}", model.description());
    return model;
}
```

**Note**: This is a simplified example. Actual implementation may vary based on the model's input and output specifications.

## Contributing

Contributions are welcome!
If you have experience with Core ML and Rust, consider helping to advance this project.
