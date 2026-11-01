# CoreML-RS

Rust-first Core ML inference for macOS. v2 uses Apple's Objective-C APIs through `objc2`; it does not build or link a Swift bridge. This is a breaking change from v1.

```toml
[dependencies]
coreml-rs = { git = "https://github.com/swarnimarun/coreml-rs", branch = "v2" }
```

```rust,no_run
use coreml_rs::{FeatureValue, Inputs, Model, ModelOptions, MultiArray, Tensor};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let compiled = Model::compile("model.mlpackage")?;
let model = compiled.load(ModelOptions::default())?;
let input = Tensor::from_vec(vec![1, 3], vec![1.0f32, 2.0, 3.0])?;
let input = MultiArray::from_tensor(&input)?;
let outputs = model.predict(&Inputs::new().with("input", FeatureValue::multi_array(&input)))?;
let result = outputs.multi_array("output")?.to_tensor::<f32>()?;
println!("{:?}", result.as_slice());
# Ok(())
# }
```

Requires macOS 12.3 or later, Rust 1.88 or later, and Apple's command line tools or Xcode. In-memory model specifications require macOS 13, compute plans require macOS 14.4, and ML Program blob mappings require macOS 15. State, CPU + Neural Engine selection, and device enumeration also have newer runtime requirements. Core ML makes the final decision about hardware placement. The current implementation uses synchronous prediction and serializes access through a non-`Send`, non-`Sync` model handle; its metrics handle can be sent to another thread.

Get and run the example models:

```sh
./scripts/download-example-models.sh
cargo run --release --example classify -- demo/models/MobileNetV2FP16.mlmodel demo/models/dog.jpg
cargo run --release --example classify -- demo/models/FastViTT8F16.mlpackage demo/models/dog.jpg
cargo run --release --example segment -- demo/models/DeepLabV3FP16.mlmodel demo/models/dog.jpg demo/models/segmentation.png
cargo run --release --example tensors
```

The downloads are kept in the ignored `demo/models` directory. See [setup](docs/setup.md), [usage](docs/usage.md), [model sources and examples](docs/examples.md), [v1 migration](docs/migration.md), and [known limitations](docs/roadmap.md).

MIT. Downloaded models and images have their own licenses; see [model sources](docs/examples.md).
