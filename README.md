# CoreML-RS

Rust bindings for Apple's [Core ML](https://developer.apple.com/documentation/coreml) framework using [swift-bridge](https://github.com/chinedufn/swift-bridge).
Load `.mlmodel` / `.mlpackage` files and run inference on macOS — with CPU, GPU, or Apple Neural Engine.

## Quick Start

```toml
[dependencies]
coreml-rs = { version = "0.5", git = "https://github.com/swarnimarun/coreml-rs" }
ndarray = "0.16"
```

```rust
use coreml_rs::{CoreMLModelWithState, CoreMLModelOptions, ComputePlatform};
use ndarray::Array4;

fn main() {
    let options = CoreMLModelOptions {
        compute_platform: ComputePlatform::CpuAndANE,
        ..Default::default()
    };

    let mut model = CoreMLModelWithState::new("model.mlpackage", options)
        .load()
        .unwrap();

    let input = Array4::<f32>::zeros((1, 3, 512, 512));
    model.add_input("image", input.into_dyn()).unwrap();

    let output = model.predict().unwrap();
    let result: ndarray::ArrayD<f32> = output.outputs["output_1"].extract_to_tensor();
}
```

## Features

- Load models from file path, raw bytes, or pre-compiled cache
- Single and batch inference
- Compute platform selection: CPU, CPU+GPU, CPU+Apple Neural Engine
- Memory management: load/unload/reload, persist buffer to disk
- Retry with configurable backoff
- Model introspection (input/output shapes, descriptions)
- f32, f16, and i32 input support (f32 output)

## Requirements

- macOS with Xcode and Command Line Tools
- `libswift_Concurrency.dylib` must be findable at runtime — see [setup guide](docs/setup.md)

## Documentation

- [Setup Guide](docs/setup.md) — prerequisites, runtime dependencies, model formats, troubleshooting
- [Usage Guide](docs/usage.md) — full API reference with examples
- [Roadmap](docs/roadmap.md) — planned improvements and known issues

## License

MIT
