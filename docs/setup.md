# Setup Guide

## Prerequisites

- **macOS** — Core ML is an Apple framework, available on macOS 10.13+ (tested on macOS 13+)
- **Xcode** with Command Line Tools — the build script runs `xcode-select --print-path` and `swift build`
- **Rust** stable (edition 2021)

Verify your tools:

```sh
xcode-select --print-path
# /Applications/Xcode.app/Contents/Developer

swift --version
# Apple Swift version 6.x

rustc --version
# rustc 1.8x.x
```

## Adding the Dependency

```toml
[dependencies]
coreml-rs = { version = "0.5", git = "https://github.com/aftershootco/coreml-rs" }
ndarray = "0.16"
```

## Runtime: `libswift_Concurrency.dylib`

The Swift code uses `async/await` to load models from `MLModelAsset` (when loading from raw bytes). This pulls in Swift Concurrency runtime support. The dynamic library is **not** automatically bundled with your binary.

### Why it's needed

When you call `CoreMLModelWithState::from_buf(...)` and then `.load()`, the Swift side runs:

```swift
Task { [weak self] in
    let res = try await MLModel.load(asset: asset, configuration: config)
    ...
}
```

This requires `libswift_Concurrency.dylib` at runtime.

### Where to find it

The library lives inside the Xcode toolchain:

```
$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx/libswift_Concurrency.dylib
```

Also available (symlinked) at:

```
/usr/lib/swift/libswift_Concurrency.dylib
```

### How to bundle it

**Option A: Copy next to binary**

```sh
cp /usr/lib/swift/libswift_Concurrency.dylib ./target/debug/
```

**Option B: Set rpath in build.rs** (for production projects)

Add to your `build.rs`:

```rust
println!("cargo:rustc-link-search=/usr/lib/swift");
println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
```

This is what `coreml-rs`'s own `build.rs` already does for the link step, but the runtime loader still needs to find it. Using `install_name_tool` or DYLD paths works too.

**Option C: DYLD_LIBRARY_PATH** (development only)

```sh
export DYLD_LIBRARY_PATH=/usr/lib/swift:$DYLD_LIBRARY_PATH
cargo run --example simple
```

## Model Formats

### `.mlmodel`

A flat Protobuf file containing the model specification. Core ML compiles it on first load to a temporary directory.

### `.mlpackage`

A directory containing the model specification plus metadata and assets. Core ML compiles it similarly.

### Compiled model cache

After compilation, Core ML produces a `.mlmodelc` directory. You can reuse this compiled path for faster loading:

```rust
let model = CoreMLModelWithState::new_compiled("./path/to/model.mlmodelc", options);
```

## Build Process

When you run `cargo build`, the `build.rs` script:

1. **swift-bridge-build** parses `src/swift.rs` and generates Swift glue code into `swift-library/Sources/swift-library/generated/`
2. **Swift compilation** runs `swift build --arch <target-arch> -Xswiftc -static` in `swift-library/`, producing a static library
3. **Rust linking** links the static Swift library and Xcode's Swift runtime search paths

## Troubleshooting

### `ld: warning: Could not find or use auto-linked library 'swiftCompatibilityConcurrency'`

Your Xcode/Swift toolchain is missing or incomplete. Install Xcode and its command line tools.

### `dyld: Library not loaded: @rpath/libswift_Concurrency.dylib`

The runtime loader can't find Swift Concurrency. Apply one of the bundling options above.

### `Swift build failed: error: terminated(72)`

The Swift compilation step can't find the Xcode toolchain. Verify `xcode-select --print-path` returns a valid path.

### `Failed to compile CoreML model at ...`

The model path is invalid, the file is not a valid Core ML model, or the model requires a newer macOS version than your system supports.

### Model loads but predict fails silently

Check that your input shape matches the model's expected input shape exactly. Use `model.input_shapes()` to inspect.

### `experimentalMLE5EngineUsage` crashes

On newer macOS versions (15+), the experimental MLE5 engine can cause crashes. Set `disable_experimental_mle: true` in `CoreMLModelOptions`. Note: the batch model currently always disables this; the single model respects the option.
