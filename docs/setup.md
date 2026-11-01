# Setup

Use macOS 12.3 or later, Rust 1.88 or later, and Xcode or the Apple command line tools with a recent macOS SDK. The crate links the system Core ML, Core Video, Metal and Foundation frameworks; it does not require a Swift runtime.

```sh
rustc --version
xcrun --show-sdk-path
MACOSX_DEPLOYMENT_TARGET=12.3 cargo build --release
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Use `MACOSX_DEPLOYMENT_TARGET` when building your final executable; a dependency cannot set the deployment target of its consumer. Newer functionality is checked at runtime: in-memory model specifications, CPU + Neural Engine and display names require macOS 13; device enumeration requires macOS 14; compute plans require macOS 14.4; and in-memory ML Program blob mappings, stateful prediction and named model functions require macOS 15. The implementation targets 12.3, but the current integration runs have only been verified on macOS 26.3.1 on Apple silicon. Test your binary on every OS and architecture you intend to ship.

Core ML compilation and inference use macOS services. In restricted sandboxes they may fail even if the crate builds. The native integration tests require access to those services. The optional downloaded-model tests are described in [Examples](examples.md).
