# Testing

## Platform and toolchain

Build and runtime checks require macOS, Xcode with Swift 6.0 or newer, and stable Rust. The Swift package declares macOS 13 as its deployment target; model and API compatibility may require a newer OS. Follow the [setup guide](setup.md), including Swift runtime library discovery.

Linux cannot validate the Core ML runtime. Even `cargo check` runs `build.rs`, which builds the Swift library against Apple frameworks. A source review or formatting check is not a macOS build or inference test.

## Checks without local model fixtures

From the repository root on macOS:

```sh
cargo fmt --all -- --check
cargo build --locked --all-targets
cargo clippy --locked --all-targets
export DYLD_LIBRARY_PATH="$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx:/usr/lib/swift${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
cargo test --locked
```

These are validation commands, not a claim that the current branch has passed them. The [macOS workflow](../.github/workflows/ci.yml) runs these checks on a standard `macos-15` GitHub-hosted runner with read-only repository permissions, no model downloads, and a three-minute timeout on the test step. The two local-fixture reload tests and the opt-in stress harness remain ignored. The default suite includes `load_empty` plus `ffi_owned`, which independently generates a tiny scalar ReLU model in memory. Its subprocess-isolated tests cover malformed buffers, single/batch CPU prediction and reload, and rejected tensor shapes/indices. Each subprocess has a 30-second deadline. This is focused synthetic coverage, not arbitrary-model, GPU/ANE, performance, or thread-safety validation. Ignored tests are still compiled by Cargo.

## Opt-in reload tests

The repository does not include or download model fixtures. Supply your own authorized local files; do not commit private models. The `demo/` directory is gitignored.

| Test | Required fixture |
| --- | --- |
| `reload_from_compiled_path` | `demo/model.zip`, containing `model.mlpackage` at the archive root |
| `reload_from_buf` | `demo/model_3.mlmodel`, a valid Core ML model specification |

Run each explicitly from the repository root:

```sh
cargo test --locked --test load reload_from_compiled_path -- --ignored --exact
cargo test --locked --test load reload_from_buf -- --ignored --exact
```

The archive test uses a shared hash-based directory under `/tmp/coreml-aftershoot/` and removes its extracted package. Avoid concurrent runs of that test. Neither test checks prediction output values.

## Retry stress test

`tests/retry.rs` is ignored by default and looks for `target/saliency.mlpackage` and `target/vitae.mlpackage`. It returns early if no model is found, so a passing result without fixtures is not runtime coverage.

The harness now locks each model for the full input-binding/prediction pair. Workers cannot alias mutable references to one shared model; different models may still run concurrently. This is not evidence that Core ML allows unsynchronized access to the same model. Do not run the entire ignored suite as a validation shortcut: provision authorized fixtures and select the intended test explicitly.

## Remaining release checks

Before treating the library as release-ready, validate representative authorized models, all supported input/output dtypes, GPU/ANE behavior, and performance. The synthetic model does not replace these checks. Review the [owned FFI compatibility changes](ffi-safety.md), remaining `Send` invariants, and cached-on-disk model paths separately.
