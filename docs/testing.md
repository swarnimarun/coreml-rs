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

These are validation commands, not a claim that the current branch has passed them. The [macOS workflow](../.github/workflows/ci.yml) runs these checks on a standard `macos-15` GitHub-hosted runner with read-only repository permissions, no model downloads, and a three-minute timeout on the test step. The two local reload tests and the unsafe stress harness remain ignored; a green workflow is not inference or concurrency coverage. The default integration tests include `load_empty`, which checks rejection of an empty model buffer. They do not establish successful inference, reload behavior, or thread safety. Ignored tests are still compiled by Cargo.

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

## Retry stress test: not a safe concurrency test yet

`tests/retry.rs` is ignored by default and looks for `target/saliency.mlpackage` and `target/vitae.mlpackage`. It returns early if no model is found, so a passing result without fixtures is not runtime coverage.

The current harness shares a model through `UnsafeCell` with unchecked `Send`/`Sync` implementations. Multiple workers can create overlapping mutable references. Do not run the entire ignored suite as a validation shortcut, or use this harness to claim thread safety. First serialize the input-binding/prediction pair or give each worker its own model. See the [concurrency roadmap item](roadmap.md#8-proper-concurrency-safety).

## Remaining release checks

Before treating the library as release-ready, add a small redistributable model and verify load, predict, unload, and reload on macOS. Cover failure paths with process-level timeouts so a Swift loading deadlock cannot hang the test suite indefinitely. Review dtype conversion and Rust/Swift buffer-ownership safety separately; these are not fixed by making local fixture tests opt-in.
