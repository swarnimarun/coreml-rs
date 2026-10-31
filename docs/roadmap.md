# Roadmap

Concrete, actionable improvements. Roughly ordered by impact.

## High Priority

### 1. Deduplicate state machine logic

`CoreMLModelWithState` (in `src/mlmodel.rs`) and `CoreMLBatchModelWithState` (in `src/mlbatchmodel.rs`) share ~90% identical load/unload/persist logic. The duplication makes bugs easy to miss and harder to fix.

- **Proposal**: Introduce a parametric `CoreMLEngine<S>` enum generic over a `ModelSession` trait with `load()`, `unload()`, `add_input()`, `predict()` methods.
- **Files**: `src/mlmodel.rs:90-318`, `src/mlbatchmodel.rs:19-216`, `src/mlmodel.rs:78-87`

### 2. Validate i32 input support across models

The Swift binder now constructs `.int32` arrays, rather than treating i32 bits as float32. Broader model-specific i32 compatibility still needs coverage.

### 3. Complete dtype support in inference pipeline

Only f32 inputs/outputs are fully wired. The current status:

| Dtype | Input (single) | Input (batch) | Output (single) | Output (batch) |
|-------|:---:|:---:|:---:|:---:|
| f32   | ✓   | ✓   | ✓   | ✓   |
| f16   | ✓   | ✗   | Partial (u16 reinterpret) | ✗   |
| i32   | ✓ | ✗ | ✗ | ✗ |
| i16   | ✗   | ✗   | ✗   | ✗   |
| i8    | ✗   | ✗   | ✗   | ✗   |
| u8    | ✗   | ✗   | ✗   | ✗   |
| u16   | ✗   | ✗   | ✗   | ✗   |
| u32   | ✗   | ✗   | ✗   | ✗   |

- Wire remaining `MLArray` variants in `add_input()` (`src/mlmodel.rs:463-523`)
- Add output type support beyond f32 in `bind_output_buffers()` (`src/mlmodel.rs:548-570`) and batch `predict()` (`src/mlbatchmodel.rs:305-349`)
- Add f16 output as a first-class type (not u16 reinterpret) in `run_bound_predict()` (`src/mlmodel.rs:590-593`)
- **Files**: `src/mlmodel.rs:463-523`, `src/mlmodel.rs:548-570`, `src/mlbatchmodel.rs:305-349`

## Medium Priority

### 4. Add retry support to batch model

`CoreMLBatchModelWithState` has no `predict_with_retry()` or `predict_with_retry_if()` methods. Retry is only available on the single model.

- **Proposal**: Mirror the retry logic from `CoreMLModel` into `CoreMLBatchModel` or (better) extract into a shared utility.
- **Files**: `src/mlbatchmodel.rs:210-215` (missing), compare with `src/mlmodel.rs:609-637`

### 5. Add shape introspection to batch model

`CoreMLBatchModelWithState` has `description()` but no `input_shapes()` or `output_shapes()` methods. The single model has both.

- **Proposal**: Add the same pattern as `CoreMLModel::input_shapes()` / `output_shapes()`.
- **Files**: `src/mlbatchmodel.rs:189-194` (missing), compare with `src/mlmodel.rs:647-668`

### 6. Fix inconsistent `disableExperimentalMLE` behavior

In `BatchModel.load()` (`swift-library/Sources/swift-library/swift_library.swift:66`), `experimentalMLE5EngineUsage` is **always** set to `1` (disabled), ignoring the `disableExperimentalMLE` flag. The single `Model.load()` (`swift_library.swift:500-502`) respects the flag.

- **Fix**: Add `self.disableExperimentalMLE` check to `BatchModel.load()`.
- **Files**: `swift-library/Sources/swift-library/swift_library.swift:66`, compare with `swift_library.swift:500-502`

### 7. Validate the owned FFI migration

The safety-fix branch replaces array transmutation with sealed typed conversions and owned-vector FFI transfers. See [Owned FFI migration](ffi-safety.md) for breaking low-level API changes and copy/performance tradeoffs. Benchmark and test representative models before adoption.

### 8. Proper concurrency safety

The retry harness now serializes each model's bind/predict pair with a mutex. This removes the harness's unchecked mutable aliasing; it does not establish thread safety of the underlying Core ML APIs. Review and document the remaining `unsafe impl Send` invariants in `src/mlmodel.rs` and `src/mlbatchmodel.rs` before promising broader concurrent use.

## Lower Priority

### 9. Output buffer reuse

`bind_output_buffers()` (`src/mlmodel.rs:548-570`) is called on every `predict()` and re-allocates output `Array::zeros()` for every output. For models with large outputs, this wastes memory and time.

- **Proposal**: Cache output arrays and reuse them across predict calls. Be careful about aliasing — the Swift side writes into these buffers.
- **File**: `src/mlmodel.rs:548-570`

### 10. Better error messages

Multiple error paths return `UnknownErrorStatic("failed to bind input to model")` with no information about which input, what shape, or why. The Swift side often prints to stdout/stderr via `print()` instead of surfacing structured errors.

- Capture the input name, shape, and dtype in bind failure errors
- Thread Swift-side error details through to Rust errors
- **Files**: `src/mlmodel.rs:442-523`, `swift-library/Sources/swift-library/swift_library.swift` (all `print()` calls)

### 11. CI/CD pipeline

A [macOS workflow](../.github/workflows/ci.yml) builds all targets, checks formatting and Clippy, and runs the fixture-free default tests. The two fixture-backed reload tests and the retry stress test are explicitly ignored by default. See [Testing](testing.md) for the macOS checks, required local fixtures, and stress-test coverage limits. Default tests now generate a tiny scalar ReLU model to validate single/batch CPU inference and reload; they do not establish arbitrary-model or GPU/ANE coverage.

- Expand the in-memory synthetic model tests to cover more dtypes, shapes, and cache paths
- Retain process-level timeouts on model-loading regression tests

### 12. Support non-MLMultiArray I/O

Core ML supports images, sequences, and dictionaries as feature types. Currently only `MLMultiArray` is supported.

- Add support for image inputs (CV pixel buffers) — relevant for most vision models
- Add support for string/sequence inputs (NLP models)
- **Files**: `swift-library/Sources/swift-library/swift_library.swift` (Model.dict always uses `MLFeatureValue(multiArray:)`)

### 13. Model compilation caching

When loading from a path (non-compiled), Core ML compiles the model on every `load()` call. The compiled path is extracted on `unload()` but only used within the same process lifetime. There's no persistent cache directory.

- **Proposal**: Cache compiled models to `options.cache_dir` and check on subsequent loads.
- **Files**: `src/mlmodel.rs:118-130`

### 14. API documentation

No docstrings exist on public types or functions. `CoreMLModelWithState`, `CoreMLModelOptions`, `MLArray`, `PredictRetryOptions`, etc. are undocumented.

- Add `///` doc comments to all public items
- Add module-level docs in `lib.rs`, `mlmodel.rs`, `mlbatchmodel.rs`, `mlarray.rs`
- Consider adding `#![warn(missing_docs)]` once docs exist

## Completed documentation cleanup

- The README and setup guide use the `0.5` dependency requirement, compatible with `Cargo.toml` version `0.5.4`, and point to this personal repository.
- Fixture-dependent reload tests are opt-in and their requirements are documented. This does not replace macOS runtime validation or a redistributable test model.
