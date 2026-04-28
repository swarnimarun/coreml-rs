# Roadmap

Concrete, actionable improvements. Roughly ordered by impact.

## High Priority

### 1. Deduplicate state machine logic

`CoreMLModelWithState` (in `src/mlmodel.rs`) and `CoreMLBatchModelWithState` (in `src/mlbatchmodel.rs`) share ~90% identical load/unload/persist logic. The duplication makes bugs easy to miss and harder to fix.

- **Proposal**: Introduce a parametric `CoreMLEngine<S>` enum generic over a `ModelSession` trait with `load()`, `unload()`, `add_input()`, `predict()` methods.
- **Files**: `src/mlmodel.rs:90-318`, `src/mlbatchmodel.rs:19-216`, `src/mlmodel.rs:78-87`

### 2. Fix `bindInputI32` creating the wrong MLMultiArray dtype (Swift bug)

In `swift_library.swift:630-635`, `bindInputI32` creates an `MLMultiArray` with `dataType: MLMultiArrayDataType.float32` instead of `.int32`. This means i32 inputs are silently cast to float, producing wrong results.

- **Fix**: Change `MLMultiArrayDataType.float32` to `.int32` on line 634.
- **File**: `swift-library/Sources/swift-library/swift_library.swift:630-635`

### 3. Complete dtype support in inference pipeline

Only f32 inputs/outputs are fully wired. The current status:

| Dtype | Input (single) | Input (batch) | Output (single) | Output (batch) |
|-------|:---:|:---:|:---:|:---:|
| f32   | ✓   | ✓   | ✓   | ✓   |
| f16   | ✓   | ✗   | Partial (u16 reinterpret) | ✗   |
| i32   | ✓ (buggy) | ✗ | ✗ | ✗ |
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

### 7. Safe MLArray conversions

`MLArray::extract_to_tensor()` and the `From` impls use `std::mem::transmute` between different `ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>` types. While all variants are `OwnedRepr` over the same dimension type (so layout is identical), this is still UB according to strict Rust rules because the type parameter differs.

- **Proposal**: Reimplement with safe `ptr::read`/`ManuallyDrop` or use `bytemuck` for the underlying buffer and `Array::from_shape_vec` to rebuild.
- **File**: `src/mlarray.rs:101-136`

### 8. Proper concurrency safety

Models use `unsafe impl Send` with no documentation of invariants. The retry stress test (`tests/retry.rs:355-374`) shares a model across threads via `UnsafeCell<CoreMLModelWithState>` without locks. This works today because `Model` (Swift side) is `@unchecked Sendable` and prediction calls happen sequentially in practice, but it's fragile.

- **Proposal**: Either:
  - Document that `CoreMLModelWithState` is `Send` but not `Sync`, and that callers must serialize access, or
  - Add `Arc<Mutex<>>` wrapper as a convenience type.
  - Investigate whether `MLModel.prediction(from:options:)` is thread-safe, and if so, make it `Sync`.
- **Files**: `src/mlmodel.rs:406`, `src/mlbatchmodel.rs:225`, `tests/retry.rs:355-374`

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

No CI configuration is present. Tests (`tests/load.rs`, `tests/retry.rs`) require local model files at `./demo/model.zip`, `./demo/model_3.mlmodel`, and `target/vitae.mlpackage`.

- Add a GitHub Actions workflow that:
  - Verifies `cargo build` passes on macOS
  - Runs unit tests with dummy model files (or a small test model checked into the repo)
  - Runs `cargo clippy` and `cargo fmt --check`
- Add a small test `.mlmodel` that exercises the load/predict/unload cycle

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

### 15. Fix README install version

The README shows `version = "0.4"` but `Cargo.toml` declares `version = "0.5.4"`. Update the README to match.
