# Owned FFI migration

This safety-fix branch intentionally changes the low-level `coreml_rs::ffi` API. It is not a release or a promise of complete memory-safety auditing.

## Compatibility

- `modelWithAssets` and `modelWithAssetsBatch` take an owned `Vec<u8>` and compute platform, not a raw pointer and length.
- Tensor binding methods take owned `Vec<f32>`, `Vec<i32>`, or `Vec<u16>` (f16 bits), without a pointer/length pair. Batch binding still takes an index.
- `MLType` is sealed to the eight supported element types. External implementations are no longer accepted.
- `MLArray::extract_to_tensor<T>` now panics on a dtype mismatch instead of reinterpreting an incompatible allocation. `try_extract_to_tensor<T>` returns the original array on mismatch.
- High-level loading/input/prediction method signatures remain unchanged. Non-contiguous ndarray inputs are serialized in logical order. Invalid output dtypes/dimensions and invalid batch indices are rejected.

## Ownership and performance

Rust vectors cross the existing swift-bridge owned-vector boundary. Swift copies model bytes into `Data` and retains that storage with its asset. Tensor values are copied into Core ML-owned arrays after checked dimension and element-count validation. Outputs are copied back through typed accessors; Core ML allocations are never reconstructed as Rust vectors.

These copies replace undocumented zero-copy ownership transfers and may increase latency and peak memory. Benchmark representative models before adopting this branch in performance-sensitive applications. Output buffer reuse remains future work.

## Loading and concurrency

The asynchronous asset-loading operation signals completion on both success and failure. Rust only transitions to `Loaded` after Swift reports successful loading. The stress harness uses a mutex around each model's full bind/predict transaction; it does not test unsynchronized sharing.

The existing `Send` implementations and other public API error paths still merit broader review. Passing focused tests does not prove general thread safety, arbitrary-model compatibility, GPU/ANE correctness, or performance equivalence.
